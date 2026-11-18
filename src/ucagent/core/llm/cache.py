"""Sqlite response cache keyed by (model, messages, temperature, sample_index).

``sample_index`` is part of the key, so ``K`` distinct samples are cacheable and
a rerun is deterministic and makes zero API calls. One connection per
``(pid, thread)``, WAL journal, ``INSERT OR IGNORE`` writes, so threads and
worker processes can share one file. ``scripts/export_cache.py`` and
``import_cache.py`` move the rows as JSONL for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from ucagent.core.llm.guard import CostGuard

log = logging.getLogger("ucagent")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  key TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  temperature REAL NOT NULL,
  sample_index INTEGER NOT NULL,
  messages TEXT NOT NULL,
  response TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


def canonical_messages(messages: list[BaseMessage]) -> str:
    """Deterministic JSON of the prompt (type and content only)."""
    return json.dumps(
        [{"type": m.type, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )


def cache_key(
    model: str, messages: list[BaseMessage], temperature: float, sample_index: int
) -> str:
    """Sha256 over the canonical request."""
    payload = f"{model}\x1f{temperature!r}\x1f{sample_index}\x1f{canonical_messages(messages)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_tag(model: str, model_kwargs: Mapping[str, Any] | None = None) -> str:
    """``model`` alone, or ``model`` plus the canonical JSON of its extra provider kwargs.

    Empty kwargs leave the tag (and so every existing cache key) unchanged;
    non-empty kwargs (a thinking mode, a context size, an output cap) change
    the sampled text, so they are part of the key and of the stored row.
    """
    if not model_kwargs:
        return model
    extra = json.dumps(dict(model_kwargs), sort_keys=True, separators=(",", ":"))
    return f"{model} {extra}"


def _enable_wal(conn: sqlite3.Connection, attempts: int = 50) -> None:
    """Switch the file to WAL mode, retrying while another process holds the lock.

    The switch needs a moment of exclusive access and SQLite's busy handler
    does not cover it, so several processes opening a fresh cache file at the
    same instant can see "database is locked" here; a short bounded retry
    settles it (the mode is persistent once any of them succeeds).
    """
    for attempt in range(attempts):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


class SqliteCache:
    """The cache file; safe for threads and processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        pid = os.getpid()
        conn = getattr(self._local, "conn", None)
        if conn is None or getattr(self._local, "pid", None) != pid:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.execute("PRAGMA busy_timeout=30000")
            _enable_wal(conn)
            self._local.conn = conn
            self._local.pid = pid
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        """Cached response row or None."""
        row = (
            self._conn()
            .execute(
                "SELECT response, input_tokens, output_tokens FROM responses WHERE key = ?", (key,)
            )
            .fetchone()
        )
        if row is None:
            return None
        return {"response": row[0], "input_tokens": row[1], "output_tokens": row[2]}

    def put(
        self,
        key: str,
        *,
        model: str,
        temperature: float,
        sample_index: int,
        messages: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Insert (ignored if the key exists)."""
        self._conn().execute(
            "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
            (
                key,
                model,
                temperature,
                sample_index,
                messages,
                response,
                input_tokens,
                output_tokens,
                time.time(),
            ),
        )

    def increment_calls(self) -> None:
        """Bump the shared real-call counter (used by the cost guard)."""
        self._conn().execute(
            "INSERT INTO counters(name, value) VALUES ('calls', 1) "
            "ON CONFLICT(name) DO UPDATE SET value = value + 1"
        )

    def total_calls(self) -> int:
        """Shared real-call count."""
        row = self._conn().execute("SELECT value FROM counters WHERE name = 'calls'").fetchone()
        return int(row[0]) if row else 0

    def reset_calls(self) -> None:
        """Zero the shared counter (start of a run)."""
        self._conn().execute("DELETE FROM counters WHERE name = 'calls'")

    def __len__(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def rows(self) -> Iterator[dict[str, Any]]:
        """All rows as dicts (for export)."""
        cur = self._conn().execute("SELECT * FROM responses ORDER BY created_at")
        names = [d[0] for d in cur.description]
        for row in cur:
            yield dict(zip(names, row, strict=True))

    def import_row(self, row: dict[str, Any]) -> None:
        """Insert an exported row (ignored if present)."""
        self._conn().execute(
            "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row["key"],
                row["model"],
                row["temperature"],
                row["sample_index"],
                row["messages"],
                row["response"],
                row.get("input_tokens", 0),
                row.get("output_tokens", 0),
                row.get("created_at", time.time()),
            ),
        )


class CachedChatModel(BaseChatModel):
    """Wraps any chat model with the sqlite cache and the cost guard.

    ``sample_index`` (a model kwarg set by the graph's ``sample`` node) is part
    of the cache key and is stripped before the inner model is called.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseChatModel
    store: SqliteCache
    model_name: str
    temperature: float = 0.7
    guard: CostGuard | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return f"cached({self.model_name})"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        sample_index = int(kwargs.pop("sample_index", 0))
        tag = model_tag(self.model_name, self.model_kwargs)
        key = cache_key(tag, messages, self.temperature, sample_index)
        hit = self.store.get(key)
        if hit is not None:
            message = AIMessage(content=hit["response"], response_metadata={"cache": "hit"})
            return ChatResult(generations=[ChatGeneration(message=message)])
        if self.guard is not None:
            self.guard.check()
        reply = self.inner.invoke(messages, stop=stop, **kwargs)
        content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
        done_reason = (getattr(reply, "response_metadata", None) or {}).get("done_reason")
        if done_reason == "length" or not content.strip():
            log.warning(
                "reply from %s (sample %d) is %s; cached as is",
                tag,
                sample_index,
                "truncated (done_reason=length)" if done_reason == "length" else "blank",
            )
        usage = getattr(reply, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        if self.guard is not None:
            self.guard.charge(input_tokens, output_tokens)
        self.store.put(
            key,
            model=tag,
            temperature=self.temperature,
            sample_index=sample_index,
            messages=canonical_messages(messages),
            response=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        message = AIMessage(content=content, response_metadata={"cache": "miss"})
        return ChatResult(generations=[ChatGeneration(message=message)])
