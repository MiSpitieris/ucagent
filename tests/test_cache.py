import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ucagent.core.llm.backend import get_chat_model
from ucagent.core.llm.cache import CachedChatModel, SqliteCache, cache_key, model_tag
from ucagent.core.llm.guard import CostGuard, CostGuardExceeded, price_for


class Counting(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        assert "sample_index" not in kwargs  # must be stripped by the wrapper
        self.calls += 1
        text = f"reply#{self.calls} to {messages[-1].content}"
        msg = AIMessage(
            content=text,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(tmp_path, **kw) -> CachedChatModel:
    return get_chat_model(
        "test:model", 0.5, cache_path=tmp_path / "c.sqlite", inner=Counting(), **kw
    )


def test_hit_miss_and_sample_index_keys(tmp_path) -> None:
    m = _model(tmp_path)
    msgs = [HumanMessage(content="hi")]
    a = m.invoke(msgs, sample_index=0)
    b = m.invoke(msgs, sample_index=0)
    c = m.invoke(msgs, sample_index=1)
    assert a.content == b.content and a.content != c.content
    assert a.response_metadata["cache"] == "miss" and b.response_metadata["cache"] == "hit"
    assert m.inner.calls == 2 and len(m.store) == 2
    assert cache_key("x", msgs, 0.5, 0) != cache_key("x", msgs, 0.6, 0)
    assert (
        m.guard is not None
        and m.guard.calls == 2
        and m.guard.usd == pytest.approx((10 * 3 + 5 * 15) * 2 / 1e6)
    )


def test_rerun_from_cache_makes_zero_calls(tmp_path) -> None:
    m = _model(tmp_path)
    for k in range(5):
        m.invoke([HumanMessage(content="q")], sample_index=k)
    again = _model(tmp_path)  # fresh model and guard, same cache file
    for k in range(5):
        again.invoke([HumanMessage(content="q")], sample_index=k)
    assert again.inner.calls == 0 and again.guard is not None and again.guard.calls == 0
    assert again.store.total_calls() == 5  # shared counter persisted


def test_threads_share_one_file(tmp_path) -> None:
    m = _model(tmp_path, max_calls=10_000)

    def work(k: int) -> str:
        return str(m.invoke([HumanMessage(content=f"q{k % 20}")], sample_index=k // 20).content)

    with ThreadPoolExecutor(16) as pool:
        out = list(pool.map(work, range(200)))
    assert len(set(out)) == 200 and len(m.store) == 200 and m.inner.calls == 200


def test_processes_share_one_file(tmp_path) -> None:
    path = tmp_path / "p.sqlite"
    code = f"""
from ucagent.core.llm.cache import SqliteCache
import sys
c = SqliteCache({str(path)!r})
for k in range(50):
    c.put(f"{{sys.argv[1]}}-{{k}}", model="m", temperature=0.0, sample_index=k,
          messages="x", response="y")
    c.increment_calls()
"""
    procs = [subprocess.Popen([sys.executable, "-c", code, str(i)]) for i in range(4)]
    assert all(p.wait() == 0 for p in procs)
    cache = SqliteCache(path)
    assert len(cache) == 200 and cache.total_calls() == 200


def test_export_import_round_trip(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    m = _model(tmp_path)
    for k in range(3):
        m.invoke([HumanMessage(content="e")], sample_index=k)
    root = Path(__file__).resolve().parents[1] / "scripts"

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    out = tmp_path / "dump.jsonl"
    assert load("export_cache").main(str(tmp_path / "c.sqlite"), str(out)) == 3
    assert len(out.read_text().splitlines()) == 3
    assert load("import_cache").main(str(out), str(tmp_path / "other.sqlite")) == 3
    other = SqliteCache(tmp_path / "other.sqlite")
    assert len(other) == 3
    row = json.loads(out.read_text().splitlines()[0])
    assert other.get(row["key"])["response"] == row["response"]


def test_cost_guard_limits(tmp_path) -> None:
    m = _model(tmp_path, max_calls=3)
    for k in range(3):
        m.invoke([HumanMessage(content="g")], sample_index=k)
    with pytest.raises(CostGuardExceeded, match="UCAGENT_MAX_LLM_CALLS=3"):
        m.invoke([HumanMessage(content="g")], sample_index=99)
    m.invoke([HumanMessage(content="g")], sample_index=0)  # cache hits are free
    usd = CostGuard(max_calls=10, max_usd=0.0001, model_name="anthropic:claude-opus-5-5")
    usd.charge(1000, 1000)
    with pytest.raises(CostGuardExceeded, match="USD"):
        usd.check()
    assert price_for("ollama:llama3.1:8b") == (0.0, 0.0)
    assert price_for("anthropic:claude-sonnet-5-5") == (3.0, 15.0)
    assert price_for("unknown:thing") == (3.0, 15.0)


def test_backend_uses_init_chat_model(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def fake(model: str, **kwargs: Any) -> BaseChatModel:
        seen.update(model=model, **kwargs)
        return Counting()

    import langchain.chat_models as lc

    monkeypatch.setattr(lc, "init_chat_model", fake)
    m = get_chat_model("ollama:llama3.1:8b", 0.3, cache_path=tmp_path / "b.sqlite")
    assert seen == {"model": "ollama:llama3.1:8b", "temperature": 0.3}
    assert m.model_name == "ollama:llama3.1:8b" and m.model_kwargs == {}
    seen.clear()
    extra = {"reasoning": "low", "num_ctx": 8192}
    m = get_chat_model(
        "ollama:gpt-oss:20b", 0.7, cache_path=tmp_path / "b.sqlite", model_kwargs=extra
    )
    assert seen == {"model": "ollama:gpt-oss:20b", "temperature": 0.7, **extra}
    assert m.model_kwargs == extra and m.model_name == "ollama:gpt-oss:20b"


def test_model_kwargs_in_key_only_when_present(tmp_path) -> None:
    import sqlite3

    msgs = [HumanMessage(content="q")]
    assert model_tag("x", {}) == "x" and model_tag("x", None) == "x"
    tag = model_tag("x", {"num_ctx": 8192, "reasoning": "low"})
    assert tag == model_tag("x", {"reasoning": "low", "num_ctx": 8192})  # order-insensitive
    assert tag.startswith("x ") and cache_key(tag, msgs, 0.5, 0) != cache_key("x", msgs, 0.5, 0)
    plain = _model(tmp_path)
    tuned = _model(tmp_path, model_kwargs={"num_ctx": 8192})
    plain.invoke(msgs, sample_index=0)
    tuned.invoke(msgs, sample_index=0)
    assert plain.inner.calls == 1 and tuned.inner.calls == 1  # distinct keys, both miss once
    assert plain.invoke(msgs, sample_index=0).response_metadata["cache"] == "hit"
    rows = sqlite3.connect(tmp_path / "c.sqlite").execute("select model from responses").fetchall()
    assert sorted(r[0] for r in rows) == sorted(["test:model", 'test:model {"num_ctx":8192}'])


def test_truncated_or_blank_reply_warns(tmp_path, caplog) -> None:
    class Truncating(Counting):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
            msg = AIMessage(content="Answer: Par", response_metadata={"done_reason": "length"})
            return ChatResult(generations=[ChatGeneration(message=msg)])

    class Blank(Counting):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="  "))])

    import logging

    with caplog.at_level(logging.WARNING, logger="ucagent"):
        m = get_chat_model("t:m", 0.5, cache_path=tmp_path / "w.sqlite", inner=Truncating())
        assert m.invoke([HumanMessage(content="q")], sample_index=3).content == "Answer: Par"
        m2 = get_chat_model("t:m", 0.5, cache_path=tmp_path / "w.sqlite", inner=Blank())
        m2.invoke([HumanMessage(content="r")], sample_index=0)
    texts = [r.getMessage() for r in caplog.records]
    assert any("truncated" in t and "sample 3" in t for t in texts)
    assert any("blank" in t for t in texts)


@pytest.mark.llm
def test_real_ollama_small_run_and_cached_rerun(tmp_path, monkeypatch) -> None:
    """Needs UCAGENT_LLM_TESTS=1 and a running Ollama with llama3.1:8b."""
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:  # pragma: no cover
        pytest.skip("Ollama not reachable")
    from tests.helpers.configs import small_config
    from ucagent.experiments.runner import run_experiment

    monkeypatch.setenv("UCAGENT_CACHE_PATH", str(tmp_path / "real.sqlite"))
    cfg = small_config(
        run_id="llm-smoke",
        task={"n_steps": 2, "ambiguity": 1.0, "n_instances": {"test": 2}},
        llm={"backend": "ollama:llama3.1:8b", "K": 2, "K_prime": 4, "temperature": 0.7},
        limits={"max_llm_calls": 50, "max_concurrency": 1},
    )
    first = run_experiment(cfg, workers=1, write=False)
    assert len(first.trajectories) == 2 and all(t.llm_calls == 4 for t in first.trajectories)
    cache = SqliteCache(tmp_path / "real.sqlite")
    calls_after_first = cache.total_calls()
    assert calls_after_first == 8
    second = run_experiment(cfg, workers=1, write=False)
    assert cache.total_calls() == calls_after_first  # zero new real calls
    assert [t.final_answer for t in second.trajectories] == [
        t.final_answer for t in first.trajectories
    ]
