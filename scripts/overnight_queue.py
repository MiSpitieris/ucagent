"""Unattended queue of experiment stages: health checks, call budgets, a watchdog, retries.

Usage::

    <python> scripts/overnight_queue.py results/queue/queue.yaml

The queue file is re-read before every stage, so stages can be appended while the queue
runs. Each stage runs one command in a frozen copy of the code (``dir`` pinned to
``commit``) with ``PYTHONPATH=<dir>/src``. Before a stage: the deadline, free disk space,
the model server (``suite`` stages) and the remaining call budget are checked. During a
stage: the lifetime call counter and the stage log are watched; a ``suite`` stage with no
progress for ``stall_minutes`` is killed (whole process group) and retried, an ``offline``
stage has a timeout. After a failure the log tail decides: transient errors wait for the
server and retry with backoff; a corrupt result file is renamed aside (never deleted) and
retried once; a full disk pauses; a budget or configuration error is final; anything else
is retried once and then final. A final failure writes ``ALERT`` and skips the stages that
depend on it; independent stages continue. At the end ``summary.md`` and ``DONE`` are
written. The script uses only the standard library and PyYAML, so it never imports the
code it runs.

Queue file keys: ``cap``, ``deadline`` (ISO time; no ``suite`` stage starts after it),
``min_free_gb``, ``stall_minutes``, ``poll_seconds``, ``offline_timeout_minutes``,
``health_url`` (null skips the check), ``model``, ``cache_db``, ``python_bin`` (prepended
to ``PATH``), ``state_dir``, ``backoff_seconds``, and ``stages``: each with ``id``,
``dir``, ``commit`` (null skips the pin check), ``kind`` (``suite`` or ``offline``),
``expected_calls``, ``retries``, ``after`` (ids it depends on) and ``cmd`` (a list).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

FINAL = ("done", "failed", "skipped")
CLASSES: list[tuple[str, re.Pattern[str]]] = [
    ("budget", re.compile(r"CostGuardExceeded|cost guard:")),
    ("config", re.compile(r"ResumeMismatch")),
    ("disk", re.compile(r"No space left|database or disk is full|disk I/O error|OperationalError")),
    (
        "corrupt",
        re.compile(
            r"ArrowInvalid|ArrowIOError|Parquet magic bytes|deserialize thrift|JSONDecodeError"
        ),
    ),
    (
        "transient",
        re.compile(
            r"ConnectError|ConnectionError|ConnectionRefused|ReadError|ReadTimeout|"
            r"RemoteProtocolError|ResponseError|BrokenProcessPool|Server disconnected",
        ),
    ),
]
CORRUPT_PATH = re.compile(r"(results/[^\s'\"]+?(?:/trajectories\.parquet|\.json))")
DEFAULTS: dict[str, Any] = {
    "cap": 0,
    "deadline": None,
    "min_free_gb": 2.0,
    "stall_minutes": 20.0,
    "poll_seconds": 30.0,
    "offline_timeout_minutes": 45.0,
    "health_url": "http://localhost:11434/api/tags",
    "model": "gpt-oss:20b",
    "cache_db": "cache/llm_cache.sqlite",
    "python_bin": None,
    "state_dir": "results/queue",
    "backoff_seconds": [60, 300, 900, 1800],
}
_stop = False


def _stopped() -> bool:
    """True once SIGTERM or SIGINT was received (read through a call so it is never cached)."""
    return _stop


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Queue:
    """The queue file, its state and its logs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conf = self.load()
        self.state_dir = Path(self.conf["state_dir"])
        (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.state: dict[str, dict[str, Any]] = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {}
        )

    def load(self) -> dict[str, Any]:
        """Re-read the queue file (stages may have been appended)."""
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        conf = {**DEFAULTS, **{k: v for k, v in data.items() if k != "stages"}}
        conf["stages"] = list(data.get("stages") or [])
        self.conf = conf
        return conf

    def log(self, message: str) -> None:
        """Append one line to ``queue.log`` and echo it."""
        line = f"{_now()} {message}"
        with open(self.state_dir / "queue.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def save(self) -> None:
        """Write the state atomically."""
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def entry(self, stage_id: str) -> dict[str, Any]:
        """The state record of a stage (created on first use)."""
        return self.state.setdefault(stage_id, {"status": "pending", "attempts": 0, "history": []})

    def alert(self, stage_id: str, cls: str, note: str) -> None:
        """Record a final failure for a human (the watcher wakes on this file)."""
        with open(self.state_dir / "ALERT", "a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {stage_id} {cls}: {note}\n")
        self.log(f"ALERT {stage_id} {cls}: {note}")


def counter(db: str) -> int:
    """The lifetime call counter of the response cache (read-only)."""
    for _ in range(5):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30) as con:
                row = con.execute("select value from counters").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            time.sleep(2)
    raise RuntimeError(f"cannot read the call counter from {db}")


def free_gb(path: Path) -> float:
    """Free space of the filesystem holding ``path``, in GB."""
    return shutil.disk_usage(path).free / 1e9


def server_ok(url: str | None, model: str) -> bool:
    """The model server answers and lists the model (``url`` None: not checked)."""
    if not url:
        return True
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return any(m.get("name", "").startswith(model) for m in data.get("models", []))
    except Exception:
        return False


def code_pinned(stage: dict[str, Any]) -> str | None:
    """None when the stage's directory is at its commit with clean tracked files."""
    commit = stage.get("commit")
    if not commit:
        return None
    where = str(stage["dir"])
    head = subprocess.run(
        ["git", "-C", where, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not head.startswith(str(commit)):
        return f"HEAD {head[:12]} is not the pinned {str(commit)[:12]}"
    dirty = subprocess.run(
        ["git", "-C", where, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"tracked files changed: {dirty.splitlines()[:3]}" if dirty else None


def classify(log_path: Path, stalled: bool, timed_out: bool) -> str:
    """Failure class from the last lines of the stage log."""
    if stalled:
        return "stall"
    if timed_out:
        return "timeout"
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
    text = "\n".join(tail)
    for name, pattern in CLASSES:
        if pattern.search(text):
            return name
    return "unknown"


def quarantine(log_path: Path, stage_dir: Path) -> list[str]:
    """Rename the run directory or artifact named in the log tail to ``*.corrupt-<time>``."""
    text = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    moved: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for rel in dict.fromkeys(CORRUPT_PATH.findall(text)):
        target = stage_dir / rel
        if target.name == "trajectories.parquet":
            target = target.parent
        if not target.exists():
            continue
        resolved = target.resolve()
        results_root = (stage_dir / "results").resolve()
        if results_root not in resolved.parents:
            continue
        aside = target.with_name(f"{target.name}.corrupt-{stamp}")
        target.rename(aside)
        moved.append(str(aside))
    return moved


def kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop a stage and every worker it started."""
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end and not _stopped():
        time.sleep(min(5.0, max(0.05, end - time.time())))


def _past(deadline: str | None) -> bool:
    return bool(deadline) and datetime.now() >= datetime.fromisoformat(str(deadline))


def wait_until(q: Queue, check: Any, what: str, poll: float, deadline: str | None) -> bool:
    """Poll ``check`` until it holds; False if the deadline passes or the queue is stopped."""
    warned = False
    while not check():
        if _stopped() or _past(deadline):
            return False
        if not warned:
            q.log(f"WAIT {what}")
            warned = True
        _wait(poll)
    if warned:
        q.log(f"RESUME {what}")
    return True


def next_stage(q: Queue) -> dict[str, Any] | None:
    """The first non-final stage whose dependencies are done (marks blocked ones skipped)."""
    stages = q.conf["stages"]
    ids = {s["id"] for s in stages}
    for stage in stages:
        rec = q.entry(stage["id"])
        if rec["status"] in FINAL:
            continue
        deps = list(stage.get("after") or [])
        missing = [d for d in deps if d not in ids]
        bad = [d for d in deps if d in ids and q.entry(d)["status"] in ("failed", "skipped")]
        if missing or bad:
            rec["status"], rec["note"] = "skipped", f"dependency {missing or bad} not done"
            q.log(f"SKIP {stage['id']}: {rec['note']}")
            q.save()
            continue
        if all(q.entry(d)["status"] == "done" for d in deps):
            ready: dict[str, Any] = stage
            return ready
    return None


def run_stage(q: Queue, stage: dict[str, Any]) -> None:
    """Run one stage to a final status (or leave it pending if the queue is stopped)."""
    conf, sid = q.conf, stage["id"]
    rec = q.entry(sid)
    kind = stage.get("kind", "suite")
    retries = int(stage.get("retries", 4 if kind == "suite" else 1))
    stage_dir = Path(stage["dir"])
    corrupt_tries = 0
    while not _stopped():
        if kind == "suite" and q.state.get("_flags", {}).get("cap"):
            rec["status"], rec["note"] = "skipped", "lifetime cap reached"
            break
        if kind == "suite" and _past(conf["deadline"]):
            rec["status"], rec["note"] = "skipped", "deadline passed"
            break
        if not wait_until(
            q,
            lambda: free_gb(q.state_dir) >= float(conf["min_free_gb"]),
            f"disk below {conf['min_free_gb']} GB free",
            600,
            conf["deadline"],
        ):
            rec["status"], rec["note"] = (
                ("pending", "stopped") if _stopped() else ("skipped", "disk")
            )
            break
        pin = code_pinned(stage)
        if pin:
            rec["status"], rec["note"] = "failed", f"code pin: {pin}"
            q.alert(sid, "code", pin)
            break
        before = counter(conf["cache_db"])
        env = dict(os.environ)
        if conf.get("python_bin"):
            env["PATH"] = f"{conf['python_bin']}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = str(stage_dir / "src")
        if kind == "suite":
            if not wait_until(
                q,
                lambda: server_ok(conf["health_url"], conf["model"]),
                "model server",
                120,
                conf["deadline"],
            ):
                rec["status"], rec["note"] = (
                    ("pending", "stopped") if _stopped() else ("skipped", "server")
                )
                break
            expected = int(stage.get("expected_calls", 0))
            remaining = int(conf["cap"]) - before
            if remaining < 1.2 * expected or remaining <= 0:
                rec["status"] = "skipped"
                rec["note"] = f"budget: {remaining} calls left, {expected} expected"
                break
            budget = min(int(conf["cap"]), before + math.ceil(1.5 * expected) + 100)
        else:
            budget = before  # offline: any model call fails
        env["UCAGENT_MAX_LLM_CALLS"] = str(budget)
        rec["attempts"] += 1
        attempt = rec["attempts"]
        log_path = q.state_dir / "logs" / f"{sid}.{attempt}.log"
        rec["status"], rec["log"] = "running", str(log_path)
        q.save()
        q.log(f"START {sid} attempt {attempt} counter {before} call limit {budget}")
        started = time.time()
        with open(log_path, "wb") as out:
            proc = subprocess.Popen(
                [str(c) for c in stage["cmd"]],
                cwd=stage_dir,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_progress, last_count, last_size = time.time(), before, 0
            stalled = timed_out = False
            while proc.poll() is None:
                _wait(float(conf["poll_seconds"]))
                if _stopped():
                    kill_group(proc)
                    break
                count, size = counter(conf["cache_db"]), log_path.stat().st_size
                if count != last_count or size != last_size:
                    last_progress, last_count, last_size = time.time(), count, size
                idle = time.time() - last_progress
                if kind == "suite" and idle > 60 * float(conf["stall_minutes"]):
                    q.log(f"STALL {sid}: no progress for {idle / 60:.1f} min; killing")
                    stalled = True
                    kill_group(proc)
                    break
                if kind != "suite" and time.time() - started > 60 * float(
                    conf["offline_timeout_minutes"]
                ):
                    q.log(f"TIMEOUT {sid}")
                    timed_out = True
                    kill_group(proc)
                    break
            rc = proc.wait()
        after = counter(conf["cache_db"])
        hist = {"attempt": attempt, "rc": rc, "calls": after - before}
        hist["seconds"] = round(time.time() - started)
        if _stopped():
            rec["status"], rec["note"] = "pending", "stopped"
            rec["history"].append({**hist, "class": "stopped"})
            break
        if rc == 0 and not stalled and not timed_out:
            rec.update(status="done", note="", calls=rec.get("calls", 0) + after - before)
            rec["history"].append({**hist, "class": "ok"})
            q.log(f"DONE {sid} calls {after - before} in {hist['seconds']} s")
            break
        cls = classify(log_path, stalled, timed_out)
        rec["history"].append({**hist, "class": cls})
        rec["calls"] = rec.get("calls", 0) + after - before
        q.log(f"FAIL {sid} attempt {attempt}: rc {rc}, class {cls}")
        if cls == "budget":
            if after >= int(conf["cap"]):
                q.state.setdefault("_flags", {})["cap"] = True
            rec["status"], rec["note"] = "failed", f"call budget reached (counter {after})"
            q.alert(sid, cls, rec["note"])
            break
        if cls == "config":
            rec["status"], rec["note"] = "failed", "resume mismatch: needs a new suite id"
            q.alert(sid, cls, rec["note"])
            break
        if cls == "corrupt":
            moved = quarantine(log_path, stage_dir)
            corrupt_tries += 1
            if moved and corrupt_tries <= 1:
                q.log(f"QUARANTINE {sid}: {moved}")
                continue
            rec["status"], rec["note"] = "failed", f"corrupt file, quarantined {moved or 'nothing'}"
            q.alert(sid, cls, rec["note"])
            break
        if cls == "disk":
            continue  # the disk check at the top of the loop pauses until space returns
        max_attempts = retries if cls in ("transient", "stall") else 2  # others: retry once
        if attempt - corrupt_tries >= max_attempts:
            rec["status"], rec["note"] = "failed", f"{cls} after {attempt} attempts"
            q.alert(sid, cls, f"{rec['note']}; see {log_path}")
            break
        backoff = list(conf["backoff_seconds"]) or [0]
        pause = backoff[min(attempt - 1, len(backoff) - 1)]
        q.log(f"RETRY {sid} in {pause} s")
        _wait(pause)
    q.save()


def summary(q: Queue) -> None:
    """Write ``summary.md``: one row per stage."""
    lines = [
        "| stage | status | attempts | calls | note |",
        "|---|---|---|---|---|",
    ]
    for stage in q.conf["stages"]:
        rec = q.entry(stage["id"])
        lines.append(
            f"| {stage['id']} | {rec['status']} | {rec['attempts']} | {rec.get('calls', 0)} | "
            f"{rec.get('note', '')} |"
        )
    (q.state_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def acquire_lock(state_dir: Path) -> Path:
    """Refuse to run beside a live queue; take over a stale lock."""
    lock = state_dir / "LOCK"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip() or 0)
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale
        else:
            raise SystemExit(f"another queue (pid {pid}) holds {lock}")
    lock.write_text(str(os.getpid()))
    return lock


def _on_signal(signum: int, frame: Any) -> None:
    global _stop
    _stop = True


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    args = parser.parse_args()
    q = Queue(args.queue)
    try:
        lock = acquire_lock(q.state_dir)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        sys.exit(3)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    q.log(f"QUEUE start pid {os.getpid()}")
    try:
        while not _stopped():
            q.load()
            stage = next_stage(q)
            if stage is None:
                break
            run_stage(q, stage)
        summary(q)
        if not _stopped():
            (q.state_dir / "DONE").write_text(_now() + "\n")
            q.log("QUEUE done")
        else:
            q.log("QUEUE stopped; rerun to resume")
    except Exception as exc:
        q.alert("queue", "crash", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        q.save()
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
