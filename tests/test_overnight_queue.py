"""Fault injection for scripts/overnight_queue.py: fake stages, a temporary counter, no model."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "overnight_queue.py"


def _setup(tmp_path: Path, stages: list[dict], **conf) -> Path:
    db = tmp_path / "cache.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("create table if not exists counters (value integer)")
        if con.execute("select count(*) from counters").fetchone()[0] == 0:
            con.execute("insert into counters values (0)")
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True, exist_ok=True)
    for st in stages:
        st.setdefault("dir", str(work))
        st.setdefault("commit", None)
        st.setdefault("kind", "offline")
    queue = {
        "cap": 1000,
        "deadline": None,
        "min_free_gb": 0,
        "stall_minutes": 0.02,
        "poll_seconds": 0.1,
        "offline_timeout_minutes": 0.2,
        "health_url": None,
        "cache_db": str(db),
        "state_dir": str(tmp_path / "queue"),
        "backoff_seconds": [0],
        **conf,
        "stages": stages,
    }
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue), encoding="utf-8")
    return path


def _run(queue: Path, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(queue)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "queue" / "state.json").read_text(encoding="utf-8"))


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_success_transient_retry_and_summary(tmp_path: Path) -> None:
    flaky = tmp_path / "flaky.txt"
    code = (
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(flaky)!r}); n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "if n < 2:\n    print('httpx.ConnectError: [Errno 111] Connection refused'); sys.exit(1)\n"
    )
    queue = _setup(
        tmp_path,
        [
            {"id": "ok", "cmd": _py("print('hello')")},
            {"id": "flaky", "kind": "suite", "expected_calls": 10, "retries": 4, "cmd": _py(code)},
        ],
    )
    result = _run(queue)
    assert result.returncode == 0, result.stderr
    state = _state(tmp_path)
    assert state["ok"]["status"] == "done" and state["flaky"]["status"] == "done"
    assert [h["class"] for h in state["flaky"]["history"]] == ["transient", "transient", "ok"]
    assert (tmp_path / "queue" / "DONE").exists() and not (tmp_path / "queue" / "ALERT").exists()
    assert "| flaky | done | 3 |" in (tmp_path / "queue" / "summary.md").read_text()


def test_stall_is_killed_and_retried_then_alerts(tmp_path: Path) -> None:
    queue = _setup(
        tmp_path,
        [
            {
                "id": "stuck",
                "kind": "suite",
                "expected_calls": 10,
                "retries": 2,
                "cmd": _py("import time; time.sleep(60)"),
            }
        ],
    )
    result = _run(queue)
    assert result.returncode == 0, result.stderr
    state = _state(tmp_path)
    assert state["stuck"]["status"] == "failed"
    assert [h["class"] for h in state["stuck"]["history"]] == ["stall", "stall"]
    assert "stuck stall" in (tmp_path / "queue" / "ALERT").read_text()


def test_budget_failure_is_final_and_skips_dependants(tmp_path: Path) -> None:
    queue = _setup(
        tmp_path,
        [
            {
                "id": "spend",
                "kind": "suite",
                "expected_calls": 10,
                "cmd": _py(
                    "import sys; print('CostGuardExceeded: cost guard: limit'); sys.exit(1)"
                ),
            },
            {"id": "after_spend", "after": ["spend"], "cmd": _py("print('never')")},
            {"id": "independent", "cmd": _py("print('runs anyway')")},
        ],
    )
    _run(queue)
    state = _state(tmp_path)
    assert state["spend"]["status"] == "failed" and state["spend"]["attempts"] == 1
    assert state["after_spend"]["status"] == "skipped"
    assert state["independent"]["status"] == "done"


def test_corrupt_file_is_quarantined_and_retried(tmp_path: Path) -> None:
    bad = tmp_path / "work" / "results" / "s" / "runs" / "r1"
    bad.mkdir(parents=True)
    (bad / "trajectories.parquet").write_bytes(b"truncated")
    code = (
        "import pathlib, sys\n"
        "if pathlib.Path('results/s/runs/r1').exists():\n"
        "    print('pyarrow.lib.ArrowInvalid: Parquet magic bytes not found: "
        "results/s/runs/r1/trajectories.parquet'); sys.exit(1)\n"
    )
    queue = _setup(tmp_path, [{"id": "resume", "cmd": _py(code)}])
    _run(queue)
    state = _state(tmp_path)
    assert state["resume"]["status"] == "done"
    assert [h["class"] for h in state["resume"]["history"]] == ["corrupt", "ok"]
    aside = [p.name for p in (tmp_path / "work" / "results" / "s" / "runs").iterdir()]
    assert len(aside) == 1 and aside[0].startswith("r1.corrupt-")


def test_call_limits_budget_skip_and_code_pin(tmp_path: Path) -> None:
    show = _py("import os; print('LIMIT', os.environ['UCAGENT_MAX_LLM_CALLS'])")
    queue = _setup(
        tmp_path,
        [
            {"id": "offline", "cmd": show},
            {"id": "suite", "kind": "suite", "expected_calls": 100, "cmd": show},
            {"id": "too_big", "kind": "suite", "expected_calls": 900, "cmd": show},
            {"id": "pinned", "commit": "deadbeef", "cmd": show},
        ],
    )
    _run(queue)
    state = _state(tmp_path)
    logs = tmp_path / "queue" / "logs"
    assert "LIMIT 0" in (logs / "offline.1.log").read_text()  # offline: no call allowed
    assert "LIMIT 250" in (logs / "suite.1.log").read_text()  # 0 + 1.5 x 100 + 100
    assert state["too_big"]["status"] == "skipped" and "budget" in state["too_big"]["note"]
    assert state["pinned"]["status"] == "failed" and "code pin" in state["pinned"]["note"]


def test_lock_refuses_a_live_queue_and_takes_over_a_stale_one(tmp_path: Path) -> None:
    queue = _setup(tmp_path, [{"id": "ok", "cmd": _py("print(1)")}])
    lock = tmp_path / "queue" / "LOCK"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))  # this test process is alive
    assert _run(queue).returncode == 3
    lock.write_text("4000000")  # no such process
    assert _run(queue).returncode == 0
    assert _state(tmp_path)["ok"]["status"] == "done" and not lock.exists()


def test_stage_appended_while_running_is_picked_up(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.yaml"
    appender = (
        "import yaml, pathlib\n"
        f"p = pathlib.Path({str(queue_path)!r}); q = yaml.safe_load(p.read_text())\n"
        "q['stages'].append({'id': 'late', 'kind': 'offline', 'dir': q['stages'][0]['dir'], "
        "'commit': None, 'cmd': ['true']})\n"
        "p.write_text(yaml.safe_dump(q))\n"
    )
    _setup(tmp_path, [{"id": "first", "cmd": _py(appender)}])
    _run(queue_path)
    state = _state(tmp_path)
    assert state["first"]["status"] == "done" and state["late"]["status"] == "done"
