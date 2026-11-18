"""``ucagent ui``: launches Streamlit, or explains how to install it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from ucagent.cli import app


def test_ui_help_lists_options() -> None:
    result = CliRunner().invoke(app, ["ui", "--help"])
    assert result.exit_code == 0, result.output
    assert "--port" in result.output and "--config" in result.output


def test_ui_without_streamlit_explains_the_extra(monkeypatch) -> None:
    real = importlib.util.find_spec

    def fake(name: str, *args, **kwargs):
        return None if name == "streamlit" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    result = CliRunner().invoke(app, ["ui"])
    assert result.exit_code == 1
    assert "uv sync --extra ui" in result.output


def test_ui_builds_the_streamlit_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    import subprocess

    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: calls.append((cmd, env)) or 0)
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("run_id: x\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["ui", "--port", "8600", "--results-dir", str(tmp_path), "--no-headless", "-c", str(cfg)],
    )
    assert result.exit_code == 0, result.output
    cmd, env = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("streamlit_app.py")
    assert cmd[cmd.index("--server.port") + 1] == "8600"
    assert cmd[cmd.index("--server.headless") + 1] == "false"
    assert env["UCAGENT_UI_RESULTS_DIR"] == str(tmp_path)
    assert env["UCAGENT_UI_CONFIG"] == str(cfg.resolve())
