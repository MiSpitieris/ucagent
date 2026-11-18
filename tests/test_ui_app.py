"""Headless drive of the Streamlit page (``streamlit.testing.v1.AppTest``), mock LLM only."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from ucagent.core.records import Action

APP = Path(__file__).resolve().parents[1] / "src" / "ucagent" / "ui" / "streamlit_app.py"


def _app(monkeypatch, tmp_path: Path, **state) -> AppTest:
    monkeypatch.setenv("UCAGENT_UI_RESULTS_DIR", str(tmp_path))  # no artifacts, no LTT table
    monkeypatch.delenv("UCAGENT_UI_CONFIG", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=120)
    preset = {
        "backend": "mock",
        "n_steps": 3,
        "k": 3,
        "k_prime": 6,
        "policy": "none",
        "supervise": True,
        "estimator": "disagreement",
        "artifact_mode": "none",
        "human_mode": "me",
        **state,
    }
    for key, value in preset.items():
        at.session_state[key] = value
    return at.run()


def _click(at: AppTest, key: str) -> AppTest:
    return at.button(key=key).click().run()


def _pause(at: AppTest):
    return at.session_state["session"].pause


def test_page_renders_without_a_run(monkeypatch, tmp_path: Path) -> None:
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.session_state["session"] is None
    assert at.button(key="btn_start")


def test_supervised_mock_run_accept_until_done(monkeypatch, tmp_path: Path) -> None:
    at = _click(_app(monkeypatch, tmp_path), "btn_start")
    assert not at.exception, at.exception
    assert at.session_state["error"] is None
    assert _pause(at).kind == "control" and _pause(at).step == 1
    assert at.button(key="btn_accept") and at.button(key="btn_override")
    for _ in range(10):
        if _pause(at).kind == "done":
            break
        at = _click(at, "btn_accept")
        assert not at.exception, at.exception
    assert _pause(at).kind == "done"
    assert any("Final answer" in m.value for m in at.markdown)
    assert at.session_state["events"][-1] == "finished"
    values = at.session_state["session"].values
    assert len(values["history"]) == 3 and values["done"] is True


def test_override_ask_human_then_answer(monkeypatch, tmp_path: Path) -> None:
    at = _click(_app(monkeypatch, tmp_path), "btn_start")
    at = at.selectbox(key="sel_action").select(Action.ASK_HUMAN.value).run()
    at = at.text_input(key="txt_reason").input("looks wrong").run()
    at = _click(at, "btn_override")
    assert not at.exception, at.exception
    assert at.session_state["error"] is None
    assert _pause(at).kind == "check_step" and _pause(at).step == 1
    assert at.text_input(key="txt_answer").value == _pause(at).payload["committed"]
    at = at.text_input(key="txt_answer").input("Zed Zed").run()
    at = _click(at, "btn_answer")
    assert not at.exception, at.exception
    values = at.session_state["session"].values
    assert values["history"][0].answer == "Zed Zed" and values["history"][0].source == "human"
    assert values["interventions"][0].numbers["override"] == 1.0
    assert "looks wrong" in values["interventions"][0].reason
    assert _pause(at).kind == "control" and _pause(at).step == 2


def test_override_escalate_and_final_answer(monkeypatch, tmp_path: Path) -> None:
    at = _click(_app(monkeypatch, tmp_path), "btn_start")
    at = at.selectbox(key="sel_action").select(Action.ESCALATE.value).run()
    at = _click(at, "btn_override")
    assert _pause(at).kind == "take_over"
    at = at.text_input(key="txt_final").input("Final Z").run()
    at = _click(at, "btn_final")
    assert not at.exception, at.exception
    values = at.session_state["session"].values
    assert values["escalated"] is True and values["final_answer"] == "Final Z"
    assert any("Final answer" in m.value for m in at.markdown)
    at = _click(at, "btn_new_run")
    assert at.session_state["session"] is None and not at.exception


def test_oracle_mode_runs_without_the_page_user(monkeypatch, tmp_path: Path) -> None:
    at = _app(monkeypatch, tmp_path, supervise=False, human_mode="oracle", policy="per_step_u")
    at.session_state["knob"] = 0.0  # ask at every uncertain step; the oracle answers
    at = _click(at.run(), "btn_start")
    assert not at.exception, at.exception
    assert at.session_state["error"] is None
    assert _pause(at).kind == "done"
    values = at.session_state["session"].values
    assert values["done"] is True and (len(values["history"]) == 3 or values["escalated"])
