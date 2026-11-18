"""Pure rendering helpers of the page (no Streamlit)."""

from __future__ import annotations

import pytest

from tests.helpers.configs import small_config
from ucagent.core.graph.session import InteractiveSession
from ucagent.core.risk.propagation import project, survival_path
from ucagent.experiments.runner import Worker
from ucagent.ui import render


@pytest.fixture(scope="module")
def run():
    worker = Worker.from_config(small_config(llm={"K": 3, "K_prime": 6}))
    session = InteractiveSession(worker.graph, worker.settings, worker.tables)
    pause = session.start(worker.task.get_instance("test-00000-s0"))
    paused = dict(session.values)
    effects = session.effects()
    proposal = session.proposal()
    while pause.kind != "done":
        pause = session.accept()
    return worker, session, dict(session.values), paused, effects, proposal


def test_steps_frame_matches_propagation(run) -> None:
    _, _, values, *_ = run
    frame = render.steps_frame(values)
    history = values["history"]
    eps = [h.eps for h in history]
    w = [h.w for h in history]
    assert frame["survival"].tolist() == pytest.approx(survival_path(eps, w).tolist())
    prev = 1.0
    for row, h in zip(frame.itertuples(), history, strict=True):
        assert row[-1] == pytest.approx(project(prev, h.eps, h.w, h.G))
        prev *= 1 - h.w * h.eps
    assert list(frame.columns) == render.STEP_COLUMNS and len(frame) == 3


def test_risk_series_shape(run) -> None:
    _, _, values, *_ = run
    series = render.risk_series(values, 0.1)
    assert list(series.columns) == ["risk so far", "projected final risk", "alpha"]
    assert series.index.name == "step" and (series["alpha"] == 0.1).all()


def test_prompt_and_cluster_and_sample_frames(run) -> None:
    _, _, _, paused, *_ = run
    text = render.prompt_text(paused["prompt"])
    assert "<ucagent" not in text and "Question:" in text
    pending = paused["pending"]
    clusters = render.clusters_frame(pending)
    assert clusters["share"].sum() == pytest.approx(1.0)
    assert clusters["share"].is_monotonic_decreasing
    samples = render.samples_frame(paused, pending.samples_key)
    assert len(samples) == 3 and list(samples.columns) == ["sample", "answer", "confidence", "text"]
    assert render.clusters_frame(None).empty
    payload = {"clusters": [c.model_dump() for c in pending.clusters]}
    assert render.clusters_from_payload(payload)["share"].tolist() == clusters["share"].tolist()


def test_effects_proposal_and_decision_frames(run) -> None:
    _, _, values, _, effects, proposal = run
    frame = render.effects_frame(effects, 0.1)
    assert frame["action"].iloc[0] == "proceed"
    for row, effect in zip(frame.itertuples(index=False), effects, strict=True):
        assert row[4] == pytest.approx(effect.delta - 0.1 * effect.cost)
    assert render.effects_frame(effects, None)["net value"].isna().all()
    assert render.proposal_frame(proposal).columns.tolist() == ["quantity", "value"]
    decisions = render.decisions_frame(values["interventions"])
    assert len(decisions) == 3 and not decisions["override"].any()
    assert decisions["action"].tolist() == ["proceed"] * 3


def test_status_numbers_trace_and_audit(run) -> None:
    worker, session, values, paused, *_ = run
    done = render.status_numbers(values, worker.settings)
    assert done["cost"] == pytest.approx(values["total_cost"]) and done["interventions"] == 0
    before = render.status_numbers(paused, worker.settings)
    assert before["cost"] == pytest.approx(paused["llm_calls"] * worker.settings.costs.llm_call)
    assert before["step"] == 1 and done["step"] == 3
    assert (
        render.pending_projection(paused) is not None and render.pending_projection(values) is None
    )
    assert (
        render.trace_summary(["prepare_step", "sample", "sample", "estimate"])
        == "prepare_step -> sample x2 -> estimate"
    )
    trajectory = worker.to_trajectory(values, "test-00000-s0", session.thread_id, 0.0)
    audit = render.audit_frame(trajectory)
    assert len(audit) == 3 and audit["decisions"].str.startswith("proceed").all()
