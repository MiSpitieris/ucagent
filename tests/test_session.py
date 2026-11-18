"""``InteractiveSession``: pause by pause driving of one trajectory."""

from __future__ import annotations

import pytest

from tests.helpers.configs import small_config
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.runtime import init_state, make_config
from ucagent.core.graph.session import InteractiveSession
from ucagent.core.records import Action, Decision
from ucagent.experiments.runner import Worker

INSTANCE = "test-00000-s0"


def _worker(**overrides):
    cfg = small_config(**overrides)
    return Worker.from_config(cfg)


def _session(worker: Worker, **kw) -> InteractiveSession:
    return InteractiveSession(worker.graph, worker.settings, worker.tables, **kw)


def test_supervised_accept_all_reproduces_the_batch_run() -> None:
    worker = _worker(llm={"K": 3, "K_prime": 6}, control={"policy": "per_step_u", "knob": 0.3})
    inst = worker.task.get_instance(INSTANCE)
    ref = run_with_human(
        worker.graph,
        init_state(inst, worker.settings),
        make_config("ref", worker.settings, worker.tables, inst.n_steps),
        worker.human,
    )
    session = _session(worker)
    pause = session.start(inst)
    kinds: list[str] = []
    for _ in range(40):
        kinds.append(pause.kind)
        if pause.kind == "done":
            break
        if pause.kind == "control":
            pause = session.accept()
        elif pause.kind == "check_step":
            pause = session.answer(worker.human.check_step(pause.payload))
        else:
            pause = session.answer(worker.human.take_over(pause.payload))
    assert kinds[-1] == "done" and kinds.count("control") == len(ref["interventions"])
    out = session.values
    assert [(h.step, h.attempt, h.answer, h.source) for h in out["history"]] == [
        (h.step, h.attempt, h.answer, h.source) for h in ref["history"]
    ]
    assert [iv.action for iv in out["interventions"]] == [iv.action for iv in ref["interventions"]]
    assert out["total_cost"] == pytest.approx(ref["total_cost"])
    assert out["final_answer"] == ref["final_answer"] and out["done"] is True


def test_pause_sequence_and_trace_with_policy_none() -> None:
    worker = _worker()
    session = _session(worker)
    inst = worker.task.get_instance(INSTANCE)
    pause = session.start(inst)
    assert (pause.kind, pause.step, pause.attempt) == ("control", 1, 0)
    assert (
        session.last_trace[:2] == ["prepare_step", "sample"] and "control" not in session.last_trace
    )
    assert session.proposal().action == Action.PROCEED
    assert session.effects()[0].action == Action.PROCEED
    pause = session.accept()
    assert session.last_trace[0] == "control"
    assert (pause.kind, pause.step) == ("control", 2)
    pause = session.accept()
    assert (pause.kind, pause.step) == ("control", 3)
    pause = session.accept()
    assert pause.kind == "done" and session.last_trace[-1] == "finish"
    assert session.values["done"] is True and len(session.values["history"]) == 3


def test_override_ask_human_then_answer() -> None:
    worker = _worker()
    session = _session(worker)
    session.start(worker.task.get_instance(INSTANCE))
    session.accept()
    pause = session.override(Decision(action=Action.ASK_HUMAN, reason="supervisor doubts it"))
    assert pause.kind == "check_step" and pause.step == 2 and pause.payload["kind"] == "check_step"
    assert pause.needs_answer and "committed" in pause.payload
    pause = session.answer("X")
    assert (pause.kind, pause.step) == ("control", 3)
    values = session.values
    assert values["history"][1].answer == "X" and values["history"][1].source == "human"
    assert values["history"][1].eps == worker.settings.p_h
    iv = next(x for x in values["interventions"] if x.step == 2)
    assert iv.action == Action.ASK_HUMAN and iv.numbers["override"] == 1.0
    assert iv.reason.startswith("human override: ask_human (supervisor doubts it); policy proposed")


def test_override_resample_and_backtrack() -> None:
    worker = _worker(llm={"K": 2, "K_prime": 4})
    session = _session(worker)
    session.start(worker.task.get_instance(INSTANCE))
    pause = session.override(Decision(action=Action.RESAMPLE))
    assert (pause.kind, pause.step, pause.attempt) == ("control", 1, 1)
    pending = session.values["pending"]
    assert pending is not None and pending.n_samples == 4 and pending.resample_count == 1
    assert pending.draw == 0 and pending.samples_key == "1:0"  # expanded in place
    session.accept()
    pause = session.accept()
    assert (pause.kind, pause.step) == ("control", 3)
    with pytest.raises(ValueError, match="nothing to expand"):
        session.override(Decision(action=Action.BACKTRACK, target_step=1))  # step 1 is full
    pause = session.override(Decision(action=Action.BACKTRACK, target_step=2))
    assert (pause.kind, pause.step, pause.attempt) == ("control", 2, 1)
    values = session.values
    assert [h.step for h in values["history"]] == [1] and values["backtrack_count"] == 1
    pending = values["pending"]
    assert pending is not None and pending.n_samples == 4
    assert pending.draw == 0 and pending.samples_key == "2:0"


def test_allowed_actions_follow_the_caps() -> None:
    worker = _worker(llm={"K": 2, "K_prime": 4}, control={"max_resamples": 1, "max_backtracks": 1})
    session = _session(worker)
    session.start(worker.task.get_instance(INSTANCE))
    allowed = session.allowed_actions()
    assert allowed[Action.BACKTRACK] is not None and allowed[Action.RESAMPLE] is None
    with pytest.raises(ValueError, match="not allowed"):
        session.override(Decision(action=Action.BACKTRACK, target_step=1))
    session.override(Decision(action=Action.RESAMPLE))
    assert session.allowed_actions()[Action.RESAMPLE] is not None  # the cap
    session.accept()
    # at step 2 the only committed step is full (K' samples): no target can be expanded
    assert "K'" in (session.allowed_actions()[Action.BACKTRACK] or "")
    session.accept()
    assert session.allowed_actions()[Action.BACKTRACK] is None
    targets = session.backtrack_targets()
    assert targets[2] is None and "nothing to expand" in (targets[1] or "")
    with pytest.raises(ValueError, match="nothing to expand"):
        session.override(Decision(action=Action.BACKTRACK, target_step=1))
    session.override(Decision(action=Action.BACKTRACK, target_step=2))
    assert session.allowed_actions()[Action.BACKTRACK] is not None  # the cap


def test_unsupervised_pauses_only_at_interrupts_and_empty_answer_keeps_committed() -> None:
    worker = _worker(control={"policy": "ask_all"})
    session = _session(worker, supervise=False)
    pause = session.start(worker.task.get_instance(INSTANCE))
    kinds = []
    committed = []
    while pause.kind != "done":
        kinds.append((pause.kind, pause.step))
        committed.append(pause.payload["committed"])
        pause = session.answer("")
    assert kinds == [("check_step", 1), ("check_step", 2), ("check_step", 3)]
    assert [h.answer for h in session.values["history"]] == committed
    assert all(h.source == "human" for h in session.values["history"])


def test_override_escalate_hands_over_the_case() -> None:
    worker = _worker()
    session = _session(worker)
    session.start(worker.task.get_instance(INSTANCE))
    pause = session.override(Decision(action=Action.ESCALATE))
    assert pause.kind == "take_over" and pause.payload["history"] == []
    with pytest.raises(RuntimeError, match="paused at 'take_over'"):
        session.accept()
    pause = session.answer("Z")
    assert pause.kind == "done"
    values = session.values
    assert values["escalated"] is True and values["final_answer"] == "Z" and values["done"]
    assert values["fixed_cost"] == pytest.approx(worker.settings.costs.escalate)


def test_wrong_pause_calls_raise() -> None:
    worker = _worker()
    session = _session(worker)
    with pytest.raises(RuntimeError, match="not started"):
        _ = session.pause
    session.start(worker.task.get_instance(INSTANCE))
    with pytest.raises(RuntimeError, match="paused at 'control'"):
        session.answer("x")


def test_done_state_converts_to_a_trajectory_and_close_deletes_the_thread() -> None:
    worker = _worker()
    session = _session(worker)
    inst = worker.task.get_instance(INSTANCE)
    session.start(inst)
    pause = session.override(Decision(action=Action.ASK_HUMAN))
    pause = session.answer("")
    while pause.kind != "done":
        pause = session.accept()
    traj = worker.to_trajectory(session.values, inst.instance_id, session.thread_id, 0.0)
    assert traj.steps[0].source == "human"
    reasons = [d["reason"] for row in traj.audit() for d in row["decisions"]]
    assert any(r.startswith("human override: ask_human") for r in reasons)
    config = session.config
    session.close()
    assert not session.started and worker.graph.get_state(config).values == {}
    with pytest.raises(RuntimeError, match="not started"):
        session.accept()
