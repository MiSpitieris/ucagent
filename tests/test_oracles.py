from ucagent.core.graph.runtime import RunSettings
from ucagent.tasks.base import EscalationOracle, GlobalOracle, LocalOracle, make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig


def _task() -> SyntheticChainTask:
    return SyntheticChainTask(SyntheticConfig(n_steps=3, ambiguity=1.0, n_instances={"test": 5}))


def _payload(inst, step: int, current: str) -> dict:
    return {"instance_id": inst.instance_id, "step": step, "attempt": 0, "current": current}


def test_local_oracle_perfect_and_noisy() -> None:
    task = _task()
    inst = task.get_instance("test-00000-s0")
    wrong_current = inst.distractors[0][1]
    assert LocalOracle(task)(_payload(inst, 2, wrong_current)) == task.local_truth(
        inst, 2, wrong_current
    )
    noisy = LocalOracle(task, p_h=1.0)
    out = noisy(_payload(inst, 2, wrong_current))
    assert out != task.local_truth(inst, 2, wrong_current)
    assert out in inst.pool


def test_global_oracle_ignores_context() -> None:
    task = _task()
    inst = task.get_instance("test-00001-s0")
    assert GlobalOracle(task)(_payload(inst, 2, inst.distractors[0][0])) == inst.entities[2]


def test_escalation_oracle() -> None:
    task = _task()
    inst = task.get_instance("test-00002-s0")
    payload = {"instance_id": inst.instance_id, "step": 2}
    assert EscalationOracle(task)(payload) == inst.entities[-1]
    assert EscalationOracle(task, p_e=1.0)(payload) != inst.entities[-1]


def test_make_human_dispatch() -> None:
    task = _task()
    inst = task.get_instance("test-00003-s0")
    human = make_human(task, RunSettings(oracle_scope="global"))
    assert human.check_step(_payload(inst, 3, "whatever")) == inst.entities[3]
    assert human.take_over({"instance_id": inst.instance_id, "step": 1}) == inst.entities[3]
