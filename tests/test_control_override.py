"""``RunSettings.force_decisions``: a human decision replaces the policy's for one attempt."""

from __future__ import annotations

import pytest

from tests.helpers.policies import register_forced
from ucagent.core.graph.build import build_graph
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.nodes import override_key, validate_override
from ucagent.core.graph.runtime import RunSettings, init_state, make_config
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import Action, Decision
from ucagent.core.tables import Tables
from ucagent.tasks.base import make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig

SCRIPT: dict[tuple[int, int], tuple[Action, int | None]] = {
    (1, 0): (Action.RESAMPLE, None),
    (2, 0): (Action.ASK_HUMAN, None),
    (3, 0): (Action.BACKTRACK, 2),  # step 1 is already expanded to K'; step 2 (K samples) is not
}


def _run(settings: RunSettings) -> dict:
    task = SyntheticChainTask(SyntheticConfig(n_steps=4, ambiguity=0.5, n_instances={"test": 3}))
    graph = build_graph(task, MockChainLLM(task=task, seed=0))
    inst = task.get_instance("test-00000-s0")
    cfg = make_config("override", settings, Tables(), 4)
    return run_with_human(graph, init_state(inst, settings), cfg, make_human(task, settings))


def test_force_decisions_reproduce_a_scripted_policy() -> None:
    name = register_forced("forced_override_reference", SCRIPT)
    ref = _run(RunSettings(policy=name, K=2, K_prime=4))
    forced = {
        override_key(s, a): Decision(action=act, target_step=t)
        for (s, a), (act, t) in SCRIPT.items()
    }
    out = _run(RunSettings(policy="none", K=2, K_prime=4, force_decisions=forced))
    assert [iv.action for iv in out["interventions"]] == [iv.action for iv in ref["interventions"]]
    assert [(h.step, h.attempt, h.answer) for h in out["history"]] == [
        (h.step, h.attempt, h.answer) for h in ref["history"]
    ]
    assert out["total_cost"] == pytest.approx(ref["total_cost"])
    flagged = [(iv.step, iv.attempt) for iv in out["interventions"] if "override" in iv.numbers]
    assert flagged == [(1, 0), (2, 0), (3, 0)]
    overrides = [iv for iv in out["interventions"] if "override" in iv.numbers]
    assert all(iv.reason.startswith("human override: ") for iv in overrides)
    assert all("policy proposed proceed" in iv.reason for iv in overrides)
    assert all(iv.numbers["override"] == 1.0 for iv in overrides)
    assert all(
        "override" not in iv.numbers
        for iv in out["interventions"]
        if (iv.step, iv.attempt) not in SCRIPT
    )


def test_override_is_consumed_only_at_its_step_attempt() -> None:
    forced = {override_key(9, 0): Decision(action=Action.ESCALATE)}
    out = _run(RunSettings(policy="none", K=1, K_prime=2, force_decisions=forced))
    assert out["escalated"] is False and len(out["history"]) == 4


@pytest.mark.parametrize("target", [None, 0, 2, 7])
def test_forced_backtrack_target_must_be_an_earlier_step(target: int | None) -> None:
    forced = {override_key(2, 0): Decision(action=Action.BACKTRACK, target_step=target)}
    with pytest.raises(ValueError, match="backtrack target"):
        _run(RunSettings(policy="none", K=1, K_prime=2, force_decisions=forced))


def test_validate_override_flags_and_keeps_reason() -> None:
    state = {"i": 3}
    out = validate_override(Decision(action=Action.BACKTRACK, target_step=1, reason="x"), state)  # type: ignore[arg-type]
    assert out.reason == "human override: backtrack to step 1 (x)" and out.target_step == 1
    assert out.numbers == {"override": 1.0}
    plain = validate_override(Decision(action=Action.PROCEED), state)  # type: ignore[arg-type]
    assert plain.reason == "human override: proceed"
