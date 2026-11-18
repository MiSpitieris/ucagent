"""The random-ask mixing baseline: nested, deterministic, and the two extremes."""

from __future__ import annotations

from itertools import pairwise

from ucagent.core.control.base import POLICIES, make_policy
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.records import Action, StepRecord
from ucagent.core.tables import Tables
from ucagent.experiments.figures import POLICY_LABEL, POLICY_ORDER


def _rec(step: int, attempt: int = 0) -> StepRecord:
    return StepRecord(
        step=step, attempt=attempt, context="", answer="a", u=0.0, entropy=0.0, eps=0.0
    )


def _asks(q: float, seed: int = 0, n_inst: int = 40) -> set[tuple[str, int]]:
    policy = make_policy(RunSettings(policy="random_ask", knob=q, seed=seed), Tables())
    asked = set()
    for k in range(n_inst):
        state = {"instance_id": f"inst-{k}", "i": 1, "n_steps": 3, "survival": 1.0}
        for step in (1, 2, 3):
            d = policy.decide(state, _rec(step))  # type: ignore[arg-type]
            assert d.action in (Action.ASK_HUMAN, Action.PROCEED) and "draw" in d.numbers
            if d.action == Action.ASK_HUMAN:
                asked.add((state["instance_id"], step))
    return asked


def test_random_ask_is_registered_and_listed() -> None:
    assert (
        "random_ask" in POLICIES and "random_ask" in POLICY_ORDER and "random_ask" in POLICY_LABEL
    )


def test_random_ask_is_nested_in_the_knob_and_deterministic() -> None:
    grid = [0.0, 0.2, 0.5, 0.8, 1.0]
    sets = [_asks(q) for q in grid]
    assert len(sets[0]) == 120 and not sets[-1]  # q = 0 asks everywhere, q = 1 never
    for small, large in pairwise(sets):
        assert large <= small  # steps asked at a larger q are asked at every smaller q
    share = len(sets[2]) / 120
    assert 0.35 < share < 0.65  # about 1 - q of the steps at q = 0.5
    assert _asks(0.5) == _asks(0.5) and _asks(0.5, seed=1) != _asks(0.5, seed=0)
