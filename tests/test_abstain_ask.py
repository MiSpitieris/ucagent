"""The abstention baseline: ask exactly when the committed answer says it cannot answer."""

from __future__ import annotations

from ucagent.core.control.base import POLICIES, make_policy
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.records import Action, StepRecord
from ucagent.core.tables import Tables
from ucagent.experiments.figures import POLICY_LABEL, POLICY_ORDER


def _decide(answer: str) -> Action:
    policy = make_policy(RunSettings(policy="abstain_ask", knob=0.0), Tables())
    state = {"instance_id": "inst-0", "i": 1, "n_steps": 2, "survival": 1.0}
    rec = StepRecord(step=1, attempt=0, context="", answer=answer, u=0.0, entropy=0.0, eps=0.0)
    return policy.decide(state, rec).action  # type: ignore[arg-type]


def test_abstain_ask_is_registered_and_listed() -> None:
    assert "abstain_ask" in POLICIES and "abstain_ask" in POLICY_ORDER
    assert "abstain_ask" in POLICY_LABEL


def test_abstain_ask_asks_only_on_abstentions() -> None:
    for answer in ("Unknown", "Not specified in the passages", "unable to determine", "Not given"):
        assert _decide(answer) == Action.ASK_HUMAN, answer
    for answer in ("Paris", "St. Louis, Missouri", "24.69", ""):
        assert _decide(answer) == Action.PROCEED, answer
