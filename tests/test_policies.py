import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ucagent.core.control  # noqa: F401
from ucagent.core.control.base import make_policy
from ucagent.core.graph.runtime import CostTable, RunSettings, init_state
from ucagent.core.protocols import Instance
from ucagent.core.records import Action, StepRecord
from ucagent.core.tables import Tables

unit = st.floats(0, 1)


def _state(survival=1.0, n=3, i=1, history=()):
    inst = Instance(instance_id="i", split="test", index=0, question="q", n_steps=n)
    st_ = init_state(inst, RunSettings())
    st_.update({"survival": survival, "i": i, "history": list(history)})
    return st_


def _step(eps=0.3, w=1.0, G=1.0, u=0.5, step=1):
    return StepRecord(
        step=step, attempt=0, context="c", answer="a", u=u, entropy=u, eps=eps, w=w, G=G
    )


def _decide(policy: str, knob: float | None, state, step, tables=None, **kw):
    settings = RunSettings(policy=policy, knob=knob, **kw)
    return make_policy(settings, tables or Tables()).decide(state, step)


def test_final_gates_only_act_on_last_step() -> None:
    last = _step(eps=0.4, u=0.9, step=3)
    assert _decide("final_gate", 0.2, _state(i=3), last).action == Action.ESCALATE
    assert _decide("final_gate", 0.5, _state(i=3), last).action == Action.PROCEED
    assert _decide("final_gate", 0.0, _state(i=1), _step(eps=0.4, step=1)).action == Action.PROCEED
    assert _decide("final_u", 0.5, _state(i=3), last).action == Action.ESCALATE
    assert _decide("final_u", 0.95, _state(i=3), last).action == Action.PROCEED


def test_per_step_gates() -> None:
    d = _decide("per_step_u", 0.4, _state(), _step(u=0.5))
    assert d.action == Action.ASK_HUMAN and "u=0.500" in d.reason and "tau=0.400" in d.reason
    assert _decide("per_step_u", 0.6, _state(), _step(u=0.5)).action == Action.PROCEED
    assert _decide("per_step_eps", 0.1, _state(), _step(eps=0.3, w=0.5)).action == Action.ASK_HUMAN
    assert _decide("per_step_eps", 0.2, _state(), _step(eps=0.3, w=0.5)).action == Action.PROCEED


def test_marginal_picks_largest_net_value_and_reports_numbers() -> None:
    costs = CostTable(ask_human=1.0, escalate=10.0, llm_call=0.01)
    step = _step(eps=0.3, w=1.0, G=1.0)
    # ask: delta 0.3, net 0.3 - tau; escalate: delta 0.3 - p_e, net 0.3 - 10 tau
    d = _decide("marginal", 0.1, _state(), step, costs=costs)
    assert d.action == Action.ASK_HUMAN and "delta_ask_human" in d.numbers
    assert _decide("marginal", 0.5, _state(), step, costs=costs).action == Action.PROCEED
    # with a cheap escalation and an unreliable human, escalate wins
    d2 = _decide(
        "marginal", 0.01, _state(), step, costs=CostTable(ask_human=5.0, escalate=1.0), p_h=0.25
    )
    assert d2.action == Action.ESCALATE
    tables = Tables(resample_bucket_edges=[0.0, 0.5], resample_effect={(10, 1): 0.05})
    d3 = _decide("marginal", 0.1, _state(), step, tables=tables, costs=costs, K_prime=10)
    assert d3.action == Action.RESAMPLE  # delta 0.25 at cost 0.1 beats ask (0.3 at cost 1)


def test_budgeted_cheapest_within_budget_else_escalate() -> None:
    costs = CostTable(ask_human=1.0, escalate=10.0, llm_call=0.01)
    step = _step(eps=0.3, step=1)
    # allowed = lambda * alpha * g(1, 3) with linear spending = 1 * 0.3 * 1/3 = 0.1 < proj 0.3
    d = _decide("budgeted", 1.0, _state(n=3, i=1), step, alpha=0.3, spending="linear", costs=costs)
    assert d.action == Action.ASK_HUMAN and "allowed=0.100" in d.reason
    assert (
        _decide("budgeted", 10.0, _state(n=3, i=1), step, alpha=0.3, costs=costs).action
        == Action.PROCEED
    )
    # a noisy human cannot restore the budget -> escalate
    d2 = _decide("budgeted", 1.0, _state(n=3, i=1), step, alpha=0.3, costs=costs, p_h=0.2)
    assert d2.action == Action.ESCALATE
    flat = _decide("budgeted", 1.0, _state(n=3, i=1), step, alpha=0.3, spending="flat", costs=costs)
    assert flat.action == Action.PROCEED  # allowed = 0.3 >= proj 0.3


@settings(max_examples=60, deadline=None)
@given(st.floats(0.001, 2), st.floats(0.001, 2), st.floats(0.01, 1), unit, unit, unit)
def test_single_step_decisions_are_monotone_in_the_knob(t1, t2, s_prev, eps, w, g) -> None:
    lo, hi = sorted([t1, t2])
    step = _step(eps=eps, w=w, G=g, u=eps)
    for policy in ["per_step_eps", "per_step_u", "marginal", "final_gate", "budgeted"]:
        state = (
            _state(survival=s_prev, n=1, i=1)
            if policy.startswith("final")
            else _state(survival=s_prev)
        )
        at_hi = _decide(policy, hi, state, step).action != Action.PROCEED
        at_lo = _decide(policy, lo, state, step).action != Action.PROCEED
        assert at_lo or not at_hi, (policy, lo, hi)


def test_default_knobs_and_unknown_policy() -> None:
    assert _decide("marginal", None, _state(), _step()).numbers["tau"] == 0.1
    with pytest.raises(KeyError):
        make_policy(RunSettings(policy="nope"), Tables())
