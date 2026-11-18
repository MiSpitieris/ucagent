import pytest
from hypothesis import given
from hypothesis import strategies as st

from ucagent.core.graph.runtime import CostTable, RunSettings, init_state
from ucagent.core.protocols import Instance
from ucagent.core.records import Action, StepRecord
from ucagent.core.risk.effects import (
    ask_effect,
    backtrack_effect,
    candidate_effects,
    escalate_effect,
    resample_effect,
)
from ucagent.core.tables import Tables

unit = st.floats(0, 1)


def _state(survival: float = 0.9, history=(), resample_count=0, backtrack_count=0, n=3):
    inst = Instance(instance_id="i", split="test", index=0, question="q", n_steps=n)
    st_ = init_state(inst, RunSettings())
    st_.update(
        {
            "survival": survival,
            "history": list(history),
            "resample_count": resample_count,
            "backtrack_count": backtrack_count,
            "i": len(history) + 1,
        }
    )
    return st_


def _step(eps=0.3, w=1.0, G=0.95, u=0.5, step=2, n_samples=5):
    return StepRecord(
        step=step,
        attempt=0,
        context="c",
        answer="a",
        u=u,
        entropy=u,
        eps=eps,
        w=w,
        G=G,
        n_samples=n_samples,
    )


def test_hand_computed_effects() -> None:
    settings = RunSettings(
        p_h=0.05, p_e=0.02, costs=CostTable(ask_human=1.0, escalate=10.0, llm_call=0.01), K_prime=20
    )
    step = _step()
    ask = ask_effect(step, 0.9, settings)
    assert ask.delta == pytest.approx(0.9 * 0.95 * (0.3 - 0.05))
    assert ask.cost == 1.0 and ask.action == Action.ASK_HUMAN
    esc = escalate_effect(step, 0.9, settings)
    assert esc.delta == pytest.approx((1 - 0.9 * 0.7 * 0.95) - 0.02)
    assert esc.projection == 0.02 and esc.cost == 10.0
    tables = Tables(resample_bucket_edges=[0.0, 0.5], resample_effect={(20, 1): 0.1})
    res = resample_effect(step, _state(), settings, tables)
    assert res is not None and res.delta == pytest.approx(0.9 * 0.95 * (0.3 - 0.1))
    assert res.cost == pytest.approx((20 - 5) * 0.01)  # only the draws that expand 5 to 20
    assert resample_effect(step, _state(resample_count=1), settings, tables) is None  # the cap
    assert resample_effect(_step(n_samples=20), _state(), settings, tables) is None  # full batch
    assert resample_effect(step, _state(), settings, Tables()) is None  # no table: never


def test_backtrack_effect_targets_the_riskiest_step() -> None:
    settings = RunSettings(
        K=5, K_prime=20, costs=CostTable(backtrack=0.5, llm_call=0.01), backtrack_min_share=0.5
    )
    h1 = _step(eps=0.4, u=0.8, step=1)
    h2 = _step(eps=0.05, u=0.1, step=2)
    survival = (1 - 0.4) * (1 - 0.05)
    state = _state(survival=survival, history=[h1, h2], n=3)
    cur = _step(eps=0.2, u=0.3, step=3)
    tables = Tables(
        resample_bucket_edges=[0.0, 0.5],
        resample_effect={(20, 1): 0.1, (20, 0): 0.03},
        eps_mean_by_position={(3, 2): 0.1, (3, 3): 0.1},
    )
    eff = backtrack_effect(cur, state, settings, tables)
    assert eff is not None and eff.target_step == 1
    expected_survival = (1 - 0.1) * (1 - 0.1) * (1 - 0.1)
    assert eff.committed_projection == pytest.approx(1 - expected_survival)
    assert eff.projection == pytest.approx(1 - expected_survival * cur.G)
    assert eff.cost == pytest.approx(0.5 + (20 - 5) * 0.01 + 2 * 5 * 0.01)
    assert (
        backtrack_effect(
            cur, _state(survival=survival, history=[h1, h2], backtrack_count=1), settings, tables
        )
        is None
    )
    full = _step(eps=0.4, u=0.8, step=1, n_samples=20)  # the riskiest step cannot be expanded
    assert (
        backtrack_effect(cur, _state(survival=survival, history=[full, h2]), settings, tables)
        is None
    )
    low_share = RunSettings(K=5, K_prime=20, backtrack_min_share=0.99)
    assert backtrack_effect(cur, state, low_share, tables) is None
    assert backtrack_effect(cur, state, settings, Tables()) is None


def test_candidates_contain_proceed_ask_escalate() -> None:
    effs = candidate_effects(_step(), _state(), RunSettings(), Tables())
    assert [e.action for e in effs] == [Action.PROCEED, Action.ASK_HUMAN, Action.ESCALATE]
    assert effs[0].delta == 0.0 and effs[0].cost == 0.0


@given(unit, unit, unit, unit, unit, unit)
def test_effect_signs(s_prev, eps, w, g, p_h, p_e) -> None:
    settings = RunSettings(p_h=p_h, p_e=p_e)
    step = _step(eps=eps, w=w, G=g)
    ask = ask_effect(step, s_prev, settings)
    assert (ask.delta >= -1e-12) == (eps >= p_h) or abs(ask.delta) < 1e-9
    esc = escalate_effect(step, s_prev, settings)
    assert esc.delta == pytest.approx(esc.projection * 0 + (1 - s_prev * (1 - w * eps) * g) - p_e)
    assert 0.0 <= ask.projection <= 1.0 and 0.0 <= ask.committed_projection <= 1.0


@given(st.floats(0.01, 1), st.floats(0.01, 1), unit, unit)
def test_sunk_risk_lowers_ask_value(s1, s2, eps, g) -> None:
    lo, hi = sorted([s1, s2])
    settings = RunSettings(p_h=0.0)
    step = _step(eps=eps, w=1.0, G=g)
    assert ask_effect(step, lo, settings).delta <= ask_effect(step, hi, settings).delta + 1e-12
