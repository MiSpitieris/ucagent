import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ucagent.api import calibrate_knob as api_calibrate
from ucagent.core.control.ltt import binomial_pvalue, calibrate_knob, hoeffding_pvalue


def test_binomial_hand_values() -> None:
    assert binomial_pvalue(5, 10, 0.5) == pytest.approx(0.623046875, abs=1e-12)
    assert binomial_pvalue(0, 20, 0.1) == pytest.approx(0.9**20)
    assert binomial_pvalue(20, 20, 0.1) == 1.0
    assert (
        binomial_pvalue(1, 100, 0.05) < 0.1 < binomial_pvalue(2, 100, 0.05)
    )  # why small configs use alpha=0.1
    assert binomial_pvalue(5, 100, 0.1) < 0.1 < binomial_pvalue(6, 100, 0.1)
    with pytest.raises(ValueError):
        binomial_pvalue(0, 0, 0.1)


@given(st.integers(1, 200), st.floats(0.01, 0.99))
def test_pvalues_are_monotone_in_errors(n, alpha) -> None:
    b = [binomial_pvalue(x, n, alpha) for x in range(n + 1)]
    h = [hoeffding_pvalue(x, n, alpha) for x in range(n + 1)]
    assert all(a <= c + 1e-12 for a, c in itertools.pairwise(b))
    assert all(a <= c + 1e-12 for a, c in itertools.pairwise(h))
    assert all(0.0 <= v <= 1.0 for v in b + h)
    assert h[-1] == 1.0


def test_hoeffding_form() -> None:
    assert hoeffding_pvalue(0, 100, 0.2) == pytest.approx(
        pytest.approx(2.718281828 ** (-2 * 100 * 0.04))
    )
    assert hoeffding_pvalue(30, 100, 0.2) == 1.0


def test_fixed_sequence_selects_last_rejected_before_first_failure() -> None:
    # knobs ascending = most conservative first; errors rise with the knob
    points = [(0.01, 100, 1), (0.03, 100, 3), (0.1, 100, 4), (0.3, 100, 12), (1.0, 100, 2)]
    res = calibrate_knob("p", points, alpha=0.1, delta=0.1)
    assert res.feasible and res.knob_star == 0.1
    assert [r.rejected for r in res.rows] == [True, True, True, False, False]
    assert [r.tested for r in res.rows] == [
        True,
        True,
        True,
        True,
        False,
    ]  # stop after the first failure
    assert res.selected is not None and res.selected.risk_hat == 0.04
    assert res.min_risk_hat == 0.01


def test_infeasible_and_errors() -> None:
    res = calibrate_knob("p", [(0.01, 50, 20), (0.1, 50, 25)], alpha=0.1, delta=0.1)
    assert not res.feasible and res.knob_star is None and res.min_risk_hat == 0.4
    assert res.selected is None
    with pytest.raises(ValueError):
        calibrate_knob("p", [], alpha=0.1, delta=0.1)
    hoe = api_calibrate("p", [(0.01, 400, 0), (0.1, 400, 60)], alpha=0.1, pvalue="hoeffding")
    assert hoe.feasible and hoe.knob_star == 0.01 and hoe.pvalue_kind == "hoeffding"


def test_unsorted_points_are_sorted_ascending() -> None:
    res = calibrate_knob("p", [(0.5, 100, 20), (0.01, 100, 0), (0.1, 100, 3)], alpha=0.1, delta=0.1)
    assert [r.knob for r in res.rows] == [0.01, 0.1, 0.5]
    assert res.knob_star == 0.1


def test_the_walk_follows_the_configured_grid_and_ends_at_a_knob_that_was_not_run() -> None:
    # every knob passes on its own: 0 errors of 200 at target 0.1
    grid = [0.1, 0.2, 0.3, 0.4]
    full = [(k, 200, 0) for k in grid]
    whole = calibrate_knob("p", full, alpha=0.1, delta=0.1, grid=grid)
    assert whole.knob_star == 0.4 and whole.missing_knobs == []
    assert whole.knob_star == calibrate_knob("p", full, alpha=0.1, delta=0.1).knob_star
    # the run at 0.2 is missing: without the grid the walk passes over the gap
    holed = [p for p in full if p[0] != 0.2]
    assert calibrate_knob("p", holed, alpha=0.1, delta=0.1).knob_star == 0.4
    res = calibrate_knob("p", holed, alpha=0.1, delta=0.1, grid=grid)
    assert res.knob_star == 0.1 and res.missing_knobs == [0.2]
    assert [r.tested for r in res.rows] == [True, False, False]
    # the most conservative knob is missing: nothing can be selected
    none = calibrate_knob("p", full[1:], alpha=0.1, delta=0.1, grid=grid)
    assert not none.feasible and none.knob_star is None and none.missing_knobs == [0.1]
    # a knob read back from a manifest may differ in the last digits
    near = [(k * (1 + 1e-12), 200, 0) for k in grid]
    assert calibrate_knob("p", near, alpha=0.1, delta=0.1, grid=grid).missing_knobs == []


def test_a_point_outside_the_configured_grid_is_an_error() -> None:
    with pytest.raises(ValueError, match="configured grid does not have"):
        calibrate_knob("p", [(0.1, 200, 0), (0.25, 200, 0)], alpha=0.1, delta=0.1, grid=[0.1, 0.2])
