import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ucagent.core.risk.propagation import ProductRiskModel, project, risk, survival, survival_path

unit = st.floats(0, 1)


def test_hand_values() -> None:
    assert survival([0.1, 0.2]) == pytest.approx(0.72)
    assert risk([0.1, 0.2], [1.0, 0.5]) == pytest.approx(1 - 0.9 * 0.9)
    assert survival([]) == 1.0 and risk([]) == 0.0
    assert project(0.9, 0.3, 1.0, 0.95) == pytest.approx(1 - 0.9 * 0.7 * 0.95)
    assert ProductRiskModel().project(1.0, 0.5, 1.0, 1.0) == 0.5
    np.testing.assert_allclose(survival_path([0.1, 0.2, 0.5]), [0.9, 0.72, 0.36])


@given(st.lists(unit, max_size=12), st.lists(unit, max_size=12))
def test_bounds(eps, w) -> None:
    m = min(len(eps), len(w))
    e, ww = eps[:m], w[:m]
    assert 0.0 <= survival(e, ww) <= 1.0
    assert 0.0 <= risk(e, ww) <= 1.0
    assert risk(e, ww) <= risk(e) + 1e-12  # less influence, less risk


@given(unit, unit, unit, unit, unit)
def test_project_monotone_in_eps(s_prev, e1, e2, w, g) -> None:
    lo, hi = sorted([e1, e2])
    assert project(s_prev, lo, w, g) <= project(s_prev, hi, w, g) + 1e-12
    assert 0.0 <= project(s_prev, lo, w, g) <= 1.0
