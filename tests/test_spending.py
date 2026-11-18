import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ucagent.core.control.spending import (
    SPENDING,
    flat,
    influence_proportional,
    linear,
    make_spending,
    obf,
    pocock,
)


@given(st.integers(1, 30))
def test_invariants(n: int) -> None:
    for name, g in SPENDING.items():
        values = [g(i, n, [1.0] * n) for i in range(1, n + 1)]
        assert values[-1] == pytest.approx(1.0), name
        assert all(0.0 <= v <= 1.0 + 1e-12 for v in values), name
        assert all(a <= b + 1e-12 for a, b in itertools.pairwise(values)), name


def test_shapes() -> None:
    n = 10
    for i in range(1, n):
        assert pocock(i, n) >= linear(i, n) >= obf(i, n)
    assert flat(1, n) == 1.0
    assert influence_proportional(2, 4, [2.0, 2.0, 1.0, 1.0]) == pytest.approx(4 / 6)
    assert influence_proportional(2, 4, []) == pytest.approx(0.5)  # falls back to linear
    with pytest.raises(KeyError):
        make_spending("nope")
