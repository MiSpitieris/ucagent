"""Spending functions ``g(i, n)`` for the budgeted controller (per-output risk budget).

All are nondecreasing in ``i`` with ``g(n, n) = 1``. The analogy with
alpha-spending in group-sequential trials is loose (instructions, Section
2.6): there the budget is over dependent looks at one hypothesis, here step
contributions are additive and independent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from scipy.stats import norm

SpendingFunction = Callable[[int, int, Sequence[float]], float]


def flat(i: int, n: int, w: Sequence[float] = ()) -> float:
    """Only the final constraint: ``g = 1`` at every step."""
    return 1.0


def linear(i: int, n: int, w: Sequence[float] = ()) -> float:
    """``i / n``."""
    return i / n


def pocock(i: int, n: int, w: Sequence[float] = ()) -> float:
    """Front-loaded Lan-DeMets Pocock shape ``ln(1 + (e - 1) t)``."""
    t = i / n
    return math.log(1.0 + (math.e - 1.0) * t)


_Z = float(norm.ppf(1.0 - 0.025))
_OBF_DENOM = 2.0 * (1.0 - float(norm.cdf(_Z)))


def obf(i: int, n: int, w: Sequence[float] = ()) -> float:
    """Back-loaded Lan-DeMets O'Brien-Fleming shape, normalised so ``g(n, n) = 1``."""
    t = i / n
    if t <= 0.0:
        return 0.0
    return 2.0 * (1.0 - float(norm.cdf(_Z / math.sqrt(t)))) / _OBF_DENOM


def influence_proportional(i: int, n: int, w: Sequence[float] = ()) -> float:
    """``cumsum(w)_i / sum(w)``; falls back to linear without influence weights."""
    if len(w) != n or sum(w) <= 0:
        return linear(i, n)
    return sum(w[:i]) / sum(w)


SPENDING: dict[str, SpendingFunction] = {
    "flat": flat,
    "linear": linear,
    "pocock": pocock,
    "obf": obf,
    "influence_proportional": influence_proportional,
}


def make_spending(name: str) -> SpendingFunction:
    """Look up a spending function by name."""
    try:
        return SPENDING[name]
    except KeyError as exc:
        raise KeyError(f"unknown spending function {name!r}; known: {sorted(SPENDING)}") from exc
