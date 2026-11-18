"""Product-form risk propagation (Section 2.3 of the instructions).

Survival after committing steps ``1..i`` is ``S_i = prod_j (1 - w_j eps_j)``
and trajectory risk is ``R_i = 1 - S_i``. This assumes independent local
errors and a single influence number per step; both are simplifications
(see ``MATH.md``, open questions).

# OPEN-QUESTION: correlated local errors would need a correlation term here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


def survival(eps: Sequence[float], w: Sequence[float] | None = None) -> float:
    """``prod_j (1 - w_j * eps_j)`` (``w = 1`` when omitted)."""
    e = np.asarray(eps, dtype=float)
    ww = np.ones_like(e) if w is None else np.asarray(w, dtype=float)
    return float(np.prod(1.0 - ww * e)) if e.size else 1.0


def risk(eps: Sequence[float], w: Sequence[float] | None = None) -> float:
    """``1 - survival``."""
    return 1.0 - survival(eps, w)


def survival_path(eps: Sequence[float], w: Sequence[float] | None = None) -> np.ndarray:
    """Vectorised cumulative survival ``S_1..S_n``."""
    e = np.asarray(eps, dtype=float)
    ww = np.ones_like(e) if w is None else np.asarray(w, dtype=float)
    return np.cumprod(1.0 - ww * e)


def project(survival_prev: float, eps_i: float, w_i: float, g_i: float = 1.0) -> float:
    """Projected final risk if step ``i`` is committed as is: ``1 - S_{i-1}(1 - w_i eps_i) G_i``."""
    value = 1.0 - survival_prev * (1.0 - w_i * eps_i) * g_i
    return float(min(1.0, max(0.0, value)))


class RiskModel(Protocol):
    """Maps the trajectory so far plus the current step to a projected final risk."""

    name: str

    def project(self, survival_prev: float, eps_i: float, w_i: float, g_i: float) -> float:
        """Projected final risk."""
        ...


class ProductRiskModel:
    """The product form above; the only risk model in v1."""

    name = "product"

    def project(self, survival_prev: float, eps_i: float, w_i: float, g_i: float) -> float:
        """See :func:`project`."""
        return project(survival_prev, eps_i, w_i, g_i)
