"""Learn-then-Test calibration of a policy's knob (Section 2.7 of the instructions).

For a grid of knob values ordered from most to least conservative, each value
is a hypothesis ``H_k: R(knob_k) > alpha`` tested on the calibration split
with a valid p-value. Fixed-sequence testing walks the grid, rejects while
``p <= delta`` and stops at the first failure; the selected knob is the least
conservative rejected one, and nothing is selected when the first test fails.
The guarantee, for a pipeline and a grid fixed before the calibration tasks
are seen and for calibration tasks drawn independently from the population of
deployment: the probability, over the calibration data, that a knob is
selected whose error rate in that population exceeds ``alpha`` is at most
``delta``. It is a statement about one controller's selection at one target;
choosing among controllers, estimators or targets afterwards is not covered.
It covers the end-to-end error of the full adaptive pipeline (resampling,
backtracking, noisy humans, escalation) as a black box in the knob, and needs
no per-trajectory monotonicity, which is why it is used instead of conformal
risk control.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field
from scipy.stats import binom

PValueKind = Literal["binomial", "hoeffding"]


def _check_counts(errors: int, n: int, alpha: float) -> None:
    """Reject counts and targets outside their domain (review finding R13)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= errors <= n:
        raise ValueError(f"errors must lie in 0..n, got {errors} of {n}")
    if not 0.0 < alpha < 1.0 or math.isnan(alpha):
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")


def binomial_pvalue(errors: int, n: int, alpha: float) -> float:
    """Exact one-sided p-value for ``H: R > alpha``: ``P(Bin(n, alpha) <= errors)``."""
    _check_counts(errors, n, alpha)
    return float(binom.cdf(errors, n, alpha))


def hoeffding_pvalue(errors: int, n: int, alpha: float) -> float:
    """Hoeffding p-value ``exp(-2 n (alpha - errors/n)_+^2)`` (looser than the exact one)."""
    _check_counts(errors, n, alpha)
    gap = max(0.0, alpha - errors / n)
    return float(math.exp(-2.0 * n * gap * gap))


class LTTRow(BaseModel):
    """One grid point's evidence."""

    knob: float
    n: int
    errors: int
    risk_hat: float
    pvalue: float
    rejected: bool
    tested: bool


class LTTResult(BaseModel):
    """Outcome of fixed-sequence testing for one policy (and seed)."""

    policy: str
    seed: int | None = None
    alpha: float
    delta: float
    pvalue_kind: PValueKind
    rows: list[LTTRow] = Field(default_factory=list)
    knob_star: float | None = None
    feasible: bool = False
    min_risk_hat: float = 1.0
    #: configured knobs that had no point; the walk ends before the first of them
    missing_knobs: list[float] = Field(default_factory=list)

    @property
    def selected(self) -> LTTRow | None:
        """The selected row, if any."""
        return next((r for r in self.rows if r.knob == self.knob_star), None)


def _same_knob(a: float, b: float) -> bool:
    """Two knobs are the same grid point (a manifest stores them as text)."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def calibrate_knob(
    policy: str,
    points: Sequence[tuple[float, int, int]],
    *,
    alpha: float,
    delta: float,
    pvalue: PValueKind = "binomial",
    seed: int | None = None,
    grid: Sequence[float] | None = None,
) -> LTTResult:
    """Fixed-sequence Learn-then-Test over ``points = [(knob, n, errors), ...]``.

    Points are sorted by knob ascending, which is the most-conservative-first
    order for every shipped policy (``KNOB_ASCENDING_IS_CONSERVATIVE``).

    ``grid`` is the configured sequence of knobs. The test is valid for that sequence,
    so with it the walk follows the grid and ends before the first knob that has no
    point: a knob that was not run cannot be passed, and no knob after it can be
    selected. ``missing_knobs`` of the result names the configured knobs without a
    point. A point at a knob that the grid does not have is an error. Without ``grid``
    the points are taken as the whole sequence, as before.
    """
    if not points:
        raise ValueError("no grid points")
    if not 0.0 < delta < 1.0 or math.isnan(delta):
        raise ValueError(f"delta must lie strictly between 0 and 1, got {delta}")
    knobs = [float(k) for k, _, _ in points]
    if any(math.isnan(k) or math.isinf(k) for k in knobs):
        raise ValueError(f"knobs must be finite, got {knobs}")
    if len(set(knobs)) != len(knobs):
        raise ValueError(f"knobs must be distinct, got {knobs}")
    missing: list[float] = []
    if grid is not None:
        configured = sorted(float(g) for g in grid)
        strangers = [k for k in knobs if not any(_same_knob(k, g) for g in configured)]
        if strangers:
            raise ValueError(
                f"{policy}: points at knobs that the configured grid does not have: "
                f"{sorted(strangers)}"
            )
        missing = [g for g in configured if not any(_same_knob(k, g) for k in knobs)]
    first_gap = min(missing) if missing else math.inf
    pv = binomial_pvalue if pvalue == "binomial" else hoeffding_pvalue
    rows: list[LTTRow] = []
    stopped = False
    knob_star: float | None = None
    for knob, n, errors in sorted(points, key=lambda t: t[0]):
        if knob > first_gap:
            stopped = True  # a configured knob before this one was never run
        p = pv(errors, n, alpha)
        rejected = (not stopped) and p <= delta
        rows.append(
            LTTRow(
                knob=knob,
                n=n,
                errors=errors,
                risk_hat=errors / n,
                pvalue=p,
                rejected=rejected,
                tested=not stopped,
            )
        )
        if stopped:
            continue
        if rejected:
            knob_star = knob
        else:
            stopped = True
    return LTTResult(
        policy=policy,
        seed=seed,
        alpha=alpha,
        delta=delta,
        pvalue_kind=pvalue,
        rows=rows,
        knob_star=knob_star,
        feasible=knob_star is not None,
        min_risk_hat=min(r.risk_hat for r in rows),
        missing_knobs=missing,
    )
