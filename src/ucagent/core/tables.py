"""Lookup tables the graph and the controllers consult at run time.

All entries are optional: with none fitted, ``eps = u``, ``w = 1``, ``G = 1``
and resampling is assumed to have no effect (so it is never chosen). Milestone
M2 fills the calibrators, the resample-effect table and the future factor; M6
fills the influence.

Calibrators are kept per sample count (``K`` and ``K'``): a step's ``u`` is
mapped through the map fitted for batches of its size, and an unfitted size
is an error rather than a silent fallback.

The effect of a resample is looked up in one of two forms. An artifact fitted
since schema version 2 carries a monotone map of the score (``resample_map``),
which is used when present. Earlier artifacts carry only a table of means over
fixed ranges of the score, read as they were fitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

#: A score within this distance below a range edge counts as on the edge: ``1 - 0.8`` is
#: ``0.19999999999999996``, so without it a four-to-one split fell into the range of the
#: unanimous batches. Tables fitted before schema version 2 are read with a tolerance of 0,
#: which is how they were fitted.
BUCKET_TOL = 1e-9


def bucket_of(u: float, edges: Sequence[float], tol: float = BUCKET_TOL) -> int:
    """Index of the range containing ``u`` (edges are lower bounds, met up to ``tol``)."""
    b = 0
    for edge in edges[1:]:
        if u >= edge - tol:
            b += 1
    return b


class Calibrator(Protocol):
    """Maps step uncertainty ``u`` to a local error probability ``eps``."""

    def __call__(self, u: float) -> float:
        """Calibrated local error probability for uncertainty ``u``."""
        ...


@dataclass
class Tables:
    """Run-time lookups; see the module docstring for defaults."""

    # sample count -> calibrator: a callable of u (isotonic) or an object with predict(features)
    calibrators: dict[int, Any] = field(default_factory=dict)
    influence: dict[tuple[int, int], float] = field(default_factory=dict)
    future_factor: dict[tuple[int, int], float] = field(default_factory=dict)
    resample_effect: dict[tuple[int, int], float] = field(default_factory=dict)
    resample_bucket_edges: list[float] = field(default_factory=list)
    resample_bucket_tol: float = BUCKET_TOL
    # expanded sample count -> map of the score at K to the expected error after the expansion
    resample_map: dict[int, Calibrator] = field(default_factory=dict)
    eps_mean_by_position: dict[tuple[int, int], float] = field(default_factory=dict)
    # sample count -> sorted train scores ``1 - p(correct cluster)`` (KnowNo-style gate)
    conformal_scores: dict[int, list[float]] = field(default_factory=dict)

    def eps(self, u: float, n_samples: int, features: Mapping[str, float] | None = None) -> float:
        """Calibrated local error probability for a batch of ``n_samples`` samples.

        Identity before calibration. Once calibrated, only the fitted sample
        counts are accepted: mapping a ``K'``-sample batch through the ``K`` map
        was review finding R1. A feature calibrator reads ``features`` (the named
        step features, see ``core.uq.features``); an isotonic map of ``u`` reads ``u``.
        """
        if not self.calibrators:
            return float(u)
        calibrator = self.calibrators.get(n_samples)
        if calibrator is None:
            raise ValueError(
                f"no calibrator fitted for batches of {n_samples} samples "
                f"(fitted: {sorted(self.calibrators)})"
            )
        predict = getattr(calibrator, "predict", None)
        if predict is not None:
            return float(predict(features if features is not None else {"u": u}))
        return float(calibrator(u))

    def w(self, n: int, i: int) -> float:
        """Downstream influence of step ``i`` in an ``n``-step task (default 1)."""
        return self.influence.get((n, i), 1.0)

    def G(self, n: int, i: int) -> float:
        """Expected survival of the steps after ``i`` (default 1)."""
        return self.future_factor.get((n, i), 1.0)

    def ehat_r(self, u: float, k_prime: int) -> float | None:
        """Expected error after expanding a batch with score ``u`` to ``k_prime`` samples.

        The monotone map when the artifact has one for ``k_prime``, else the mean of the
        range of ``u`` in the table of an earlier artifact; None if neither knows.
        """
        fitted = self.resample_map.get(k_prime)
        if fitted is not None:
            return float(fitted(u))
        if not self.resample_bucket_edges:
            return None
        bucket = bucket_of(u, self.resample_bucket_edges, self.resample_bucket_tol)
        return self.resample_effect.get((k_prime, bucket))

    def ebar(self, n: int, pos: int) -> float | None:
        """Mean calibrated ``eps`` at position ``pos`` (None if unknown)."""
        return self.eps_mean_by_position.get((n, pos))

    def qhat(self, n_samples: int, eps: float) -> float | None:
        """Conformal quantile at miscoverage ``eps`` for steps with ``n_samples`` samples.

        The ``ceil((m + 1)(1 - eps))``-th smallest of the ``m`` train scores; ``1.0``
        (every cluster qualifies) when that rank exceeds ``m``; None without scores.
        """
        scores = self.conformal_scores.get(n_samples)
        if not scores:
            return None
        m = len(scores)
        rank = math.ceil((m + 1) * (1.0 - eps))
        if rank > m:
            return 1.0
        return float(scores[max(rank, 1) - 1])
