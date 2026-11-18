"""Verbalized-confidence estimator (baseline): ``u = 1 - mean stated confidence``.

The mean runs over every sample that states a confidence, whichever cluster
it belongs to; samples without one are left out, and when none states one the
estimator falls back to disagreement (``1 - share of the largest cluster``).
Fitting it requires the confidences to be persisted with the samples
(``Trajectory.sample_confidences``), which review finding R3 found missing.
"""

from __future__ import annotations

from collections.abc import Sequence

from ucagent.core.records import Sample
from ucagent.core.uq.base import (
    StepContext,
    StepUncertainty,
    normalized_entropy,
    register_estimator,
)
from ucagent.core.uq.equivalence import EquivalenceJudge, cluster_samples


class VerbalizedEstimator:
    """Uses the confidence the model states; falls back to disagreement if absent."""

    name = "verbalized"
    reads = frozenset({"answer", "confidence"})

    def __init__(self, judge: EquivalenceJudge) -> None:
        self.judge = judge

    def estimate(self, samples: Sequence[Sample], context: StepContext) -> StepUncertainty:
        """One minus the mean stated confidence over the samples that state one."""
        if not samples:
            raise ValueError("estimate() needs at least one sample")
        clusters = cluster_samples(samples, self.judge, context)
        top = clusters[0]
        confs = [s.confidence for s in samples if s.confidence is not None]
        u = 1.0 - sum(confs) / len(confs) if confs else 1.0 - top.proportion
        return StepUncertainty(
            u=float(min(1.0, max(0.0, u))),
            entropy=normalized_entropy([c.proportion for c in clusters], len(samples)),
            clusters=clusters,
            committed=top.representative,
            extra={"n_with_confidence": float(len(confs))},
        )


register_estimator("verbalized", VerbalizedEstimator)
