"""Disagreement estimator: ``u = 1 - max_j p_j`` over equivalence clusters."""

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


class DisagreementEstimator:
    """``u = 1 - max_j p_j``; also reports the normalised semantic entropy."""

    name = "disagreement"
    reads = frozenset({"answer"})

    def __init__(self, judge: EquivalenceJudge) -> None:
        self.judge = judge

    def estimate(self, samples: Sequence[Sample], context: StepContext) -> StepUncertainty:
        """Cluster the samples and return disagreement as ``u``."""
        if not samples:
            raise ValueError("estimate() needs at least one sample")
        clusters = cluster_samples(samples, self.judge, context)
        top = clusters[0]
        entropy = normalized_entropy([c.proportion for c in clusters], len(samples))
        return StepUncertainty(
            u=float(1.0 - top.proportion),
            entropy=entropy,
            clusters=clusters,
            committed=top.representative,
            extra={"n_clusters": float(len(clusters))},
        )


register_estimator("disagreement", DisagreementEstimator)
