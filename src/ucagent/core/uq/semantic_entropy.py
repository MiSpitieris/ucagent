"""Semantic-entropy estimator: ``u = H / log K`` over equivalence clusters."""

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


class SemanticEntropyEstimator:
    """``u`` is the normalised entropy of the cluster proportions."""

    name = "semantic_entropy"
    reads = frozenset({"answer"})

    def __init__(self, judge: EquivalenceJudge) -> None:
        self.judge = judge

    def estimate(self, samples: Sequence[Sample], context: StepContext) -> StepUncertainty:
        """Cluster the samples and return normalised semantic entropy as ``u``."""
        if not samples:
            raise ValueError("estimate() needs at least one sample")
        clusters = cluster_samples(samples, self.judge, context)
        entropy = normalized_entropy([c.proportion for c in clusters], len(samples))
        return StepUncertainty(
            u=entropy,
            entropy=entropy,
            clusters=clusters,
            committed=clusters[0].representative,
            extra={
                "n_clusters": float(len(clusters)),
                "disagreement": 1.0 - clusters[0].proportion,
            },
        )


register_estimator("semantic_entropy", SemanticEntropyEstimator)
