"""Confidence-weighted agreement: one minus the confidence-weighted share of the top cluster.

``u = 1 - (sum of the stated confidences of the samples in the largest cluster) / K``.

The disagreement estimator reads only the share of the largest cluster; the
verbalized estimator reads only the stated confidences. This one weights the
majority share by how sure the model said it was: a unanimous batch with mean
confidence 0.9 gets ``u = 0.1``, a 4-of-5 split at confidence 1.0 gets
``u = 0.2``, and a unanimous batch at confidence 1.0 gets ``u = 0``. A sample
without a stated confidence counts as fully confident, so without any
confidences the estimator reduces to disagreement.
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


class ConfidenceAgreementEstimator:
    """Majority share weighted by the stated confidence of the majority's samples."""

    name = "confidence_agreement"
    reads = frozenset({"answer", "confidence"})

    def __init__(self, judge: EquivalenceJudge) -> None:
        self.judge = judge

    def estimate(self, samples: Sequence[Sample], context: StepContext) -> StepUncertainty:
        """``u = 1 - sum_{s in top cluster} conf_s / K`` with ``conf_s = 1`` when unstated."""
        if not samples:
            raise ValueError("estimate() needs at least one sample")
        clusters = cluster_samples(samples, self.judge, context)
        top = clusters[0]
        by_index = {s.sample_index: s for s in samples}
        weighted = 0.0
        for member in top.members:
            confidence = by_index[member].confidence
            weighted += confidence if confidence is not None else 1.0
        u = 1.0 - weighted / len(samples)
        n_conf = sum(1 for s in samples if s.confidence is not None)
        return StepUncertainty(
            u=float(min(1.0, max(0.0, u))),
            entropy=normalized_entropy([c.proportion for c in clusters], len(samples)),
            clusters=clusters,
            committed=top.representative,
            extra={"n_with_confidence": float(n_conf), "disagreement": 1.0 - top.proportion},
        )


register_estimator("confidence_agreement", ConfidenceAgreementEstimator)
