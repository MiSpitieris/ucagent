import math

import pytest

from ucagent.core.records import Sample
from ucagent.core.uq.agreement import DisagreementEstimator
from ucagent.core.uq.base import StepContext, make_estimator, normalized_entropy
from ucagent.core.uq.equivalence import NormalizedExactMatch, cluster_samples, make_judge
from ucagent.core.uq.semantic_entropy import SemanticEntropyEstimator
from ucagent.core.uq.verbalized import VerbalizedEstimator

CTX = StepContext(instance_id="x", step=1, attempt=0, current="A", n_steps=2)


def _samples(answers: list[str], confs: list[float] | None = None) -> list[Sample]:
    return [
        Sample(sample_index=k, text=a, answer=a, confidence=None if confs is None else confs[k])
        for k, a in enumerate(answers)
    ]


def test_hand_computed_entropy_and_disagreement() -> None:
    samples = _samples(["B", "B", "C", "D"])  # clusters 2/1/1
    h = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert normalized_entropy([0.5, 0.25, 0.25], 4) == pytest.approx(h / math.log(4))
    assert normalized_entropy([0.5, 0.25, 0.25], 4) == pytest.approx(0.75)
    dis = DisagreementEstimator(NormalizedExactMatch()).estimate(samples, CTX)
    assert (
        dis.u == pytest.approx(0.5) and dis.committed == "B" and dis.entropy == pytest.approx(0.75)
    )
    se = SemanticEntropyEstimator(NormalizedExactMatch()).estimate(samples, CTX)
    assert se.u == pytest.approx(0.75) and se.committed == "B"
    assert se.extra["disagreement"] == pytest.approx(0.5)


def test_unanimous_and_single_sample() -> None:
    est = DisagreementEstimator(NormalizedExactMatch())
    assert est.estimate(_samples(["B"] * 10), CTX).u == 0.0
    assert est.estimate(_samples(["B"]), CTX).entropy == 0.0
    assert SemanticEntropyEstimator(NormalizedExactMatch()).estimate(
        _samples(["a", "b"]), CTX
    ).u == pytest.approx(1.0)
    with pytest.raises(ValueError):
        est.estimate([], CTX)


def test_clustering_uses_judge_and_is_order_independent() -> None:
    class Loose:
        name = "loose"

        def same(self, a: str, b: str, context: object) -> bool:
            return a[0].lower() == b[0].lower()

    samples = _samples(["Bob", "bill", "Carl", "ben"])
    clusters = cluster_samples(reversed(samples), Loose(), CTX)
    assert [c.members for c in clusters] == [[0, 1, 3], [2]]
    assert clusters[0].representative == "Bob" and clusters[0].proportion == 0.75
    exact = cluster_samples(samples, NormalizedExactMatch(), CTX)
    assert len(exact) == 4 and exact[0].members == [0]
    assert (
        cluster_samples(_samples(["The Cat", "cat", "Cat."]), NormalizedExactMatch(), CTX)[
            0
        ].proportion
        == 1.0
    )


def test_verbalized_estimator() -> None:
    est = VerbalizedEstimator(NormalizedExactMatch())
    out = est.estimate(_samples(["B", "B", "C"], [0.9, 0.7, 0.5]), CTX)
    assert out.u == pytest.approx(1 - 0.7) and out.committed == "B"
    fallback = est.estimate(_samples(["B", "B", "C"]), CTX)
    assert fallback.u == pytest.approx(1 / 3)


def test_confidence_agreement_estimator() -> None:
    est = make_estimator("confidence_agreement", NormalizedExactMatch())
    out = est.estimate(_samples(["B", "B", "C"], [0.9, 0.7, 0.5]), CTX)
    assert out.u == pytest.approx(1 - (0.9 + 0.7) / 3) and out.committed == "B"
    assert out.extra["n_with_confidence"] == 3.0 and out.extra["disagreement"] == pytest.approx(
        1 / 3
    )
    sure = est.estimate(_samples(["B"] * 5, [1.0] * 5), CTX)
    assert sure.u == 0.0
    hesitant = est.estimate(_samples(["B"] * 5, [0.9] * 5), CTX)
    assert hesitant.u == pytest.approx(0.1)
    split = est.estimate(_samples(["B", "B", "B", "B", "C"], [1.0] * 5), CTX)
    assert split.u == pytest.approx(0.2)
    # without confidences it is the disagreement estimator
    fallback = est.estimate(_samples(["B", "B", "C"]), CTX)
    assert fallback.u == pytest.approx(1 / 3)


def test_registries() -> None:
    judge = make_judge("normalized_exact")
    for name in ["disagreement", "semantic_entropy", "verbalized", "confidence_agreement"]:
        assert make_estimator(name, judge).name == name
    with pytest.raises(KeyError):
        make_estimator("nope", judge)
    with pytest.raises(KeyError):
        make_judge("nope")
