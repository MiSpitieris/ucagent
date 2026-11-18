"""Estimator interfaces: ``StepUncertaintyEstimator`` and its registry."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from ucagent.core.records import Cluster, Sample
from ucagent.core.uq.equivalence import EquivalenceJudge


class StepContext(BaseModel, frozen=True):
    """What an estimator may know about the step besides the samples."""

    instance_id: str
    step: int
    attempt: int
    current: str
    n_steps: int


class StepUncertainty(BaseModel, frozen=True):
    """Output of an estimator for one step attempt."""

    u: float
    entropy: float
    clusters: list[Cluster]
    committed: str
    extra: dict[str, float] = Field(default_factory=dict)


class StepUncertaintyEstimator(Protocol):
    """Turns ``K`` samples into ``u`` in ``[0, 1]`` plus clusters and a committed answer.

    ``reads`` names the sample fields the estimator uses. A run records ``answer`` and
    ``confidence`` (``RECORDED_FIELDS``), and the fit re-estimates every step from them. The
    fit therefore refuses an estimator that declares another field, such as ``text`` (the raw
    reply, which is not recorded), and one that reads stated confidences when the training run
    recorded none. Register a new estimator with :func:`register_estimator` from a plugin
    module.
    """

    name: str
    reads: frozenset[str]

    def estimate(self, samples: Sequence[Sample], context: StepContext) -> StepUncertainty:
        """Estimate the uncertainty of one step attempt."""
        ...


def normalized_entropy(proportions: Sequence[float], k: int) -> float:
    """Semantic entropy ``-sum p log p`` divided by ``log K`` (0 when ``K <= 1``)."""
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in proportions if p > 0)
    return float(min(1.0, max(0.0, h / math.log(k))))


EstimatorFactory = Callable[[EquivalenceJudge], StepUncertaintyEstimator]
ESTIMATORS: dict[str, EstimatorFactory] = {}
# the sample fields a run records and a fit can therefore re-estimate from
RECORDED_FIELDS = frozenset({"answer", "confidence"})


class _SameText:
    """A judge for building an estimator only to read what it declares."""

    name = "same_text"

    def same(self, a: str, b: str, context: object) -> bool:
        return a == b


def estimator_reads(name: str) -> frozenset[str]:
    """Sample fields the named estimator reads (``answer`` when it does not say).

    The declaration is taken from the registered factory when it carries one, as a class
    does, and otherwise from the estimator the factory builds. So ``MyEstimator`` and
    ``lambda judge: MyEstimator(judge)`` declare the same (code review of 2026-11-11, C05:
    a lambda used to read as ``answer`` only, which let an estimator of stated confidences
    be fitted on runs that recorded none).
    """
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; known: {sorted(ESTIMATORS)}")
    factory = ESTIMATORS[name]
    declared = getattr(factory, "reads", None)
    if declared is None:
        declared = getattr(factory(_SameText()), "reads", None)
    return frozenset({"answer"} if declared is None else declared)


def register_estimator(name: str, factory: EstimatorFactory) -> None:
    """Register an estimator factory under ``name``."""
    ESTIMATORS[name] = factory


def make_estimator(name: str, judge: EquivalenceJudge) -> StepUncertaintyEstimator:
    """Instantiate a registered estimator."""
    try:
        return ESTIMATORS[name](judge)
    except KeyError as exc:
        raise KeyError(f"unknown estimator {name!r}; known: {sorted(ESTIMATORS)}") from exc
