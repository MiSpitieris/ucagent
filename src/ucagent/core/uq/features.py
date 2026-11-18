"""Named step features: what a calibrator may read about one step.

An estimator returns a score ``u`` (plus clusters and a committed answer). The
framework then computes a fixed set of named features from the same samples, so
that a calibrator can map any subset of them, not only ``u``, to the probability
that the step is locally wrong. The default pipeline reads ``u`` alone, which is
the isotonic map of the earlier versions.

A missing stated confidence counts as 1 (fully confident) in the confidence
features, the convention of the confidence-weighted agreement estimator;
``has_conf`` records the share of samples that stated one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ucagent.core.records import Cluster, Sample

FEATURE_NAMES: tuple[str, ...] = (
    "u",
    "entropy",
    "disagreement",
    "n_clusters",
    "conf_mean",
    "conf_min",
    "conf_top",
    "conf_agreement",
    "has_conf",
    "position",
)

# features computed from stated confidences: a pipeline that reads them needs them persisted
CONFIDENCE_FEATURES: frozenset[str] = frozenset(
    {"conf_mean", "conf_min", "conf_top", "conf_agreement", "has_conf"}
)


def _normalized_entropy(proportions: Sequence[float], k: int) -> float:
    import math

    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in proportions if p > 0)
    return float(min(1.0, max(0.0, h / math.log(k))))


def step_features(
    u: float,
    samples: Sequence[Sample],
    clusters: Sequence[Cluster],
    step: int,
    n_steps: int,
) -> dict[str, float]:
    """The named features of one step's batch (see the module docstring)."""
    k = len(samples)
    if k == 0 or not clusters:
        raise ValueError("step_features needs at least one sample and one cluster")
    top = clusters[0]
    by_index = {s.sample_index: s for s in samples}
    stated = [c for c in (s.confidence for s in samples) if c is not None]
    top_conf: list[float] = []
    for m in top.members:
        conf = by_index[m].confidence
        top_conf.append(1.0 if conf is None else conf)
    return {
        "u": float(u),
        "entropy": _normalized_entropy([c.proportion for c in clusters], k),
        "disagreement": 1.0 - float(top.proportion),
        "n_clusters": float(len(clusters)),
        "conf_mean": float(sum(stated) / len(stated)) if stated else 1.0,
        "conf_min": float(min(stated)) if stated else 1.0,
        "conf_top": float(sum(top_conf) / len(top_conf)),
        "conf_agreement": 1.0 - float(sum(top_conf)) / k,
        "has_conf": len(stated) / k,
        "position": (step - 1) / max(n_steps - 1, 1),
    }


def check_features(names: Sequence[str]) -> list[str]:
    """Validate a feature list (non-empty, known names, no duplicates); returns it as a list."""
    names = list(names)
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(f"features must be distinct names from {list(FEATURE_NAMES)}, got {names}")
    return names


def select(features: Mapping[str, float], names: Sequence[str]) -> list[float]:
    """The values of ``names`` in order (a missing name is an error)."""
    try:
        return [float(features[n]) for n in names]
    except KeyError as exc:
        raise KeyError(f"feature {exc.args[0]!r} missing from {sorted(features)}") from exc
