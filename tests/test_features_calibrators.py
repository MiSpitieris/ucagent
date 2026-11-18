"""Named step features, the calibrator registry, and their use through the artifact and graph."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.helpers.configs import small_config
from ucagent.api import fit_calibrators
from ucagent.core.records import Cluster, Sample
from ucagent.core.uq.base import ESTIMATORS, estimator_reads, register_estimator
from ucagent.core.uq.calibration import StepObservation, fit_artifact, load_artifact
from ucagent.core.uq.calibrators import (
    CALIBRATORS,
    fit_calibrator,
    load_calibrator,
    register_calibrator,
)
from ucagent.core.uq.features import FEATURE_NAMES, check_features, step_features
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import Worker


def _samples(answers: list[str], confs: list[float | None]) -> list[Sample]:
    return [
        Sample(sample_index=k, text=a, answer=a, confidence=c)
        for k, (a, c) in enumerate(zip(answers, confs, strict=True))
    ]


def test_step_features_by_hand() -> None:
    samples = _samples(["B", "B", "B", "C"], [0.9, 0.7, None, 0.5])
    clusters = [
        Cluster(representative="B", members=[0, 1, 2], proportion=0.75),
        Cluster(representative="C", members=[3], proportion=0.25),
    ]
    f = step_features(0.4, samples, clusters, step=2, n_steps=3)
    assert set(f) == set(FEATURE_NAMES)
    assert f["u"] == 0.4 and f["disagreement"] == pytest.approx(0.25) and f["n_clusters"] == 2.0
    assert f["conf_mean"] == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert f["conf_min"] == pytest.approx(0.5)
    assert f["conf_top"] == pytest.approx((0.9 + 0.7 + 1.0) / 3)  # a missing confidence counts as 1
    assert f["conf_agreement"] == pytest.approx(1 - (0.9 + 0.7 + 1.0) / 4)
    assert f["has_conf"] == pytest.approx(0.75) and f["position"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        check_features(["u", "u"])
    with pytest.raises(ValueError):
        check_features(["nope"])


def _rows(n: int = 120) -> tuple[list[dict[str, float]], list[bool]]:
    rows, labels = [], []
    for k in range(n):
        u = (k % 10) / 10
        conf = 1.0 - ((k * 7) % 10) / 20
        rows.append({"u": u, "conf_mean": conf, "position": float(k % 2)})
        labels.append(u + (1 - conf) > 0.6)
    return rows, labels


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_each_calibrator_fits_predicts_and_round_trips(name: str) -> None:
    rows, labels = _rows()
    features = ["u"] if name == "isotonic" else ["u", "conf_mean", "position"]
    cal = fit_calibrator(name, features, rows, labels)
    probs = [cal.predict(r) for r in rows]
    assert all(0.0 <= p <= 1.0 for p in probs)
    hi = [p for p, y in zip(probs, labels, strict=True) if y]
    lo = [p for p, y in zip(probs, labels, strict=True) if not y]
    assert sum(hi) / len(hi) > sum(lo) / len(lo)  # it separates the classes on average
    back = load_calibrator(cal.to_data())
    assert [back.predict(r) for r in rows] == pytest.approx(probs)


def test_single_class_labels_give_a_constant() -> None:
    rows, _ = _rows(20)
    cal = fit_calibrator("logistic", ["u"], rows, [False] * 20)
    assert cal.predict(rows[0]) == 0.0 and load_calibrator(cal.to_data()).predict(rows[3]) == 0.0
    with pytest.raises(KeyError, match="unknown calibrator"):
        fit_calibrator("nope", ["u"], rows, [True] * 20)


def test_a_plugin_calibrator_fits_and_round_trips() -> None:
    class ThresholdStep:
        """A plugin calibrator (test only): the training error rate above and below a cut."""

        name = "test_threshold"

        def __init__(self, features, cut, lo, hi) -> None:
            self.features, self.cut, self.lo, self.hi = features, cut, lo, hi

        @classmethod
        def fit(cls, features, rows, labels):
            xs = [r[features[0]] for r in rows]
            cut = sorted(xs)[len(xs) // 2]
            above = [y for x, y in zip(xs, labels, strict=True) if x >= cut]
            below = [y for x, y in zip(xs, labels, strict=True) if x < cut] or [False]
            return cls(features, cut, sum(below) / len(below), sum(above) / len(above))

        def predict(self, x):
            return self.hi if x[self.features[0]] >= self.cut else self.lo

        def to_data(self):
            return {
                "name": self.name,
                "features": self.features,
                "cut": self.cut,
                "lo": self.lo,
                "hi": self.hi,
            }

        @classmethod
        def from_data(cls, data):
            return cls(list(data["features"]), data["cut"], data["lo"], data["hi"])

    register_calibrator("test_threshold", ThresholdStep)
    try:
        rows, labels = _rows()
        cal = fit_calibrator("test_threshold", ["u"], rows, labels)
        back = load_calibrator(cal.to_data())
        assert [back.predict(r) for r in rows] == [cal.predict(r) for r in rows]
        assert back.predict({"u": 0.9}) > back.predict({"u": 0.0})
    finally:
        del CALIBRATORS["test_threshold"]
    with pytest.raises(KeyError, match="not registered"):
        load_calibrator(cal.to_data())


def test_config_validates_calibrator_and_features() -> None:
    cfg = ExperimentConfig(
        run_id="x", uq={"calibrator": "logistic", "features": ["u", "conf_mean"]}
    )
    assert cfg.uq.features == ["u", "conf_mean"]
    with pytest.raises(ValidationError):
        ExperimentConfig(run_id="x", uq={"calibrator": "isotonic", "features": ["u", "conf_mean"]})
    with pytest.raises(ValidationError):
        ExperimentConfig(run_id="x", uq={"features": ["not_a_feature"]})


def _obs() -> list[StepObservation]:
    rows, labels = _rows(80)
    return [
        StepObservation(
            n_steps=2,
            step=1 + k % 2,
            u_K=r["u"],
            wrong_K=y,
            u_K_prime=r["u"],
            wrong_K_prime=y,
            x_K=r,
            x_K_prime=r,
        )
        for k, (r, y) in enumerate(zip(rows, labels, strict=True))
    ]


def test_artifact_with_a_feature_calibrator_routes_features(tmp_path) -> None:
    art = fit_artifact(
        _obs(),
        estimator="disagreement",
        equivalence="normalized_exact",
        K=5,
        K_prime=10,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=40,
        calibrator="logistic",
        features=["u", "conf_mean"],
    )
    assert not art.plain_isotonic and set(art.step_models) == {"K", "K_prime"}
    back = load_artifact(art.save(tmp_path / "a.json"))
    tables = back.to_tables()
    confident = tables.eps(0.2, 5, {"u": 0.2, "conf_mean": 1.0, "position": 0.0})
    unsure = tables.eps(0.2, 5, {"u": 0.2, "conf_mean": 0.5, "position": 0.0})
    assert unsure > confident  # the same u, a lower stated confidence: a higher error estimate
    with pytest.raises(ValueError, match="calibrator"):
        back.check_compatible("disagreement", 5, 10, calibrator="isotonic", features=["u"])
    # the default artifact is unchanged: isotonic on u, no stored models
    plain = fit_artifact(
        _obs(),
        estimator="disagreement",
        equivalence="normalized_exact",
        K=5,
        K_prime=10,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=40,
    )
    assert plain.plain_isotonic and plain.step_models == {}


def test_feature_calibrator_end_to_end_on_the_mock() -> None:
    cfg = small_config(
        task={"n_steps": 2, "n_instances": {"train": 12, "calib": 2, "test": 4}},
        llm={"K": 3, "K_prime": 6},
        uq={
            "estimator": "semantic_entropy",
            "calibrator": "logistic",
            "features": ["u", "conf_mean", "position"],
        },
    )
    art = fit_calibrators(cfg, workers=1)
    assert art.calibrator == "logistic" and art.features == ["u", "conf_mean", "position"]
    worker = Worker.from_config(cfg.model_copy(update={"split": "test"}))
    worker.tables = art.to_tables(cfg)
    traj = worker.run_instance(next(iter(worker.task.instances("test"))).instance_id)
    assert all(0.0 <= s.eps <= 1.0 for s in traj.steps)
    assert all("f_conf_mean" in s.extra and "f_position" in s.extra for s in traj.steps)


def test_estimators_declare_what_they_read() -> None:
    assert estimator_reads("disagreement") == {"answer"}
    assert estimator_reads("semantic_entropy") == {"answer"}
    assert estimator_reads("verbalized") == {"answer", "confidence"}
    assert estimator_reads("confidence_agreement") == {"answer", "confidence"}

    class LengthEstimator:
        """A plugin estimator: longer answers are less certain (for the test only)."""

        name = "test_length"
        reads = frozenset({"answer", "text"})

        def __init__(self, judge) -> None:
            self.inner = ESTIMATORS["disagreement"](judge)

        def estimate(self, samples, context):
            out = self.inner.estimate(samples, context)
            return out.model_copy(update={"u": min(1.0, len(out.committed) / 50)})

    register_estimator("test_length", LengthEstimator)
    assert estimator_reads("test_length") == {"answer", "text"}
    with pytest.raises(KeyError):
        estimator_reads("not_registered")


def test_a_factory_function_declares_what_its_estimator_declares() -> None:
    # code review of 2026-11-11, C05: a lambda used to read as "answer" only
    register_estimator("test_verbalized_factory", lambda judge: ESTIMATORS["verbalized"](judge))
    assert estimator_reads("test_verbalized_factory") == estimator_reads("verbalized")
    cfg = small_config(
        task={"n_steps": 1, "ambiguity": 0.0, "n_instances": {"train": 3, "calib": 1, "test": 1}},
        llm={"K": 1, "K_prime": 2, "verbalized": False},
        uq={"estimator": "test_verbalized_factory"},
    )
    with pytest.raises(ValueError, match="reads stated confidences"):
        fit_calibrators(cfg, workers=1)


def test_fit_refuses_an_estimator_that_reads_a_field_no_run_records() -> None:
    class RawTextEstimator:
        """Reads the raw reply, which a run does not record (for the test only)."""

        name = "test_raw_text"
        reads = frozenset({"answer", "text"})

        def __init__(self, judge) -> None:
            self.inner = ESTIMATORS["disagreement"](judge)

        def estimate(self, samples, context):
            return self.inner.estimate(samples, context)

    register_estimator("test_raw_text", RawTextEstimator)
    cfg = small_config(
        task={"n_steps": 1, "ambiguity": 0.0, "n_instances": {"train": 3, "calib": 1, "test": 1}},
        llm={"K": 1, "K_prime": 2},
        uq={"estimator": "test_raw_text"},
    )
    with pytest.raises(ValueError, match=r"reads \['text'\], which a run does not record"):
        fit_calibrators(cfg, workers=1)


def test_fit_refuses_confidence_features_without_confidences() -> None:
    cfg = small_config(
        task={"n_steps": 1, "ambiguity": 0.0, "n_instances": {"train": 3, "calib": 1, "test": 1}},
        llm={"K": 1, "K_prime": 2, "verbalized": False},
        uq={"estimator": "disagreement", "calibrator": "logistic", "features": ["u", "conf_mean"]},
    )
    with pytest.raises(ValueError, match="reads stated"):
        fit_calibrators(cfg, workers=1)
