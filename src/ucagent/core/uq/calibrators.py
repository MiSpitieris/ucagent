"""Step calibrators: named features of a step to the probability that it is locally wrong.

The calibrator is the one stage between an estimator and the controllers. It is
fitted on the training split for each batch size (``K`` and ``K'``) and stored in
the calibration artifact. Four are registered:

- ``isotonic``: monotone step function of one feature (default: ``u``), the
  calibrator of the earlier versions;
- ``logistic``: logistic regression on standardised features, stored as
  coefficients;
- ``boosted``: gradient-boosted trees (small, regularised), stored as a pickle;
- ``gp``: a Gaussian-process classifier with an ARD RBF kernel on standardised
  features (Laplace approximation), stored as a pickle.

Pickled models are tied to the installed scikit-learn version; a version change
means refitting the artifact.

A plugin calibrator is a class with a ``fit(features, rows, labels)`` class method,
``predict(row)``, ``to_data()`` and a ``from_data(data)`` class method that inverts
``to_data``, registered with :func:`register_calibrator`.
"""

from __future__ import annotations

import base64
import math
import pickle
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self

import numpy as np

from ucagent.core.uq.features import check_features, select


class StepCalibrator(Protocol):
    """A fitted map from named step features to a local error probability."""

    name: str
    features: list[str]

    def predict(self, x: Mapping[str, float]) -> float:
        """Probability that the step is locally wrong."""
        ...

    def to_data(self) -> dict[str, Any]:
        """JSON-serialisable state (see :func:`load_calibrator`)."""
        ...


def _matrix(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([select(r, names) for r in rows], dtype=float)


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


class _Constant:
    """Fallback when the training labels have a single class."""

    name = "constant"

    def __init__(self, features: list[str], value: float) -> None:
        self.features = features
        self.value = float(value)

    def predict(self, x: Mapping[str, float]) -> float:
        """The single training label's value."""
        return self.value

    def to_data(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {"name": self.name, "features": self.features, "value": self.value}


class IsotonicStep:
    """Isotonic regression of the label on one feature (clipped linear interpolation)."""

    name = "isotonic"

    def __init__(self, features: list[str], x: list[float], y: list[float], n: int = 0) -> None:
        if len(features) != 1:
            raise ValueError("the isotonic calibrator reads exactly one feature")
        self.features, self.x, self.y, self.n = features, x, y, n

    @classmethod
    def fit(
        cls, features: list[str], rows: Sequence[Mapping[str, float]], labels: Sequence[bool]
    ) -> Self:
        """Fit on named feature rows and 0/1 labels."""
        from ucagent.core.uq.calibration import IsotonicCalibrator

        cal = IsotonicCalibrator.fit(
            [select(r, features)[0] for r in rows], [int(v) for v in labels]
        )
        return cls(features, cal.data.x, cal.data.y, cal.data.n)

    def predict(self, x: Mapping[str, float]) -> float:
        """Probability that the step is locally wrong."""
        value = select(x, self.features)[0]
        return float(np.clip(np.interp(value, self.x, self.y), 0.0, 1.0))

    def to_data(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {"name": self.name, "features": self.features, "x": self.x, "y": self.y, "n": self.n}


class LogisticStep:
    """Logistic regression on standardised features."""

    name = "logistic"

    def __init__(
        self,
        features: list[str],
        mean: list[float],
        scale: list[float],
        coef: list[float],
        b: float,
    ) -> None:
        self.features, self.mean, self.scale, self.coef, self.b = features, mean, scale, coef, b

    @classmethod
    def fit(
        cls, features: list[str], rows: Sequence[Mapping[str, float]], labels: Sequence[bool]
    ) -> Self:
        """Fit on named feature rows and 0/1 labels."""
        from sklearn.linear_model import LogisticRegression

        x = _matrix(rows, features)
        mean, scale = _standardise(x)
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit((x - mean) / scale, np.asarray(labels, dtype=int))
        return cls(
            features,
            [float(v) for v in mean],
            [float(v) for v in scale],
            [float(v) for v in model.coef_[0]],
            float(model.intercept_[0]),
        )

    def predict(self, x: Mapping[str, float]) -> float:
        """Probability that the step is locally wrong."""
        values = select(x, self.features)
        z = self.b + sum(
            c * (v - m) / s
            for c, v, m, s in zip(self.coef, values, self.mean, self.scale, strict=True)
        )
        return float(1.0 / (1.0 + math.exp(-z)))

    def to_data(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {
            "name": self.name,
            "features": self.features,
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "b": self.b,
        }


class _PickledStep:
    """Shared machinery of the scikit-learn models stored as pickles."""

    name = ""

    def __init__(
        self, features: list[str], model: Any, mean: list[float], scale: list[float]
    ) -> None:
        self.features, self.model, self.mean, self.scale = features, model, mean, scale

    @classmethod
    def _new_model(cls, n_features: int) -> Any:
        raise NotImplementedError

    @classmethod
    def fit(
        cls, features: list[str], rows: Sequence[Mapping[str, float]], labels: Sequence[bool]
    ) -> Self:
        """Fit on named feature rows and 0/1 labels."""
        x = _matrix(rows, features)
        mean, scale = _standardise(x)
        model = cls._new_model(len(features))
        model.fit((x - mean) / scale, np.asarray(labels, dtype=int))
        return cls(features, model, [float(v) for v in mean], [float(v) for v in scale])

    def predict(self, x: Mapping[str, float]) -> float:
        """Probability that the step is locally wrong."""
        row = (np.asarray(select(x, self.features), dtype=float) - self.mean) / self.scale
        return float(self.model.predict_proba(row.reshape(1, -1))[0, 1])

    def to_data(self) -> dict[str, Any]:
        """JSON-serialisable state (the model as a base64 pickle)."""
        return {
            "name": self.name,
            "features": self.features,
            "mean": self.mean,
            "scale": self.scale,
            "pickle": base64.b64encode(pickle.dumps(self.model)).decode("ascii"),
        }


class BoostedStep(_PickledStep):
    """Small, regularised gradient-boosted trees."""

    name = "boosted"

    @classmethod
    def _new_model(cls, n_features: int) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0
        )


class GPStep(_PickledStep):
    """Gaussian-process classifier, ARD RBF kernel, Laplace approximation."""

    name = "gp"

    @classmethod
    def _new_model(cls, n_features: int) -> Any:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        return GaussianProcessClassifier(kernel=kernel, random_state=0)


CALIBRATORS: dict[str, Any] = {
    "isotonic": IsotonicStep,
    "logistic": LogisticStep,
    "boosted": BoostedStep,
    "gp": GPStep,
}


def register_calibrator(name: str, cls: Any) -> None:
    """Make a calibrator class available by name (from a plugin module)."""
    CALIBRATORS[name] = cls


def fit_calibrator(
    name: str,
    features: Sequence[str],
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[bool],
) -> StepCalibrator:
    """Fit the named calibrator on ``rows`` (named features) and 0/1 ``labels``."""
    if name not in CALIBRATORS:
        raise KeyError(f"unknown calibrator {name!r}; known: {sorted(CALIBRATORS)}")
    names = check_features(features)
    if not rows or len(rows) != len(labels):
        raise ValueError("fit_calibrator needs equally long, non-empty rows and labels")
    if len({bool(v) for v in labels}) < 2:
        return _Constant(names, float(bool(labels[0])))
    fitted: StepCalibrator = CALIBRATORS[name].fit(names, rows, labels)
    return fitted


def load_calibrator(data: Mapping[str, Any]) -> StepCalibrator:
    """Inverse of ``to_data``."""
    name, features = data["name"], list(data["features"])
    if name == "constant":
        return _Constant(features, data["value"])
    if name == "isotonic":
        return IsotonicStep(features, list(data["x"]), list(data["y"]), int(data.get("n", 0)))
    if name == "logistic":
        return LogisticStep(
            features, list(data["mean"]), list(data["scale"]), list(data["coef"]), float(data["b"])
        )
    cls = CALIBRATORS.get(name)
    if cls is None:
        raise KeyError(f"cannot load calibrator {name!r}: not registered (is its plugin listed?)")
    if isinstance(cls, type) and issubclass(cls, _PickledStep):
        model = pickle.loads(base64.b64decode(data["pickle"]))  # own artifact files only
        loaded: StepCalibrator = cls(features, model, list(data["mean"]), list(data["scale"]))
        return loaded
    if not hasattr(cls, "from_data"):
        raise KeyError(f"cannot load calibrator {name!r}: its class has no from_data")
    plugin: StepCalibrator = cls.from_data(data)
    return plugin
