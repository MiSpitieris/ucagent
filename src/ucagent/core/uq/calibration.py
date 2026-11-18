"""Calibration artifacts: isotonic ``u -> eps``, resample effect, future factor.

Everything here is fitted on the ``train`` split and persisted as JSON with
the data and config fingerprints it was fitted for; the runner refuses an
artifact fitted for a different ``(estimator, K, K', equivalence, task)``.
``eps`` is always the probability of a *local* error (Section 2.1 of the
instructions), fitted once for batches of ``K`` samples and once for ``K'``.

The effect of a resample is the probability that the answer of the expanded
batch is locally wrong, given the score ``u`` of the batch at ``K`` samples,
where the ``K'`` batch is the ``K`` batch expanded by ``K'-K`` further draws
of the same prompt. That is exactly what the runtime resample action does (it
appends samples ``K..K'-1`` under the same key and re-estimates), so the
fitted effect is the effect of the action being valued (review finding R2).
It is fitted as ``resample_map``: an isotonic regression of the ``K'`` error
label on the ``K`` score, the twin of the ``K`` map on the same steps. The
difference of the two is then the net rate at which the expansion turns a
wrong answer into a right one, and it is zero where the expansion cannot
change the answer.

Until schema version 1 the effect was a table of the mean ``K'``-map ``eps``
over five fixed ranges of the ``K`` score (``resample_table``). One value per
range overstated the effect on the steps a controller chose to expand, and a
score of ``1 - 0.8`` fell below the edge at 0.2. The table is still fitted,
with the edge met up to a tolerance, as a diagnostic; the artifacts of earlier
runs carry only the table and are read as they were fitted.

Assumption flagged for ``MATH.md``: the resample effect and the future
factor are estimated under policy ``none``; actions change the distribution
of later steps, which the Learn-then-Test guarantee covers end to end but
these tables do not.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.isotonic import IsotonicRegression

from ucagent.core.tables import BUCKET_TOL, Tables, bucket_of
from ucagent.core.uq.calibrators import fit_calibrator, load_calibrator
from ucagent.core.utils import stable_digest

__all__ = [
    "DEFAULT_BUCKET_EDGES",
    "CalibrationArtifact",
    "IsotonicCalibrator",
    "IsotonicData",
    "ResampleTableData",
    "StepObservation",
    "bucket_of",
    "fit_artifact",
    "load_artifact",
]

DEFAULT_BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8]
#: Artifacts from this version on carry ``resample_map`` and a range table whose edges are met
#: up to ``BUCKET_TOL``; earlier tables are read with a tolerance of 0, as they were fitted.
SCHEMA_VERSION = 2


class IsotonicData(BaseModel):
    """A fitted isotonic step function (breakpoints) with its fit size."""

    x: list[float]
    y: list[float]
    n: int = 0

    def __call__(self, u: float) -> float:
        """Interpolated, clipped prediction."""
        if not self.x:
            return float(u)
        return float(np.clip(np.interp(u, self.x, self.y), 0.0, 1.0))


class IsotonicCalibrator:
    """Isotonic regression of local-error labels on ``u``; monotone, in ``[0, 1]``."""

    def __init__(self, data: IsotonicData | None = None) -> None:
        self.data = data or IsotonicData(x=[], y=[])

    @classmethod
    def fit(cls, u: Sequence[float], is_wrong: Sequence[bool | int]) -> IsotonicCalibrator:
        """Fit on ``(u, label)`` pairs (``label`` = 1 if the step was locally wrong)."""
        if len(u) != len(is_wrong) or not u:
            raise ValueError("fit() needs equally long, non-empty u and labels")
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        reg.fit(np.asarray(u, dtype=float), np.asarray(is_wrong, dtype=float))
        return cls(
            IsotonicData(
                x=[float(v) for v in reg.X_thresholds_],
                y=[float(v) for v in reg.y_thresholds_],
                n=len(u),
            )
        )

    def __call__(self, u: float) -> float:
        """Calibrated local error probability."""
        return self.data(u)


class ResampleTableData(BaseModel):
    """Mean calibrated ``eps`` after resampling, by range of ``u`` at ``K`` (diagnostic)."""

    bucket_edges: list[float] = Field(default_factory=lambda: list(DEFAULT_BUCKET_EDGES))
    k_prime: int = 0
    ehat: dict[str, float] = Field(default_factory=dict)  # bucket index -> mean eps at K'
    counts: dict[str, int] = Field(default_factory=dict)


class CalibrationArtifact(BaseModel):
    """Everything fitted on the train split, with fingerprints."""

    schema_version: int = SCHEMA_VERSION
    estimator: str
    equivalence: str
    K: int
    K_prime: int
    task_config_hash: str
    config_hash: str
    data_hash: str
    # what the fit was computed from: digests of the data files, the questions and the code
    # (``ucagent.experiments.provenance.record``); empty in artifacts written before 2026-11-12
    inputs: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    n_trajectories: int = 0
    isotonic_K: IsotonicData
    isotonic_K_prime: IsotonicData
    resample_table: ResampleTableData
    # isotonic regression of the K' error label on the K score: the expected error after a
    # resample; None in artifacts written before schema version 2, which use the table
    resample_map: IsotonicData | None = None
    future_factor: dict[str, float] = Field(default_factory=dict)  # "n:i"
    eps_mean_by_position: dict[str, float] = Field(default_factory=dict)  # "n:l"
    influence: dict[str, float] = Field(default_factory=dict)  # "n:i"
    influence_method: str = "ones"
    # "K" / "K_prime" -> sorted train scores 1 - p(correct cluster), for the knowno gate
    conformal_scores: dict[str, list[float]] = Field(default_factory=dict)
    # the step calibrator: ``isotonic`` on ``u`` reproduces the earlier artifacts exactly; any
    # other choice is stored in ``step_models`` ("K" / "K_prime" -> serialised calibrator)
    calibrator: str = "isotonic"
    features: list[str] = Field(default_factory=lambda: ["u"])
    step_models: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def plain_isotonic(self) -> bool:
        """Whether the step calibrator is the isotonic map of ``u`` (the stored isotonic fits)."""
        return self.calibrator == "isotonic" and self.features == ["u"]

    def check_compatible(
        self,
        estimator: str,
        K: int,
        K_prime: int,
        equivalence: str | None = None,
        task_config_hash: str | None = None,
        calibrator: str | None = None,
        features: Sequence[str] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Refuse use with a different estimator, sample counts, judge or task.

        ``equivalence`` and ``task_config_hash`` are compared when given (review
        finding R4: a table fitted on another task, model or matching rule must
        not be applied silently). ``inputs``, when given, must equal the record
        of the fit's inputs; an artifact without one does not pass.
        """
        if (estimator, K, K_prime) != (self.estimator, self.K, self.K_prime):
            raise ValueError(
                f"artifact fitted for estimator={self.estimator!r}, K={self.K}, "
                f"K_prime={self.K_prime}; run uses estimator={estimator!r}, "
                f"K={K}, K_prime={K_prime}"
            )
        if equivalence is not None and equivalence != self.equivalence:
            raise ValueError(
                f"artifact fitted with equivalence={self.equivalence!r}; "
                f"run uses equivalence={equivalence!r}"
            )
        if task_config_hash is not None and task_config_hash != self.task_config_hash:
            raise ValueError(
                f"artifact fitted on task config {self.task_config_hash}; "
                f"run uses task config {task_config_hash}"
            )
        if calibrator is not None and (calibrator, list(features or ["u"])) != (
            self.calibrator,
            self.features,
        ):
            raise ValueError(
                f"artifact fitted with calibrator {self.calibrator!r} on {self.features}; "
                f"run uses calibrator {calibrator!r} on {list(features or ['u'])}"
            )
        if inputs is not None and inputs != self.inputs:
            raise ValueError(
                "artifact fitted on other inputs (data files, questions or code)"
                + ("" if self.inputs else "; it records none")
            )

    def to_tables(self, config: Any = None) -> Tables:
        """Run-time tables; ``config`` (an ExperimentConfig) is checked when given.

        One calibrator per fitted sample count (``K`` and ``K'``), so a step is
        mapped through the map fitted for a batch of its size.
        """
        if config is not None:
            self.check_compatible(
                config.uq.estimator,
                config.llm.K,
                config.llm.K_prime,
                config.uq.equivalence,
                stable_digest(config.task.model_dump()),
                config.uq.calibrator,
                config.uq.features,
            )
        calibrators: dict[int, Any]
        if self.plain_isotonic:
            calibrators = {
                self.K: IsotonicCalibrator(self.isotonic_K),
                self.K_prime: IsotonicCalibrator(self.isotonic_K_prime),
            }
        else:
            calibrators = {
                self.K: load_calibrator(self.step_models["K"]),
                self.K_prime: load_calibrator(self.step_models["K_prime"]),
            }
        return Tables(
            calibrators=calibrators,
            influence={_key(k): v for k, v in self.influence.items()},
            future_factor={_key(k): v for k, v in self.future_factor.items()},
            resample_effect={
                (self.resample_table.k_prime, int(b)): v
                for b, v in self.resample_table.ehat.items()
            },
            resample_bucket_edges=list(self.resample_table.bucket_edges),
            resample_bucket_tol=BUCKET_TOL if self.schema_version >= SCHEMA_VERSION else 0.0,
            resample_map=(
                {self.K_prime: IsotonicCalibrator(self.resample_map)}
                if self.resample_map is not None
                else {}
            ),
            eps_mean_by_position={_key(k): v for k, v in self.eps_mean_by_position.items()},
            conformal_scores={
                k: list(v)
                for k, v in (
                    (self.K, self.conformal_scores.get("K", [])),
                    (self.K_prime, self.conformal_scores.get("K_prime", [])),
                )
                if v
            },
        )

    def save(self, path: str | Path) -> Path:
        """Write JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=1), encoding="utf-8")
        return path


def _key(text: str) -> tuple[int, int]:
    a, b = text.split(":")
    return int(a), int(b)


def load_artifact(path: str | Path) -> CalibrationArtifact:
    """Read an artifact JSON."""
    return CalibrationArtifact.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class StepObservation(BaseModel):
    """One step of a train trajectory, re-estimated at ``K`` and ``K'``."""

    n_steps: int
    step: int
    u_K: float
    wrong_K: bool
    u_K_prime: float
    wrong_K_prime: bool
    p_true_K: float = 0.0  # share of the locally correct cluster at K (0 if none)
    p_true_K_prime: float = 0.0
    x_K: dict[str, float] = Field(default_factory=dict)  # named step features at K
    x_K_prime: dict[str, float] = Field(default_factory=dict)  # and at K'


def fit_artifact(
    observations: Sequence[StepObservation],
    *,
    estimator: str,
    equivalence: str,
    K: int,
    K_prime: int,
    task_config_hash: str,
    config_hash: str,
    data_hash: str,
    n_trajectories: int,
    inputs: dict[str, Any] | None = None,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    trajectories: Sequence[Sequence[StepObservation]] = (),
    influence: dict[tuple[int, int], float] | None = None,
    influence_method: str = "ones",
    calibrator: str = "isotonic",
    features: Sequence[str] = ("u",),
) -> CalibrationArtifact:
    """Fit all tables from step observations (see ``api.fit_calibrators``).

    ``calibrator`` and ``features`` choose the step calibrator; the default (isotonic on
    ``u``) gives the step maps of the earlier versions. Any other choice is fitted
    on the observations' named features at ``K`` and at ``K'`` and drives the resample
    table, the position means and the future factor as the isotonic map does. The effect of
    a resample (``resample_map``) is fitted on the score and the ``K'`` label in either case.
    """
    if not observations:
        raise ValueError("no observations to fit on")
    iso_k = IsotonicCalibrator.fit([o.u_K for o in observations], [o.wrong_K for o in observations])
    iso_kp = IsotonicCalibrator.fit(
        [o.u_K_prime for o in observations], [o.wrong_K_prime for o in observations]
    )
    features = list(features)
    step_models: dict[str, dict[str, Any]] = {}
    if calibrator == "isotonic" and features == ["u"]:

        def eps_k(o: StepObservation) -> float:
            return iso_k(o.u_K)

        def eps_kp(o: StepObservation) -> float:
            return iso_kp(o.u_K_prime)

    else:
        if any(not o.x_K or not o.x_K_prime for o in observations):
            raise ValueError("a feature calibrator needs observations with named features")
        model_k = fit_calibrator(
            calibrator, features, [o.x_K for o in observations], [o.wrong_K for o in observations]
        )
        model_kp = fit_calibrator(
            calibrator,
            features,
            [o.x_K_prime for o in observations],
            [o.wrong_K_prime for o in observations],
        )
        step_models = {"K": model_k.to_data(), "K_prime": model_kp.to_data()}

        def eps_k(o: StepObservation) -> float:
            return model_k.predict(o.x_K)

        def eps_kp(o: StepObservation) -> float:
            return model_kp.predict(o.x_K_prime)

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for o in observations:
        b = bucket_of(o.u_K, bucket_edges)
        sums[b] = sums.get(b, 0.0) + eps_kp(o)
        counts[b] = counts.get(b, 0) + 1
    table = ResampleTableData(
        bucket_edges=list(bucket_edges),
        k_prime=K_prime,
        ehat={str(b): sums[b] / counts[b] for b in sorted(sums)},
        counts={str(b): counts[b] for b in sorted(counts)},
    )
    # the K map's twin: the same scores, the label of the batch expanded to K' samples
    resample_map = IsotonicCalibrator.fit(
        [o.u_K for o in observations], [o.wrong_K_prime for o in observations]
    ).data
    pos_sums: dict[tuple[int, int], float] = {}
    pos_counts: dict[tuple[int, int], int] = {}
    for o in observations:
        key = (o.n_steps, o.step)
        pos_sums[key] = pos_sums.get(key, 0.0) + eps_k(o)
        pos_counts[key] = pos_counts.get(key, 0) + 1
    ebar = {
        f"{n}:{pos}": pos_sums[(n, pos)] / pos_counts[(n, pos)] for (n, pos) in sorted(pos_sums)
    }
    g_sums: dict[tuple[int, int], float] = {}
    g_counts: dict[tuple[int, int], int] = {}
    for traj in trajectories:
        if not traj:
            continue
        n = traj[0].n_steps
        eps = {o.step: eps_k(o) for o in traj}
        for i in range(1, n + 1):
            survival = 1.0
            for later in range(i + 1, n + 1):
                w_later = (influence or {}).get((n, later), 1.0)
                survival *= 1.0 - w_later * eps.get(later, 0.0)
            g_sums[(n, i)] = g_sums.get((n, i), 0.0) + survival
            g_counts[(n, i)] = g_counts.get((n, i), 0) + 1
    future = {f"{n}:{i}": g_sums[(n, i)] / g_counts[(n, i)] for (n, i) in sorted(g_sums)}
    conformal = {
        "K": sorted(1.0 - o.p_true_K for o in observations),
        "K_prime": sorted(1.0 - o.p_true_K_prime for o in observations),
    }
    return CalibrationArtifact(
        estimator=estimator,
        equivalence=equivalence,
        K=K,
        K_prime=K_prime,
        task_config_hash=task_config_hash,
        config_hash=config_hash,
        data_hash=data_hash,
        inputs=dict(inputs or {}),
        n_trajectories=n_trajectories,
        isotonic_K=iso_k.data,
        isotonic_K_prime=iso_kp.data,
        resample_table=table,
        resample_map=resample_map,
        future_factor=future,
        eps_mean_by_position=ebar,
        influence={f"{n}:{i}": v for (n, i), v in (influence or {}).items()},
        influence_method=influence_method if influence else "ones",
        conformal_scores=conformal,
        calibrator=calibrator,
        features=features,
        step_models=step_models,
    )
