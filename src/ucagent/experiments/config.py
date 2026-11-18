"""Experiment configuration: the YAML of Appendix B as pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ucagent.core.graph.runtime import CostTable, RunSettings


class TaskConfig(BaseModel):
    """``task`` block: ``name`` selects a registered task; other fields belong to that task.

    ``n_steps``, ``n_instances`` and ``seed`` are common to every task; the
    synthetic chain adds ``ambiguity``, ``distractors`` and ``reconvergence``
    (see :class:`ucagent.tasks.synthetic_chain.SyntheticConfig`).
    """

    model_config = ConfigDict(extra="allow")

    name: str = "synthetic_chain"
    n_steps: int = Field(5, ge=1)
    n_instances: dict[str, int] = Field(
        default_factory=lambda: {"train": 300, "calib": 300, "test": 300}
    )
    seed: int = 0

    @field_validator("n_instances")
    @classmethod
    def _counts_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        bad = {k: v for k, v in value.items() if v < 0}
        if bad:
            raise ValueError(f"n_instances must be non-negative, got {bad}")
        return value


class LLMConfig(BaseModel):
    """``llm`` block. ``backend`` is ``mock`` or a ``provider:model`` string.

    ``K`` samples per step; a resample expands the batch to ``K_prime``, so
    ``K_prime >= K`` (equal means the resample and backtrack actions are never
    feasible).
    """

    backend: str = "mock"
    difficulty_beta: list[float] = Field(default_factory=lambda: [2.0, 5.0])
    clean_factor: float = Field(0.2, ge=0.0, le=1.0)
    verbalized: bool = True
    verbalized_sigma: float = Field(0.1, ge=0.0)
    mock_seed: int | None = None
    K: int = Field(5, ge=1)
    K_prime: int = Field(10, ge=1)
    temperature: float = Field(0.7, ge=0.0)
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments for the provider class (ChatOllama: reasoning, "
        "num_ctx, num_predict); part of the cache key when non-empty.",
    )

    @field_validator("difficulty_beta")
    @classmethod
    def _beta_shape(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or any(v <= 0 for v in value):
            raise ValueError(f"difficulty_beta must be two positive shape parameters, got {value}")
        return value

    @model_validator(mode="after")
    def _k_prime_covers_k(self) -> LLMConfig:
        if self.K_prime < self.K:
            raise ValueError(f"K_prime ({self.K_prime}) must be at least K ({self.K})")
        return self


class UQConfig(BaseModel):
    """``uq`` block.

    ``calibrator`` (``isotonic``, ``logistic``, ``boosted``, ``gp`` or a registered plugin)
    maps the named step ``features`` (see ``core.uq.features``) to the probability that a
    step is locally wrong; the default, isotonic on ``u``, is the calibrator of the earlier
    versions.
    """

    estimator: str = "disagreement"
    equivalence: str = "normalized_exact"
    artifact: str | None = None
    calibrator: str = "isotonic"
    features: list[str] = Field(default_factory=lambda: ["u"])

    @model_validator(mode="after")
    def _calibrator_and_features(self) -> UQConfig:
        from ucagent.core.uq.features import check_features

        check_features(self.features)
        if self.calibrator == "isotonic" and len(self.features) != 1:
            raise ValueError("the isotonic calibrator reads exactly one feature")
        return self


class RiskConfig(BaseModel):
    """``risk`` block."""

    propagation: Literal["weighted", "naive"] = "weighted"
    influence: str = "ones"
    influence_estimator: Literal["paired", "absolute"] = "paired"


class ControlConfig(BaseModel):
    """``control`` block."""

    policy: str = "none"
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    knob: float | None = None
    spending: str = "linear"
    costs: CostTable = Field(default_factory=CostTable)
    # one expansion per batch is fitted, so values above 1 are inert
    max_resamples: int = Field(1, ge=0)
    max_backtracks: int = Field(1, ge=0)
    backtrack_min_share: float = Field(0.5, ge=0.0, le=1.0)


class OracleConfig(BaseModel):
    """``oracle`` block; ``scope: global`` is refused outside tests."""

    scope: Literal["local", "global"] = "local"
    p_h: float = Field(0.0, ge=0.0, le=1.0)
    p_e: float = Field(0.0, ge=0.0, le=1.0)


class GridConfig(BaseModel):
    """Knob grid for calibration."""

    kind: Literal["log", "linear"] = "log"
    min: float = 0.001
    max: float = 10.0
    points: int = Field(12, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> GridConfig:
        if self.max < self.min:
            raise ValueError(f"grid max ({self.max}) must be at least min ({self.min})")
        if self.kind == "log" and self.min <= 0:
            raise ValueError("a log grid needs min > 0")
        return self


class CalibrationConfig(BaseModel):
    """``calibration`` block (Learn-then-Test)."""

    method: Literal["ltt"] = "ltt"
    delta: float = Field(0.1, gt=0.0, lt=1.0)
    pvalue: Literal["binomial", "hoeffding"] = "binomial"
    grid: GridConfig = Field(default_factory=GridConfig)


class LimitsConfig(BaseModel):
    """``limits`` block; the minimum of ``max_llm_calls`` and ``UCAGENT_MAX_LLM_CALLS`` wins."""

    max_llm_calls: int = Field(20000, ge=1)
    max_concurrency: int = Field(8, ge=1)


class ExperimentConfig(BaseModel):
    """One run: a split of a task through the graph under one policy and knob."""

    run_id: str
    split: str = "test"
    seed: int = 0
    plugins: list[str] = Field(
        default_factory=list,
        description="Modules imported in every worker before building (register policies, tasks).",
    )
    task: TaskConfig = Field(default_factory=TaskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    uq: UQConfig = Field(default_factory=UQConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialise (round-trips through :meth:`from_yaml`)."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def to_run_settings(self) -> RunSettings:
        """The per-run settings handed to the graph."""
        return RunSettings(
            policy=self.control.policy,
            knob=self.control.knob,
            estimator=self.uq.estimator,
            equivalence=self.uq.equivalence,
            K=self.llm.K,
            K_prime=self.llm.K_prime,
            costs=self.control.costs,
            max_resamples=self.control.max_resamples,
            max_backtracks=self.control.max_backtracks,
            backtrack_min_share=self.control.backtrack_min_share,
            p_h=self.oracle.p_h,
            p_e=self.oracle.p_e,
            oracle_scope=self.oracle.scope,
            propagation=self.risk.propagation,
            spending=self.control.spending,
            alpha=self.control.alpha,
            seed=self.seed,
        )

    def with_updates(self, **sections: dict[str, Any]) -> ExperimentConfig:
        """Copy with some section fields replaced, e.g. ``control={"knob": 0.1}``."""
        data = self.model_dump()
        for section, fields in sections.items():
            if isinstance(data.get(section), dict):
                data[section].update(fields)
            else:
                data[section] = fields
        return ExperimentConfig.model_validate(data)
