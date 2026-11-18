"""Run-time configuration: everything a run needs that is not baked into the graph.

``RunSettings`` (policy, knob, K, costs, oracle noise, ...) and ``Tables``
travel in ``config["configurable"]``, so one compiled graph serves every run
and a forked checkpoint can be resumed under a different policy.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from ucagent.core.graph.state import SCHEMA_VERSION, TrajectoryState
from ucagent.core.protocols import Instance
from ucagent.core.records import Decision
from ucagent.core.tables import Tables

SETTINGS_KEY = "ucagent_settings"
TABLES_KEY = "ucagent_tables"


class CostTable(BaseModel, frozen=True):
    """Costs in one abstract unit (Appendix B defaults); none may be negative."""

    llm_call: float = Field(0.01, ge=0.0)
    resample: float = Field(0.0, ge=0.0)
    backtrack: float = Field(0.5, ge=0.0)
    ask_human: float = Field(1.0, ge=0.0)
    escalate: float = Field(10.0, ge=0.0)


class RunSettings(BaseModel, frozen=True):
    """Per-run settings read by the nodes and the policies.

    ``K`` samples are drawn per step; a resample (or a backtrack's redo of its
    target) expands the batch to ``K'`` samples, so ``K' >= K`` is required and
    ``K' == K`` means those two actions are never feasible. One expansion per
    batch is all the fitted tables describe, so ``max_resamples`` above 1 is
    inert.
    """

    policy: str = "none"
    knob: float | None = None
    estimator: str = "disagreement"
    equivalence: str = "normalized_exact"
    K: int = Field(5, ge=1)
    K_prime: int = Field(10, ge=1)
    costs: CostTable = Field(default_factory=CostTable)
    max_resamples: int = Field(1, ge=0)
    max_backtracks: int = Field(1, ge=0)
    backtrack_min_share: float = Field(0.5, ge=0.0, le=1.0)
    p_h: float = Field(0.0, ge=0.0, le=1.0)
    p_e: float = Field(0.0, ge=0.0, le=1.0)
    oracle_scope: Literal["local", "global"] = "local"
    propagation: Literal["weighted", "naive"] = "weighted"
    spending: str = "linear"
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    seed: int = 0
    record_samples: bool = True
    # step -> forced answer, applied at decision round 0 (counterfactual replays)
    force_answers: dict[int, str] = Field(default_factory=dict)
    # "step:attempt" -> decision used instead of the policy's (human override, see session.py)
    force_decisions: dict[str, Decision] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _k_prime_covers_k(self) -> RunSettings:
        if self.K_prime < self.K:
            raise ValueError(f"K_prime ({self.K_prime}) must be at least K ({self.K})")
        return self


def recursion_limit_for(n_steps: int, settings: RunSettings) -> int:
    """Worst-case superstep count of one uninterrupted invoke, plus a margin.

    One controlled step costs 4 supersteps (prepare, sample fan-out, estimate,
    control), a resample 5 more (resample, prepare, sample, estimate, control)
    and an ask 1 (its interrupt then starts a fresh budget). A backtrack can
    revisit every step once more, with its own resamples and asks, so the
    per-step allowance is multiplied by ``1 + max_backtracks``; the backtrack
    nodes, ``finish`` and ``escalate`` are added on top. Review finding R11:
    the earlier additive bound rejected legal sequences.
    """
    visits = 1 + settings.max_backtracks
    per_visit = 4 + 5 * settings.max_resamples + 1
    return visits * n_steps * per_visit + settings.max_backtracks + 2 + 10


def make_config(
    thread_id: str, settings: RunSettings, tables: Tables, n_steps: int
) -> RunnableConfig:
    """Build the invoke config for one trajectory."""
    return {
        "configurable": {"thread_id": thread_id, SETTINGS_KEY: settings, TABLES_KEY: tables},
        "recursion_limit": recursion_limit_for(n_steps, settings),
    }


def with_settings(config: RunnableConfig, **updates: Any) -> RunnableConfig:
    """Copy of ``config`` with some ``RunSettings`` fields replaced (for forks)."""
    conf = dict(config.get("configurable", {}))
    settings = cast(RunSettings, conf[SETTINGS_KEY])
    conf[SETTINGS_KEY] = settings.model_copy(update=updates)
    out: RunnableConfig = {**config, "configurable": conf}
    return out


def get_settings(config: RunnableConfig) -> RunSettings:
    """The ``RunSettings`` of the current invocation."""
    return cast(RunSettings, config["configurable"][SETTINGS_KEY])


def get_tables(config: RunnableConfig) -> Tables:
    """The lookup ``Tables`` of the current invocation."""
    return cast(Tables, config["configurable"][TABLES_KEY])


def init_state(instance: Instance, settings: RunSettings) -> TrajectoryState:
    """Initial state for one trajectory."""
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "question": instance.question,
        "n_steps": instance.n_steps,
        "i": 1,
        "attempt": 0,
        "draw": 0,
        "k_requested": settings.K,
        "k_next": None,
        "current": "",
        "prompt": [],
        "history": [],
        "samples": {},
        "pending": None,
        "survival": 1.0,
        "interventions": [],
        "fixed_cost": 0.0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "resample_count": 0,
        "backtrack_count": 0,
        "backtrack_target": None,
        "final_answer": None,
        "final_correct": None,
        "escalated": False,
        "done": False,
    }
