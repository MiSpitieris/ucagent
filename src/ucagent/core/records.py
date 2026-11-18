"""Data records shared by the graph, the controllers and the harness.

Everything that ends up in a checkpoint, a parquet row or an audit log is a
frozen pydantic model defined here, so one definition serves the graph state,
the policies and the analysis code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2  # 2: ``StepRecord.draw`` and ``Trajectory.sample_confidences`` (2026-11-08)


class Action(StrEnum):
    """Controller actions (Section 2.4 of the instructions)."""

    PROCEED = "proceed"
    RESAMPLE = "resample"
    BACKTRACK = "backtrack"
    ASK_HUMAN = "ask_human"
    ESCALATE = "escalate"


class Sample(BaseModel, frozen=True):
    """One LLM sample for one step attempt."""

    sample_index: int
    text: str
    answer: str
    confidence: float | None = None


class Cluster(BaseModel, frozen=True):
    """A semantic-equivalence cluster of samples."""

    representative: str
    members: list[int]
    proportion: float


class StepRecord(BaseModel, frozen=True):
    """A committed (or provisional) step.

    ``eps`` is the calibrated probability that ``answer`` is *locally* wrong,
    ``w`` the downstream influence and ``G`` the future factor used by the
    projection. ``attempt`` is the decision round of the step (0, then 1 after
    a resample or a backtrack), ``draw`` the id of its sample batch
    (``samples_key == f"{step}:{draw}"``) and ``n_samples`` the batch size
    (``K``, or ``K'`` once expanded). ``locally_correct`` / ``globally_correct``
    are analysis labels filled by the harness after the run; policies never
    read them. Rows written before schema version 2 have no ``draw`` (it reads
    as 0) and their ``attempt`` also numbered the batch.
    """

    step: int
    attempt: int
    draw: int = 0
    context: str
    answer: str
    u: float
    entropy: float
    eps: float
    w: float = 1.0
    G: float = 1.0
    clusters: list[Cluster] = Field(default_factory=list)
    samples_key: str = ""
    n_samples: int = 0
    resample_count: int = 0
    source: Literal["llm", "human", "counterfactual"] = "llm"
    locally_correct: bool | None = None
    globally_correct: bool | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class Intervention(BaseModel, frozen=True):
    """One controller decision with its cost and human-readable reason."""

    step: int
    attempt: int
    action: Action
    target_step: int | None = None
    cost: float = 0.0
    reason: str = ""
    numbers: dict[str, float] = Field(default_factory=dict)


class Decision(BaseModel, frozen=True):
    """What the controller decided for the current step and why.

    Policies return one; a human override (``RunSettings.force_decisions``)
    supplies one in its place, flagged with ``numbers["override"] = 1``.
    """

    action: Action
    target_step: int | None = None
    reason: str = ""
    numbers: dict[str, float] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """One finished trajectory: the audit record and the parquet row."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    instance_id: str
    split: str
    seed: int
    n_steps: int
    policy: str
    knob: float | None
    estimator: str
    K: int
    K_prime: int
    steps: list[StepRecord]
    interventions: list[Intervention]
    final_answer: str | None
    final_correct: bool
    escalated: bool
    total_cost: float
    fixed_cost: float
    llm_calls: int
    u_final: float | None
    risk_naive_final: float
    risk_weighted_final: float
    survival_final: float
    duration_s: float = 0.0
    thread_id: str = ""
    samples: dict[str, list[str]] = Field(default_factory=dict)  # samples_key -> answers
    # samples_key -> stated confidences, aligned with ``samples`` (None where none was parsed)
    sample_confidences: dict[str, list[float | None]] = Field(default_factory=dict)

    def count(self, action: Action) -> int:
        """Number of decisions with the given action."""
        return sum(1 for x in self.interventions if x.action == action)

    @property
    def n_interventions(self) -> int:
        """Number of non-``proceed`` decisions."""
        return sum(1 for x in self.interventions if x.action != Action.PROCEED)

    def audit(self) -> list[dict[str, Any]]:
        """Per-step view: every decision with its reason and the numbers that drove it.

        One row per step that was committed or decided on, in step order. A step that was
        never committed has no record: the case was escalated there, or a backtrack discarded
        the step and the trajectory ended before it was reached again. Such a row has
        ``committed`` false and no answer, but it lists its decisions; until 2026-11-17 it was
        left out, so an escalation did not appear in this view. The decisions of a step are in
        the order taken and include those of rounds that a backtrack discarded.
        """
        by_step: dict[int, list[Intervention]] = {}
        for iv in self.interventions:
            by_step.setdefault(iv.step, []).append(iv)
        records = {rec.step: rec for rec in self.steps}
        out: list[dict[str, Any]] = []
        for step in sorted(set(records) | set(by_step)):
            rec = records.get(step)
            out.append(
                {
                    "step": step,
                    "committed": rec is not None,
                    "answer": rec.answer if rec else None,
                    "u": rec.u if rec else None,
                    "eps": rec.eps if rec else None,
                    "w": rec.w if rec else None,
                    "source": rec.source if rec else None,
                    "decisions": [
                        {
                            "action": iv.action.value,
                            "attempt": iv.attempt,
                            "target_step": iv.target_step,
                            "cost": iv.cost,
                            "reason": iv.reason,
                            "numbers": dict(iv.numbers),
                        }
                        for iv in by_step.get(step, [])
                    ],
                }
            )
        return out

    def to_row(self) -> dict[str, Any]:
        """Flatten to one parquet row (records as JSON strings)."""
        row = self.model_dump(exclude={"steps", "interventions", "samples", "sample_confidences"})
        row["steps_json"] = json.dumps([s.model_dump() for s in self.steps])
        row["interventions_json"] = json.dumps([i.model_dump() for i in self.interventions])
        row["samples_json"] = json.dumps(self.samples)
        row["sample_confidences_json"] = json.dumps(self.sample_confidences)
        row["n_committed"] = len(self.steps)
        row["n_interventions"] = self.n_interventions
        for action in Action:
            if action != Action.PROCEED:
                row[f"n_{action.value}"] = self.count(action)
        first_wrong = next((s.step for s in self.steps if s.locally_correct is False), None)
        row["first_local_error_step"] = first_wrong
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Trajectory:
        """Inverse of :meth:`to_row`."""
        data = {k: v for k, v in row.items() if k in cls.model_fields}
        data["steps"] = [StepRecord.model_validate(s) for s in json.loads(row["steps_json"])]
        data["interventions"] = [
            Intervention.model_validate(i) for i in json.loads(row["interventions_json"])
        ]
        data["samples"] = json.loads(row.get("samples_json") or "{}")
        # rows of schema version 1 carry no confidences; they load with an empty mapping
        data["sample_confidences"] = json.loads(row.get("sample_confidences_json") or "{}")
        knob = data.get("knob")
        if knob is not None and knob != knob:  # NaN from parquet
            data["knob"] = None
        return cls.model_validate(data)
