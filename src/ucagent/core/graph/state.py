"""LangGraph state of one trajectory and its reducers (Section 5.3).

Only three channels have reducers: ``samples`` (per-key list merge so the
``K`` parallel sample nodes can each append), ``interventions`` (append) and
``llm_calls`` (sum). Everything else is written whole by exactly one node per
superstep.

Two counters describe where a step stands. ``draw`` identifies the sample
batch: it goes into the prompt header and into the samples key
``"{step}:{draw}"``. A resample keeps the draw and *expands* the batch (the
sample nodes append indices ``K..K'-1`` under the same key), which is exactly
the experiment the resample-effect table was fitted on; a backtrack keeps the
target step's draw for the same reason and gives the redone later steps fresh
draws. ``attempt`` is the decision round of a step (0, then 1 after a resample
or a backtrack): it keys human overrides, scripted decisions and pauses, so it
must be unique per control visit, and it is derived from the append-only
``interventions`` channel so a backtrack cannot make it repeat.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from ucagent.core.records import Intervention, Sample, StepRecord

SCHEMA_VERSION = 2


def merge_samples(
    left: dict[str, list[Sample]] | None, right: dict[str, list[Sample]] | None
) -> dict[str, list[Sample]]:
    """Per-key list concatenation; never drops a key."""
    out = {k: list(v) for k, v in (left or {}).items()}
    for k, v in (right or {}).items():
        out.setdefault(k, []).extend(v)
    return out


class TrajectoryState(TypedDict, total=False):
    """Graph state; see the module docstring for the reducer policy."""

    schema_version: int
    instance_id: str
    question: str
    n_steps: int
    i: int
    attempt: int
    draw: int
    k_requested: int
    k_next: int | None
    current: str
    prompt: list[BaseMessage]
    history: list[StepRecord]
    samples: Annotated[dict[str, list[Sample]], merge_samples]
    pending: StepRecord | None
    survival: float
    interventions: Annotated[list[Intervention], operator.add]
    fixed_cost: float
    total_cost: float
    llm_calls: Annotated[int, operator.add]
    resample_count: int
    backtrack_count: int
    backtrack_target: int | None
    final_answer: str | None
    final_correct: bool | None
    escalated: bool
    done: bool


class SamplePayload(TypedDict):
    """Input of one fan-out ``sample`` task."""

    messages: list[BaseMessage]
    sample_index: int
    key: str


def samples_key(step: int, draw: int) -> str:
    """Key of the samples channel for one sample batch (a step and its draw)."""
    return f"{step}:{draw}"


def next_draw(samples: dict[str, list[Sample]], step: int) -> int:
    """First unused draw number for ``step`` (fresh samples after a backtrack)."""
    draws = [int(k.split(":")[1]) for k in samples if int(k.split(":")[0]) == step]
    return 1 + max(draws) if draws else 0


def next_attempt(interventions: list[Intervention], step: int) -> int:
    """Next decision round for ``step``: one more than any decision already taken there."""
    rounds = [iv.attempt for iv in interventions if iv.step == step]
    return 1 + max(rounds) if rounds else 0
