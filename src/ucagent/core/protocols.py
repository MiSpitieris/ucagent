"""Protocols that adopters implement: ``Task`` and ``HumanInterface``.

The core only ever talks to a task through these methods, which is what keeps
``ucagent.core`` free of task-specific code.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ucagent.core.records import StepRecord


class Instance(BaseModel):
    """A task instance: a question whose answer needs ``n_steps`` steps."""

    instance_id: str
    split: str
    index: int
    question: str
    n_steps: int


@runtime_checkable
class Task(Protocol):
    """A multi-step task (Section 5.2 of the instructions).

    ``local_truth`` answers step ``i``'s sub-question given the context the
    agent actually has (``current``), even if that context is wrong;
    ``global_truth`` is the ground-truth chain. Both are needed: the first for
    local labels and local oracles, the second for final correctness.
    """

    name: str

    def instances(self, split: str) -> Iterable[Instance]:
        """Iterate over the instances of a split."""
        ...

    def get_instance(self, instance_id: str) -> Instance:
        """Return one instance by id (cheap; nodes call this on every step)."""
        ...

    def n_steps(self, instance: Instance) -> int:
        """Number of steps of an instance."""
        ...

    def initial_context(self, instance: Instance) -> str:
        """The context before step 1 (for a chain: the start entity)."""
        ...

    def step_prompt(
        self, instance: Instance, i: int, history: Sequence[StepRecord], attempt: int
    ) -> list[BaseMessage]:
        """Messages for step ``i`` given the committed history and the draw id.

        ``attempt`` is the id of the step's sample batch (the *draw*), not the
        decision round: equal values must give byte-identical prompts, because
        a resample expands the batch under the same prompt and the cache keys
        samples by prompt and index. Put it in the header and nowhere else.
        """
        ...

    def parse_answer(self, text: str) -> str:
        """Extract the committed answer string from a model output."""
        ...

    def local_truth(self, instance: Instance, i: int, current: str) -> str:
        """Correct answer to step ``i`` given the (possibly wrong) current context."""
        ...

    def global_truth(self, instance: Instance, i: int) -> str:
        """Ground-truth answer of step ``i`` on the true chain."""
        ...

    def distractor_names(self, instance: Instance, i: int, current: str) -> list[str]:
        """Wrong answers present in the context of step ``i`` (empty if unambiguous)."""
        ...

    def random_entity(self, instance: Instance, rng: random.Random) -> str:
        """A random entity name of the instance (used for hallucinations)."""
        ...

    def is_final_correct(self, instance: Instance, answer: str | None) -> bool:
        """Whether a final answer is correct."""
        ...

    def is_locally_correct(self, instance: Instance, i: int, current: str, answer: str) -> bool:
        """Whether ``answer`` is the local truth of step ``i`` given ``current``."""
        ...


class HumanInterface(Protocol):
    """Where human answers come from: an oracle in experiments, a person in production."""

    def check_step(self, payload: dict[str, Any]) -> str:
        """Review one step (payload from ``ask_human``) and return the corrected answer."""
        ...

    def take_over(self, payload: dict[str, Any]) -> str:
        """Take over the whole case (payload from ``escalate``) and return the final answer."""
        ...
