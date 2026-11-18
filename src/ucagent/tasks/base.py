"""Task-side protocols re-exported for adopters, plus the simulated humans.

``LocalOracle`` answers a step's sub-question given the context the agent
actually had; ``GlobalOracle`` returns the ground-truth chain entity and is
only allowed in tests (it makes "ask at the last step" trivially optimal, see
Section 2.1 of the instructions); ``EscalationOracle`` produces the final
answer when a case is escalated.
"""

from __future__ import annotations

import random
from typing import Any

from ucagent.core.graph.human import OracleResumer
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.protocols import HumanInterface, Instance, Task
from ucagent.core.utils import stable_hash

__all__ = [
    "EscalationOracle",
    "GlobalOracle",
    "HumanInterface",
    "Instance",
    "LocalOracle",
    "Task",
    "make_human",
]


def _wrong_answer(task: Task, instance: Instance, i: int, current: str, rng: random.Random) -> str:
    truth = task.local_truth(instance, i, current)
    wrong = [d for d in task.distractor_names(instance, i, current) if d != truth]
    if not wrong:
        candidate = task.random_entity(instance, rng)
        while candidate == truth:
            candidate = task.random_entity(instance, rng)
        return candidate
    return rng.choice(wrong)


class LocalOracle:
    """Checks one step locally; wrong with probability ``p_h``."""

    def __init__(self, task: Task, p_h: float = 0.0, seed: int = 0) -> None:
        self.task, self.p_h, self.seed = task, p_h, seed

    def __call__(self, payload: dict[str, Any]) -> str:
        """Answer a ``check_step`` payload."""
        instance = self.task.get_instance(payload["instance_id"])
        i, current = int(payload["step"]), str(payload["current"])
        has_local_truth = getattr(self.task, "has_local_truth", None)
        if has_local_truth is not None and not has_local_truth(instance, i, current):
            # the chain is already off track: a local reviewer can only confirm the step
            return str(payload.get("committed", ""))
        if self.p_h > 0:
            rng = random.Random(
                stable_hash(self.seed, "local_oracle", instance.instance_id, i, payload["attempt"])
            )
            if rng.random() < self.p_h:
                return _wrong_answer(self.task, instance, i, current, rng)
        return self.task.local_truth(instance, i, current)


class GlobalOracle:
    """Returns the ground-truth chain entity regardless of context. Tests only."""

    def __init__(self, task: Task) -> None:
        self.task = task

    def __call__(self, payload: dict[str, Any]) -> str:
        """Answer a ``check_step`` payload with the global truth."""
        instance = self.task.get_instance(payload["instance_id"])
        return self.task.global_truth(instance, int(payload["step"]))


class EscalationOracle:
    """Produces the final answer of an escalated case; wrong with probability ``p_e``."""

    def __init__(self, task: Task, p_e: float = 0.0, seed: int = 0) -> None:
        self.task, self.p_e, self.seed = task, p_e, seed

    def __call__(self, payload: dict[str, Any]) -> str:
        """Answer a ``take_over`` payload."""
        instance = self.task.get_instance(payload["instance_id"])
        n = instance.n_steps
        if self.p_e > 0:
            rng = random.Random(stable_hash(self.seed, "escalation", instance.instance_id))
            if rng.random() < self.p_e:
                prev = self.task.global_truth(instance, n - 1) if n > 1 else ""
                current = prev or self.task.initial_context(instance)
                return _wrong_answer(self.task, instance, n, current, rng)
        return self.task.global_truth(instance, n)


def make_human(task: Task, settings: RunSettings) -> OracleResumer:
    """The simulated human for a run, from the oracle settings."""
    check: LocalOracle | GlobalOracle
    if settings.oracle_scope == "global":
        check = GlobalOracle(task)
    else:
        check = LocalOracle(task, p_h=settings.p_h, seed=settings.seed)
    return OracleResumer(check, EscalationOracle(task, p_e=settings.p_e, seed=settings.seed))
