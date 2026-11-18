"""Policy interface and the policy registry (``Decision`` itself lives in ``records``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.records import Action, Decision, StepRecord
from ucagent.core.tables import Tables

__all__ = [
    "POLICIES",
    "Decision",
    "NonePolicy",
    "Policy",
    "PolicyFactory",
    "make_policy",
    "register_policy",
]


class Policy(Protocol):
    """A controller: decides an action for the provisional step record."""

    name: str

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Return the action for ``step`` given the trajectory so far."""
        ...


class NonePolicy:
    """Always proceed."""

    name = "none"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings = settings
        self.tables = tables

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Proceed unconditionally."""
        return Decision(action=Action.PROCEED, reason="none: always proceed")


PolicyFactory = Callable[[RunSettings, Tables], Policy]
POLICIES: dict[str, PolicyFactory] = {"none": NonePolicy}


def register_policy(name: str, factory: PolicyFactory) -> None:
    """Register a policy factory under ``name`` (tests and adopters use this too)."""
    POLICIES[name] = factory


def make_policy(settings: RunSettings, tables: Tables) -> Policy:
    """Instantiate the policy named in ``settings``."""
    try:
        factory = POLICIES[settings.policy]
    except KeyError as exc:
        raise KeyError(f"unknown policy {settings.policy!r}; known: {sorted(POLICIES)}") from exc
    return factory(settings, tables)
