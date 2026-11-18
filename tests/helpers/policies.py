"""Test-only policies, registered through the public registry."""

from __future__ import annotations

from ucagent.core.control.base import Decision, register_policy
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.records import Action, StepRecord
from ucagent.core.tables import Tables


class AskLastPolicy:
    name = "ask_last"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        pass

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        if step.step == state["n_steps"]:
            return Decision(action=Action.ASK_HUMAN, reason="ask_last: last step")
        return Decision(action=Action.PROCEED, reason="ask_last: proceed")


class AskAllPolicy:
    name = "ask_all"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        pass

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        return Decision(action=Action.ASK_HUMAN, reason="ask_all: every step")


register_policy("ask_last", AskLastPolicy)
register_policy("ask_all", AskAllPolicy)


FORCED: dict[str, dict[tuple[int, int], tuple[Action, int | None]]] = {}


class ForcedSequencePolicy:
    """Replays a scripted decision per (step, attempt); proceeds otherwise."""

    name = "forced"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.script = FORCED[settings.policy]

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        action, target = self.script.get((step.step, step.attempt), (Action.PROCEED, None))
        return Decision(action=action, target_step=target, reason=f"forced: {action.value}")


def register_forced(name: str, script: dict[tuple[int, int], tuple[Action, int | None]]) -> str:
    FORCED[name] = script
    register_policy(name, ForcedSequencePolicy)
    return name
