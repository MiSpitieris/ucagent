"""Human-in-the-loop plumbing: the resume loop and the shipped human interfaces.

``run_with_human`` drives one trajectory to completion, answering every
``interrupt`` through a ``HumanInterface``. ``OracleResumer`` wraps two
callables (simulated humans in experiments); ``TerminalHuman`` asks a person
at the terminal (``ucagent demo``). A production review queue implements the
same protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ucagent.core.graph.build import Graph
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.protocols import HumanInterface

Durability = Literal["sync", "async", "exit"]


def run_with_human(
    graph: Graph,
    state: TrajectoryState | None,
    config: RunnableConfig,
    human: HumanInterface,
    *,
    durability: Durability = "exit",
) -> dict[str, Any]:
    """Invoke the graph and answer interrupts until the trajectory is done."""
    result = graph.invoke(state, config, durability=durability)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if payload.get("kind") == "check_step":
            answer = human.check_step(payload)
        else:
            answer = human.take_over(payload)
        result = graph.invoke(Command(resume=answer), config, durability=durability)
    return result


class OracleResumer:
    """``HumanInterface`` built from two callables (simulated humans)."""

    def __init__(
        self,
        check_step: Callable[[dict[str, Any]], str],
        take_over: Callable[[dict[str, Any]], str],
    ) -> None:
        self._check = check_step
        self._take = take_over

    def check_step(self, payload: dict[str, Any]) -> str:
        """Delegate a step check."""
        return self._check(payload)

    def take_over(self, payload: dict[str, Any]) -> str:
        """Delegate a take-over."""
        return self._take(payload)


class TerminalHuman:
    """Asks a person at the terminal (used by ``ucagent demo``)."""

    def __init__(self, prompt: Callable[[str], str] = input, echo: Callable[[str], None] = print):
        self._input = prompt
        self._echo = echo

    def check_step(self, payload: dict[str, Any]) -> str:
        """Show the step and read the corrected answer."""
        self._echo(f"\n[review] step {payload['step']}/{payload['n_steps']}")
        self._echo(f"  question : {payload['question']}")
        self._echo(f"  context  : {payload['current']}")
        self._echo(f"  committed: {payload['committed']}")
        for c in payload.get("clusters", []):
            self._echo(f"    cluster {c['proportion']:.2f}: {c['representative']}")
        answer = self._input("  your answer (empty keeps the committed one): ").strip()
        return answer or str(payload["committed"])

    def take_over(self, payload: dict[str, Any]) -> str:
        """Show the case and read the final answer."""
        self._echo(f"\n[escalation] {payload['question']}")
        self._echo(f"  chain so far: {payload['history']}")
        return self._input("  final answer: ").strip()
