"""Drive one trajectory interactively: one pause at a time, any user interface on top.

``InteractiveSession`` is the non-blocking counterpart of ``run_with_human``
for interfaces that cannot block on a person (a web page, a review service).
Each call advances the graph until the next *pause*:

- ``control``: supervisor mode only; the graph stopped *before* the controller
  decided (per-call ``interrupt_before=["control"]``). The caller inspects the
  provisional step, the policy's proposal and the candidate effects, then
  ``accept()`` lets the policy decide or ``override(decision)`` forces one.
- ``check_step`` / ``take_over``: the ``ask_human`` / ``escalate`` nodes called
  ``interrupt``; ``answer(text)`` resumes.
- ``done``: the trajectory finished; read ``values``.

Overrides travel in ``RunSettings.force_decisions`` for exactly one invoke, so
the thread's checkpoints record them like any other decision (flagged with
``numbers["override"] = 1``). Sessions use ``durability="sync"`` so every
superstep is checkpointed and the history can be inspected later.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from ucagent.core.control.base import make_policy
from ucagent.core.graph.build import Graph
from ucagent.core.graph.feasibility import (
    backtrack_blocked,
    backtrack_targets,
    resample_blocked,
)
from ucagent.core.graph.human import Durability
from ucagent.core.graph.nodes import override_key, validate_override
from ucagent.core.graph.runtime import RunSettings, init_state, make_config, with_settings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.protocols import Instance
from ucagent.core.records import Action, Decision
from ucagent.core.risk.effects import Effect, candidate_effects
from ucagent.core.tables import Tables

PauseKind = Literal["control", "check_step", "take_over", "done"]


class Pause(BaseModel, frozen=True):
    """Where the graph stopped and what it needs from the caller."""

    kind: PauseKind
    step: int
    attempt: int
    payload: dict[str, Any] = Field(default_factory=dict)  # the interrupt value, if any

    @property
    def needs_answer(self) -> bool:
        """True at ``check_step`` and ``take_over``."""
        return self.kind in ("check_step", "take_over")


class InteractiveSession:
    """One trajectory, driven pause by pause (see the module docstring)."""

    def __init__(
        self,
        graph: Graph,
        settings: RunSettings,
        tables: Tables,
        *,
        supervise: bool = True,
        durability: Durability = "sync",
        run_id: str = "ui",
    ) -> None:
        self.graph = graph
        self.settings = settings
        self.tables = tables
        self.supervise = supervise
        self.durability: Durability = durability
        self.run_id = run_id
        self.thread_id = ""
        self.last_trace: list[str] = []
        self._config: RunnableConfig | None = None
        self._pause: Pause | None = None

    # --- state -------------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """True between ``start`` and ``close``."""
        return self._config is not None

    @property
    def config(self) -> RunnableConfig:
        """The thread's invoke config (settings and tables included)."""
        if self._config is None:
            raise RuntimeError("session not started")
        return self._config

    @property
    def pause(self) -> Pause:
        """The current pause."""
        if self._pause is None:
            raise RuntimeError("session not started")
        return self._pause

    @property
    def values(self) -> TrajectoryState:
        """The thread's current state values."""
        return cast(TrajectoryState, self.graph.get_state(self.config).values)

    # --- driving -----------------------------------------------------------------------

    def start(self, instance: Instance) -> Pause:
        """Begin a new thread for ``instance`` and run to the first pause."""
        suffix = uuid.uuid4().hex[:8]
        self.thread_id = f"{self.run_id}:{instance.instance_id}:{self.settings.seed}:{suffix}"
        self._config = make_config(self.thread_id, self.settings, self.tables, instance.n_steps)
        return self._advance(init_state(instance, self.settings), self._config)

    def accept(self) -> Pause:
        """At a ``control`` pause: let the policy decide and run on."""
        self._expect("control")
        return self._advance(None, self.config)

    def override(self, decision: Decision) -> Pause:
        """At a ``control`` pause: apply ``decision`` instead of the policy's and run on."""
        pause = self._expect("control")
        blocked = self.allowed_actions().get(decision.action)
        if blocked is not None:
            raise ValueError(f"{decision.action.value} is not allowed here: {blocked}")
        values = self.values
        validate_override(decision, values)
        if decision.action == Action.BACKTRACK:
            why = backtrack_blocked(values, self.settings, decision.target_step)
            if why is not None:
                raise ValueError(f"backtrack to step {decision.target_step} is not allowed: {why}")
        key = override_key(pause.step, pause.attempt)
        cfg = with_settings(self.config, force_decisions={key: decision})
        return self._advance(None, cfg)

    def answer(self, text: str) -> Pause:
        """At ``check_step`` / ``take_over``: resume with the human's answer.

        An empty answer keeps the committed step answer (``check_step``) or the
        pending answer (``take_over``).
        """
        pause = self._expect("check_step", "take_over")
        value = text.strip()
        if not value:
            fallback = pause.payload.get("committed" if pause.kind == "check_step" else "pending")
            if not fallback:
                raise ValueError("an answer is required")
            value = str(fallback)
        return self._advance(Command(resume=value), self.config)

    def close(self) -> None:
        """Delete the thread's checkpoints; the session cannot be driven afterwards."""
        if self._config is None:
            return
        checkpointer = self.graph.checkpointer
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(self.thread_id)
        self._config = None
        self._pause = None

    # --- inspection at a control pause ------------------------------------------------

    def proposal(self) -> Decision:
        """What the policy would decide now (read-only; policies are stateless)."""
        self._expect("control")
        values = self.values
        pending = values["pending"]
        assert pending is not None
        return make_policy(self.settings, self.tables).decide(values, pending)

    def effects(self) -> list[Effect]:
        """The feasible actions with their expected effect, ``proceed`` first."""
        self._expect("control")
        values = self.values
        pending = values["pending"]
        assert pending is not None
        return candidate_effects(pending, values, self.settings, self.tables)

    def allowed_actions(self) -> dict[Action, str | None]:
        """Every action mapped to ``None`` (allowed) or the reason it is not.

        The reasons are the caps and the batch-size rule (a batch with ``K'``
        samples can be expanded no further); see ``core.graph.feasibility``.
        """
        self._expect("control")
        values = self.values
        s = self.settings
        return {
            Action.PROCEED: None,
            Action.ASK_HUMAN: None,
            Action.ESCALATE: None,
            Action.RESAMPLE: resample_blocked(values, s),
            Action.BACKTRACK: backtrack_blocked(values, s),
        }

    def backtrack_targets(self) -> dict[int, str | None]:
        """Each committed step mapped to ``None`` (a feasible target) or why it is not."""
        self._expect("control")
        return backtrack_targets(self.values, self.settings)

    # --- internals ---------------------------------------------------------------------

    def _expect(self, *kinds: str) -> Pause:
        pause = self.pause
        if pause.kind not in kinds:
            raise RuntimeError(f"session is paused at {pause.kind!r}, not {' or '.join(kinds)}")
        return pause

    def _advance(self, inp: Any, cfg: RunnableConfig) -> Pause:
        kwargs: dict[str, Any] = {"durability": self.durability, "stream_mode": "updates"}
        if self.supervise:
            kwargs["interrupt_before"] = ["control"]
        trace: list[str] = []
        interrupts: tuple[Any, ...] = ()
        for chunk in self.graph.stream(inp, cfg, **kwargs):
            if not isinstance(chunk, dict):
                continue
            for name, update in chunk.items():
                if name == "__interrupt__":
                    interrupts = tuple(update)
                else:
                    trace.append(str(name))
        self.last_trace = trace
        snap = self.graph.get_state(self.config)
        values = snap.values
        if interrupts:
            payload = dict(interrupts[0].value)
            kind: PauseKind = "take_over" if payload.get("kind") == "take_over" else "check_step"
            self._pause = Pause(
                kind=kind,
                step=int(payload.get("step", values.get("i", 0))),
                attempt=int(payload.get("attempt", values.get("attempt", 0))),
                payload=payload,
            )
        elif snap.next:
            if tuple(snap.next) != ("control",):
                raise RuntimeError(f"unexpected pause before {snap.next}")
            self._pause = Pause(kind="control", step=values["i"], attempt=values["attempt"])
        else:
            self._pause = Pause(kind="done", step=values["i"], attempt=values["attempt"])
        return self._pause
