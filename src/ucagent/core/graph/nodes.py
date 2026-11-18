"""Node functions of the control loop (Section 5.3 of the instructions).

Read ``build.py`` for the wiring. Nodes are methods of ``GraphNodes`` so they
share the task and the model; everything run-specific comes from
``config["configurable"]`` through :mod:`ucagent.core.graph.runtime`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send, interrupt

from ucagent.core.control.base import make_policy
from ucagent.core.graph.feasibility import backtrack_blocked, resample_blocked
from ucagent.core.graph.runtime import RunSettings, get_settings, get_tables
from ucagent.core.graph.state import (
    SamplePayload,
    TrajectoryState,
    next_attempt,
    next_draw,
    samples_key,
)
from ucagent.core.protocols import Task
from ucagent.core.records import Action, Decision, Intervention, Sample, StepRecord
from ucagent.core.uq.base import StepContext, make_estimator
from ucagent.core.uq.equivalence import make_judge
from ucagent.core.uq.features import step_features

ControlTarget = Literal["prepare_step", "resample", "backtrack", "ask_human", "escalate", "finish"]

log = logging.getLogger(__name__)

_CONF_RE = re.compile(r"Confidence:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_confidence(text: str) -> float | None:
    """Verbalized confidence stated as ``Confidence: <number in [0, 1]>``."""
    m = _CONF_RE.search(text)
    if m is None:
        return None
    try:
        return min(1.0, max(0.0, float(m.group(1))))
    except ValueError:
        return None


def action_cost(
    action: Action, settings: RunSettings, *, n_samples: int = 0, steps_after: int = 0
) -> float:
    """Expected cost of an action for the audit record (Section 2.4).

    A resample expands a batch of ``n_samples`` to ``K'`` and pays the extra
    draws; a backtrack expands its target the same way and redoes the
    ``steps_after`` later steps with ``K`` fresh samples each.
    """
    c = settings.costs
    extra_draws = max(settings.K_prime - n_samples, 0) * c.llm_call
    if action == Action.PROCEED:
        return 0.0
    if action == Action.RESAMPLE:
        return c.resample + extra_draws
    if action == Action.BACKTRACK:
        return c.backtrack + extra_draws + steps_after * settings.K * c.llm_call
    if action == Action.ASK_HUMAN:
        return c.ask_human
    return c.escalate


def fixed_cost_of(action: Action, settings: RunSettings) -> float:
    """The non-LLM part of an action's cost (LLM calls are charged at ``finish``)."""
    c = settings.costs
    return {
        Action.PROCEED: 0.0,
        Action.RESAMPLE: c.resample,
        Action.BACKTRACK: c.backtrack,
        Action.ASK_HUMAN: c.ask_human,
        Action.ESCALATE: c.escalate,
    }[action]


def commit_update(state: TrajectoryState, record: StepRecord) -> dict[str, Any]:
    """State update that commits ``record`` and moves to the next step."""
    history = [*state["history"], record]
    survival = state["survival"] * (1.0 - record.w * record.eps)
    new_i = state["i"] + 1
    return {
        "history": history,
        "survival": survival,
        "i": new_i,
        "attempt": next_attempt(state["interventions"], new_i),
        "draw": next_draw(state["samples"], new_i),
        "resample_count": 0,
        "pending": None,
    }


def override_key(step: int, attempt: int) -> str:
    """Key of ``RunSettings.force_decisions`` for one decision round of a step."""
    return f"{step}:{attempt}"


def validate_override(
    decision: Decision, state: TrajectoryState, proposal: Decision | None = None
) -> Decision:
    """Check a forced decision against the state and flag it as a human override.

    Only the structural rule is enforced here (a backtrack target must be an
    earlier step); feasibility caps are the caller's job, see
    ``InteractiveSession.allowed_actions``. The audit reason records what the
    policy would have done (``proposal``) next to the human's note.
    """
    if decision.action == Action.BACKTRACK:
        j = decision.target_step
        if j is None or not 1 <= j < state["i"]:
            raise ValueError(f"backtrack target {j!r} is not in 1..{state['i'] - 1}")
    target = f" to step {decision.target_step}" if decision.target_step is not None else ""
    note = f" ({decision.reason})" if decision.reason else ""
    proposed = (
        f"; policy proposed {proposal.action.value}: {proposal.reason}"
        if proposal is not None
        else ""
    )
    numbers = {**(proposal.numbers if proposal is not None else {}), **decision.numbers}
    return decision.model_copy(
        update={
            "reason": f"human override: {decision.action.value}{target}{note}{proposed}",
            "numbers": {**numbers, "override": 1.0},
        }
    )


def route_after_commit(state: TrajectoryState) -> Literal["prepare_step", "finish"]:
    """After a human-committed step: next step or finish."""
    return "finish" if state["i"] > state["n_steps"] else "prepare_step"


class GraphNodes:
    """The node functions, bound to one task and one chat model."""

    def __init__(self, task: Task, llm: BaseChatModel) -> None:
        self.task = task
        self.llm = llm

    # --- sampling -----------------------------------------------------------------

    def prepare_step(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Build the prompt of the current step's draw and set the target batch size.

        The prompt depends on the draw id, not on the decision round, so an
        expansion (resample, or a backtrack's redo of its target) reuses the
        prompt and only adds samples.
        """
        settings = get_settings(config)
        instance = self.task.get_instance(state["instance_id"])
        history = state["history"]
        current = history[-1].answer if history else self.task.initial_context(instance)
        draw = state["draw"]
        messages = self.task.step_prompt(instance, state["i"], history, draw)
        k = state.get("k_next") or settings.K
        key = samples_key(state["i"], draw)
        existing = len(state["samples"].get(key, []))
        if k <= existing:
            raise RuntimeError(
                f"nothing to sample: batch {key} already has {existing} samples, {k} requested"
            )
        return {"prompt": messages, "k_requested": k, "k_next": None, "current": current}

    def fan_out(self, state: TrajectoryState) -> list[Send]:
        """Fan out one ``sample`` task per missing index up to the target batch size."""
        key = samples_key(state["i"], state["draw"])
        start = len(state["samples"].get(key, []))
        return [
            Send("sample", {"messages": state["prompt"], "sample_index": k, "key": key})
            for k in range(start, state["k_requested"])
        ]

    def sample(self, state: SamplePayload, config: RunnableConfig) -> dict[str, Any]:
        """One LLM call; ``sample_index`` is a model kwarg, never part of the prompt."""
        message = self.llm.invoke(state["messages"], sample_index=state["sample_index"])
        content = message.content
        text = content if isinstance(content, str) else str(content)
        sample = Sample(
            sample_index=state["sample_index"],
            text=text,
            answer=self.task.parse_answer(text),
            confidence=parse_confidence(text),
        )
        return {"samples": {state["key"]: [sample]}, "llm_calls": 1}

    # --- estimation ---------------------------------------------------------------

    def estimate(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Cluster the current batch; write the provisional record.

        The calibrator is chosen by the batch size (``K`` or ``K'``), see
        ``Tables.eps``.
        """
        settings = get_settings(config)
        tables = get_tables(config)
        i, attempt, n = state["i"], state["attempt"], state["n_steps"]
        key = samples_key(i, state["draw"])
        samples = sorted(state["samples"][key], key=lambda s: s.sample_index)
        indices = [s.sample_index for s in samples]
        if indices != list(range(len(samples))):
            raise RuntimeError(f"samples of batch {key} are not contiguous: {indices}")
        estimator = make_estimator(
            settings.estimator, make_judge(settings.equivalence, llm=self.llm)
        )
        context = StepContext(
            instance_id=state["instance_id"],
            step=i,
            attempt=attempt,
            current=state["current"],
            n_steps=n,
        )
        est = estimator.estimate(samples, context)
        committed = est.committed
        extra = dict(est.extra)
        features = step_features(est.u, samples, est.clusters, i, n)
        extra.update({f"f_{name}": value for name, value in features.items() if name != "u"})
        forced = settings.force_answers.get(i)
        if forced is not None and attempt == 0:
            committed, extra["forced"] = forced, 1.0  # counterfactual replay (influence estimation)
        w = tables.w(n, i) if settings.propagation == "weighted" else 1.0
        record = StepRecord(
            step=i,
            attempt=attempt,
            draw=state["draw"],
            context=state["current"],
            answer=committed,
            u=est.u,
            entropy=est.entropy,
            eps=tables.eps(est.u, len(samples), features),
            w=w,
            G=tables.G(n, i),
            clusters=est.clusters,
            samples_key=key,
            n_samples=len(samples),
            resample_count=state["resample_count"],
            extra=extra,
        )
        return {"pending": record}

    # --- control --------------------------------------------------------------------

    def control(self, state: TrajectoryState, config: RunnableConfig) -> Command[ControlTarget]:
        """Ask the policy (or apply a human override) and route to the chosen action."""
        settings = get_settings(config)
        tables = get_tables(config)
        pending = state["pending"]
        assert pending is not None
        proposal = make_policy(settings, tables).decide(state, pending)
        forced = settings.force_decisions.get(override_key(state["i"], state["attempt"]))
        decision = proposal if forced is None else validate_override(forced, state, proposal)
        action = decision.action
        blocked = None
        if action == Action.RESAMPLE:
            blocked = resample_blocked(state, settings)
        elif action == Action.BACKTRACK:
            blocked = backtrack_blocked(state, settings, decision.target_step)
        if blocked is not None:
            # A forced or scripted action the caps or the batch size rule out: proceed and
            # say so in the audit record rather than letting the graph dead-end.
            log.warning(
                "step %d: infeasible %s (%s); proceeding", state["i"], action.value, blocked
            )
            decision = decision.model_copy(
                update={
                    "action": Action.PROCEED,
                    "target_step": None,
                    "reason": f"infeasible {action.value}: {blocked}; proceeding "
                    f"(decided: {decision.reason})",
                    "numbers": {**decision.numbers, "infeasible": 1.0},
                }
            )
            action = Action.PROCEED
        n_samples, steps_after = pending.n_samples, 0
        if action == Action.BACKTRACK and decision.target_step is not None:
            n_samples = state["history"][decision.target_step - 1].n_samples
            steps_after = state["i"] - decision.target_step
        record = Intervention(
            step=state["i"],
            attempt=state["attempt"],
            action=action,
            target_step=decision.target_step,
            cost=action_cost(action, settings, n_samples=n_samples, steps_after=steps_after),
            reason=decision.reason,
            numbers=decision.numbers,
        )
        update: dict[str, Any] = {
            "interventions": [record],
            "fixed_cost": state["fixed_cost"] + fixed_cost_of(action, settings),
        }
        goto: ControlTarget
        if action == Action.PROCEED:
            update.update(commit_update(state, pending))
            goto = "finish" if state["i"] >= state["n_steps"] else "prepare_step"
        elif action == Action.RESAMPLE:
            goto = "resample"
        elif action == Action.BACKTRACK:
            update["backtrack_target"] = decision.target_step
            goto = "backtrack"
        elif action == Action.ASK_HUMAN:
            goto = "ask_human"
        else:
            goto = "escalate"
        return Command(goto=goto, update=update)

    # --- actions ----------------------------------------------------------------------

    def resample(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Expand the current step's batch to ``K'`` samples under the same prompt.

        The draw (and so the prompt and the samples key) is unchanged; the next
        decision round starts, and ``prepare_step`` / ``fan_out`` add the
        missing sample indices.
        """
        settings = get_settings(config)
        return {
            "attempt": next_attempt(state["interventions"], state["i"]),
            "k_next": settings.K_prime,
            "resample_count": state["resample_count"] + 1,
            "pending": None,
        }

    def backtrack(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Discard steps ``j..i``, expand step ``j``'s batch to ``K'`` and redo from there.

        Step ``j``'s prompt is unchanged (its context ``history[:j-1]`` is kept),
        so the expansion is the experiment the resample table was fitted on;
        the later steps get fresh draws through ``commit_update``.
        """
        j = state["backtrack_target"]
        assert j is not None and 1 <= j <= state["i"]
        target = state["history"][j - 1]
        history = state["history"][: j - 1]
        survival = 1.0
        for rec in history:
            survival *= 1.0 - rec.w * rec.eps
        settings = get_settings(config)
        return {
            "history": history,
            "survival": survival,
            "i": j,
            "attempt": next_attempt(state["interventions"], j),
            "draw": target.draw,
            "k_next": settings.K_prime,
            "resample_count": 0,
            "pending": None,
            "backtrack_count": state["backtrack_count"] + 1,
            "backtrack_target": None,
        }

    def ask_human(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Local human check of the current step via ``interrupt``; commits the answer."""
        settings = get_settings(config)
        pending = state["pending"]
        assert pending is not None
        payload = {
            "kind": "check_step",
            "instance_id": state["instance_id"],
            "question": state["question"],
            "n_steps": state["n_steps"],
            "step": state["i"],
            "attempt": state["attempt"],
            "current": pending.context,
            "committed": pending.answer,
            "clusters": [c.model_dump() for c in pending.clusters],
        }
        answer = str(interrupt(payload))
        record = pending.model_copy(
            update={"answer": answer, "eps": settings.p_h, "source": "human"}
        )
        return commit_update(state, record)

    def escalate(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Hand the whole case to a human via ``interrupt``; ends the trajectory."""
        pending = state["pending"]
        payload = {
            "kind": "take_over",
            "instance_id": state["instance_id"],
            "question": state["question"],
            "n_steps": state["n_steps"],
            "step": state["i"],
            "history": [h.answer for h in state["history"]],
            "pending": pending.answer if pending is not None else None,
        }
        answer = str(interrupt(payload))
        return {"final_answer": answer, "escalated": True, "pending": None}

    def finish(self, state: TrajectoryState, config: RunnableConfig) -> dict[str, Any]:
        """Final answer, correctness (analysis only) and exact total cost."""
        settings = get_settings(config)
        instance = self.task.get_instance(state["instance_id"])
        history = state["history"]
        final = state.get("final_answer")
        if final is None and history:
            final = history[-1].answer
        total = state["fixed_cost"] + state["llm_calls"] * settings.costs.llm_call
        return {
            "final_answer": final,
            "final_correct": self.task.is_final_correct(instance, final),
            "fixed_cost": state["fixed_cost"],
            "done": True,
            "total_cost": total,
        }
