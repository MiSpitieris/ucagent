"""The controllers and baselines of Sections 2.5 and 2.6 of the instructions.

Every policy has one knob. For every policy here the smallest knob value is
the most conservative setting (most interventions), which is the order
Learn-then-Test walks the grid in: ``calibrate_knob`` sorts the knobs ascending
and does not consult a policy. ``KNOB_ASCENDING_IS_CONSERVATIVE`` records the
convention and nothing reads it, so a policy registered from a plugin has to
follow it; a knob that runs the other way would still give a valid selection,
but not the least conservative one.

Every decision carries a ``reason`` string with the numbers that drove it.
"""

from __future__ import annotations

import random
import re

from ucagent.core.control.base import Decision, register_policy
from ucagent.core.control.spending import make_spending
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.records import Action, StepRecord
from ucagent.core.risk.effects import Effect, candidate_effects
from ucagent.core.risk.propagation import project
from ucagent.core.tables import Tables
from ucagent.core.utils import stable_hash

KNOB_ASCENDING_IS_CONSERVATIVE = True
DEFAULT_KNOB = {
    "final_gate": 0.2,
    "final_u": 0.5,
    "per_step_u": 0.5,
    "per_step_eps": 0.1,
    "marginal": 0.1,
    "budgeted": 1.0,
    "knowno": 0.1,
    "random_ask": 0.5,
    "abstain_ask": 0.0,
}

#: An explicit inability to answer: the diagnostic pattern of the external review of the
#: 2WikiMultiHopQA results (2026-11-09), frozen verbatim. It was found after those runs, so
#: results of the ``abstain_ask`` baseline on them are development, not evidence.
ABSTAIN_PATTERN = re.compile(
    r"unknown|not (?:specified|provided|stated)|unable to determine"
    r"|cannot (?:determine|be determined)|not (?:mentioned|given)",
    re.IGNORECASE,
)


def _knob(settings: RunSettings, name: str) -> float:
    return settings.knob if settings.knob is not None else DEFAULT_KNOB[name]


def _fmt(values: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in values.items())


class FinalGate:
    """Score and route: at the last step, escalate iff ``R_n > tau``."""

    name = "final_gate"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings, self.tables = settings, tables
        self.tau = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Escalate at the last step when the committed risk exceeds the knob."""
        risk_n = project(state["survival"], step.eps, step.w, 1.0)
        numbers = {"R_n": risk_n, "tau": self.tau, "S_prev": state["survival"], "eps": step.eps}
        if step.step == state["n_steps"] and risk_n > self.tau:
            return Decision(
                action=Action.ESCALATE,
                reason=f"final_gate: escalate, {_fmt(numbers)}",
                numbers=numbers,
            )
        return Decision(
            action=Action.PROCEED, reason=f"final_gate: proceed, {_fmt(numbers)}", numbers=numbers
        )


class FinalU:
    """At the last step, escalate iff ``u_n > tau``."""

    name = "final_u"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.tau = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Escalate at the last step when raw uncertainty exceeds the knob."""
        numbers = {"u": step.u, "tau": self.tau}
        if step.step == state["n_steps"] and step.u > self.tau:
            return Decision(
                action=Action.ESCALATE,
                reason=f"final_u: escalate, {_fmt(numbers)}",
                numbers=numbers,
            )
        return Decision(
            action=Action.PROCEED, reason=f"final_u: proceed, {_fmt(numbers)}", numbers=numbers
        )


class PerStepU:
    """Ask a human whenever ``u_i > tau``."""

    name = "per_step_u"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.tau = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Gate on raw uncertainty."""
        numbers = {"u": step.u, "tau": self.tau}
        if step.u > self.tau:
            return Decision(
                action=Action.ASK_HUMAN, reason=f"per_step_u: ask, {_fmt(numbers)}", numbers=numbers
            )
        return Decision(
            action=Action.PROCEED, reason=f"per_step_u: proceed, {_fmt(numbers)}", numbers=numbers
        )


class PerStepEps:
    """Ask a human whenever ``w_i * eps_i > tau`` (the strong fixed gate)."""

    name = "per_step_eps"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.tau = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Gate on calibrated, influence-weighted local error."""
        stat = step.w * step.eps
        numbers = {"w_eps": stat, "w": step.w, "eps": step.eps, "tau": self.tau}
        if stat > self.tau:
            return Decision(
                action=Action.ASK_HUMAN,
                reason=f"per_step_eps: ask, {_fmt(numbers)}",
                numbers=numbers,
            )
        return Decision(
            action=Action.PROCEED, reason=f"per_step_eps: proceed, {_fmt(numbers)}", numbers=numbers
        )


class Marginal:
    """Value-of-intervention controller: ``argmax_a (Delta_a - tau * c_a)``, proceed on ties."""

    name = "marginal"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings, self.tables = settings, tables
        self.tau = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Pick the action with the largest net value; proceed unless it is positive."""
        effects = candidate_effects(step, state, self.settings, self.tables)
        best: Effect = effects[0]
        best_value = 0.0
        for eff in effects[1:]:
            value = eff.net_value(self.tau)
            if value > best_value + 1e-12:
                best, best_value = eff, value
        numbers = {
            "S_prev": state["survival"],
            "G": step.G,
            "w": step.w,
            "eps": step.eps,
            "proj": effects[0].projection,
            "tau": self.tau,
        }
        for eff in effects[1:]:
            numbers[f"delta_{eff.action.value}"] = eff.delta
            numbers[f"net_{eff.action.value}"] = eff.net_value(self.tau)
        if best.action == Action.PROCEED:
            return Decision(
                action=Action.PROCEED,
                reason=f"marginal: proceed (no positive net value), {_fmt(numbers)}",
                numbers=numbers,
            )
        return Decision(
            action=best.action,
            target_step=best.target_step,
            reason=(
                f"marginal: {best.action.value}{' ' + best.note if best.note else ''}, "
                f"delta={best.delta:.3f} > tau*c={self.tau * best.cost:.3f}, {_fmt(numbers)}"
            ),
            numbers=numbers,
        )


class Budgeted:
    """Spending-function controller for the model-based per-output risk budget (objective B)."""

    name = "budgeted"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings, self.tables = settings, tables
        self.lam = _knob(settings, self.name)
        self.g = make_spending(settings.spending)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Proceed within budget; else the cheapest action that restores it; else escalate."""
        n, i = state["n_steps"], step.step
        w_vec = [self.tables.w(n, k) for k in range(1, n + 1)]
        allowed = self.lam * self.settings.alpha * self.g(i, n, w_vec)
        effects = candidate_effects(step, state, self.settings, self.tables)
        proj = effects[0].committed_projection
        numbers = {
            "proj": proj,
            "allowed": allowed,
            "lambda": self.lam,
            "g": self.g(i, n, w_vec),
            "S_prev": state["survival"],
            "eps": step.eps,
        }
        if proj <= allowed + 1e-12:
            return Decision(
                action=Action.PROCEED,
                reason=f"budgeted: proceed within budget, {_fmt(numbers)}",
                numbers=numbers,
            )
        qualifying = [e for e in effects[1:] if e.committed_projection <= allowed + 1e-12]
        if not qualifying:
            return Decision(
                action=Action.ESCALATE,
                reason=f"budgeted: escalate (no action restores budget), {_fmt(numbers)}",
                numbers=numbers,
            )
        best = min(qualifying, key=lambda e: (e.cost, -e.delta))
        return Decision(
            action=best.action,
            target_step=best.target_step,
            reason=(
                f"budgeted: {best.action.value} (cheapest within budget, "
                f"post={best.committed_projection:.3f}), {_fmt(numbers)}"
            ),
            numbers=numbers,
        )


class KnowNo:
    """KnowNo-style per-step conformal gate (Ren et al., 2023), on answer clusters.

    Candidates are the clusters with proportions ``p_j``; the nonconformity of
    a candidate is ``1 - p_j``. The train split provides the scores of the
    locally correct cluster, and ``qhat`` is their ``ceil((m+1)(1-eps))``-th
    smallest (``Tables.qhat``). The prediction set is the top cluster plus every
    cluster with ``1 - p_j <= qhat``; the step proceeds iff the set is a
    singleton, else a human is asked. The knob is the miscoverage ``eps``:
    smaller means larger sets and more asks. Without fitted scores the gate
    asks when ``1 - p_max > eps`` and flags the decision as uncalibrated.
    """

    name = "knowno"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.tables = tables
        self.eps = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Ask iff the conformal prediction set over the clusters is not a singleton."""
        shares = sorted((c.proportion for c in step.clusters), reverse=True) or [1.0]
        p_max = shares[0]
        qhat = self.tables.qhat(step.n_samples, self.eps)
        if qhat is None:
            set_size = 1 + sum(1 for p in shares[1:] if 1.0 - p <= 1.0 - self.eps)
            ask = 1.0 - p_max > self.eps or set_size > 1
            numbers = {"p_max": p_max, "eps": self.eps, "set_size": float(set_size)}
            numbers["uncalibrated"] = 1.0
        else:
            set_size = 1 + sum(1 for p in shares[1:] if 1.0 - p <= qhat)
            ask = set_size > 1
            numbers = {
                "qhat": qhat,
                "p_max": p_max,
                "eps": self.eps,
                "set_size": float(set_size),
                "m": float(len(self.tables.conformal_scores.get(step.n_samples, []))),
            }
        if ask:
            return Decision(
                action=Action.ASK_HUMAN, reason=f"knowno: ask, {_fmt(numbers)}", numbers=numbers
            )
        return Decision(
            action=Action.PROCEED, reason=f"knowno: proceed, {_fmt(numbers)}", numbers=numbers
        )


class RandomAsk:
    """Mixing baseline: ask a human at a step with probability ``1 - q``, independent of any signal.

    The knob ``q`` is the probability of proceeding. The draw is a deterministic
    function of the seed, instance, step and attempt, so the steps asked at
    ``q`` are asked at every smaller ``q`` (nested), which keeps the ascending
    knob less conservative as Learn-then-Test requires. Its expected cost is
    linear in ``q``; its error is not, because a chain is right only when every
    unchecked step is. Its measured curve of error against cost is the
    reference a selective controller has to beat. (The straight line between
    never asking and always asking belongs to another construction: choosing
    at random, per question, between checking no step and checking every step.)
    """

    name = "random_ask"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings = settings
        self.q = _knob(settings, self.name)

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Ask iff the step's uniform draw is at least ``q``."""
        draw = random.Random(
            stable_hash(
                self.settings.seed, "random_ask", state["instance_id"], step.step, step.attempt
            )
        ).random()
        numbers = {"q": self.q, "draw": draw}
        if draw >= self.q:
            return Decision(
                action=Action.ASK_HUMAN, reason=f"random_ask: ask, {_fmt(numbers)}", numbers=numbers
            )
        return Decision(
            action=Action.PROCEED, reason=f"random_ask: proceed, {_fmt(numbers)}", numbers=numbers
        )


class AbstainAsk:
    """Baseline: ask a human at a step whose committed answer is an explicit abstention.

    "Unknown", "not specified in the passages" and the like match
    :data:`ABSTAIN_PATTERN`. The rule has no knob (a one-point grid): it measures what
    a controller gains beyond reacting to the model's own statement that it cannot answer.
    """

    name = "abstain_ask"

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings = settings

    def decide(self, state: TrajectoryState, step: StepRecord) -> Decision:
        """Ask iff the committed answer states that it cannot answer."""
        abstained = bool(ABSTAIN_PATTERN.search(step.answer or ""))
        numbers = {"abstained": float(abstained)}
        if abstained:
            return Decision(
                action=Action.ASK_HUMAN,
                reason=f"abstain_ask: ask, {_fmt(numbers)}",
                numbers=numbers,
            )
        return Decision(
            action=Action.PROCEED, reason=f"abstain_ask: proceed, {_fmt(numbers)}", numbers=numbers
        )


for _cls in (
    FinalGate,
    FinalU,
    PerStepU,
    PerStepEps,
    Marginal,
    Budgeted,
    KnowNo,
    RandomAsk,
    AbstainAsk,
):
    register_policy(_cls.name, _cls)
