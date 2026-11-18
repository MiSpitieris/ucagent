"""Expected effect of each action on the projected final risk (Section 2.4).

Each candidate carries ``delta`` (expected reduction of the projected final
risk versus proceeding), its ``cost``, the ``projection`` after the action
(with the future factor) and the ``committed_projection`` (without it, for
the per-output risk budget). The formulas are the effect models of the
instructions. A resample expands the step's batch from its current size to
``K'`` samples (the experiment the resample-effect table was fitted on) and
costs only the extra draws; backtracking expands the target step the same way
and redoes the steps after it with fresh draws at their position means (see
``MATH.md``). A batch that already holds ``K'`` samples can be expanded no
further, so neither action is feasible for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.records import Action, StepRecord
from ucagent.core.risk.propagation import project
from ucagent.core.tables import Tables


@dataclass(frozen=True)
class Effect:
    """A feasible action and what the model expects from it."""

    action: Action
    target_step: int | None
    delta: float
    cost: float
    projection: float
    committed_projection: float
    note: str = ""

    def net_value(self, tau: float) -> float:
        """``delta - tau * cost``: positive means worth doing at price ``tau``."""
        return self.delta - tau * self.cost


def proceed_effect(step: StepRecord, survival_prev: float) -> Effect:
    """The reference: commit the step as is."""
    return Effect(
        action=Action.PROCEED,
        target_step=None,
        delta=0.0,
        cost=0.0,
        projection=project(survival_prev, step.eps, step.w, step.G),
        committed_projection=project(survival_prev, step.eps, step.w, 1.0),
    )


def ask_effect(step: StepRecord, survival_prev: float, settings: RunSettings) -> Effect:
    """Human check sets ``eps_i -> p_h``: ``delta = S G w (eps - p_h)``."""
    base = project(survival_prev, step.eps, step.w, step.G)
    after = project(survival_prev, settings.p_h, step.w, step.G)
    return Effect(
        action=Action.ASK_HUMAN,
        target_step=None,
        delta=base - after,
        cost=settings.costs.ask_human,
        projection=after,
        committed_projection=project(survival_prev, settings.p_h, step.w, 1.0),
    )


def resample_effect(
    step: StepRecord, state: TrajectoryState, settings: RunSettings, tables: Tables
) -> Effect | None:
    """Expand the batch to ``K'``: ``eps_i -> ehat_r(u_i, K')``; None when infeasible or unknown."""
    if state["resample_count"] >= settings.max_resamples:
        return None
    if step.n_samples >= settings.K_prime:
        return None  # nothing left to expand: the table describes one expansion
    ehat = tables.ehat_r(step.u, settings.K_prime)
    if ehat is None:
        return None
    survival_prev = state["survival"]
    base = project(survival_prev, step.eps, step.w, step.G)
    after = project(survival_prev, ehat, step.w, step.G)
    extra_calls = max(settings.K_prime - step.n_samples, 0)
    return Effect(
        action=Action.RESAMPLE,
        target_step=None,
        delta=base - after,
        cost=settings.costs.resample + extra_calls * settings.costs.llm_call,
        projection=after,
        committed_projection=project(survival_prev, ehat, step.w, 1.0),
        note=f"ehat_r={ehat:.3f}",
    )


def backtrack_effect(
    step: StepRecord, state: TrajectoryState, settings: RunSettings, tables: Tables
) -> Effect | None:
    """Backtrack to the riskiest committed step if it holds a large enough share of the risk.

    New survival: ``S_{j-1} (1 - w_j ehat_r(u_j, K')) prod_{l=j+1..i} (1 - w_l ebar_l)``.
    The target's batch is expanded to ``K'`` (so a target that already holds
    ``K'`` samples makes the action infeasible; no other step is substituted)
    and the later steps are redone with ``K`` fresh samples each.
    """
    history = state["history"]
    if not history or state["backtrack_count"] >= settings.max_backtracks:
        return None
    n = state["n_steps"]
    contributions = [(h.w * h.eps, idx) for idx, h in enumerate(history)]
    top, idx = max(contributions)
    risk_prev = 1.0 - state["survival"]
    if risk_prev <= 0.0 or top / risk_prev < settings.backtrack_min_share:
        return None
    target = history[idx]
    if target.n_samples >= settings.K_prime:
        return None
    j = target.step
    ehat = tables.ehat_r(target.u, settings.K_prime)
    if ehat is None:
        return None
    survival_new = 1.0
    for h in history[:idx]:
        survival_new *= 1.0 - h.w * h.eps
    survival_new *= 1.0 - target.w * ehat
    for later in range(j + 1, step.step + 1):
        ebar = tables.ebar(n, later)
        if ebar is None:
            return None
        survival_new *= 1.0 - tables.w(n, later) * ebar
    base = project(state["survival"], step.eps, step.w, step.G)
    after = float(min(1.0, max(0.0, 1.0 - survival_new * step.G)))
    steps_after = step.step - j
    extra_calls = max(settings.K_prime - target.n_samples, 0)
    cost = (
        settings.costs.backtrack
        + extra_calls * settings.costs.llm_call
        + steps_after * settings.K * settings.costs.llm_call
    )
    return Effect(
        action=Action.BACKTRACK,
        target_step=j,
        delta=base - after,
        cost=cost,
        projection=after,
        committed_projection=float(min(1.0, max(0.0, 1.0 - survival_new))),
        note=f"j={j} share={top / risk_prev:.2f}",
    )


def escalate_effect(step: StepRecord, survival_prev: float, settings: RunSettings) -> Effect:
    """Full human take-over: final risk ``-> p_e``."""
    base = project(survival_prev, step.eps, step.w, step.G)
    return Effect(
        action=Action.ESCALATE,
        target_step=None,
        delta=base - settings.p_e,
        cost=settings.costs.escalate,
        projection=settings.p_e,
        committed_projection=settings.p_e,
    )


def candidate_effects(
    step: StepRecord, state: TrajectoryState, settings: RunSettings, tables: Tables
) -> list[Effect]:
    """Every feasible action for the current step, ``proceed`` first."""
    survival_prev = state["survival"]
    out = [proceed_effect(step, survival_prev)]
    resample = resample_effect(step, state, settings, tables)
    if resample is not None:
        out.append(resample)
    backtrack = backtrack_effect(step, state, settings, tables)
    if backtrack is not None:
        out.append(backtrack)
    out.append(ask_effect(step, survival_prev, settings))
    out.append(escalate_effect(step, survival_prev, settings))
    return out
