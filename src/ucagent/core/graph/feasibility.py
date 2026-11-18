"""Feasibility of the two sampling actions: resample and backtrack.

A resample expands the current step's batch from ``K`` to ``K'`` samples and a
backtrack expands its target's batch the same way before redoing the later
steps, so both need a batch that is still below ``K'`` and an unused cap. The
same rules serve the policies (through the effect models), the interactive
session (allowed actions and overrides) and the ``control`` node, which turns
an infeasible forced or scripted action into ``proceed`` rather than letting
the graph dead-end.
"""

from __future__ import annotations

from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState, samples_key


def batch_size(state: TrajectoryState) -> int:
    """Samples drawn so far for the current step's batch."""
    pending = state.get("pending")
    if pending is not None:
        return pending.n_samples
    key = samples_key(state["i"], state.get("draw", 0))
    return len(state.get("samples", {}).get(key, []))


def resample_blocked(state: TrajectoryState, settings: RunSettings) -> str | None:
    """``None`` when a resample is feasible, else the reason it is not."""
    if state["resample_count"] >= settings.max_resamples:
        return f"max_resamples={settings.max_resamples} reached for this step"
    n = batch_size(state)
    if n >= settings.K_prime:
        return f"the batch already has {n} samples (K'={settings.K_prime}); nothing to expand"
    return None


def backtrack_blocked(
    state: TrajectoryState, settings: RunSettings, target: int | None = None
) -> str | None:
    """``None`` when a backtrack (to ``target``, or to some step) is feasible, else why not."""
    history = state["history"]
    if not history:
        return "no committed step to go back to"
    if state["backtrack_count"] >= settings.max_backtracks:
        return f"max_backtracks={settings.max_backtracks} reached"
    if target is not None:
        if not 1 <= target <= len(history):
            return f"backtrack target {target!r} is not in 1..{len(history)}"
        record = history[target - 1]
        if record.n_samples >= settings.K_prime:
            return (
                f"step {target} already has {record.n_samples} samples "
                f"(K'={settings.K_prime}); nothing to expand"
            )
        return None
    if all(record.n_samples >= settings.K_prime for record in history):
        return f"every committed step already has K'={settings.K_prime} samples"
    return None


def backtrack_targets(state: TrajectoryState, settings: RunSettings) -> dict[int, str | None]:
    """Each committed step mapped to ``None`` (a feasible target) or the blocking reason."""
    return {
        record.step: backtrack_blocked(state, settings, record.step) for record in state["history"]
    }
