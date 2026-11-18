"""Metrics over trajectories and steps (Section 5.8 of the instructions)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from ucagent.core.records import Action, Trajectory


def auroc(scores: Sequence[float], labels: Sequence[bool | int]) -> float:
    """Rank-based AUROC with tie handling; 0.5 when one class is absent."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(probs: Sequence[float], labels: Sequence[bool | int], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def final_error(trajectories: Sequence[Trajectory]) -> float:
    """Fraction of trajectories whose final answer is wrong."""
    if not trajectories:
        return float("nan")
    return sum(not t.final_correct for t in trajectories) / len(trajectories)


def selective_error(trajectories: Sequence[Trajectory]) -> float:
    """Error among non-escalated outputs (what is delivered automatically)."""
    kept = [t for t in trajectories if not t.escalated]
    return final_error(kept) if kept else float("nan")


def escalation_rate(trajectories: Sequence[Trajectory]) -> float:
    """Fraction of trajectories handed to a human entirely."""
    if not trajectories:
        return float("nan")
    return sum(t.escalated for t in trajectories) / len(trajectories)


def interventions_per_traj(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """Mean non-``proceed`` decisions per trajectory, total and by action."""
    n = max(len(trajectories), 1)
    out = {"total": sum(t.n_interventions for t in trajectories) / n}
    for action in Action:
        if action != Action.PROCEED:
            out[action.value] = sum(t.count(action) for t in trajectories) / n
    return out


def total_cost(trajectories: Sequence[Trajectory]) -> float:
    """Mean total cost per trajectory."""
    if not trajectories:
        return float("nan")
    return sum(t.total_cost for t in trajectories) / len(trajectories)


def step_pairs(trajectories: Sequence[Trajectory]) -> tuple[list[float], list[float], list[bool]]:
    """``(u, eps, locally_wrong)`` for every committed LLM step with a label."""
    u: list[float] = []
    eps: list[float] = []
    wrong: list[bool] = []
    for t in trajectories:
        for s in t.steps:
            if s.source == "llm" and s.locally_correct is not None:
                u.append(s.u)
                eps.append(s.eps)
                wrong.append(not s.locally_correct)
    return u, eps, wrong


def certificate_metrics(trajectories: Sequence[Trajectory], alpha: float) -> dict[str, float]:
    """Coverage and validity of the per-output certificate ``R_n <= alpha``."""
    if not trajectories:
        return {"certificate_coverage": float("nan"), "certificate_validity": float("nan")}
    certified = [t for t in trajectories if not t.escalated and t.risk_weighted_final <= alpha]
    return {
        "certificate_coverage": len(certified) / len(trajectories),
        "certificate_validity": final_error(certified) if certified else float("nan"),
    }


def summary(trajectories: Sequence[Trajectory], alpha: float | None = None) -> dict[str, float]:
    """The headline numbers of one run."""
    out: dict[str, float] = {
        "n": float(len(trajectories)),
        "final_error": final_error(trajectories),
        "selective_error": selective_error(trajectories),
        "escalation_rate": escalation_rate(trajectories),
        "total_cost": total_cost(trajectories),
    }
    out.update({f"interventions_{k}": v for k, v in interventions_per_traj(trajectories).items()})
    u, eps, wrong = step_pairs(trajectories)
    if u:
        out["auroc_u_local"] = auroc(u, wrong)
        out["ece_u"] = ece(u, wrong)
        out["ece_eps"] = ece(eps, wrong)
    finals = [t for t in trajectories if t.steps]
    if finals:
        wrong_final = [not t.final_correct for t in finals]
        out["auroc_u_final"] = auroc([t.u_final or 0.0 for t in finals], wrong_final)
        out["auroc_risk_naive"] = auroc([t.risk_naive_final for t in finals], wrong_final)
        out["auroc_risk_weighted"] = auroc([t.risk_weighted_final for t in finals], wrong_final)
    if alpha is not None:
        out.update(certificate_metrics(trajectories, alpha))
    return out
