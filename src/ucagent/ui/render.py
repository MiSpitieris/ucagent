"""Turn session state into tables and series for the page (pandas only, no Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.messages import BaseMessage

from ucagent.core.graph.runtime import RunSettings
from ucagent.core.graph.state import TrajectoryState
from ucagent.core.llm.mock import HEADER_RE
from ucagent.core.records import Action, Decision, Intervention, StepRecord, Trajectory
from ucagent.core.risk.effects import Effect
from ucagent.core.risk.propagation import project

STEP_COLUMNS = [
    "step",
    "attempt",
    "draw",
    "answer",
    "source",
    "u",
    "entropy",
    "eps",
    "w",
    "G",
    "samples",
    "survival",
    "risk so far",
    "projected final risk",
]


def prompt_text(prompt: list[BaseMessage]) -> str:
    """The step's human message without the ``<ucagent ...>`` header line."""
    for message in reversed(prompt):
        if message.type == "human":
            content = message.content if isinstance(message.content, str) else str(message.content)
            return HEADER_RE.sub("", content).strip()
    return ""


def steps_frame(values: TrajectoryState) -> pd.DataFrame:
    """One row per committed step with the numbers that drove the risk."""
    rows: list[dict[str, Any]] = []
    survival = 1.0
    for h in values["history"]:
        projected = project(survival, h.eps, h.w, h.G)
        survival *= 1.0 - h.w * h.eps
        rows.append(
            {
                "step": h.step,
                "attempt": h.attempt,
                "draw": h.draw,
                "answer": h.answer,
                "source": h.source,
                "u": h.u,
                "entropy": h.entropy,
                "eps": h.eps,
                "w": h.w,
                "G": h.G,
                "samples": h.n_samples,
                "survival": survival,
                "risk so far": 1.0 - survival,
                "projected final risk": projected,
            }
        )
    return pd.DataFrame(rows, columns=STEP_COLUMNS)


def risk_series(values: TrajectoryState, alpha: float) -> pd.DataFrame:
    """Risk so far and projected final risk per committed step, plus the target ``alpha``."""
    frame = steps_frame(values)
    out = frame[["step", "risk so far", "projected final risk"]].set_index("step")
    out["alpha"] = alpha
    return out


def pending_projection(values: TrajectoryState) -> float | None:
    """Projected final risk if the provisional step is committed as is."""
    pending = values.get("pending")
    if pending is None:
        return None
    return project(values["survival"], pending.eps, pending.w, pending.G)


def clusters_frame(step: StepRecord | None) -> pd.DataFrame:
    """The answer clusters of a provisional step, largest first."""
    if step is None:
        return pd.DataFrame(columns=["answer", "share", "samples"])
    rows = [
        {"answer": c.representative, "share": c.proportion, "samples": len(c.members)}
        for c in step.clusters
    ]
    frame = pd.DataFrame(rows, columns=["answer", "share", "samples"])
    return frame.sort_values("share", ascending=False).reset_index(drop=True)


def clusters_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Same table from a ``check_step`` interrupt payload."""
    rows = [
        {"answer": c["representative"], "share": c["proportion"], "samples": len(c["members"])}
        for c in payload.get("clusters", [])
    ]
    frame = pd.DataFrame(rows, columns=["answer", "share", "samples"])
    return frame.sort_values("share", ascending=False).reset_index(drop=True)


def samples_frame(values: TrajectoryState, key: str) -> pd.DataFrame:
    """The raw samples of one step attempt (``key`` is ``"step:attempt"``)."""
    samples = sorted(values["samples"].get(key, []), key=lambda s: s.sample_index)
    rows = [
        {"sample": s.sample_index, "answer": s.answer, "confidence": s.confidence, "text": s.text}
        for s in samples
    ]
    return pd.DataFrame(rows, columns=["sample", "answer", "confidence", "text"])


def effects_frame(effects: list[Effect], tau: float | None) -> pd.DataFrame:
    """Every feasible action with its expected effect; net value at price ``tau``."""
    rows = [
        {
            "action": e.action.value,
            "target": e.target_step,
            "delta (risk reduction)": e.delta,
            "cost": e.cost,
            "net value": e.net_value(tau) if tau is not None else None,
            "projected final risk": e.projection,
            "committed risk": e.committed_projection,
            "note": e.note,
        }
        for e in effects
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "action",
            "target",
            "delta (risk reduction)",
            "cost",
            "net value",
            "projected final risk",
            "committed risk",
            "note",
        ],
    )


def proposal_frame(decision: Decision) -> pd.DataFrame:
    """The numbers a policy reported with its decision."""
    rows = [{"quantity": k, "value": v} for k, v in decision.numbers.items()]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def decisions_frame(interventions: list[Intervention]) -> pd.DataFrame:
    """Every controller decision so far, human overrides flagged."""
    rows = [
        {
            "step": iv.step,
            "attempt": iv.attempt,
            "action": iv.action.value,
            "target": iv.target_step,
            "cost": iv.cost,
            "override": iv.numbers.get("override", 0.0) == 1.0,
            "reason": iv.reason,
        }
        for iv in interventions
    ]
    return pd.DataFrame(
        rows, columns=["step", "attempt", "action", "target", "cost", "override", "reason"]
    )


def status_numbers(values: TrajectoryState, settings: RunSettings) -> dict[str, Any]:
    """The headline numbers of the run so far."""
    survival = float(values["survival"])
    llm_cost = values["llm_calls"] * settings.costs.llm_call
    total = values["total_cost"] if values.get("done") else values["fixed_cost"] + llm_cost
    return {
        "step": min(values["i"], values["n_steps"]),
        "n_steps": values["n_steps"],
        "committed": len(values["history"]),
        "survival": survival,
        "risk_so_far": 1.0 - survival,
        "cost": float(total),
        "llm_calls": int(values["llm_calls"]),
        "interventions": sum(1 for iv in values["interventions"] if iv.action != Action.PROCEED),
    }


def trace_summary(trace: list[str]) -> str:
    """``prepare_step -> sample x5 -> estimate`` from the node names of one advance."""
    parts: list[list[Any]] = []
    for name in trace:
        if parts and parts[-1][0] == name:
            parts[-1][1] += 1
        else:
            parts.append([name, 1])
    return " -> ".join(f"{n} x{c}" if c > 1 else str(n) for n, c in parts)


def audit_frame(trajectory: Trajectory) -> pd.DataFrame:
    """``Trajectory.audit()`` as a table (decisions joined into one cell)."""
    rows = []
    for row in trajectory.audit():
        decisions = "; ".join(f"{d['action']}: {d['reason']}" for d in row["decisions"])
        rows.append({**{k: v for k, v in row.items() if k != "decisions"}, "decisions": decisions})
    return pd.DataFrame(rows, columns=["step", "answer", "u", "eps", "w", "source", "decisions"])
