"""Markdown tables for a suite's results, real model or mock, from its files only.

Usage: ``uv run python scripts/real_results.py results/<suite_id> [--label TEXT]
[--alpha-extra A] [--paired-with results/<other_suite>]``
or, for a single ``policy none`` run directory, ``results/<run_id>``.

Reads ``manifest.csv``, ``ltt.csv``, ``coverage.json``, ``artifacts/seed0.json`` and the
parquet files of the ``none`` policy (and of the selected knobs); writes the tables to
stdout. Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ucagent.core.control.ltt import calibrate_knob
from ucagent.core.records import Trajectory
from ucagent.core.uq.base import normalized_entropy
from ucagent.experiments.figures import POLICY_ORDER, aggregate_curves
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.suite import configured_knobs


def _auroc(y: pd.Series, score: pd.Series) -> float | None:
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y.astype(int), score.astype(float)))


def _resolve(suite_dir: Path, path: str) -> Path:
    """A manifest path, or the same run under ``suite_dir`` when the suite was moved."""
    p = Path(path)
    if p.exists():
        return p
    moved = suite_dir / "runs" / p.parent.name / p.name
    return moved if moved.exists() else p


def _task_of(config_path: Path, key: str | None = None) -> Any:
    """The task of a run or suite config file, or None when the file is missing."""
    if not config_path.exists():
        return None
    import yaml

    from ucagent.tasks import make_task

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = data[key]["task"] if key else data["task"]
    try:
        return make_task(block)
    except (KeyError, FileNotFoundError, ValueError):
        return None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _on_track(task: Any, trajectory: Trajectory, step: Any) -> bool:
    """Whether the step had a local truth (recomputed, so archived files report alike)."""
    has = getattr(task, "has_local_truth", None) if task is not None else None
    if has is None:
        return True
    return bool(has(task.get_instance(trajectory.instance_id), step.step, step.context))


def signal_table(trajectories: list[Trajectory], label: str, task: Any = None) -> list[str]:
    """Uncontrolled error and how predictive u / R_n are, from ``none`` trajectories.

    Local-error AUROCs are given over every labelled hop (archived runs stored
    off-track hops as wrong) and over the hops that have a local truth: first
    hops, later on-track hops, and both pooled. The final-failure comparison
    carries a paired bootstrap interval over questions.
    """
    steps = pd.DataFrame(
        [
            {
                "step": s.step,
                "u": s.u,
                "eps": s.eps,
                "local_ok": s.locally_correct,
                "on_track": _on_track(task, t, s),
            }
            for t in trajectories
            for s in t.steps
            if s.source == "llm"
        ]
    )
    labelled = steps.dropna(subset=["local_ok"])
    wrong = ~labelled["local_ok"].astype(bool)
    fin = pd.DataFrame(
        {
            "wrong": [not t.final_correct for t in trajectories],
            "u_n": [t.u_final if t.u_final is not None else 0.0 for t in trajectories],
            "R_n": [t.risk_weighted_final for t in trajectories],
            # simple aggregates over the steps, as baselines for the propagated risk
            "u_first": [t.steps[0].u if t.steps else 0.0 for t in trajectories],
            "u_max": [max((s.u for s in t.steps), default=0.0) for t in trajectories],
            "u_mean": [
                float(np.mean([s.u for s in t.steps])) if t.steps else 0.0 for t in trajectories
            ],
            "eps_max": [max((s.eps for s in t.steps), default=0.0) for t in trajectories],
        }
    )
    per_step = labelled.groupby("step").agg(
        local_error=("local_ok", lambda x: 1 - x.astype(bool).mean()),
        u_mean=("u", "mean"),
        unanimous=("u", lambda x: (x == 0).mean()),
    )
    calls = np.mean([t.llm_calls for t in trajectories])
    cost = np.mean([t.total_cost for t in trajectories])
    lines = [
        f"**{label}**: {len(trajectories)} trajectories, final error {fin['wrong'].mean():.3f}, "
        f"mean LLM calls {calls:.1f}, mean cost {cost:.2f}.",
        "",
        "| step | local error | mean u | share of unanimous steps |",
        "|---|---|---|---|",
    ]
    for step, r in per_step.iterrows():
        lines.append(
            f"| {step} | {r['local_error']:.3f} | {r['u_mean']:.3f} | {r['unanimous']:.2f} |"
        )
    n_lab = len(labelled)
    lines += [
        "",
        "| statistic | AUROC |",
        "|---|---|",
        f"| u -> local error (all stored labels, {n_lab} steps; archived runs stored off-track "
        f"hops as wrong, the rows below use the hops with a local truth) | "
        f"{_fmt(_auroc(wrong, labelled['u']))} |",
        f"| eps -> local error | {_fmt(_auroc(wrong, labelled['eps']))} |",
    ]
    first = labelled[labelled["step"] == 1]
    later = labelled[(labelled["step"] > 1) & labelled["on_track"]]
    track = labelled[labelled["on_track"]]
    for name, sub in (
        ("first hops", first),
        ("later hops with a local truth (on-track prefix)", later),
        ("all hops with a local truth", track),
    ):
        w = ~sub["local_ok"].astype(bool)
        lines.append(
            f"| u -> local error, {name} ({len(sub)} hops, {int(w.sum())} wrong) | "
            f"{_fmt(_auroc(w, sub['u']))} |"
        )
    gain, lo, hi = _auroc_gain_ci(fin)
    gain_max, lo_max, hi_max = _auroc_gain_ci(fin, other="u_max")
    lines += [
        f"| last-step u_n -> final failure | {_fmt(_auroc(fin['wrong'], fin['u_n']))} |",
        f"| propagated R_n -> final failure | {_fmt(_auroc(fin['wrong'], fin['R_n']))} |",
        f"| gain of R_n over u_n, paired bootstrap over questions (95 %) | "
        f"{_fmt(gain)} [{_fmt(lo)}, {_fmt(hi)}] |",
        f"| first-step u -> final failure | {_fmt(_auroc(fin['wrong'], fin['u_first']))} |",
        f"| largest step u -> final failure | {_fmt(_auroc(fin['wrong'], fin['u_max']))} |",
        f"| mean step u -> final failure | {_fmt(_auroc(fin['wrong'], fin['u_mean']))} |",
        f"| largest step eps -> final failure | {_fmt(_auroc(fin['wrong'], fin['eps_max']))} |",
        f"| gain of R_n over the largest step u, paired bootstrap over questions (95 %) | "
        f"{_fmt(gain_max)} [{_fmt(lo_max)}, {_fmt(hi_max)}] |",
    ]
    return lines


def _auroc_gain_ci(
    fin: pd.DataFrame, resamples: int = 5000, seed: int = 0, other: str = "u_n"
) -> tuple[float | None, float | None, float | None]:
    """AUROC(R_n) - AUROC(other) for final failure, percentile interval over questions."""
    if fin["wrong"].nunique() < 2:
        return None, None, None
    gain = float(_auroc(fin["wrong"], fin["R_n"]) or 0.0) - float(
        _auroc(fin["wrong"], fin[other]) or 0.0
    )
    rng = np.random.default_rng(seed)
    draws = []
    idx = np.arange(len(fin))
    for _ in range(resamples):
        b = fin.iloc[rng.choice(idx, size=len(idx), replace=True)]
        if b["wrong"].nunique() < 2:
            continue
        draws.append(
            roc_auc_score(b["wrong"].astype(int), b["R_n"])
            - roc_auc_score(b["wrong"].astype(int), b[other])
        )
    return gain, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def final_errors(trajectories: list[Trajectory], task: Any) -> list[str]:
    """Final error under both label rules, recomputed from the stored final answers.

    ``strict`` is MuSiQue's exact match against the answer and its aliases;
    ``containment`` is the task's whole-word containment rule (either way), the one
    the per-hop labels use when ``local_match="contains"``. The suite's own final
    label (``final_match``) is one of the two; the other shows how much of the
    error is verbose-but-right final answers.
    """
    matches, truths = getattr(task, "_matches", None), getattr(task, "_truths", None)
    if matches is None or truths is None or not trajectories:
        return []
    strict = containment = 0
    for t in trajectories:
        inst = task.get_instance(t.instance_id)
        gold = truths(inst, inst.n_steps)
        strict += not matches(t.final_answer or "", gold, "exact")
        containment += not matches(t.final_answer or "", gold, "contains")
    n = len(trajectories)
    return [
        f"Final error by rule: strict (exact match with aliases) {strict / n:.3f}, "
        f"containment (the hop rule) {containment / n:.3f}; this suite's final label is "
        f"`{getattr(task, 'final_match', 'exact')}`."
    ]


def _partitions(n: int, largest: int | None = None) -> Iterator[list[int]]:
    """Integer partitions of ``n`` in non-increasing order."""
    largest = n if largest is None else largest
    if n == 0:
        yield []
        return
    for k in range(min(n, largest), 0, -1):
        for rest in _partitions(n - k, k):
            yield [k, *rest]


def u_lattice(k: int) -> dict[float, list[list[int]]]:
    """Attainable values of the normalised cluster entropy at ``K = k``, by cluster sizes."""
    out: dict[float, list[list[int]]] = {}
    for p in _partitions(k):
        u = round(normalized_entropy([x / k for x in p], k), 4)
        out.setdefault(u, []).append(p)
    return out


def _split_label(u: float, lattice: dict[float, list[list[int]]]) -> str:
    parts = lattice.get(round(u, 4))
    return " or ".join("-".join(map(str, p)) for p in parts) if parts else "?"


def signal_by_level(trajectories: list[Trajectory], task: Any, k: int) -> list[str]:
    """Local error at each attainable disagreement level, over labelled on-track hops.

    Shows whether the signal is graded beyond "unanimous or not" (the isotonic fit on
    a small train split can pool graded levels into one).
    """
    rows = [
        (round(s.u, 4), int(not s.locally_correct))
        for t in trajectories
        for s in t.steps
        if s.source == "llm" and s.locally_correct is not None and _on_track(task, t, s)
    ]
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["u", "wrong"])
    lattice = u_lattice(k)
    lines = [
        f"Local error by disagreement level ({len(df)} labelled on-track hops, K = {k}):",
        "",
        "| u | samples split as | hops | local error |",
        "|---|---|---|---|",
    ]
    for u, g in df.groupby("u"):
        lines.append(
            f"| {u:.3f} | {_split_label(float(u), lattice)} | {len(g)} | {g['wrong'].mean():.3f} |"
        )
    non_unanimous = df[df["u"] > 0]
    auroc = _auroc(non_unanimous["wrong"].astype(bool), non_unanimous["u"])
    lines += [
        "",
        f"AUROC of u for local error within the {len(non_unanimous)} non-unanimous hops: "
        f"{_fmt(auroc)}.",
    ]
    return lines


def _implied_counts(levels: list[float], n: int) -> list[str]:
    """Train hops per level implied by the stored level values (``wrong / hops`` as a
    reduced fraction, scaled so that the hop counts add up to ``n``; unique or n/a)."""
    reduced = []
    tolerance = max(1e-12, 0.5 / (n * n))  # distinct fractions with denominators <= n differ more
    for y in levels:
        found = None
        for b in range(1, n + 1):
            a = round(y * b)
            if abs(a / b - y) < tolerance:
                g = gcd(a, b) or 1
                found = (a // g, b // g)
                break
        if found is None:
            return ["n/a"] * len(levels)
        reduced.append(found)
    solutions: list[list[tuple[int, int]]] = []

    def walk(i: int, left: int, acc: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if i == len(reduced):
            if left == 0:
                solutions.append(list(acc))
            return
        a, b = reduced[i]
        for m in range(1, left // b + 1):
            walk(i + 1, left - m * b, [*acc, (m * a, m * b)])

    walk(0, n, [])
    if len(solutions) != 1:
        return [f"{a}/{b} (reduced)" for a, b in reduced]
    return [f"{a}/{b}" for a, b in solutions[0]]


def isotonic_levels(suite_dir: Path, seed: int = 0) -> list[str]:
    """The artifact's isotonic maps evaluated on every attainable value of u.

    A level is a distinct value of eps; the controllers can only distinguish steps
    whose u falls on different levels.
    """
    from ucagent.experiments.suite import load_artifact_for

    try:
        art = load_artifact_for(suite_dir, seed)
    except FileNotFoundError:
        return []
    if art.estimator not in ("semantic_entropy", "disagreement"):
        return [
            f"Calibration map for estimator `{art.estimator}`: {len(art.isotonic_K.x)} "
            f"breakpoints at K = {art.K} and {len(art.isotonic_K_prime.x)} at K' = {art.K_prime} "
            f"({art.isotonic_K.n} labelled train hops); its u is not confined to the cluster-share "
            "lattice, so the levels are counted on the test split (the distinct eps values above).",
            "",
        ]
    lines = [
        "Calibration map levels (the artifact's isotonic fits evaluated on every way the "
        "samples can split):",
        "",
    ]
    for name, data, k in (("K", art.isotonic_K, art.K), ("K'", art.isotonic_K_prime, art.K_prime)):
        lattice = u_lattice(k)
        levels: dict[float, list[float]] = {}
        exact: dict[float, float] = {}
        for key, parts in sorted(lattice.items()):
            u = normalized_entropy([x / k for x in parts[0]], k)  # unrounded, as at run time
            eps = data(u)
            levels.setdefault(round(eps, 6), []).append(key)
            exact.setdefault(round(eps, 6), eps)
        xs = ", ".join(f"{x:.3f}" for x in data.x)
        ys = ", ".join(f"{y:.3f}" for y in data.y)
        lines += [
            f"map for {name} = {k} ({data.n} labelled train hops; breakpoints u = [{xs}], "
            f"eps = [{ys}]): {len(levels)} levels",
            "",
            "| eps level | u range | attainable u values | train hops wrong/total |",
            "|---|---|---|---|",
        ]
        counts = _implied_counts([exact[key] for key in sorted(levels)], data.n)
        for (eps, us), count in zip(sorted(levels.items()), counts, strict=True):
            lines.append(
                f"| {eps:.3f} | {min(us):.3f} to {max(us):.3f} | {len(us)} of {len(lattice)} | "
                f"{count} |"
            )
        lines.append("")
    return lines


def operating_points(suite_dir: Path, split: str = "test") -> list[str]:
    """Distinct (error, cost) pairs along each policy's grid: how many ways the knob can
    actually move the policy on this task."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    grid = manifest[(manifest["label"] == "grid") & (manifest["split"] == split)]
    if "variant" in grid.columns:
        grid = grid[grid["variant"] == "main"]
    lines = [
        f"Operating points on the {split} split (distinct (error, cost) pairs over each "
        "policy's grid, seeds averaged, 3 decimals):",
        "",
        "| policy | grid points | distinct (error, cost) | distinct ask rates | "
        "points, error at cost |",
        "|---|---|---|---|---|",
    ]
    for policy in POLICY_ORDER:
        sub = grid[grid["policy"] == policy].dropna(subset=["knob"])
        if sub.empty:
            continue
        agg = sub.groupby("knob").agg(
            final_error=("final_error", "mean"),
            total_cost=("total_cost", "mean"),
            asks=("interventions_ask_human", "mean"),
        )
        pairs = sorted(
            {
                (round(e, 3), round(c, 3))
                for e, c in zip(agg["final_error"], agg["total_cost"], strict=True)
            },
            key=lambda p: p[1],
        )
        asks = {round(a, 3) for a in agg["asks"]}
        points = ", ".join(f"{e:.3f} at {c:.2f}" for e, c in pairs)
        lines.append(f"| {policy} | {len(agg)} | {len(pairs)} | {len(asks)} | {points} |")
    return lines


def _breakdown(trajectories: list[Trajectory]) -> dict[str, int]:
    """Wrong trajectories by what went wrong at the step level."""
    out = {
        "n": len(trajectories),
        "wrong": 0,
        "unanimous_wrong": 0,
        "other_wrong": 0,
        "label_gap": 0,
        "escalated": 0,
        "unlabelled": 0,
    }
    for t in trajectories:
        if t.final_correct:
            continue
        out["wrong"] += 1
        llm = [s for s in t.steps if s.source == "llm"]
        if t.escalated:
            out["escalated"] += 1
        elif any(s.u == 0 and s.locally_correct is False for s in llm):
            out["unanimous_wrong"] += 1
        elif any(s.locally_correct is False for s in llm):
            out["other_wrong"] += 1
        elif llm and all(s.locally_correct for s in llm):
            out["label_gap"] += 1
        else:
            out["unlabelled"] += 1
    return out


def wrong_breakdown(suite_dir: Path) -> list[str]:
    """Why the final answer is wrong, at the uncontrolled run and at each policy's
    Learn-then-Test knob (test split): a unanimous-and-wrong hop (no disagreement gate
    can see it), a non-unanimous wrong hop (seen but not acted on, or the fix failed),
    or every LLM hop locally correct yet the final answer judged wrong (the final label
    is stricter than the hop labels)."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    test = manifest[manifest["split"] == "test"]
    targets: list[tuple[str, pd.DataFrame]] = [("none", test[test["policy"] == "none"])]
    ltt_path = suite_dir / "ltt.csv"
    if ltt_path.exists():
        ltt = pd.read_csv(ltt_path)
        for policy in POLICY_ORDER:
            sel = ltt[(ltt["policy"] == policy) & ltt["feasible"]]
            if sel.empty:
                continue
            rows = pd.concat(
                [
                    test[
                        (test["policy"] == policy)
                        & (test["seed"] == r["seed"])
                        & (np.isclose(test["knob"].astype(float), float(r["knob_star"])))
                    ]
                    for _, r in sel.iterrows()
                ]
            )
            targets.append((f"{policy} at knob* {sel['knob_star'].median():.4g}", rows))
    lines = [
        "Wrong final answers by cause (test split, seeds pooled):",
        "",
        "| run | wrong / n | unanimous-wrong hop | other wrong hop | all hops locally correct "
        "| escalated |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in targets:
        trajectories: list[Trajectory] = []
        for path in rows["path"]:
            trajectories.extend(read_trajectories(_resolve(suite_dir, str(path))))
        if not trajectories:
            continue
        b = _breakdown(trajectories)
        extra = f" (+{b['unlabelled']} unlabelled)" if b["unlabelled"] else ""
        lines.append(
            f"| {name} | {b['wrong']}/{b['n']} | {b['unanimous_wrong']} | {b['other_wrong']} | "
            f"{b['label_gap']} | {b['escalated']}{extra} |"
        )
    return lines


def _hop_frame(suite_dir: Path, split: str = "test") -> pd.DataFrame:
    """Labelled on-track LLM hops of the ``none`` runs: instance, step, u, wrong."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    none = manifest[(manifest["policy"] == "none") & (manifest["split"] == split)]
    task = _task_of(suite_dir / "suite.yaml", key="base")
    rows = []
    for path in none["path"]:
        for t in read_trajectories(_resolve(suite_dir, str(path))):
            for s in t.steps:
                if s.source == "llm" and s.locally_correct is not None and _on_track(task, t, s):
                    rows.append((t.instance_id, s.step, float(s.u), int(not s.locally_correct)))
    return pd.DataFrame(rows, columns=["instance_id", "step", "u", "wrong"])


def paired_signal(suite_a: Path, suite_b: Path, resamples: int = 5000, seed: int = 0) -> list[str]:
    """Signal statistics of two suites on their common test questions, with a paired
    bootstrap over questions for the AUROC difference (this suite minus the other)."""
    a, b = _hop_frame(suite_a), _hop_frame(suite_b)
    common = sorted(set(a["instance_id"]) & set(b["instance_id"]))
    a = a[a["instance_id"].isin(common)].reset_index(drop=True)
    b = b[b["instance_id"].isin(common)].reset_index(drop=True)
    lines = [
        f"Paired signal comparison, {suite_a.name} (this) against {suite_b.name} (other), "
        f"on the {len(common)} common test questions:",
        "",
        "| statistic | this | other | difference [95 %, paired bootstrap over questions] |",
        "|---|---|---|---|",
    ]
    unanimous_a, unanimous_b = a[a["u"] == 0], b[b["u"] == 0]
    for name, frame_a, frame_b in (
        ("labelled on-track hops", a, b),
        ("unanimous hops (u = 0)", unanimous_a, unanimous_b),
        (
            "unanimous-and-wrong hops",
            unanimous_a[unanimous_a["wrong"] == 1],
            unanimous_b[unanimous_b["wrong"] == 1],
        ),
    ):
        lines.append(f"| {name} | {len(frame_a)} | {len(frame_b)} | - |")
    lines.append(
        f"| local error, labelled on-track hops | {a['wrong'].mean():.3f} | "
        f"{b['wrong'].mean():.3f} | - |"
    )
    rng = np.random.default_rng(seed)
    groups_a = {k: v.index.to_numpy() for k, v in a.groupby("instance_id")}
    groups_b = {k: v.index.to_numpy() for k, v in b.groupby("instance_id")}
    ids = np.array(common)
    for name, mask_a, mask_b in (
        ("AUROC of u for local error, all labelled on-track hops", None, None),
        ("AUROC of u for local error, first hops", a["step"] == 1, b["step"] == 1),
    ):
        fa = a if mask_a is None else a[mask_a]
        fb = b if mask_b is None else b[mask_b]
        auc_a = _auroc(fa["wrong"].astype(bool), fa["u"])
        auc_b = _auroc(fb["wrong"].astype(bool), fb["u"])
        keep_a = np.ones(len(a), dtype=bool) if mask_a is None else mask_a.to_numpy()
        keep_b = np.ones(len(b), dtype=bool) if mask_b is None else mask_b.to_numpy()
        ua, wa = a["u"].to_numpy(), a["wrong"].to_numpy()
        ub, wb = b["u"].to_numpy(), b["wrong"].to_numpy()
        draws = []
        for _ in range(resamples):
            draw = rng.choice(ids, size=len(ids), replace=True)
            ia = np.concatenate([groups_a[i] for i in draw])
            ib = np.concatenate([groups_b[i] for i in draw])
            ia, ib = ia[keep_a[ia]], ib[keep_b[ib]]
            if len(set(wa[ia])) < 2 or len(set(wb[ib])) < 2:
                continue
            draws.append(roc_auc_score(wa[ia], ua[ia]) - roc_auc_score(wb[ib], ub[ib]))
        diff = None if auc_a is None or auc_b is None else auc_a - auc_b
        lo = float(np.percentile(draws, 2.5)) if draws else None
        hi = float(np.percentile(draws, 97.5)) if draws else None
        lines.append(
            f"| {name} | {_fmt(auc_a)} | {_fmt(auc_b)} | {_fmt(diff)} [{_fmt(lo)}, {_fmt(hi)}] |"
        )
    return lines


def unanimity_floor(trajectories: list[Trajectory]) -> list[str]:
    """Steps that were unanimous (u = 0) and wrong, per step: the floor of any gate that
    proceeds on agreement (a KnowNo-style set is a singleton whenever the samples agree)."""
    lines = []
    for step in sorted({s.step for t in trajectories for s in t.steps}):
        rows = [s for t in trajectories for s in t.steps if s.step == step and s.source == "llm"]
        lab = [s for s in rows if s.locally_correct is not None]
        uw = sum(1 for s in lab if s.u == 0 and not s.locally_correct)
        lines.append(f"step {step}: {uw}/{len(lab)} labelled steps unanimous and wrong")
    return ["Unanimous-and-wrong steps (floor for an agreement gate): " + "; ".join(lines)]


def ltt_at_alpha(suite_dir: Path, alpha: float) -> list[str]:
    """Learn-then-Test re-run at a looser target from the same calibration runs.

    Supplementary reading only: the files on disk keep the configured alpha. The
    ``budgeted`` policy spends ``lambda * alpha`` inside its decisions, so its knob
    means a different budget here; the certificate itself is unchanged.
    """
    import yaml

    manifest = pd.read_csv(suite_dir / "manifest.csv")
    cal = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))["base"].get(
        "calibration", {}
    )
    delta = float(cal.get("delta", 0.1))
    pvalue = cal.get("pvalue", "binomial")
    grid = manifest[manifest["label"] == "grid"]
    if "variant" in grid.columns:
        grid = grid[grid["variant"] == "main"]
    lines = [
        f"Learn-then-Test at the looser target alpha = {alpha:g} (delta = {delta:g}), "
        "from the same runs:",
        "",
        "| policy | feasible seeds | knob* | calib risk | test error at knob* | test cost "
        "| interventions |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy in POLICY_ORDER:
        sub = grid[grid["policy"] == policy]
        if sub.empty or policy == "none":
            continue
        rows = []
        for seed, g in sub.groupby("seed"):
            c = g[g["split"] == "calib"].dropna(subset=["knob"])
            if c.empty:
                continue
            points = [
                (float(k), int(n), round(float(e) * int(n)))
                for k, n, e in zip(c["knob"], c["n"], c["final_error"], strict=True)
            ]
            res = calibrate_knob(
                policy,
                points,
                alpha=alpha,
                delta=delta,
                pvalue=pvalue,
                seed=int(seed),
                grid=configured_knobs(suite_dir, policy),
            )
            t = g[(g["split"] == "test") & (g["knob"] == res.knob_star)]
            rows.append((res, t))
        feasible = [(r, t) for r, t in rows if r.feasible and not t.empty]
        if not feasible:
            lines.append(f"| {policy} | 0/{len(rows)} | - | - | - | - | - |")
            continue
        knob = float(np.median([r.knob_star for r, _ in feasible]))
        calib = float(np.mean([r.selected.risk_hat for r, _ in feasible if r.selected]))
        err = float(np.mean([t["final_error"].iloc[0] for _, t in feasible]))
        cost = float(np.mean([t["total_cost"].iloc[0] for _, t in feasible]))
        ivs = float(np.mean([t["interventions_total"].iloc[0] for _, t in feasible]))
        lines.append(
            f"| {policy} | {len(feasible)}/{len(rows)} | {knob:.4g} | {calib:.3f} | {err:.3f} | "
            f"{cost:.2f} | {ivs:.2f} |"
        )
    return lines


def ltt_table(suite_dir: Path) -> list[str]:
    """Learn-then-Test operating points per policy."""
    ltt_path = suite_dir / "ltt.csv"
    if not ltt_path.exists():
        return ["(no ltt.csv: run `ucagent calibrate`)"]
    ltt = pd.read_csv(ltt_path)
    cov = json.loads((suite_dir / "coverage.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    none_rows = manifest[(manifest["policy"] == "none") & (manifest["split"] == "test")]
    alpha = float(ltt["alpha"].iloc[0]) if len(ltt) else float("nan")
    lines = [
        f"Learn-then-Test at alpha = {alpha:g}, delta = 0.1; uncontrolled (policy none) test "
        f"error {none_rows['final_error'].mean():.3f}, cost {none_rows['total_cost'].mean():.2f}.",
        "",
        "| policy | feasible seeds | min calib risk | knob* | test error at knob* | test cost "
        "| interventions | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in POLICY_ORDER:
        sub = ltt[ltt["policy"] == policy]
        if sub.empty:
            continue
        feasible = sub[sub["feasible"]]
        c = cov.get(policy, {})
        knob = f"{feasible['knob_star'].median():.4g}" if len(feasible) else "-"
        err = f"{feasible['test_error_at_star'].mean():.3f}" if len(feasible) else "-"
        cost = f"{feasible['test_cost_at_star'].mean():.2f}" if len(feasible) else "-"
        ivs = f"{feasible['test_interventions_at_star'].mean():.2f}" if len(feasible) else "-"
        coverage = c.get("guarantee_coverage")
        cov_txt = "-" if coverage is None else f"{coverage:.2f}"
        lines.append(
            f"| {policy} | {len(feasible)}/{len(sub)} | {sub['min_calib_risk'].min():.3f} | "
            f"{knob} | {err} | {cost} | {ivs} | {cov_txt} |"
        )
    return lines


def grid_table(suite_dir: Path) -> list[str]:
    """Error and cost along each policy's knob grid (test split)."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    curves = aggregate_curves(manifest, split="test")
    lines = ["| policy | knob | error | cost | interventions |", "|---|---|---|---|---|"]
    for policy in POLICY_ORDER:
        sub = curves[(curves["policy"] == policy) & (curves["label"] == "grid")]
        for _, r in sub.sort_values("knob").iterrows():
            knob = "-" if pd.isna(r["knob"]) else f"{r['knob']:.4g}"
            lines.append(
                f"| {policy} | {knob} | {r['final_error']:.3f} | {r['total_cost']:.2f} | "
                f"{r['interventions_total']:.2f} |"
            )
    return lines


def off_track_share(trajectories: list[Trajectory], task: Any = None) -> str:
    """Later steps that started off track, under both definitions.

    Strict: the previous committed answer is not the exact gold answer (the step's
    global label). Local truth: the step has no local truth under the task's own
    rule (``has_local_truth``, containment when ``local_match="contains"``), which is
    the population the local-error labels and AUROCs use. The two differ by the
    verbose-but-right previous answers.
    """
    later = [
        (prev.globally_correct, cur, t)
        for t in trajectories
        for prev, cur in zip(t.steps, t.steps[1:], strict=False)
    ]
    if not later:
        return "no multi-step trajectories"
    strict = sum(1 for ok, _, _ in later if ok is False)
    text = (
        f"{strict}/{len(later)} later steps started from a previous answer that is not the "
        f"exact gold ({strict / len(later):.2f})"
    )
    if task is not None and getattr(task, "has_local_truth", None) is not None:
        no_truth = sum(1 for _, cur, t in later if not _on_track(task, t, cur))
        rule = getattr(task, "local_match", "exact")
        text += (
            f"; {no_truth}/{len(later)} have no local truth under the task's `{rule}` rule "
            f"({no_truth / len(later):.2f})"
        )
    return text


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--alpha-extra",
        type=float,
        default=None,
        help="also run Learn-then-Test at this second target, looser or stricter (reading only)",
    )
    parser.add_argument(
        "--paired-with",
        type=Path,
        default=None,
        help="another suite on the same test questions: paired signal comparison",
    )
    args = parser.parse_args()
    suite_dir: Path = args.suite_dir
    label = args.label or suite_dir.name
    out = [f"### {label}", ""]
    if not (suite_dir / "manifest.csv").exists():  # a single run directory (policy none)
        trajectories = read_trajectories(suite_dir / "trajectories.parquet")
        task = _task_of(suite_dir / "config.yaml")
        out += [
            *signal_table(trajectories, "single run", task),
            "",
            off_track_share(trajectories, task),
        ]
        out += unanimity_floor(trajectories)
        print("\n".join(out))
        return
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    none_test = manifest[(manifest["policy"] == "none") & (manifest["split"] == "test")]
    none_calib = manifest[(manifest["policy"] == "none") & (manifest["split"] == "calib")]
    trajectories = []
    for path in none_test["path"]:
        trajectories.extend(read_trajectories(_resolve(suite_dir, str(path))))
    if trajectories:
        backend = none_test["backend"].iloc[0]
        suite_yaml = suite_dir / "suite.yaml"
        task = _task_of(suite_yaml, key="base") if suite_yaml.exists() else None
        out += signal_table(trajectories, f"policy none, test split ({backend})", task)
        out += ["", off_track_share(trajectories, task), *unanimity_floor(trajectories)]
        if task is not None:
            import yaml

            block = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))["base"]["task"]
            out += final_errors(trajectories, task)
            rule, rendered = block.get("local_match", "exact"), block.get("render_relations", "n/a")
            out.append(
                f"Labels: per-hop rule `{rule}`, final answer rule "
                f"`{block.get('final_match', 'exact')}` (with aliases); "
                f"sub-questions rendered: {rendered}."
            )
        both = list(trajectories)
        for path in none_calib["path"]:
            both.extend(read_trajectories(_resolve(suite_dir, str(path))))
        out += ["", *signal_by_level(both, task, int(trajectories[0].K))]
        levels = sorted(
            {round(s.eps, 6) for t in trajectories for s in t.steps if s.source == "llm"}
        )
        shown = ", ".join(f"{v:.3f}" for v in levels[:12]) + (", ..." if len(levels) > 12 else "")
        out.append(
            f"Distinct calibrated eps values on the test split (policy none, {len(levels)} "
            f"values): {shown}."
        )
        out.append("")
    out += [*isotonic_levels(suite_dir), *ltt_table(suite_dir), ""]
    if args.alpha_extra is not None:
        out += [*ltt_at_alpha(suite_dir, args.alpha_extra), ""]
    out += [*grid_table(suite_dir), "", *operating_points(suite_dir), ""]
    out += wrong_breakdown(suite_dir)
    if args.paired_with is not None:
        out += ["", *paired_signal(suite_dir, args.paired_with)]
    timing = suite_dir / "timing.txt"
    if timing.exists():
        text = timing.read_text(encoding="utf-8").replace("\n", "; ").strip("; ")
        out += ["", "Timing: " + text]
    print("\n".join(out))


if __name__ == "__main__":
    main()
