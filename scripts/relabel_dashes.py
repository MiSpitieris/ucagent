"""Relabel recorded MuSiQue suites with ``canonical_dashes`` on, at zero calls.

Usage::

    uv run python scripts/relabel_dashes.py results/musique_confirm2_confagree [more suites]

The decisions stay as they were made; only the labels are recomputed. For every run of each
suite, every step label (``is_locally_correct`` on the stored committed answer and context)
and every final label (``is_final_correct`` on the stored final answer) are recomputed twice:
with the task as run (a sanity check that must reproduce the stored labels) and with
``canonical_dashes: true`` (Unicode hyphens read as ASCII ones, and a hyphen also separating
words when labels are matched). Prints the labels that flip, the AUROC of ``u`` for local
error over the labelled on-track test hops of the uncontrolled run, and the Learn-then-Test
selections at 0.2 and 0.1 of every controller before and after.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from ucagent.core.control.ltt import calibrate_knob
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.suite import configured_knobs
from ucagent.tasks import make_task

TARGETS = (0.2, 0.1)


def _tasks(suite: Path) -> tuple[Any, Any, dict]:
    base = yaml.safe_load((suite / "suite.yaml").read_text(encoding="utf-8"))["base"]
    block = dict(base["task"])
    as_run = make_task({**block, "canonical_dashes": bool(block.get("canonical_dashes", False))})
    fixed = make_task({**block, "canonical_dashes": True})
    return as_run, fixed, base.get("calibration", {})


def _labels(task: Any, trajectory: Any) -> tuple[list[bool | None], bool]:
    inst = task.get_instance(trajectory.instance_id)
    steps: list[bool | None] = []
    for s in trajectory.steps:
        if s.source != "llm" or not task.has_local_truth(inst, s.step, s.context):
            steps.append(None)
        else:
            steps.append(task.is_locally_correct(inst, s.step, s.context, s.answer))
    return steps, task.is_final_correct(inst, trajectory.final_answer)


def relabel(suite: Path) -> list[str]:
    """Markdown lines for one suite."""
    as_run, fixed, cal = _tasks(suite)
    manifest = pd.read_csv(suite / "manifest.csv")
    grid = manifest[manifest["label"] == "grid"]
    errors: dict[str, tuple[float, float, int]] = {}  # run path -> (error as run, fixed, n)
    mismatch = flips_local = flips_final = local_total = 0
    for path in grid["path"]:
        run = Path(path)
        if not run.is_absolute():
            run = suite.parent.parent / run
        trajs = read_trajectories(str(run))
        wrong_run = wrong_fixed = 0
        for t in trajs:
            steps_run, final_run = _labels(as_run, t)
            steps_fixed, final_fixed = _labels(fixed, t)
            stored = [s.locally_correct if s.source == "llm" else None for s in t.steps]
            if steps_run != stored or final_run != t.final_correct:
                mismatch += 1
            local_total += sum(x is not None for x in steps_run)
            flips_local += sum(a != b for a, b in zip(steps_run, steps_fixed, strict=True))
            flips_final += final_run != final_fixed
            wrong_run += not final_run
            wrong_fixed += not final_fixed
        errors[str(path)] = (wrong_run / len(trajs), wrong_fixed / len(trajs), len(trajs))
    lines = [
        f"### {suite.name}",
        "",
        f"Sanity check, labels recomputed as run: {mismatch} trajectories differ from the stored "
        "labels (must be 0).",
        f"With canonical dashes: {flips_local} step labels change (of {local_total} labelled "
        f"steps over all runs) and {flips_final} final labels change.",
    ]
    none_test = grid[(grid["policy"] == "none") & (grid["split"] == "test")]["path"].iloc[0]
    run = Path(none_test)
    if not run.is_absolute():
        run = suite.parent.parent / run
    aurocs = []
    for task in (as_run, fixed):
        u, y = [], []
        for t in read_trajectories(str(run)):
            labels, _ = _labels(task, t)
            for s, lab in zip(t.steps, labels, strict=True):
                if lab is not None:
                    u.append(s.u)
                    y.append(int(not lab))
        aurocs.append(roc_auc_score(np.array(y), np.array(u)))
    lines.append(
        f"AUROC of u for local error, uncontrolled test run: {aurocs[0]:.3f} as run, "
        f"{aurocs[1]:.3f} with canonical dashes."
    )
    lines += ["", "| controller | target | as run | with canonical dashes |", "|---|---|---|---|"]
    for policy in sorted(set(grid["policy"]) - {"none"}):
        sub = grid[grid["policy"] == policy]
        calib = sub[sub["split"] == "calib"].dropna(subset=["knob"])
        for alpha in TARGETS:
            cells = []
            for which in (0, 1):
                points = [
                    (float(k), int(n), round(errors[str(p)][which] * int(n)))
                    for k, n, p in zip(calib["knob"], calib["n"], calib["path"], strict=True)
                ]
                res = calibrate_knob(
                    policy,
                    points,
                    alpha=alpha,
                    delta=float(cal.get("delta", 0.1)),
                    pvalue=cal.get("pvalue", "binomial"),
                    grid=configured_knobs(suite, policy),
                )
                test = sub[(sub["split"] == "test") & (sub["knob"] == res.knob_star)]
                if not res.feasible or test.empty:
                    cells.append("--")
                    continue
                err = errors[str(test["path"].iloc[0])][which]
                cost = float(test["total_cost"].iloc[0])
                cells.append(f"{err:.3f} at {cost:.2f} (knob {res.knob_star:.4g})")
            lines.append(f"| {policy} | {alpha:g} | {cells[0]} | {cells[1]} |")
    return [*lines, ""]


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", type=Path)
    args = parser.parse_args()
    out: list[str] = []
    for suite in args.suites:
        out += relabel(suite)
    print("\n".join(out))
    sys.exit(0)


if __name__ == "__main__":
    main()
