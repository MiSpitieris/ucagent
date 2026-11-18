"""Does the task's label rule accept the answers the simulated reviewer writes?

Usage::

    uv run python scripts/reviewer_audit.py results/finqa_confagree --targets 0.1,0.05

A checked step that is on track is committed with the task's local truth, written by the
simulated reviewer. The label rule should accept that answer: as the final answer when the
step is the last one, and otherwise as the previous answer of the next step, which is what
keeps the next step on track and its own check meaningful. This script counts the reviewer's
answers that the rule rejects, and the wrong final answers that contain one.

Runs audited, on the calibration and test splits: random asking at knob 0, which checks every
step, and the Learn-then-Test selections of the calibrated gate and of the
value-of-intervention rule at each target. Read from the run records; no model calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.records import Trajectory
from ucagent.experiments.io import read_trajectories
from ucagent.tasks import make_task


def _run(suite_dir: Path, policy: str, split: str, knob: float) -> Path:
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split)]
    rows = rows[(rows["knob"] - knob).abs() < 1e-9]
    path = Path(str(rows["path"].iloc[0]))
    return path if path.is_absolute() else suite_dir.parent.parent / path


def audit(task: Any, traj: Trajectory) -> tuple[int, int]:
    """Reviewer-written answers of a trajectory, and those the label rule rejects."""
    instance = task.get_instance(traj.instance_id)
    committed = {record.step: record for record in traj.steps}  # the last record of each step
    written = rejected = 0
    previous = ""
    for step in sorted(committed):
        record = committed[step]
        answer = str(record.answer)
        truth = task.local_truth(instance, step, previous)
        if record.source == "human" and answer == truth:
            written += 1
            if step == max(committed):
                accepted = task.is_final_correct(instance, answer)
            else:
                accepted = task.has_local_truth(instance, step + 1, answer)
            rejected += not accepted
        previous = answer
    return written, rejected


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path, nargs="+")
    parser.add_argument("--targets", default="0.1,0.05")
    args = parser.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    lines = [
        "| suite | run | split | wrong / n | reviewer-written answers | rejected by the label "
        "rule | wrong answers with a rejected one |",
        "|---|---|---|---|---|---|---|",
    ]
    for suite_dir in args.suite_dir:
        task = make_task(pr._base(suite_dir)["task"])
        runs: list[tuple[str, str, float]] = [("every step checked", "random_ask", 0.0)]
        for policy, name in (("per_step_eps", "gate"), ("marginal", "rule")):
            for alpha in targets:
                sel = pr.select(suite_dir, policy, alpha)
                if sel.feasible and sel.knob is not None:
                    runs.append((f"{name} certified at {alpha:g}", policy, sel.knob))
        seen: set[tuple[str, float]] = set()
        for label, policy, knob in runs:
            if (policy, knob) in seen:
                continue
            seen.add((policy, knob))
            for split in ("calib", "test"):
                trajs = read_trajectories(str(_run(suite_dir, policy, split, knob)))
                counts = [audit(task, t) for t in trajs]
                wrong = [not t.final_correct for t in trajs]
                lines.append(
                    f"| {suite_dir.name} | {label} ({policy}, knob {knob:g}) | {split} "
                    f"| {sum(wrong)} / {len(trajs)} | {sum(w for w, _ in counts)} "
                    f"| {sum(r for _, r in counts)} "
                    f"| {sum(1 for (_, r), bad in zip(counts, wrong, strict=True) if bad and r)} |"
                )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
