"""Two suites on the same questions: where their uncontrolled runs differ.

Usage::

    uv run python scripts/compare_suites.py \\
        results/musique_confirm2_confagree results/musique_confirm2d_confagree

Prints, for the calibration and the test split, the questions the two suites share, the
questions whose committed answer differs at each step of the uncontrolled run (policy
``none``), the questions whose final label differs, and the wrong answers of each suite. Used
for the rerun with the hyphen fix, where the first suite is the run as first made. Read from
the run records; no model calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.records import Trajectory
from ucagent.experiments.io import read_trajectories


def _uncontrolled(suite_dir: Path, split: str) -> dict[str, Trajectory]:
    grid = pr._grid(suite_dir)
    row = grid[(grid["policy"] == "none") & (grid["split"] == split)].iloc[0]
    path = Path(str(row["path"]))
    path = path if path.is_absolute() else suite_dir.parent.parent / path
    return {t.instance_id: t for t in read_trajectories(str(path))}


def _answers(traj: Trajectory) -> dict[int, str]:
    return {record.step: str(record.answer) for record in traj.steps}  # last record of each step


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    lines = [
        f"Uncontrolled runs of {args.first.name} (first) and {args.second.name} (second):",
        "",
        "| split | questions in common | committed answer differs, by step | final label differs "
        "| wrong, first | wrong, second |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("calib", "test"):
        a, b = _uncontrolled(args.first, split), _uncontrolled(args.second, split)
        common = sorted(set(a) & set(b))
        steps = sorted({s for i in common for s in _answers(a[i])})
        differs = {
            s: sum(1 for i in common if _answers(a[i]).get(s) != _answers(b[i]).get(s))
            for s in steps
        }
        lines.append(
            f"| {split} | {len(common)} | "
            + ", ".join(f"step {s}: {n}" for s, n in differs.items())
            + f" | {sum(1 for i in common if a[i].final_correct != b[i].final_correct)} "
            f"| {sum(1 for i in common if not a[i].final_correct)} "
            f"| {sum(1 for i in common if not b[i].final_correct)} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
