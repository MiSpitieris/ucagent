"""A replayed suite against the recorded one: which controllers' runs are the same (no model calls).

Usage::

    uv run python scripts/replay_check.py \\
        results/musique_confirm2d_confagree results/fx_musique_confirm2d_confagree \\
        --changed marginal budgeted --targets 0.2,0.1

A suite that is replayed from the response cache after a change of code has to reproduce every
controller the change does not touch. For each controller of the recorded suite that is not
named in ``--changed``, this compares, at every knob and on every split:

- the manifest's error, cost and action counts;
- every question's final answer, its label, its cost, its committed steps (answer, score,
  estimated error, source and batch size) and its decisions (step, round, action and target);
- the Learn-then-Test selection at each target, as ``scripts/preprint_results.py`` makes it.

It also compares the parts of the calibration artifact that a change of the effect model
leaves alone: the two step maps, the future factor, the position means, the conformal scores
and the influence. Prints one line per controller, then the verdict, and exits with status 1
when anything differs. Reading every run of a suite takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
import real_results as rr
from ucagent.core.records import Trajectory
from ucagent.experiments.io import read_trajectories

MANIFEST_COLUMNS = (
    "n",
    "final_error",
    "total_cost",
    "interventions_total",
    "interventions_resample",
    "interventions_backtrack",
    "interventions_ask_human",
    "interventions_escalate",
)
ARTIFACT_PARTS = (
    "isotonic_K",
    "isotonic_K_prime",
    "future_factor",
    "eps_mean_by_position",
    "conformal_scores",
    "influence",
)
TOLERANCE = 1e-12


def signature(trajectory: Trajectory) -> tuple[Any, ...]:
    """What a question's run decided and produced, without ids, times or the samples."""
    steps = tuple(
        (s.step, s.answer, round(s.u, 12), round(s.eps, 12), s.source, s.n_samples)
        for s in trajectory.steps
    )
    decisions = tuple(
        (d.step, d.attempt, d.action.value, d.target_step) for d in trajectory.interventions
    )
    return (
        trajectory.final_answer,
        trajectory.final_correct,
        trajectory.escalated,
        round(trajectory.total_cost, 12),
        trajectory.llm_calls,
        steps,
        decisions,
    )


def differing(old: Sequence[Trajectory], new: Sequence[Trajectory]) -> list[str]:
    """The questions whose runs differ between two readings of one run, by instance id."""
    a = {t.instance_id: signature(t) for t in old}
    b = {t.instance_id: signature(t) for t in new}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def same_number(x: Any, y: Any) -> bool:
    """Two manifest entries are the same number, or both missing."""
    if pd.isna(x) and pd.isna(y):
        return True
    if pd.isna(x) or pd.isna(y):
        return False
    return abs(float(x) - float(y)) <= TOLERANCE


def match_row(rows: pd.DataFrame, knob: Any, split: str) -> pd.DataFrame:
    """The rows of one controller at a knob (missing for a controller without one) and a split."""
    at_split = rows[rows["split"] == split]
    if pd.isna(knob):
        return at_split[at_split["knob"].isna()]
    return at_split[(at_split["knob"] - float(knob)).abs() <= TOLERANCE]


def compare_policy(
    old_dir: Path, new_dir: Path, policy: str, targets: Sequence[float]
) -> list[str]:
    """Everything that differs for one controller between the two suites."""
    problems: list[str] = []
    old, new = pr._grid(old_dir), pr._grid(new_dir)
    old_rows, new_rows = old[old["policy"] == policy], new[new["policy"] == policy]
    if len(old_rows) != len(new_rows):
        problems.append(f"{len(old_rows)} runs recorded, {len(new_rows)} replayed")
    for _, row in old_rows.iterrows():
        found = match_row(new_rows, row["knob"], str(row["split"]))
        where = f"knob {row['knob']:g} {row['split']}"
        if len(found) != 1:
            problems.append(f"{where}: {len(found)} replayed runs")
            continue
        other = found.iloc[0]
        for column in MANIFEST_COLUMNS:
            if not same_number(row[column], other[column]):
                problems.append(f"{where}: {column} {row[column]} against {other[column]}")
        questions = differing(
            read_trajectories(rr._resolve(old_dir, str(row["path"]))),
            read_trajectories(rr._resolve(new_dir, str(other["path"]))),
        )
        if questions:
            problems.append(f"{where}: {len(questions)} questions differ, first {questions[0]}")
    for alpha in targets:
        a, b = pr.select(old_dir, policy, alpha), pr.select(new_dir, policy, alpha)
        if (a.feasible, a.knob, a.calib_errors, a.test_error, a.test_cost) != (
            b.feasible,
            b.knob,
            b.calib_errors,
            b.test_error,
            b.test_cost,
        ):
            problems.append(f"selection at {alpha:g} differs")
    return problems


def compare_artifacts(old_dir: Path, new_dir: Path) -> tuple[list[str], list[str]]:
    """Artifact parts that had to stay and differ, and the parts that changed as expected."""
    old = json.loads((old_dir / "artifacts" / "seed0.json").read_text(encoding="utf-8"))
    new = json.loads((new_dir / "artifacts" / "seed0.json").read_text(encoding="utf-8"))
    problems = [part for part in ARTIFACT_PARTS if old.get(part) != new.get(part)]
    changed = [
        part
        for part in ("resample_table", "resample_map", "schema_version")
        if old.get(part) != new.get(part)
    ]
    return problems, changed


def main(argv: Sequence[str] | None = None) -> int:
    """CLI; returns the exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("recorded", type=Path)
    parser.add_argument("replayed", type=Path)
    parser.add_argument("--changed", nargs="*", default=["marginal", "budgeted"])
    parser.add_argument("--targets", default="0.2,0.1,0.05,0.02")
    args = parser.parse_args(argv)
    targets = [float(x) for x in args.targets.split(",")]
    policies = sorted(set(pr._grid(args.recorded)["policy"]) - set(args.changed))
    out = [f"### {args.replayed.name} against {args.recorded.name}", ""]
    failed = False
    for policy in policies:
        problems = compare_policy(args.recorded, args.replayed, policy, targets)
        failed = failed or bool(problems)
        out.append(
            f"- {policy}: " + ("the same" if not problems else f"{len(problems)} differences")
        )
        out += [f"  - {p}" for p in problems[:10]]
    problems, changed = compare_artifacts(args.recorded, args.replayed)
    failed = failed or bool(problems)
    out.append(
        "- artifact: "
        + ("the parts that had to stay are the same" if not problems else f"differs in {problems}")
        + f"; changed as expected: {', '.join(changed) or 'nothing'}"
    )
    out += ["", "Verdict: " + ("DIFFERENT" if failed else "the same")]
    print("\n".join(out))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
