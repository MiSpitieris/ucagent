"""An agreement-only gate that can split ties, evaluated from the recorded runs (after the fact).

Usage::

    uv run python scripts/randomized_gate.py                 # every two-hop benchmark and target
    uv run python scripts/randomized_gate.py --targets 0.1   # one target
    uv run python scripts/randomized_gate.py --suites recorded --seeds 100

The entropy gate is a deterministic threshold on one score, and every unanimous batch has the
same score. So it has no operating point between checking every step whose samples disagree and
checking every step, and when the first leaves too many calibration errors it has to check
everything. This script evaluates the alternative that the preprint names and did not run:

    check a step whose samples disagree; check a unanimous step with probability ``1 - q``.

It needs no model call and no new run. On a two-hop chain a check-or-proceed rule can meet only
two prompts at the second hop: the one after the model's own first answer and the one after the
reviewer's. The uncontrolled run holds the first and the run that checks every step the second,
each with its samples, so the outcome of any such rule on any question can be read from the
records. Before anything is reported the script reproduces every recorded random-asking run this
way, question by question (final label and number of checks), and stops if one differs.

The knob ``q`` runs over the grid of random asking and is selected by the same Learn-then-Test
walk as every controller, from the 200 calibration questions; the cost is read on the test
questions. The coins are those of random asking (a draw per question and step from the seed), so
at seed 0 the rule and random asking flip the same coins. Seeds 0 to ``--seeds - 1`` change the
coins only: questions, model replies and labels stay as recorded.

What this is not: a controller of the package, a pre-registered run, or a result on chains of
more than two steps (FinQA is left out). The rule was proposed by the preprint and first
computed by an independent review of 2026-11-18 (``docs/reviews/second_pass_2026-11-18``); this
script repeats that computation through the task's own label rule.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_benchmarks as pb
import preprint_results as pr
import real_results as rr
from ucagent.core.control.ltt import calibrate_knob
from ucagent.core.utils import stable_hash
from ucagent.experiments.runner import build_task
from ucagent.experiments.suite import SuiteConfig, configured_knobs

AGREEMENT = "entropy"  # the suite read: its uncontrolled and random-asking runs, its clusters
STATED = "confagree"  # the reference gate that reads the stated confidence
RANDOM = "random_ask"
QUANTILES = (0, 5, 50, 95, 100)


@dataclass(frozen=True)
class Tree:
    """What a check-or-proceed rule can meet on the questions of one split of a two-hop task."""

    ids: tuple[str, ...]
    #: one cluster at hop 1; at hop 2 after an unchecked hop 1; at hop 2 after a checked hop 1
    unanimous: np.ndarray
    #: the final answer is wrong, by [hop 1 checked][hop 2 checked]
    wrong: np.ndarray


def coins(ids: Sequence[str], seed: int) -> np.ndarray:
    """The uniform draws of random asking for the two hops of each question (first attempt)."""
    return np.array(
        [
            [random.Random(stable_hash(seed, RANDOM, iid, step, 0)).random() for step in (1, 2)]
            for iid in ids
        ]
    )


def decide(tree: Tree, q: float, draws: np.ndarray, ties: bool) -> tuple[np.ndarray, np.ndarray]:
    """Final errors and checks per question under one knob.

    A step is checked when its draw is at least ``q``, as random asking does. With ``ties`` a
    step whose samples disagree is checked whatever its draw, so the coin decides the unanimous
    steps only. The second hop is the one that follows what was done at the first.
    """
    rows = np.arange(len(tree.ids))
    first = draws[:, 0] >= q
    if ties:
        first = first | ~tree.unanimous[:, 0]
    second = draws[:, 1] >= q
    if ties:
        second = second | ~tree.unanimous[rows, 1 + first.astype(int)]
    return tree.wrong[rows, first.astype(int), second.astype(int)], first.astype(int) + second


def select(
    calib: Tree, grid: Sequence[float], seed: int, alpha: float, delta: float, ties: bool = True
) -> float | None:
    """The Learn-then-Test knob of the rule on the calibration questions (None: none passes)."""
    draws = coins(calib.ids, seed)
    points = [(q, len(calib.ids), int(decide(calib, q, draws, ties)[0].sum())) for q in grid]
    return calibrate_knob("randomized_gate", points, alpha=alpha, delta=delta, grid=grid).knob_star


def _run(suite_dir: Path, policy: str, split: str, knob: float | None) -> pd.DataFrame:
    grid = pd.read_csv(suite_dir / "manifest.csv")
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split) & (grid["label"] == "grid")]
    if knob is not None:
        rows = rows[(rows["knob"] - knob).abs() < 1e-9]
    if len(rows) != 1:
        raise SystemExit(f"{suite_dir.name}: {len(rows)} runs of {policy} on {split} at {knob}")
    path = rr._resolve(suite_dir, str(rows["path"].iloc[0]))
    return pd.read_parquet(path).set_index("instance_id").sort_index()


def build_tree(suite_dir: Path, task: Any, split: str, grid: Sequence[float]) -> Tree:
    """Both branches of every question, from the uncontrolled run and the run that checks all."""
    none = _run(suite_dir, "none", split, None)
    every = _run(suite_dir, RANDOM, split, min(grid))
    if not none.index.equals(every.index):
        raise SystemExit(f"{suite_dir.name}: the two runs of {split} hold other questions")
    has_truth = getattr(task, "has_local_truth", None)
    ids = tuple(str(i) for i in none.index)
    unanimous = np.zeros((len(ids), 3), dtype=bool)
    wrong = np.zeros((len(ids), 2, 2), dtype=bool)
    for n, iid in enumerate(ids):
        inst = task.get_instance(iid)
        if inst.n_steps != 2:
            raise SystemExit(f"{suite_dir.name}: {iid} has {inst.n_steps} steps; two are needed")
        steps = [json.loads(frame.at[iid, "steps_json"]) for frame in (none, every)]
        if steps[0][0]["clusters"] != steps[1][0]["clusters"]:
            raise SystemExit(f"{suite_dir.name}: {iid} has another first batch in the two runs")
        unanimous[n] = [len(s["clusters"]) == 1 for s in (steps[0][0], steps[0][1], steps[1][1])]
        for checked_first, records in enumerate(steps):
            hop = records[1]
            context = records[0]["answer"]  # the model's first answer, or the reviewer's
            if hop["context"] != context:
                raise SystemExit(f"{suite_dir.name}: {iid} has an unexpected second prompt")
            unchecked = hop["clusters"][0]["representative"]
            labelled = has_truth is None or has_truth(inst, 2, context)
            # the simulated reviewer returns the local truth, and leaves an off-track step as it is
            checked = task.local_truth(inst, 2, context) if labelled else unchecked
            wrong[n, checked_first] = [
                not task.is_final_correct(inst, answer) for answer in (unchecked, checked)
            ]
    return Tree(ids, unanimous, wrong)


def reproduce(suite_dir: Path, tree: Tree, split: str, grid: Sequence[float]) -> tuple[int, int]:
    """Recorded random-asking runs read from the tree: questions compared, and those that differ."""
    draws = coins(tree.ids, 0)
    compared = differing = 0
    for q in grid:
        recorded = _run(suite_dir, RANDOM, split, q)
        errors, checks = decide(tree, q, draws, ties=False)
        compared += len(tree.ids)
        differing += int(
            (
                (errors != ~recorded["final_correct"].to_numpy(dtype=bool))
                | (checks != recorded["n_ask_human"].to_numpy(dtype=int))
            ).sum()
        )
    return compared, differing


def _cell(sel: pr.Selection) -> str:
    if not sel.feasible or sel.test_error is None or sel.test_cost is None:
        return "not selected"
    return f"{sel.test_error:.3f} at {sel.test_cost:.2f}"


def report(results: Path, bench: pb.Benchmark, targets: Sequence[float], seeds: int) -> list[str]:
    """The rule on one benchmark: the reproduction, then each target."""
    suite_dir = results / bench.pattern.format(AGREEMENT)
    stored = SuiteConfig.model_validate(
        yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    )
    task = build_task(stored.base)
    grid = configured_knobs(suite_dir, RANDOM)
    if grid is None:
        raise SystemExit(f"{suite_dir.name}: random asking has no grid")
    delta = float(stored.base.calibration.delta)
    price = float(stored.base.control.costs.ask_human)
    trees = {split: build_tree(suite_dir, task, split, grid) for split in ("calib", "test")}
    compared = differing = 0
    for split, tree in trees.items():
        c, d = reproduce(suite_dir, tree, split, grid)
        compared, differing = compared + c, differing + d
    out = [
        f"**{bench.label}** (`{suite_dir.name}`)",
        "",
        f"- Recorded random-asking runs read from the two branches: {compared} questions compared "
        f"over {len(grid)} knobs and both splits, {differing} with another final label or "
        "another number of checks.",
    ]
    if differing:
        raise SystemExit("\n".join(out) + "\nThe records do not determine the rule here: stop.")
    n = len(trees["calib"].ids)
    floor = int(decide(trees["calib"], 1.0, coins(trees["calib"].ids, 0), ties=True)[0].sum())
    base = float(_run(suite_dir, "none", "test", None)["total_cost"].mean())
    every_step = base + 2 * price
    out += [
        f"- Calibration questions left wrong when only disagreements are checked: {floor} of {n}.",
        "",
        "| target | allowed | seed 0: q, held-out error at cost | seeds with a selection "
        "| cost over those seeds: min, 5 %, median, 95 %, max | held-out error above target "
        "| below every step | cheaper than random asking | cheaper than the stated-confidence gate "
        "| entropy gate | stated-confidence gate | random asking |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    stated_dir = results / bench.pattern.format(STATED)
    for alpha in targets:
        rows = []
        for seed in range(seeds):
            star = select(trees["calib"], grid, seed, alpha, delta)
            if star is None:
                continue
            errors, checks = decide(trees["test"], star, coins(trees["test"].ids, seed), ties=True)
            rows.append((seed, star, int(errors.sum()), base + price * float(checks.mean())))
        refs = [
            pr.select(suite_dir, pb.GATE, alpha),
            pr.select(stated_dir, pb.GATE, alpha),
            pr.select(suite_dir, RANDOM, alpha),
        ]
        first = next((r for r in rows if r[0] == 0), None)
        seed0 = (
            "not selected"
            if first is None
            else f"{first[1]:g}, {first[2] / len(trees['test'].ids):.3f} at {first[3]:.3f}"
        )
        costs = np.array([r[3] for r in rows])
        quantiles = (
            ", ".join(f"{x:.3f}" for x in np.percentile(costs, QUANTILES)) if len(rows) else "-"
        )
        above = sum(r[2] / len(trees["test"].ids) > alpha + 1e-12 for r in rows)

        def cheaper(ref: pr.Selection, costs: np.ndarray = costs) -> str:
            if not ref.feasible or ref.test_cost is None or not len(costs):
                return "-"
            return f"{int((costs < ref.test_cost - 1e-9).sum())} of {len(costs)}"

        out.append(
            f"| {alpha:g} | {pb.allowance(alpha, n, delta)} | {seed0} | {len(rows)} of {seeds} | "
            f"{quantiles} | {above} | {int((costs < every_step - 1e-9).sum())} of {len(costs)} | "
            f"{cheaper(refs[2])} | {cheaper(refs[1])} | "
            + " | ".join(_cell(r) for r in refs)
            + " |"
        )
    out += ["", f"Every step checked costs {every_step:.2f} here.", ""]
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--suites", choices=sorted(pb.GENERATIONS), default=pb.DEFAULT_GENERATION)
    parser.add_argument("--seeds", type=int, default=100, help="coin seeds 0 to this minus one")
    parser.add_argument(
        "--targets", default=None, help="comma-separated; default: each benchmark's own targets"
    )
    args = parser.parse_args(argv)
    only = None if args.targets is None else [float(x) for x in args.targets.split(",")]
    out = [
        f"### An agreement-only gate with randomised ties ({args.suites} suites, after the fact)",
        "",
        "The rule checks a step whose samples disagree and a unanimous step with probability "
        "1 - q. Costs and errors are read from the recorded runs; one seed of coins is one "
        "pipeline. The three reference columns are the recorded selections at the same target.",
        "",
    ]
    for bench in pb.benchmarks(args.suites):
        if "FinQA" in bench.label:
            continue  # chains of two to four steps: the two recorded branches do not cover them
        targets = [t for t in bench.targets if only is None or t in only]
        if targets:
            out += report(args.results, bench, targets, args.seeds)
    print("\n".join(out))


if __name__ == "__main__":
    main()
