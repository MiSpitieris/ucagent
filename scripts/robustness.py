"""How robust are the certified costs? Dependence among questions and a calibration bootstrap.

Usage::

    uv run python scripts/robustness.py --table docs/figures/preprint/table_robustness.tex

Learn-then-Test assumes independent calibration questions and gives one selection per
calibration draw. This script measures how far the benchmarks are from the first and how much
the second moves the reported costs. For each benchmark it prints:

1. **Dependence.** Questions that share a sub-question (MuSiQue: a constituent single-hop
   question; 2WikiMultiHopQA: the first-hop sub-question, or the first-hop answer with the
   second relation) or a report page (FinQA) form a cluster. Printed: questions that share
   with another question of their split, with the other split and with the fit's questions;
   the correlation of final errors within pairs of a cluster; and the design effect, the
   cluster-robust variance of the calibration error count over the binomial variance.
2. **One question per cluster.** Learn-then-Test on a random question from each cluster, which
   are independent units: the certified test cost over the draws.
3. **Cluster bootstrap.** Clusters of the calibration half resampled with replacement, the
   selection redone, the test cost of the selected knob read from the recorded test run: the
   median and the 5 % and 95 % points of the certified cost, how often nothing is certified,
   and how often a controller is cheaper than random asking (nothing certified counts as
   checking every step). Shares are printed as counts of the draws; the LaTeX table rounds
   them to percentages and writes 0 or 100 only when the count is none or all.
4. **Shared questions on the test half.** The held-out error at the confidence-weighted gate's
   selection on the test questions that share a cluster key with a calibration question, and
   on the others.

A run that a corrected run has replaced as the measurement (``REPLACED``: FinQA with the
adapter as first written) is reported like the others and left out of the LaTeX table.

Everything is read from the run records; no model is called. The draws are seeded.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.control.ltt import calibrate_knob
from ucagent.experiments.suite import configured_knobs
from ucagent.tasks import make_task

GATE = ("confidence-weighted agreement", "per_step_eps")
RANDOM = ("random asking", "random_ask")
ROWS = (
    ("confidence-weighted agreement", "confagree", "per_step_eps"),
    ("confidence-weighted agreement", "confagree", "marginal"),
    ("verbalized confidence", "verbalized", "per_step_eps"),
    ("verbalized confidence", "verbalized", "marginal"),
    ("cluster entropy", "entropy", "per_step_eps"),
    ("cluster entropy", "entropy", "marginal"),
    ("random asking", "confagree", "random_ask"),
)
CONTROLLER = {"per_step_eps": "gate", "marginal": "rule", "random_ask": "-"}
REPORTED = (
    ("MuSiQue", "musique_confirm2d_{}", (0.2, 0.1)),
    ("2WikiMultiHopQA, split 1", "twowiki_{}", (0.1, 0.05)),
    ("2WikiMultiHopQA, split 2", "twowiki_split2_{}", (0.1, 0.05)),
    ("FinQA", "finqa2_{}", (0.1, 0.05)),
)
FIRST_RUNS = (("FinQA, as first run", "finqa_{}", (0.1, 0.05)),)  # in the report, not the table
REPLACED = ("FinQA, as first run",)
#: Prefix of the suite ids of each generation (``scripts/preprint_benchmarks.py``): the replay
#: with the repaired effect model of the rule, and the runs as recorded before it.
GENERATIONS = {"repaired": "fx_", "recorded": ""}
DEFAULT_GENERATION = "repaired"  # the replay of 2026-11-17; "recorded" for the first runs
#: Benchmarks whose dependence is counted without a bootstrap (suites of the recorded
#: generation: the questions are the same in both).
DEPENDENCE_ONLY = (("2WikiMultiHopQA, split 3 (filtered)", "twowiki_split3_{}"),)


def benchmarks(
    generation: str = DEFAULT_GENERATION,
) -> tuple[tuple[str, str, tuple[float, ...]], ...]:
    """The benchmarks of the report in one generation of their suites, then the first runs."""
    prefix = GENERATIONS[generation]
    return (*((label, prefix + pattern, t) for label, pattern, t in REPORTED), *FIRST_RUNS)


BENCHMARKS = benchmarks()
TOLERANCE = 0.005  # two costs printed alike are the same point


def cluster_keys(task_name: str) -> Callable[[Any], list[Hashable]]:
    """What two questions of a benchmark can share."""
    if task_name == "musique":
        return lambda i: list(re.findall(r"\d+", i.qid.split("__", 1)[1]))
    if task_name == "twowiki":
        return lambda i: [i.hops[0].template, (i.hops[0].answer, i.hops[1].template)]
    if task_name == "finqa":
        return lambda i: [i.qid.rsplit("-", 1)[0]]
    raise KeyError(f"no cluster key for task {task_name!r}")


def clusters(keys: dict[str, Sequence[Hashable]]) -> list[list[str]]:
    """Connected components of the questions under shared keys (union-find)."""
    parent = {i: i for i in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[Hashable, str] = {}
    for i, ks in keys.items():
        for k in ks:
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups: dict[str, list[str]] = defaultdict(list)
    for i in keys:
        groups[find(i)].append(i)
    return list(groups.values())


def window_keys(config: Path, keyfn: Callable[[Any], list[Hashable]]) -> set[Hashable]:
    """The keys of the calibration and test questions that a configuration runs."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if data.get("kind") == "suite":
        block = dict(data["base"]["task"])
        splits = [s for s in data.get("splits", ["calib", "test"]) if s != "train"]
    else:
        block, splits = dict(data["task"]), [str(data.get("split", "test"))]
    task = make_task(block)
    return {k for split in splits for i in task.instances(split) for k in keyfn(i)}


def development_overlap(
    keys: dict[str, dict[str, Sequence[Hashable]]],
    keyfn: Callable[[Any], list[Hashable]],
    configs: Sequence[Path],
) -> str:
    """Confirmatory questions that share a key with a question of an earlier run.

    ``configs`` are the configurations of the earlier runs of the benchmark; their
    calibration and test questions were answered, and in development read, before the
    confirmatory questions were run.
    """
    earlier: set[Hashable] = set()
    for config in configs:
        earlier |= window_keys(config, keyfn)
    shared = {
        split: sum(1 for ks in keys[split].values() if any(k in earlier for k in ks))
        for split in ("calib", "test")
    }
    return (
        f"- Earlier runs ({len(configs)} configurations): {shared['calib']} of the "
        f"{len(keys['calib'])} calibration questions and {shared['test']} of the "
        f"{len(keys['test'])} test questions share a key with a calibration or test question "
        "of an earlier run."
    )


def dependence_lines(
    keys: dict[str, dict[str, Sequence[Hashable]]], groups: list[list[str]]
) -> list[str]:
    """Two lines: sharing within the calibration half, and across the halves and the fit."""
    flat = {split: {k for ks in keys[split].values() for k in ks} for split in keys}

    def shares(a: str, b: str) -> int:
        return sum(1 for ks in keys[a].values() if any(k in flat[b] for k in ks))

    multi = [g for g in groups if len(g) > 1]
    return [
        f"- Calibration half: {len(keys['calib'])} questions in {len(groups)} clusters; "
        f"{sum(len(g) for g in multi)} questions share a key with another calibration "
        f"question ({len(multi)} clusters of more than one, the largest of "
        f"{max(len(g) for g in groups)}).",
        f"- {shares('calib', 'test')} calibration questions share a key with a test "
        f"question, and {shares('test', 'calib')} test questions with a calibration "
        f"question; {shares('calib', 'train')} calibration and "
        f"{shares('test', 'train')} test questions share one with the "
        f"{len(keys['train'])} questions of the fit.",
    ]


def design_effect(errors: dict[str, float], groups: list[list[str]]) -> tuple[float, float]:
    """Cluster-robust over binomial variance of the error count; error correlation in pairs."""
    p = float(np.mean(list(errors.values())))
    binomial = len(errors) * p * (1 - p)
    robust = sum(sum(errors[i] - p for i in g) ** 2 for g in groups)
    pairs = [(errors[a], errors[b]) for g in groups for x, a in enumerate(g) for b in g[x + 1 :]]
    rho = float("nan")
    if len(pairs) > 5:
        a, b = np.array(pairs).T
        if a.std() > 0 and b.std() > 0:
            rho = float(np.corrcoef(a, b)[0, 1])
    return (robust / binomial if binomial else float("nan")), rho


def _errors(suite_dir: Path, row: Any) -> dict[str, float]:
    path = Path(str(row.path))
    path = path if path.is_absolute() else suite_dir.parent.parent / path
    frame = pd.read_parquet(path, columns=["instance_id", "final_correct"])
    pairs = zip(frame["instance_id"], frame["final_correct"], strict=True)
    return {i: float(not ok) for i, ok in pairs}


def _row(suite_dir: Path, policy: str, split: str, knob: float | None) -> Any:
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split)]
    rows = rows[rows["knob"].isna()] if knob is None else rows[(rows["knob"] - knob).abs() < 1e-9]
    return rows.iloc[0]


def load(
    suite_dir: Path, policy: str, ids: list[str]
) -> tuple[dict[float, np.ndarray], dict[float, tuple[float, float]]]:
    """Calibration errors per knob as arrays over ``ids``; test cost and error per knob."""
    grid = pr._grid(suite_dir)
    sub = grid[grid["policy"] == policy].dropna(subset=["knob"])
    errors = {}
    for row in sub[sub["split"] == "calib"].sort_values("knob").itertuples():
        e = _errors(suite_dir, row)
        errors[float(row.knob)] = np.array([e[i] for i in ids])
    test = sub[sub["split"] == "test"]
    triples = zip(test["knob"], test["total_cost"], test["final_error"], strict=True)
    return errors, {float(k): (float(c), float(e)) for k, c, e in triples}


def select(
    errors: dict[float, np.ndarray],
    idx: np.ndarray,
    alpha: float,
    delta: float,
    grid: list[float] | None = None,
) -> float | None:
    """The Learn-then-Test knob on the calibration questions ``idx`` (None: not certified).

    ``grid`` is the suite's configured grid: a knob without a calibration run ends the walk.
    """
    points = [(k, len(idx), int(e[idx].sum())) for k, e in errors.items()]
    return calibrate_knob("x", points, alpha=alpha, delta=delta, grid=grid).knob_star


def _cost(point: dict[float, tuple[float, float]], knob: float | None) -> float:
    return point[knob][0] if knob is not None else float("nan")


def _q(values: np.ndarray) -> str:
    if np.all(np.isnan(values)):
        return "never certified"
    lo, mid, hi = np.nanpercentile(values, [5, 50, 95])
    return f"{mid:.2f} [{lo:.2f}, {hi:.2f}]"


def count(mask: np.ndarray) -> str:
    """A share of the draws as a count, so that 399 of 400 is not printed as all of them."""
    return f"{int(np.sum(mask))}/{len(mask)}"


def percent(mask: np.ndarray) -> str:
    """A share of the draws as a LaTeX percentage that reads 0 or 100 only when it is exact."""
    hits, n = int(np.sum(mask)), len(mask)
    text = f"{100 * hits / n:.0f}"
    if text == "0" and hits > 0:
        text = "$<$1"
    elif text == "100" and hits < n:
        text = "$>$99"
    return text + "\\,\\%"


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", type=Path, default=None, help="write the LaTeX table body")
    parser.add_argument(
        "--suites",
        choices=sorted(GENERATIONS),
        default=DEFAULT_GENERATION,
        help="the generation of the suites: after the repair of the effect model, or before",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="where the configurations of the earlier MuSiQue runs are",
    )
    args = parser.parse_args()
    out: list[str] = ["### Robustness of the certified costs", ""]
    tex: list[str] = [
        "% written by scripts/robustness.py; do not edit by hand",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{conf.-weighted gate} & \multicolumn{2}{c}{random asking} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"benchmark & target & reported & bootstrap & none & reported & bootstrap "
        r"& gate cheaper \\",
        r"\midrule",
    ]
    for label, pattern, targets in benchmarks(args.suites):
        rng = np.random.default_rng(args.seed)
        first = args.results / pattern.format("confagree")
        base = pr._base(first)
        delta = float(base.get("calibration", {}).get("delta", 0.1))
        task = make_task(base["task"])
        keyfn = cluster_keys(str(base["task"]["name"]))
        keys = {
            split: {i.instance_id: keyfn(i) for i in task.instances(split)}
            for split in ("train", "calib", "test")
        }
        groups = clusters(keys["calib"])
        flat = {split: {k for ks in keys[split].values() for k in ks} for split in keys}

        def shares(a: str, b: str, keys: dict = keys, flat: dict = flat) -> set[str]:
            return {i for i, ks in keys[a].items() if any(k in flat[b] for k in ks)}

        out += [f"**{label}** (`{pattern.format('*')}`)", "", *dependence_lines(keys, groups)]
        if label == "MuSiQue":
            earlier = sorted(
                c
                for c in args.configs.glob("musique_*.yaml")
                if not c.name.startswith("musique_confirm2")
            )
            out.append(development_overlap(keys, keyfn, earlier))
        ids = sorted(keys["calib"])
        everyone = np.arange(len(ids))
        position = {i: n for n, i in enumerate(ids)}
        index = [np.array([position[i] for i in g]) for g in groups]
        thin = [np.array([rng.choice(g) for g in index]) for _ in range(args.resamples)]
        boot = [
            np.concatenate([index[j] for j in rng.integers(0, len(index), len(index))])
            for _ in range(args.resamples)
        ]
        data = {
            (est, policy): load(args.results / pattern.format(suffix), policy, ids)
            for est, suffix, policy in ROWS
        }
        grids = {
            (est, policy): configured_knobs(args.results / pattern.format(suffix), policy)
            for est, suffix, policy in ROWS
        }
        every = max(cost for cost, _ in data[RANDOM][1].values())

        # design effect at the uncontrolled run and at the reported selections
        effects = [("no checks", _errors(first, _row(first, "none", "calib", None)))]
        for alpha in targets:
            for key in (GATE, RANDOM):
                knob = select(data[key][0], everyone, alpha, delta, grids[key])
                if knob is not None:
                    name = f"{key[0]} {CONTROLLER[key[1]]}, certified at {alpha:g}"
                    effects.append((name, dict(zip(ids, data[key][0][knob], strict=True))))
        out.append("- Design effect of the calibration error count (1: independent questions):")
        for name, errs in effects:
            de, rho = design_effect(errs, groups)
            out.append(
                f"  - {name}: {int(sum(errs.values()))} wrong; design effect {de:.2f}; "
                f"error correlation within pairs of a cluster {rho:.2f}"
            )
        out.append("")

        for alpha in targets:
            rows = []
            for est, _, policy in ROWS:
                errors, point = data[(est, policy)]
                grid = grids[(est, policy)]
                full = select(errors, everyone, alpha, delta, grid)
                thin_knobs = [select(errors, t, alpha, delta, grid) for t in thin]
                b_cost = np.array(
                    [_cost(point, select(errors, b, alpha, delta, grid)) for b in boot]
                )
                rows.append(
                    {
                        "key": (est, policy),
                        "reported": "not certified"
                        if full is None
                        else f"{point[full][1]:.3f} at {point[full][0]:.2f}",
                        "thin": np.array([_cost(point, k) for k in thin_knobs]),
                        "same": float(np.mean([k == full for k in thin_knobs])),
                        "boot": b_cost,
                        "filled": np.nan_to_num(b_cost, nan=every),
                    }
                )
            by_key = {r["key"]: r for r in rows}
            random = by_key[RANDOM]["filled"]
            out += [
                f"Target {alpha:g} ({args.resamples} draws; test costs as recorded; every step "
                f"checked costs {every:.2f}):",
                "",
                "| estimator | controller | reported | one question per cluster: median "
                "[5 %, 95 %] | not certified | same knob as reported | cluster bootstrap: "
                "median [5 %, 95 %] | not certified | cheaper than random asking "
                "| below every step |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for r in rows:
                est, policy = r["key"]
                cheaper = "-" if policy == "random_ask" else count(random - r["filled"] > TOLERANCE)
                out.append(
                    f"| {est} | {CONTROLLER[policy]} | {r['reported']} | {_q(r['thin'])} "
                    f"| {count(np.isnan(r['thin']))} "
                    f"| {r['same']:.2f} | {_q(r['boot'])} | {count(np.isnan(r['boot']))} "
                    f"| {cheaper} | {count(r['filled'] < every - TOLERANCE)} |"
                )
            out.append("")
            gate, rand = by_key[GATE], by_key[RANDOM]
            if label in REPLACED:
                continue  # a run that a corrected run has replaced: reported, not tabulated
            tex.append(
                f"{label} & {alpha:g} & {gate['reported'].split(' at ')[-1]} & {_q(gate['boot'])} "
                f"& {percent(np.isnan(gate['boot']))} "
                f"& {rand['reported'].split(' at ')[-1]} & {_q(rand['boot'])} "
                f"& {percent(random - gate['filled'] > TOLERANCE)} \\\\"
            )

        # held-out error on the test questions that share a key with calibration
        strict = targets[-1]
        knob = select(data[GATE][0], everyone, strict, delta, grids[GATE])
        shared = shares("test", "calib")
        if knob is not None and shared:
            errs = _errors(first, _row(first, "per_step_eps", "test", knob))
            a = [e for i, e in errs.items() if i in shared]
            b = [e for i, e in errs.items() if i not in shared]
            out += [
                f"At the confidence-weighted gate's selection for {strict:g}, the held-out error "
                f"is {int(sum(a))} of {len(a)} ({np.mean(a):.3f}) on the test questions that "
                f"share a key with a calibration question and {int(sum(b))} of {len(b)} "
                f"({np.mean(b):.3f}) on the others.",
                "",
            ]
    for label, pattern in DEPENDENCE_ONLY:
        base = pr._base(args.results / pattern.format("confagree"))
        task = make_task(base["task"])
        keyfn = cluster_keys(str(base["task"]["name"]))
        keys = {
            split: {i.instance_id: keyfn(i) for i in task.instances(split)}
            for split in ("train", "calib", "test")
        }
        out += [
            f"**{label}** (`{pattern.format('*')}`; dependence only)",
            "",
            *dependence_lines(keys, clusters(keys["calib"])),
            "",
        ]
    print("\n".join(out))
    if args.table is not None:
        tex += [r"\bottomrule", r"\end{tabular}"]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(f"\nwrote {args.table}")


if __name__ == "__main__":
    main()
