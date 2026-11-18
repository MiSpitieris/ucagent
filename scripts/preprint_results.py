"""Preprint results from finished estimator suites: the estimator table and readings.

Usage::

    uv run python scripts/preprint_results.py \\
        --suite "disagreement=results/musique_confirm2_disagreement" \\
        --suite "cluster entropy=results/musique_confirm2_entropy" \\
        --suite "verbalized confidence=results/musique_confirm2_verbalized" \\
        --suite "confidence-weighted agreement=results/musique_confirm2_confagree" \\
        --table docs/figures/preprint/table_estimators.tex

Prints, in markdown: the signal of each estimator over the labelled on-track test hops and,
within the hops that are unanimous under entropy, the AUROC of the stated confidence; the
Learn-then-Test selection of every controller at both targets with its held-out error, cost
and gap to the random-asking curve at matched cost; the guarantee check over every certified
point; the uncontrolled agent and random asking, which do not depend on the estimator.
Writes the LaTeX body of the estimator table. Every number is computed from the suites'
manifests and trajectories with the helpers of ``scripts/real_results.py``; nothing is
typed in.

It also prints two facts about the batches of the uncontrolled test run. Clusters are the
connected components of the judge's pairwise verdicts, so a cluster can hold two answers that
the judge does not equate, joined through a third: the script counts the batches with such a
pair, and the unanimous hops that are unanimous only through one. And it counts the samples
that state no confidence, which the confidence-weighted score counts as fully confident.

The census (``--census``) gives, beside the number of certified points, how many of them are
distinct: the estimator suites of a benchmark share random asking and the KnowNo-style gate,
and a suite replayed after a repair repeats every controller the repair did not touch.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_results as rr
from ucagent.core.control.ltt import calibrate_knob
from ucagent.core.uq.equivalence import make_judge
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.suite import configured_knobs

TARGETS: tuple[float, ...] = (0.2, 0.1)  # --targets overrides; the table shows the first two
CENSUS_TARGETS = (0.2, 0.1)
#: Suites replayed with the repaired effect model of the rule and the budgeted controller
#: (2026-11-17). The census counts them beside the suites as first run and says how many.
REPLAY_PREFIX = "fx_"
CONTROLLERS = (
    "final_gate",
    "per_step_eps",
    "marginal",
    "knowno",
    "budgeted",
    "random_ask",
    "abstain_ask",
)
STEP_LEVEL = ("per_step_eps", "marginal", "knowno")


@dataclass
class Selection:
    """The Learn-then-Test selection of one controller at one target, with its test point."""

    policy: str
    alpha: float
    feasible: bool
    knob: float | None = None
    calib_errors: int | None = None
    n_calib: int | None = None
    test_error: float | None = None
    test_cost: float | None = None
    interventions: float | None = None


def _base(suite_dir: Path) -> dict:
    return yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))["base"]


def _grid(suite_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    grid = manifest[manifest["label"] == "grid"]
    if "variant" in grid.columns:
        grid = grid[grid["variant"] == "main"]
    if grid["seed"].nunique() != 1:
        raise ValueError(f"{suite_dir.name}: expected one seed, found {sorted(grid['seed'])}")
    return grid


def select(suite_dir: Path, policy: str, alpha: float) -> Selection:
    """Learn-then-Test on the calibration runs (as ``real_results.ltt_at_alpha``), test point."""
    cal = _base(suite_dir).get("calibration", {})
    grid = _grid(suite_dir)
    sub = grid[grid["policy"] == policy]
    calib = sub[sub["split"] == "calib"].dropna(subset=["knob"])
    if calib.empty:
        return Selection(policy, alpha, feasible=False)
    points = [
        (float(k), int(n), round(float(e) * int(n)))
        for k, n, e in zip(calib["knob"], calib["n"], calib["final_error"], strict=True)
    ]
    res = calibrate_knob(
        policy,
        points,
        alpha=alpha,
        delta=float(cal.get("delta", 0.1)),
        pvalue=cal.get("pvalue", "binomial"),
        seed=int(sub["seed"].iloc[0]),
        grid=configured_knobs(suite_dir, policy),
    )
    if res.missing_knobs:
        warnings.warn(
            f"{suite_dir.name}: {policy} has no calibration run at {len(res.missing_knobs)} "
            f"configured knob(s), the first at {min(res.missing_knobs):g}; nothing at or "
            "beyond it can be selected",
            stacklevel=2,
        )
    test = sub[(sub["split"] == "test") & (sub["knob"] == res.knob_star)]
    if not res.feasible or res.knob_star is None or test.empty or res.selected is None:
        return Selection(policy, alpha, feasible=False)
    n_calib = int(calib.loc[calib["knob"] == res.knob_star, "n"].iloc[0])
    return Selection(
        policy,
        alpha,
        feasible=True,
        knob=float(res.knob_star),
        calib_errors=round(float(res.selected.risk_hat) * n_calib),
        n_calib=n_calib,
        test_error=float(test["final_error"].iloc[0]),
        test_cost=float(test["total_cost"].iloc[0]),
        interventions=float(test["interventions_total"].iloc[0]),
    )


def random_curve(suite_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """The random-asking test curve, sorted by cost."""
    grid = _grid(suite_dir)
    sub = grid[(grid["policy"] == "random_ask") & (grid["split"] == "test")]
    sub = sub.sort_values("total_cost")
    return sub["total_cost"].to_numpy(float), sub["final_error"].to_numpy(float)


def gap_to_random(suite_dir: Path, sel: Selection) -> float | None:
    """Random asking's test error interpolated at the selection's cost, minus the selection's."""
    if not sel.feasible or sel.test_cost is None or sel.test_error is None:
        return None
    costs, errors = random_curve(suite_dir)
    if len(costs) < 2 or not costs[0] <= sel.test_cost <= costs[-1]:
        return None
    return float(np.interp(sel.test_cost, costs, errors)) - sel.test_error


def uncontrolled(suite_dir: Path) -> tuple[float, float]:
    """Test error and cost of the agent without control (policy ``none``)."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    row = manifest[(manifest["policy"] == "none") & (manifest["split"] == "test")]
    return float(row["final_error"].iloc[0]), float(row["total_cost"].iloc[0])


def selection_key(suite_dir: Path, sel: Selection) -> tuple[object, ...]:
    """What makes a certified point distinct: its questions, controller, target and outcome.

    Two suites on the same task block (the estimator suites of a benchmark, or a suite and its
    replay) give the same key when a controller reaches the same knob with the same errors
    and cost, as random asking does in every estimator suite.
    """
    task = json.dumps(_base(suite_dir).get("task", {}), sort_keys=True)
    return (
        task,
        sel.policy,
        sel.alpha,
        sel.knob,
        sel.calib_errors,
        sel.n_calib,
        sel.test_error,
        sel.test_cost,
        sel.interventions,
    )


def closure_report(suite_dir: Path, split: str = "test") -> list[str]:
    """Batches of the uncontrolled run whose clusters hold a pair the judge does not equate.

    Also the labelled on-track hops with a single cluster that are unanimous only through such
    a pair, those of them the label calls wrong, and the samples without a stated confidence.
    """
    judge = make_judge(str(_base(suite_dir)["uq"]["equivalence"]))
    task = rr._task_of(suite_dir / "suite.yaml", key="base")
    grid = _grid(suite_dir)
    none = grid[(grid["policy"] == "none") & (grid["split"] == split)]
    batches = joined = hops = hops_joined = wrong = wrong_joined = samples = silent = 0
    for path in none["path"]:
        for trajectory in read_trajectories(rr._resolve(suite_dir, str(path))):
            for step in trajectory.steps:
                if step.source != "llm":
                    continue
                answers = trajectory.samples.get(step.samples_key, [])
                stated = trajectory.sample_confidences.get(step.samples_key) or []
                batches += 1
                samples += len(answers)
                silent += sum(1 for c in stated[: len(answers)] if c is None) + max(
                    0, len(answers) - len(stated)
                )
                loose = any(
                    not judge.same(answers[a], answers[b], None)
                    for cluster in step.clusters
                    for a, b in combinations(cluster.members, 2)
                    if a < len(answers) and b < len(answers)
                )
                joined += loose
                labelled = step.locally_correct is not None and rr._on_track(task, trajectory, step)
                if labelled and len(step.clusters) == 1:
                    hops += 1
                    hops_joined += loose
                    if not step.locally_correct:
                        wrong += 1
                        wrong_joined += loose
    return [
        f"Clusters as connected components (policy none, {split} split): {joined} of {batches} "
        "batches have a cluster with a pair of samples that the judge does not equate. Of the "
        f"{hops} labelled on-track hops with a single cluster, {hops_joined} are unanimous only "
        f"through such a pair; of the {wrong} of them that the label calls wrong, {wrong_joined}.",
        f"Samples without a stated confidence (same run): {silent} of {samples}.",
    ]


def census_split(rows: list[tuple[str, int, int]]) -> str:
    """The census by kind of suite, from rows of (name, certified points, points above).

    A replay repeats the suite it replays: the controllers that the repair did not touch
    reach the same selections, which the count of distinct points takes once.
    """
    kinds = (
        ("as first run", [r for r in rows if not r[0].startswith(REPLAY_PREFIX)]),
        (
            f"replays with the repaired effect model (`{REPLAY_PREFIX}*`)",
            [r for r in rows if r[0].startswith(REPLAY_PREFIX)],
        ),
    )
    return (
        "By kind: "
        + "; ".join(
            f"{len(part)} suites {name}, {sum(r[1] for r in part)} certified points, "
            f"{sum(r[2] for r in part)} above the target"
            for name, part in kinds
        )
        + "."
    )


def census(results_dir: Path) -> list[str]:
    """Certified points and target violations at both targets over every real-model suite.

    A suite counts when it has a manifest, a suite file whose backend is not the mock, and a
    ``ltt.csv`` (it finished and was calibrated); the others are listed as skipped.
    """
    rows, skipped = [], []
    distinct: set[tuple[object, ...]] = set()
    distinct_above: set[tuple[object, ...]] = set()
    for suite_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if not (suite_dir / "suite.yaml").exists() or not (suite_dir / "manifest.csv").exists():
            continue
        backend = str(_base(suite_dir).get("llm", {}).get("backend", "mock"))
        if backend == "mock":
            continue
        if not (suite_dir / "ltt.csv").exists():
            skipped.append(suite_dir.name)
            continue
        policies = sorted(set(_grid(suite_dir)["policy"]) - {"none"})
        sels = [select(suite_dir, policy, alpha) for policy in policies for alpha in CENSUS_TARGETS]
        certified = [x for x in sels if x.feasible]
        above = [x for x in certified if x.test_error is not None and x.test_error > x.alpha]
        distinct.update(selection_key(suite_dir, x) for x in certified)
        distinct_above.update(selection_key(suite_dir, x) for x in above)
        by_target = {a: sum(1 for x in certified if x.alpha == a) for a in CENSUS_TARGETS}
        rows.append((suite_dir.name, backend, len(policies), by_target, len(above)))
    lines = [
        "Guarantee census over every finished real-model suite (Learn-then-Test recomputed at "
        "both targets from the calibration runs):",
        "",
        "| suite | backend | controllers | certified at 0.2 | certified at 0.1 | above target |",
        "|---|---|---|---|---|---|",
    ]
    for name, backend, n_pol, by_target, n_above in rows:
        lines.append(
            f"| {name} | {backend} | {n_pol} | {by_target[0.2]} | {by_target[0.1]} | {n_above} |"
        )
    total = sum(sum(r[3].values()) for r in rows)
    lines += [
        "",
        f"{len(rows)} suites, {total} certified points, "
        f"{sum(r[4] for r in rows)} with held-out error above the target.",
        census_split([(r[0], sum(r[3].values()), r[4]) for r in rows]),
        f"Distinct in questions, controller, target, knob and outcome: {len(distinct)} certified "
        f"points, {len(distinct_above)} above the target.",
    ]
    if skipped:
        lines.append(f"Skipped (not finished or not calibrated): {', '.join(skipped)}.")
    return lines


def _cell(sel: Selection) -> str:
    if not sel.feasible:
        return "--"
    return f"{sel.test_error:.3f} at {sel.test_cost:.2f}"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--census", type=Path, default=None, help="results directory to census")
    parser.add_argument(
        "--targets",
        default="0.2,0.1",
        help="comma-separated targets; the table shows the first two",
    )
    parser.add_argument("--table", type=Path, default=None, help="write the LaTeX table body")
    args = parser.parse_args()
    global TARGETS
    TARGETS = tuple(float(x) for x in args.targets.split(","))
    if len(TARGETS) < 2:
        parser.error("give at least two targets")
    if args.census is not None:
        print("\n".join(census(args.census)))
        if not args.suite:
            return
    if not args.suite:
        parser.error("give --suite at least once, or --census")
    suites = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.suite]
    estimator = {label: _base(path)["uq"]["estimator"] for label, path in suites}
    by_estimator = {estimator[label]: (label, path) for label, path in suites}
    out: list[str] = ["### Preprint results", ""]

    # signal
    frames = {label: rr._hop_frame(path) for label, path in suites}
    auroc = {label: rr._auroc(f["wrong"].astype(bool), f["u"]) for label, f in frames.items()}
    out += [
        "Signal over the labelled on-track test hops (policy none):",
        "",
        "| estimator | hops | wrong | AUROC of u for local error |",
        "|---|---|---|---|",
    ]
    for label, f in frames.items():
        out.append(f"| {label} | {len(f)} | {int(f['wrong'].sum())} | {_num(auroc[label])} |")
    out.append("")
    if "semantic_entropy" in by_estimator:
        ent_label, _ = by_estimator["semantic_entropy"]
        unanimous = frames[ent_label][frames[ent_label]["u"] == 0][["instance_id", "step", "wrong"]]
        out.append(
            f"Hops unanimous under entropy: {len(unanimous)}, of which wrong: "
            f"{int(unanimous['wrong'].sum())}."
        )
        for name in ("verbalized", "confidence_agreement"):
            if name not in by_estimator:
                continue
            label, _ = by_estimator[name]
            joined = unanimous.merge(
                frames[label][["instance_id", "step", "u"]], on=["instance_id", "step"]
            )
            value = rr._auroc(joined["wrong"].astype(bool), joined["u"])
            out.append(
                f"- AUROC within them of the {label} score (on these hops, one minus the mean "
                f"stated confidence): {_num(value)} over {len(joined)} hops."
            )
        out.append("")
        for name in ("confidence_agreement", "verbalized", "disagreement"):
            if name in by_estimator:
                out += [
                    *rr.paired_signal(by_estimator[name][1], by_estimator["semantic_entropy"][1]),
                    "",
                ]
    out += [*closure_report(suites[0][1]), ""]

    # selections, gaps and the guarantee (only the controllers each suite ran)
    present = {label: set(_grid(path)["policy"]) for label, path in suites}
    selections = {
        (label, policy, alpha): select(path, policy, alpha)
        for label, path in suites
        for policy in CONTROLLERS
        if policy in present[label]
        for alpha in TARGETS
    }
    for alpha in TARGETS:
        out += [
            f"Learn-then-Test at alpha = {alpha:g}: held-out error and cost at the selected knob, "
            "and the gap below the random-asking test curve at matched cost:",
            "",
            "| estimator | controller | knob | calib errors | test error | test cost "
            "| interventions | gap below random asking |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label, path in suites:
            for policy in CONTROLLERS:
                if policy not in present[label]:
                    continue
                s = selections[(label, policy, alpha)]
                if not s.feasible:
                    out.append(f"| {label} | {policy} | not certified | - | - | - | - | - |")
                    continue
                gap = gap_to_random(path, s) if policy != "random_ask" else None
                out.append(
                    f"| {label} | {policy} | {s.knob:.4g} | {s.calib_errors}/{s.n_calib} | "
                    f"{s.test_error:.3f} | {s.test_cost:.2f} | {s.interventions:.2f} | "
                    f"{_num(gap)} |"
                )
        out.append("")
    certified = [s for s in selections.values() if s.feasible]
    above = [
        (key, s)
        for key, s in selections.items()
        if s.feasible and s.test_error is not None and s.test_error > s.alpha
    ]
    out += [
        f"Guarantee check: {len(certified)} certified points over {len(suites)} suites, "
        f"the controllers they ran and {len(TARGETS)} targets; held-out error above the "
        f"target at {len(above)} of them.",
        *[f"- above the target: {key} {s}" for key, s in above],
        "",
    ]

    # estimator-independent rows
    base_rows = {label: uncontrolled(path) for label, path in suites}
    with_random = [(label, path) for label, path in suites if "random_ask" in present[label]]
    ask_points = {
        label: tuple(
            (
                selections[(label, "random_ask", a)].test_error,
                selections[(label, "random_ask", a)].test_cost,
            )
            for a in TARGETS
        )
        for label, _ in with_random
    }
    out.append(
        "Uncontrolled test error and cost by suite: "
        + "; ".join(f"{label} {e:.3f} at {c:.2f}" for label, (e, c) in base_rows.items())
        + (" (identical)" if len(set(base_rows.values())) == 1 else " (differ)")
    )
    out.append(
        "Random asking's selections by suite identical: "
        + ("yes" if len(set(ask_points.values())) == 1 else f"no: {ask_points}")
    )
    print("\n".join(out))

    if args.table is not None:
        first = (with_random or suites)[0][0]
        lines = [
            "% written by scripts/preprint_results.py; do not edit by hand",
            r"\begin{tabular}{@{}lccccc@{}}",
            r"\toprule",
            r" & & \multicolumn{2}{c}{target "
            + f"{TARGETS[0]:g}"
            + r"} & \multicolumn{2}{c}{target "
            + f"{TARGETS[1]:g}"
            + r"} \\",
            r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
            r"estimator & AUROC & gate & rule & gate & rule \\",
            r"\midrule",
        ]
        for label, _ in suites:
            cells = [
                _cell(selections[(label, policy, alpha)])
                for alpha in TARGETS[:2]
                for policy in ("per_step_eps", "marginal")
            ]
            lines.append(f"{label} & {_num(auroc[label])} & " + " & ".join(cells) + r" \\")
        r20 = selections[(first, "random_ask", TARGETS[0])]
        r10 = selections[(first, "random_ask", TARGETS[1])]
        err, cost = base_rows[first]
        lines += [
            r"\midrule",
            r"random asking & -- & \multicolumn{2}{c}{"
            + _cell(r20)
            + r"} & \multicolumn{2}{c}{"
            + _cell(r10)
            + r"} \\",
            r"no checks & -- & \multicolumn{4}{c}{" + f"{err:.3f} at {cost:.2f}" + r"} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.table}")


if __name__ == "__main__":
    main()
