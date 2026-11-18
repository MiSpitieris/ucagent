"""Cross-benchmark tables of the preprint: selected cost by estimator, and the error floor.

Usage::

    uv run python scripts/preprint_benchmarks.py \\
        --table docs/figures/preprint/table_benchmarks.tex \\
        --table-rule docs/figures/preprint/table_benchmarks_rule.tex \\
        --table-first-runs docs/figures/preprint/table_benchmarks_first_runs.tex \\
        --table-first-runs-rule docs/figures/preprint/table_benchmarks_first_runs_rule.tex

A benchmark is a set of suites that differ only in the estimator. For each benchmark and each
of its pre-registered targets the script prints the number of calibration errors that
Learn-then-Test allows, the selection of the calibrated gate and that of the
value-of-intervention rule with every estimator (held-out error at cost per question), random
asking's selection, and the cost of checking every step. The two controllers are written to
separate tables: each is selected on its own, and a choice between them after the calibration
runs is covered by neither selection. A summary compares the two cell by cell: which costs less
on the test questions, as the costs are printed, and whether the cheaper one also leaves more
held-out errors. It is given over the cells of the tables and over distinct pairs of
selections, because one selection can serve several targets.

Per benchmark it also prints the *confident-error floor* of the agreement-only gate. Entropy
over answer clusters gives every unanimous step the same calibrated level, so a threshold on it
has no operating point between checking every non-unanimous step and checking every step.
The floor is the smallest number of calibration questions the gate leaves wrong while it
checks fewer than all steps. The gate can certify a target below checking everything only if
the floor is within the target's allowance; the script counts the benchmark-target pairs in
which the selection agrees with that. Last, it tallies the certified points of every
controller and how many have a
held-out error above their target, by target, and prints what puts such points in proportion:
Fisher's exact test for the uncontrolled errors of the calibration and test halves, the
probability that a pipeline whose risk equals the target is certified, and the probability
that such a pipeline shows the observed number of held-out errors.

A run that a corrected run has replaced as the measurement (``FIRST_RUNS``: FinQA with the
adapter as first written) is printed after the benchmarks and written to its own table; the
floor account and the tallies do not count it.

Last, it sets the calibrated gate beside the other controllers at the one target that every
benchmark has (``BASELINE_TARGET``), with confidence-weighted agreement where a controller reads
a score: the selection of each controller as held-out error at cost, with the checks and the
handovers per question, and the agent without checks and with every step checked for reference.
A handover gives the whole case to a person, so a selection is marked as checking every step by
its checks, never by its cost.

Two generations of the suites exist: the *recorded* one, and the *repaired* one (suite ids
``fx_*``), a replay after the effect model of the rule and the budgeted controller was
repaired (2026-11-17). ``--suites`` chooses; ``DEFAULT_GENERATION`` says which one the
tables are made from. The abstention suites are not replayed and are always read from the
recorded generation. When both generations are on disk the report ends with how far the
repair moved the selections of the gate, the rule and the budgeted controller.

Per benchmark the script also prints each estimator's own floor, and over the tables how many
selections sit at the least conservative knob of their grid, where a wider grid might have
selected a cheaper point. Beside the chance that a pipeline at its target is selected it
prints the chance for a pipeline at half its target, which is the power of the test there.

Everything is read from the suites' manifests with the helpers of
``scripts/preprint_results.py``; nothing is typed in, and no model is called.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binom, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.control.ltt import binomial_pvalue

ESTIMATORS = (
    ("confagree", "confidence-weighted agreement"),
    ("verbalized", "verbalized confidence"),
    ("entropy", "cluster entropy"),
    ("disagreement", "disagreement"),
)
AGREEMENT_ONLY = "entropy"  # the suite whose gate defines the floor
TOLERANCE = 0.005  # two costs printed alike are the same point
GATE, RULE = "per_step_eps", "marginal"


@dataclass(frozen=True)
class Benchmark:
    """Suites ``pattern.format(<estimator suffix>)`` under the results directory."""

    label: str
    pattern: str
    targets: tuple[float, ...]  # as pre-registered for the benchmark
    recorded: str = ""  # the pattern of the recorded generation (its abstention suite is read)

    @property
    def abstain(self) -> str:
        """The suite of the abstention baseline, which was not replayed."""
        return (self.recorded or self.pattern).format("abstain")


#: Prefix of the suite ids of each generation: the replay with the repaired effect model of
#: the rule and the budgeted controller, and the runs as recorded before it.
GENERATIONS = {"repaired": "fx_", "recorded": ""}
#: The generation the tables are made from: the replay of 2026-11-17. The tables of the
#: preprint's first version (``docs/figures/preprint/``) come from ``--suites recorded``.
DEFAULT_GENERATION = "repaired"


def benchmarks(generation: str = DEFAULT_GENERATION) -> tuple[Benchmark, ...]:
    """The reported benchmarks in one generation of their suites."""
    prefix = GENERATIONS[generation]
    return tuple(
        Benchmark(label, prefix + pattern, targets, recorded=pattern)
        for label, pattern, targets in (
            ("MuSiQue", "musique_confirm2d_{}", (0.2, 0.1)),
            ("2WikiMultiHopQA, split 1", "twowiki_{}", (0.2, 0.1, 0.05, 0.02)),
            ("2WikiMultiHopQA, split 2", "twowiki_split2_{}", (0.2, 0.1, 0.05, 0.02)),
            ("2WikiMultiHopQA, split 3 (filtered)", "twowiki_split3_{}", (0.2, 0.1, 0.05, 0.02)),
            ("FinQA", "finqa2_{}", (0.2, 0.1, 0.05)),
        )
    )


BENCHMARKS = benchmarks()
# Runs that a corrected run has replaced as the measurement. They are printed after the
# benchmarks above and written to their own table; the tallies do not count them.
FIRST_RUNS = (Benchmark("FinQA, as first run", "finqa_{}", (0.2, 0.1, 0.05), recorded="finqa_{}"),)
# The comparison of controllers: the one target of every benchmark (pre-registered on MuSiQue
# and FinQA, reported beside the stricter ones on 2WikiMultiHopQA), on the suites of the
# estimator that the gate reports first. Random asking and the KnowNo-style gate read no score.
BASELINE_TARGET = 0.1
BASELINE_SUFFIX = "confagree"
BASELINES = (
    (GATE, "calibrated gate"),
    (RULE, "value-of-intervention rule"),
    ("budgeted", "budgeted controller"),
    ("knowno", "KnowNo-style gate"),
    ("final_gate", "final gate (hands the case over)"),
    ("random_ask", "random asking"),
)
HANDOVER_ONLY = ("final_gate",)  # controllers that never check a step


@dataclass(frozen=True)
class GridPoint:
    """One knob of a controller on the calibration split."""

    knob: float
    errors: int
    asks: float  # human checks per question


@dataclass(frozen=True)
class Cell:
    """The Learn-then-Test selection of one controller at one target, as its test point."""

    feasible: bool
    error: float = 0.0
    cost: float = 0.0
    all_steps: bool = False  # the selection checks every step
    above: bool = False  # held-out error above the target
    no_anchor: bool = False  # nothing certified, and the grid has no point checking every step


def allowance(alpha: float, n: int, delta: float) -> int:
    """Largest number of calibration errors whose binomial p-value is at most delta (-1: none)."""
    k = -1
    while k + 1 <= n and binomial_pvalue(k + 1, n, alpha) <= delta:
        k += 1
    return k


def floor_point(points: list[GridPoint], all_steps: float) -> GridPoint | None:
    """The point with the fewest errors among those that check fewer than all steps.

    Ties go to the point with fewer checks. None when every point checks every step.
    """
    partial = [p for p in points if p.asks < all_steps - 1e-9]
    return min(partial, key=lambda p: (p.errors, p.asks), default=None)


def grid_points(suite_dir: Path, policy: str = GATE) -> list[GridPoint]:
    """A controller's knob grid on the calibration split (the calibrated gate's by default)."""
    grid = pr._grid(suite_dir)
    sub = grid[(grid["policy"] == policy) & (grid["split"] == "calib")]
    sub = sub.dropna(subset=["knob"])
    return [
        GridPoint(float(k), round(float(e) * int(n)), float(a))
        for k, n, e, a in zip(
            sub["knob"], sub["n"], sub["final_error"], sub["interventions_ask_human"], strict=True
        )
    ]


def every_step(suite_dir: Path, split: str) -> tuple[float, int, float]:
    """Checks per question, wrong answers and cost when every step is checked.

    Read from random asking at knob 0, which asks at every step.
    """
    grid = pr._grid(suite_dir)
    sub = grid[(grid["policy"] == "random_ask") & (grid["split"] == split) & (grid["knob"] == 0)]
    if sub.empty:
        raise ValueError(f"{suite_dir.name}: no random_ask run at knob 0 on the {split} split")
    row = sub.iloc[0]
    return (
        float(row["interventions_ask_human"]),
        round(float(row["final_error"]) * int(row["n"])),
        float(row["total_cost"]),
    )


def uncontrolled_errors(suite_dir: Path, split: str) -> tuple[int, int]:
    """Wrong answers and questions of the agent without control on a split."""
    grid = pr._grid(suite_dir)
    row = grid[(grid["policy"] == "none") & (grid["split"] == split)].iloc[0]
    return round(float(row["final_error"]) * int(row["n"])), int(row["n"])


def _point(sel: pr.Selection, all_cost: float) -> Cell:
    if not sel.feasible or sel.test_error is None or sel.test_cost is None:
        return Cell(False)
    return Cell(
        True,
        sel.test_error,
        sel.test_cost,
        all_steps=sel.test_cost >= all_cost - TOLERANCE,
        above=sel.test_error > sel.alpha + 1e-12,
    )


def checks_every_step(suite_dir: Path, all_asks: float, policy: str = GATE) -> bool:
    """Whether a controller's most conservative grid point checks every step (calibration)."""
    points = grid_points(suite_dir, policy)
    return bool(points) and max(p.asks for p in points) >= all_asks - 1e-9


def controller_cell(
    suite_dir: Path, policy: str, alpha: float, all_cost: float, anchored: bool = True
) -> Cell:
    """One controller's selection at a target; nothing selected is marked when unanchored."""
    cell = _point(pr.select(suite_dir, policy, alpha), all_cost)
    return cell if cell.feasible else Cell(False, no_anchor=not anchored)


def compare(pairs: list[tuple[Cell, Cell]]) -> dict[str, int]:
    """Gate against rule over (gate, rule) cells in which both have a selection.

    Two costs that print alike (two decimals, as in the tables) are a tie, so the counts can
    be checked against the tables. For each side that is cheaper, ``*_more_errors`` counts the
    cells in which it also leaves more held-out errors than the other.
    """
    out = dict.fromkeys(
        ("both", "gate_cheaper", "rule_cheaper", "tie", "gate_more_errors", "rule_more_errors"), 0
    )
    for gate, rule in pairs:
        if not (gate.feasible and rule.feasible):
            continue
        out["both"] += 1
        diff = rule.cost - gate.cost
        if f"{gate.cost:.2f}" == f"{rule.cost:.2f}":
            out["tie"] += 1
        elif diff > 0:
            out["gate_cheaper"] += 1
            out["gate_more_errors"] += gate.error > rule.error + 1e-9
        else:
            out["rule_cheaper"] += 1
            out["rule_more_errors"] += rule.error > gate.error + 1e-9
    return out


def distinct_pairs(rows: list[tuple[str, Cell, Cell]]) -> list[tuple[str, Cell, Cell]]:
    """Rows of (estimator, gate cell, rule cell) with repeated pairs of selections dropped.

    A selection is the same when its held-out error and cost are; the flag that depends on the
    target (held-out error above it) is ignored.
    """
    seen: set[tuple[object, ...]] = set()
    out = []
    for suffix, gate, rule in rows:
        key = (suffix, *((c.feasible, round(c.error, 6), round(c.cost, 6)) for c in (gate, rule)))
        if key not in seen:
            seen.add(key)
            out.append((suffix, gate, rule))
    return out


def compare_md(label: str, counts: dict[str, int]) -> str:
    """One line of the gate-against-rule summary."""
    return (
        f"- {label}: {counts['both']} with both selected; gate cheaper in "
        f"{counts['gate_cheaper']} (more held-out errors than the rule in "
        f"{counts['gate_more_errors']} of them), rule cheaper in {counts['rule_cheaper']} "
        f"(more held-out errors than the gate in {counts['rule_more_errors']} of them), "
        f"same cost in {counts['tie']}."
    )


def md_cell(cell: Cell) -> str:
    """Markdown: error at cost, with the flags spelled out."""
    if not cell.feasible:
        return "not certified" + (" (no grid point checks every step)" if cell.no_anchor else "")
    flags = [
        name for name, on in (("every step", cell.all_steps), ("above target", cell.above)) if on
    ]
    return f"{cell.error:.3f} at {cell.cost:.2f}" + (f" ({', '.join(flags)})" if flags else "")


def tex_cell(cell: Cell) -> str:
    """LaTeX: error at cost; italic when every step is checked, a dagger above the target.

    Nothing certified is a dash, with a g when the grid has no point that checks every step.
    """
    if not cell.feasible:
        return "--$^{g}$" if cell.no_anchor else "--"
    text = f"{cell.error:.3f} at {cell.cost:.2f}"
    if cell.all_steps:
        text = r"\textit{" + text + "}"
    return text + (r"$^{\dagger}$" if cell.above else "")


@dataclass(frozen=True)
class Baseline:
    """One controller's selection at the baseline target, with what it asks of people."""

    cell: Cell
    checks: float = 0.0  # human checks per question on the test questions
    handovers: float = 0.0  # cases handed to a person per question on the test questions


def every_step_by_checks(checks: float, all_checks: float) -> bool:
    """Whether a selection checks every step, judged by its checks so a handover never counts."""
    return checks >= all_checks - 1e-9


def short_label(label: str) -> str:
    """A column head: ``2WikiMultiHopQA, split 3 (filtered)`` becomes ``2Wiki 3``."""
    if label.startswith("2WikiMultiHopQA, split "):
        return "2Wiki " + label.removeprefix("2WikiMultiHopQA, split ").split(" ")[0]
    return label


def baseline_row(label: str, cells: list[str]) -> str:
    """One LaTeX row of the comparison of controllers."""
    return f"{label} & " + " & ".join(cells) + r" \\"


def baseline(
    suite_dir: Path, policy: str, alpha: float, all_checks: float, anchored: bool
) -> Baseline:
    """A controller's selection at ``alpha``, with its test checks and handovers per question."""
    sel = pr.select(suite_dir, policy, alpha)
    if not sel.feasible or sel.test_error is None or sel.test_cost is None:
        return Baseline(Cell(False, no_anchor=not anchored))
    grid = pr._grid(suite_dir)
    row = grid[(grid["policy"] == policy) & (grid["split"] == "test") & (grid["knob"] == sel.knob)]
    checks = float(row["interventions_ask_human"].iloc[0])
    handovers = float(row["interventions_escalate"].iloc[0])
    return Baseline(
        Cell(
            True,
            sel.test_error,
            sel.test_cost,
            all_steps=every_step_by_checks(checks, all_checks),
            above=sel.test_error > alpha + 1e-12,
        ),
        checks,
        handovers,
    )


def baseline_tables(
    results: Path, reported: Sequence[Benchmark] = BENCHMARKS, alpha: float = BASELINE_TARGET
) -> tuple[list[str], list[str]]:
    """The comparison of controllers at ``alpha``: markdown lines and the LaTeX table body."""
    columns: list[tuple[str, dict[str, Baseline], tuple[float, float], tuple[float, float]]] = []
    for bench in reported:
        suite_dir = results / bench.pattern.format(BASELINE_SUFFIX)
        all_checks_cal, _, _ = every_step(suite_dir, "calib")
        all_checks, all_wrong, all_cost = every_step(suite_dir, "test")
        _, n_test = uncontrolled_errors(suite_dir, "test")
        present = set(pr._grid(suite_dir)["policy"])
        sels = {
            policy: baseline(
                suite_dir,
                policy,
                alpha,
                all_checks,
                anchored=policy in HANDOVER_ONLY
                or checks_every_step(suite_dir, all_checks_cal, policy),
            )
            for policy, _ in BASELINES
            if policy in present
        }
        columns.append(
            (bench.label, sels, pr.uncontrolled(suite_dir), (all_wrong / n_test, all_cost))
        )
    md = [
        f"### Controllers at the target {alpha:g} (`*_{BASELINE_SUFFIX}` suites)",
        "",
        "| benchmark | controller | held-out error at cost | checks per question "
        "| handovers per question |",
        "|---|---|---|---|---|",
    ]
    for label, sels, _, _ in columns:
        for policy, name in BASELINES:
            if policy not in sels:
                md.append(f"| {label} | {name} | not run | - | - |")
                continue
            b = sels[policy]
            md.append(
                f"| {label} | {name} | {md_cell(b.cell)} | "
                + (f"{b.checks:.3f} | {b.handovers:.3f} |" if b.cell.feasible else "- | - |")
            )
    md += [
        "",
        "Without checks: "
        + "; ".join(f"{label} {e:.3f} at {c:.2f}" for label, _, (e, c), _ in columns)
        + ". Every step checked: "
        + "; ".join(f"{label} {e:.3f} at {c:.2f}" for label, _, _, (e, c) in columns)
        + ".",
    ]
    tex = [
        "% written by scripts/preprint_benchmarks.py; do not edit by hand",
        r"\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}",
        r"\toprule",
        baseline_row("controller", [short_label(label) for label, _, _, _ in columns]),
        r"\midrule",
    ]
    for policy, name in BASELINES:
        tex.append(
            baseline_row(
                name,
                [
                    tex_cell(sels[policy].cell) if policy in sels else ""
                    for _, sels, _, _ in columns
                ],
            )
        )
    tex += [
        r"\midrule",
        baseline_row("no checks", [f"{e:.3f} at {c:.2f}" for _, _, (e, c), _ in columns]),
        baseline_row("every step checked", [f"{e:.3f} at {c:.2f}" for _, _, _, (e, c) in columns]),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return md, tex


def at_loose_end(suite_dir: Path, policy: str, alpha: float) -> bool | None:
    """Whether a selection is the least conservative knob of its grid; None without one.

    At that knob the walk of Learn-then-Test ran out of grid, so a grid that reached further
    might have selected a cheaper point.
    """
    sel = pr.select(suite_dir, policy, alpha)
    if not sel.feasible or sel.knob is None:
        return None
    return sel.knob >= max(p.knob for p in grid_points(suite_dir, policy)) - 1e-12


@dataclass(frozen=True)
class Moves:
    """How far a change of code moved one controller's selections, over labelled cells."""

    cells: int = 0
    both: int = 0  # a selection before and after
    one: int = 0  # a selection on one side only
    changed: int = 0  # of ``both``: the held-out error or the cost prints differently
    cost: float = 0.0  # the largest move of the cost, as printed
    cost_where: str = ""
    error: float = 0.0  # the largest move of the held-out error, as printed
    error_where: str = ""


def moves(rows: Sequence[tuple[str, Cell, Cell]]) -> Moves:
    """Selections before and after a change, as rows of (where, before, after).

    The moves are those of the numbers as the tables print them (the error to three decimals,
    the cost to two), so that they can be checked against the tables.
    """
    both = one = changed = 0
    cost, error, cost_where, error_where = 0.0, 0.0, "", ""
    for where, before, after in rows:
        if before.feasible != after.feasible:
            one += 1
        if not (before.feasible and after.feasible):
            continue
        both += 1
        d_cost = abs(round(after.cost, 2) - round(before.cost, 2))
        d_error = abs(round(after.error, 3) - round(before.error, 3))
        changed += d_cost > 1e-9 or d_error > 1e-9
        shown = f"{where}: {md_cell(before)} before, {md_cell(after)} after"
        if d_cost > cost + 1e-9:
            cost, cost_where = d_cost, shown
        if d_error > error + 1e-9:
            error, error_where = d_error, shown
    return Moves(len(rows), both, one, changed, cost, cost_where, error, error_where)


def moves_md(name: str, m: Moves) -> str:
    """One line of the comparison of the two generations."""
    line = (
        f"- {name}: {m.cells} cells; a selection in both generations in {m.both}, in one only "
        f"in {m.one}; the selection prints differently in {m.changed} of the {m.both}"
    )
    if m.changed:
        line += (
            f"; largest move of the cost {m.cost:.2f} ({m.cost_where}); largest move of the "
            f"held-out error {m.error:.3f} ({m.error_where})"
        )
    return line + "."


def generation_moves(results: Path) -> list[str]:
    """Recorded against repaired selections of the controllers, when both are on disk."""
    before, after = benchmarks("recorded"), benchmarks("repaired")
    needed = [
        results / b.pattern.format(suffix) / "manifest.csv"
        for b in (*before, *after)
        for suffix, _ in ESTIMATORS
    ]
    if not all(path.exists() for path in needed):
        return []
    out = [
        "",
        "Recorded against repaired suites, over every benchmark, target and estimator (numbers "
        "as printed):",
        "",
    ]
    for policy, name in ((GATE, "gate"), (RULE, "rule"), ("budgeted", "budgeted controller")):
        rows = []
        for old, new in zip(before, after, strict=True):
            for alpha in old.targets:
                for suffix, label in ESTIMATORS:
                    cells = [
                        Cell(True, sel.test_error, sel.test_cost)
                        if sel.feasible and sel.test_error is not None and sel.test_cost is not None
                        else Cell(False)
                        for sel in (
                            pr.select(results / b.pattern.format(suffix), policy, alpha)
                            for b in (old, new)
                        )
                    ]
                    rows.append((f"{old.label}, {label}, {alpha:g}", cells[0], cells[1]))
        out.append(moves_md(name, moves(rows)))
    return out


def selection_chance(alpha: float, n: int, delta: float, risk: float) -> float:
    """The probability that a pipeline of a given risk has at most the allowed errors."""
    return float(binom.cdf(allowance(alpha, n, delta), n, risk))


def tally(bench: Benchmark, results: Path) -> dict[float, tuple[int, list[float]]]:
    """Certified points per target over every suite and controller, and the errors above it."""
    out: dict[float, tuple[int, list[float]]] = {}
    suites = [results / bench.pattern.format(suffix) for suffix, _ in ESTIMATORS]
    suites.append(results / bench.abstain)
    for alpha in bench.targets:
        certified, above = 0, []
        for suite_dir in suites:
            if not (suite_dir / "manifest.csv").exists():
                continue
            present = set(pr._grid(suite_dir)["policy"])
            for policy in pr.CONTROLLERS:
                if policy not in present:
                    continue
                sel = pr.select(suite_dir, policy, alpha)
                if sel.feasible and sel.test_error is not None:
                    certified += 1
                    if sel.test_error > alpha + 1e-12:
                        above.append(sel.test_error)
        out[alpha] = (certified, above)
    return out


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--suites",
        choices=sorted(GENERATIONS),
        default=DEFAULT_GENERATION,
        help="the generation of the suites: after the repair of the effect model, or before",
    )
    parser.add_argument("--table", type=Path, default=None, help="write the gate's LaTeX table")
    parser.add_argument("--table-rule", type=Path, default=None, help="the same for the rule")
    parser.add_argument(
        "--table-first-runs",
        type=Path,
        default=None,
        help="write the gate's table for the runs that a corrected run has replaced",
    )
    parser.add_argument(
        "--table-first-runs-rule", type=Path, default=None, help="the same for the rule"
    )
    parser.add_argument(
        "--table-baselines",
        type=Path,
        default=None,
        help=f"write the comparison of controllers at the target {BASELINE_TARGET:g}",
    )
    args = parser.parse_args()

    reported = benchmarks(args.suites)
    out: list[str] = [f"### Benchmarks ({args.suites} suites)", ""]
    loose: dict[str, list[bool]] = {GATE: [], RULE: []}
    head = [
        "% written by scripts/preprint_benchmarks.py; do not edit by hand",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"target & allowed & conf.-weighted agreement & verbalized & entropy & disagreement "
        r"& random asking \\",
    ]
    # one table per controller: [benchmarks, replaced runs]
    texs: dict[str, list[list[str]]] = {p: [list(head), list(head)] for p in (GATE, RULE)}
    # per benchmark: (estimator, gate cell, rule cell) for every target
    compared: dict[str, list[tuple[str, Cell, Cell]]] = {}
    pairs = agree = 0
    totals: dict[float, list[int]] = {}
    for bench in (*reported, *FIRST_RUNS):
        counted = bench in reported
        if not counted and bench is FIRST_RUNS[0]:
            out += ["#### Runs replaced by a corrected run (not in the tallies)", ""]
        dirs = {suffix: args.results / bench.pattern.format(suffix) for suffix, _ in ESTIMATORS}
        first = dirs[ESTIMATORS[0][0]]
        delta = float(pr._base(first).get("calibration", {}).get("delta", 0.1))
        cal_err, n_cal = uncontrolled_errors(first, "calib")
        test_wrong, n_test = uncontrolled_errors(first, "test")
        halves = fisher_exact([[cal_err, n_cal - cal_err], [test_wrong, n_test - test_wrong]])[1]
        test_err, test_cost = pr.uncontrolled(first)
        all_asks_cal, all_err_cal, _ = every_step(first, "calib")
        _, all_err_test, all_cost = every_step(first, "test")
        floor = floor_point(grid_points(dirs[AGREEMENT_ONLY]), all_asks_cal)
        anchored = {
            policy: {
                suffix: checks_every_step(dirs[suffix], all_asks_cal, policy)
                for suffix, _ in ESTIMATORS
            }
            for policy in (GATE, RULE)
        }
        floor_md = (
            "none (every knob checks every step)"
            if floor is None
            else f"{floor.errors} of {n_cal} at {floor.asks:.2f} checks per question "
            f"(knob {floor.knob:g})"
        )
        out += [
            f"**{bench.label}** (`{bench.pattern.format('*')}`)",
            "",
            f"- Without checks: held-out error {test_err:.3f} at cost {test_cost:.2f}; "
            f"{cal_err} of {n_cal} calibration questions and {test_wrong} of {n_test} test "
            f"questions wrong (Fisher's exact test for the two halves, two-sided p = "
            f"{halves:.2f}).",
            f"- Every step checked: cost {all_cost:.2f}; {all_err_cal} of {n_cal} calibration "
            f"questions and {all_err_test} test questions still wrong.",
            f"- Floor of the agreement-only gate ({AGREEMENT_ONLY}): {floor_md}.",
            "- Floor of each estimator's own gate (the fewest calibration questions it leaves "
            "wrong while it checks fewer than all steps): "
            + "; ".join(
                f"{label} "
                + (
                    "none"
                    if own is None
                    else f"{own.errors} at {own.asks:.2f} checks (knob {own.knob:g})"
                )
                for suffix, label in ESTIMATORS
                for own in [floor_point(grid_points(dirs[suffix]), all_asks_cal)]
            )
            + ".",
            *(
                f"- Estimators whose {name} grid has no point that checks every step, with the "
                "checks per question and calibration errors of their most conservative knob: "
                + (
                    "; ".join(
                        f"{label} ({top.asks:.2f} checks, {top.errors} errors)"
                        for suffix, label in ESTIMATORS
                        if not anchored[policy][suffix]
                        for top in [max(grid_points(dirs[suffix], policy), key=lambda p: p.asks)]
                    )
                    or "none"
                )
                + "."
                for policy, name in ((GATE, "gate"), (RULE, "rule"))
            ),
            "",
            "| target | allowed errors | floor within | controller | "
            + " | ".join(label for _, label in ESTIMATORS)
            + " | random asking |",
            "|---|---|---|---|" + "---|" * len(ESTIMATORS) + "---|",
        ]
        floor_tex = (
            "no floor" if floor is None else f"floor {floor.errors} at {floor.asks:.2f} checks"
        )
        for policy in (GATE, RULE):
            texs[policy][0 if counted else 1] += [
                r"\midrule",
                r"\multicolumn{7}{@{}l}{\textit{"
                + bench.label
                + f"}}: no checks {test_err:.3f} at {test_cost:.2f}; {floor_tex}; "
                + f"every step checked {all_cost:.2f}"
                + r"} \\",
            ]
        for alpha in bench.targets:
            allowed = allowance(alpha, n_cal, delta)
            cells = {
                policy: [
                    controller_cell(dirs[suffix], policy, alpha, all_cost, anchored[policy][suffix])
                    for suffix, _ in ESTIMATORS
                ]
                for policy in (GATE, RULE)
            }
            if counted:
                for policy in (GATE, RULE):
                    for suffix, _ in ESTIMATORS:
                        end = at_loose_end(dirs[suffix], policy, alpha)
                        if end is not None:
                            loose[policy].append(end)
                compared.setdefault(bench.label, []).extend(
                    (suffix, gate_cell, rule_cell)
                    for (suffix, _), gate_cell, rule_cell in zip(
                        ESTIMATORS, cells[GATE], cells[RULE], strict=True
                    )
                )
            random = _point(pr.select(first, "random_ask", alpha), all_cost)
            gate = cells[GATE][[suffix for suffix, _ in ESTIMATORS].index(AGREEMENT_ONLY)]
            within = floor is not None and floor.errors <= allowed
            below = gate.feasible and not gate.all_steps
            pairs += counted
            agree += counted and within == below
            for policy, name in ((GATE, "gate"), (RULE, "rule")):
                out.append(
                    f"| {alpha:g} | {allowed} | {'yes' if within else 'no'} | {name} | "
                    + " | ".join(md_cell(c) for c in cells[policy])
                    + f" | {md_cell(random)} |"
                )
                texs[policy][0 if counted else 1].append(
                    f"{alpha:g} & {allowed} & "
                    + " & ".join(tex_cell(c) for c in cells[policy])
                    + f" & {tex_cell(random)}"
                    + r" \\"
                )
        counts = tally(bench, args.results)
        out += [
            "",
            "Certified points over every controller, and those with held-out error above the "
            "target: "
            + "; ".join(f"{a:g}: {c} certified, {len(b)} above" for a, (c, b) in counts.items())
            + ".",
        ]
        for alpha, (certified, above) in counts.items():
            if counted:
                slot = totals.setdefault(alpha, [0, 0])
                slot[0] += certified
                slot[1] += len(above)
            if above:
                low, high = round(min(above) * n_test), round(max(above) * n_test)
                out.append(
                    f"At {alpha:g} the held-out errors above the target are {low} to {high} of "
                    f"{n_test}; a pipeline whose risk equals the target shows at least {low} "
                    f"with probability {binom.sf(low - 1, n_test, alpha):.2f} and at least "
                    f"{high} with probability {binom.sf(high - 1, n_test, alpha):.2f}."
                )
        out.append("")
    n_ref, delta_ref = 200, 0.1
    out.append(
        "Gate against rule, by cost on the test questions (two costs that print alike are the "
        "same; the replaced runs are not counted)."
    )
    for title, distinct in (
        ("Over the cells of the two tables", False),
        (
            "Over distinct pairs of selections (a pair that serves several targets of a "
            "benchmark is counted once)",
            True,
        ),
    ):
        per_bench = {
            label: [(g, r) for _, g, r in (distinct_pairs(rows) if distinct else rows)]
            for label, rows in compared.items()
        }
        everything = [pair for cells in per_bench.values() for pair in cells]
        out += [
            "",
            f"{title}:",
            "",
            *(compare_md(label, compare(cells)) for label, cells in per_bench.items()),
            compare_md("all benchmarks", compare(everything)),
        ]
    out += [
        "",
        "Selections at the least conservative knob of their grid, over the cells of the two "
        "tables: "
        + "; ".join(
            f"{name} {sum(loose[policy])} of {len(loose[policy])}"
            for policy, name in ((GATE, "gate"), (RULE, "rule"))
        )
        + ".",
    ]
    out.append("")
    out += [
        f"The entropy gate certifies below checking every step exactly when its floor is within "
        f"the allowance in {agree} of {pairs} benchmark-target pairs.",
        "",
        "Certified points by target over the benchmarks (without the replaced runs): "
        + "; ".join(
            f"{a:g}: {c} certified, {b} above" for a, (c, b) in sorted(totals.items(), reverse=True)
        )
        + ".",
        "",
        f"With {n_ref} calibration questions and delta {delta_ref:g}, a pipeline whose risk "
        "equals the target is certified (at most the allowed errors) with probability: "
        + "; ".join(
            f"{a:g}: {binomial_pvalue(allowance(a, n_ref, delta_ref), n_ref, a):.3f}"
            for a in (0.2, 0.1, 0.05, 0.02)
        )
        + ". A pipeline whose risk is half the target is certified with probability: "
        + "; ".join(
            f"{a:g}: {selection_chance(a, n_ref, delta_ref, a / 2):.2f}"
            for a in (0.2, 0.1, 0.05, 0.02)
        )
        + ".",
    ]
    baseline_md, baseline_tex = baseline_tables(args.results, reported)
    out += ["", *baseline_md, *generation_moves(args.results)]
    print("\n".join(out))
    for path, lines in (
        (args.table, texs[GATE][0]),
        (args.table_rule, texs[RULE][0]),
        (args.table_first_runs, texs[GATE][1]),
        (args.table_first_runs_rule, texs[RULE][1]),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {path}")
    if args.table_baselines is not None:
        args.table_baselines.parent.mkdir(parents=True, exist_ok=True)
        args.table_baselines.write_text("\n".join(baseline_tex) + "\n", encoding="utf-8")
        print(f"\nwrote {args.table_baselines}")


if __name__ == "__main__":
    main()
