"""Which actions the value-of-intervention rule takes at its selections (no model calls).

Usage::

    uv run python scripts/rule_actions.py

For every cell of the cross-benchmark tables (each benchmark of
``scripts/preprint_benchmarks.py``, in the generation of suites that ``--suites`` names, each
estimator and each pre-registered target), the rule's
Learn-then-Test selection is taken as ``select`` of ``scripts/preprint_results.py`` makes it,
and its test run is read from the suite's manifest: checks by a person, resamples, backtracks
and escalations per question. The summary gives, by target, how many selections take each
action at all and the mean per question.

The rule chooses among five actions, the calibrated gate only between proceeding and asking;
this shows how much of the rule's wider choice its selections use.

With ``--labels`` it replays the rule's test grid and labels, by the task's own local rule, the
answer of each distinct expanded batch before and after its expansion: what a resample did to
correctness, set beside the error the effect model expected. One batch is met at many knobs, so
the distinct cases are far fewer than the decisions. With ``--future`` it prints the stored
future factor of a first hop beside the survival of the second hops that have a label.

With ``--resamples`` it also reads every run of the rule's knob grid (calibration and test) and
sets what the rule expected of each resample beside what the expanded batch showed. A decision
records the step's estimated error ``eps``, the survival ``S``, the future factor ``G``, the
weight ``w`` and the expected reduction of the projected risk, ``delta = S G w (eps - ehat)``,
so the error it expected after the resample is ``ehat = eps - delta / (S G w)``. What the
expanded batch showed is the ``eps`` of the next decision at the same step. The score of the
batch as it was before the expansion is recomputed from its first ``K`` recorded samples, to
count the resamples decided at a score that sits just below an edge of the effect table's
ranges (``1 - 0.8`` is below 0.2 in floating point), which the table as run put in the range
below. Reading the grids takes some minutes.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_benchmarks as pb
import preprint_results as pr
import real_results as rr
from ucagent.core.records import Action, Sample, Trajectory
from ucagent.core.uq.base import StepContext, StepUncertaintyEstimator, make_estimator
from ucagent.core.uq.equivalence import cluster_samples, make_judge
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.runner import build_task
from ucagent.experiments.suite import SuiteConfig

ACTIONS = (
    ("ask_human", "checks"),
    ("resample", "resamples"),
    ("backtrack", "backtracks"),
    ("escalate", "escalations"),
)
RULE = "marginal"
RANGE_EDGES = (0.2, 0.4, 0.6, 0.8)  # the inner edges of the effect table's ranges, as run


@dataclass(frozen=True)
class Cell:
    """The rule's selection in one cell, with its actions per question on the test run."""

    benchmark: str
    estimator: str
    target: float
    knob: float
    error: float
    cost: float
    actions: dict[str, float]  # action -> per question


def selection_actions(suite_dir: Path, target: float) -> tuple[pr.Selection, dict[str, float]]:
    """The rule's selection at a target and the actions per question of its test run."""
    sel = pr.select(suite_dir, RULE, target)
    if not sel.feasible or sel.knob is None:
        return sel, {}
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == RULE) & (grid["split"] == "test")]
    row = rows[(rows["knob"] - sel.knob).abs() < 1e-9].iloc[0]
    return sel, {name: float(row[f"interventions_{name}"]) for name, _ in ACTIONS}


def summary(cells: Sequence[Cell]) -> list[str]:
    """By target: selections, how many take each action at all, and the means per question."""
    out = []
    for target in sorted({c.target for c in cells}, reverse=True):
        at = [c for c in cells if c.target == target]
        n = len(at)
        parts = [f"{sum(c.actions[name] > 0 for c in at)} take {label}" for name, label in ACTIONS]
        means = [f"{label} {sum(c.actions[name] for c in at) / n:.3f}" for name, label in ACTIONS]
        only_ask = sum(  # checks, and no other action
            c.actions["ask_human"] > 0
            and all(c.actions[name] == 0 for name, _ in ACTIONS if name != "ask_human")
            for c in at
        )
        out.append(
            f"- Target {target:g}: {n} selections; "
            + ", ".join(parts)
            + f"; {only_ask} only check; per question: "
            + ", ".join(means)
            + "."
        )
    return out


@dataclass(frozen=True)
class Resample:
    """One resample decision of the rule: what it expected and what the expanded batch showed."""

    before: float  # the step's estimated error when the rule decided
    predicted: float  # the error the effect table expected after the resample
    realised: float | None  # the estimated error of the expanded batch; None: not decided again
    below_edge: bool | None  # the score sat just below a range edge; None: batch not rebuilt


def expected_after(numbers: dict[str, float]) -> float | None:
    """The error a decision expected after a resample, from the numbers it recorded."""
    scale = numbers.get("S_prev", 0.0) * numbers.get("G", 0.0) * numbers.get("w", 0.0)
    if "delta_resample" not in numbers or "eps" not in numbers or scale <= 0.0:
        return None
    return numbers["eps"] - numbers["delta_resample"] / scale


def just_below_edge(u: float, edges: Sequence[float] = RANGE_EDGES) -> bool:
    """Whether a score equals a range edge in exact arithmetic but lies below it as a float."""
    return any(edge - 1e-9 < u < edge for edge in edges)


def score_before(
    trajectory: Trajectory, step: int, estimator: StepUncertaintyEstimator
) -> float | None:
    """The score of a step's batch before its expansion, from its first ``K`` recorded samples.

    None when the step was not committed with an expanded batch (it was escalated, or a
    backtrack replaced it), so the batch of the decision cannot be told apart.
    """
    record = next((s for s in trajectory.steps if s.step == step), None)
    if record is None or record.n_samples <= trajectory.K:
        return None
    answers = trajectory.samples.get(record.samples_key, [])
    if len(answers) < trajectory.K:
        return None
    stated = trajectory.sample_confidences.get(record.samples_key) or [None] * len(answers)
    samples = [
        Sample(sample_index=j, text=a, answer=a, confidence=c)
        for j, (a, c) in enumerate(zip(answers, stated, strict=True))
    ]
    context = StepContext(
        instance_id=trajectory.instance_id,
        step=step,
        attempt=record.attempt,
        current=record.context,
        n_steps=trajectory.n_steps,
    )
    return float(estimator.estimate(samples[: trajectory.K], context).u)


def resamples_of(
    trajectory: Trajectory, estimator: StepUncertaintyEstimator | None = None
) -> list[Resample]:
    """Every resample decision of one trajectory, in the order taken."""
    out = []
    decisions = trajectory.interventions
    for k, decision in enumerate(decisions):
        if decision.action != Action.RESAMPLE:
            continue
        predicted = expected_after(decision.numbers)
        if predicted is None:
            continue
        later = next((d for d in decisions[k + 1 :] if d.step == decision.step), None)
        realised = later.numbers.get("eps") if later is not None else None
        u = score_before(trajectory, decision.step, estimator) if estimator is not None else None
        out.append(
            Resample(
                before=decision.numbers["eps"],
                predicted=predicted,
                realised=realised,
                below_edge=None if u is None else just_below_edge(u),
            )
        )
    return out


def resample_row(label: str, records: Sequence[Resample]) -> str:
    """One markdown row: counts and means of what was expected and what was seen."""
    seen = [r for r in records if r.realised is not None]
    rebuilt = [r for r in records if r.below_edge is not None]
    if not seen:
        return f"| {label} | {len(records)} | - | - | - | - | - | - |"
    n = len(seen)
    predicted = sum(r.predicted for r in seen) / n
    realised = sum(r.realised or 0.0 for r in seen) / n
    return (
        f"| {label} | {len(records)} | {n} | {sum(r.before for r in seen) / n:.3f} | "
        f"{predicted:.3f} | {realised:.3f} | {realised - predicted:+.3f} | "
        f"{sum(bool(r.below_edge) for r in rebuilt)} of {len(rebuilt)} |"
    )


def grid_resamples(suite_dir: Path) -> list[Resample]:
    """The resample decisions of every run of the rule's grid in a suite, both splits."""
    base = pr._base(suite_dir)
    estimator = make_estimator(base["uq"]["estimator"], make_judge(base["uq"]["equivalence"]))
    grid = pr._grid(suite_dir)
    out: list[Resample] = []
    for path in grid[grid["policy"] == RULE]["path"]:
        for trajectory in read_trajectories(rr._resolve(suite_dir, str(path))):
            out.extend(resamples_of(trajectory, estimator))
    return out


def resample_report(results: Path, reported: Sequence[pb.Benchmark]) -> list[str]:
    """Expected against seen after a resample, by benchmark and estimator, over the grids."""
    out = [
        "",
        "### What the rule expected of a resample, and what the expanded batch showed",
        "",
        "Every run of the rule's knob grid, calibration and test. *decided again*: resamples "
        "after which the step was decided again, which the means are over. *seen minus "
        "expected*: the difference of the two means before rounding; above zero, the rule "
        "expected more of the resample than the expanded batch showed. *below an edge*: "
        "resamples decided at a score just below an edge of the effect table's ranges, of those "
        "whose batch could be rebuilt (the repaired effect model does not use the ranges).",
        "",
        "| benchmark, estimator | resamples | decided again | error before | expected after "
        "| seen after | seen minus expected | below an edge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bench in reported:
        pooled: list[Resample] = []
        for suffix, label in pb.ESTIMATORS:
            records = grid_resamples(results / bench.pattern.format(suffix))
            pooled.extend(records)
            out.append(resample_row(f"{bench.label}, {label}", records))
        out.append(resample_row(f"{bench.label}, all four", pooled))
    return out


@dataclass(frozen=True)
class Expansion:
    """A resample decision whose batch has a label, replayed from its trajectory."""

    #: question, step, the answers committed before it, draw and samples: one distinct case,
    #: however many knobs of the grid arrive at it
    key: tuple[object, ...]
    eps_before: float
    expected_after: float
    seen_after: float
    wrong_before: bool
    wrong_after: bool


def replay_expansions(
    trajectory: Trajectory, task: Any, representative: Callable[[tuple[str, ...]], str]
) -> tuple[int, list[Expansion]]:
    """The resample decisions of a trajectory (their number) and those with a local label.

    The decisions are replayed in order: which batch each one saw, what had been committed
    before it, and what a check or a backtrack did. The answer of a batch is the
    representative of its largest cluster, before the expansion over its first ``K`` samples
    and after it over all ``K'``; each is labelled by the task's own local rule in the context
    of the decision. A step that is off track has no local label and is counted only as a
    decision. The replay must end at the recorded final answer, or it raises.
    """
    inst = task.get_instance(trajectory.instance_id)
    has_truth = getattr(task, "has_local_truth", None)
    k, k_prime = trajectory.K, trajectory.K_prime
    history: list[tuple[str, int]] = []  # committed (answer, draw) per step
    next_draw: dict[int, int] = {}
    draw, size = 0, k
    decisions = trajectory.interventions
    n_resamples, out = 0, []
    for position, d in enumerate(decisions):
        step = len(history) + 1
        if d.step != step:
            raise ValueError(f"{trajectory.instance_id}: decision {position} is not at step {step}")
        next_draw[step] = max(next_draw.get(step, 0), draw + 1)
        batch = tuple(trajectory.samples[f"{step}:{draw}"])
        context = history[-1][0] if history else task.initial_context(inst)
        labelled = has_truth is None or has_truth(inst, step, context)
        answer = representative(batch[:size])
        if d.action == Action.RESAMPLE:
            n_resamples += 1
            expected = expected_after(d.numbers)
            later = decisions[position + 1] if position + 1 < len(decisions) else None
            if size != k or len(batch) != k_prime or later is None or later.step != step:
                raise ValueError(f"{trajectory.instance_id}: an expansion that cannot be replayed")
            if labelled and expected is not None:
                out.append(
                    Expansion(
                        (trajectory.instance_id, step, tuple(a for a, _ in history), draw, batch),
                        d.numbers["eps"],
                        expected,
                        later.numbers["eps"],
                        not task.is_locally_correct(inst, step, context, answer),
                        not task.is_locally_correct(inst, step, context, representative(batch)),
                    )
                )
            size = k_prime
        elif d.action == Action.BACKTRACK:
            target = int(d.target_step or 0)
            draw = history[target - 1][1]
            history = history[: target - 1]
            size = k_prime
        elif d.action in (Action.PROCEED, Action.ASK_HUMAN):
            if d.action == Action.ASK_HUMAN and labelled:
                answer = task.local_truth(inst, step, context)
            history.append((answer, draw))
            draw, size = next_draw.get(step + 1, 0), k
        elif d.action == Action.ESCALATE:
            if position + 1 != len(decisions):
                raise ValueError(f"{trajectory.instance_id}: a decision after a handover")
        else:
            raise ValueError(f"{trajectory.instance_id}: unknown action {d.action}")
    ended = bool(decisions) and decisions[-1].action != Action.ESCALATE
    if ended and (not history or history[-1][0] != trajectory.final_answer):
        raise ValueError(f"{trajectory.instance_id}: the replay ends at another answer")
    return n_resamples, out


def label_row(label: str, decisions: int, labelled: int, cases: Sequence[Expansion]) -> str:
    """One markdown row over distinct expansions: what the expansion did to correctness."""
    head = f"| {label} | {decisions} | {labelled} | {len(cases)} "
    if not cases:
        return head + "| - | - | - | - | - | - | - |"
    n = len(cases)
    return head + (
        f"| {len({c.key[0] for c in cases})} | {sum(c.wrong_before for c in cases)} | "
        f"{sum(c.wrong_after for c in cases)} | "
        f"{sum(c.wrong_before and not c.wrong_after for c in cases)} | "
        f"{sum(c.wrong_after and not c.wrong_before for c in cases)} | "
        f"{sum(c.expected_after for c in cases) / n:.3f} | "
        f"{sum(c.wrong_after for c in cases) / n:.3f} |"
    )


def suite_expansions(suite_dir: Path) -> tuple[int, int, list[Expansion]]:
    """Resample decisions on the rule's test grid, those with a label, and the distinct cases."""
    stored = SuiteConfig.model_validate(
        yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    )
    task = build_task(stored.base)
    judge = make_judge(stored.base.uq.equivalence)

    @functools.lru_cache(maxsize=20000)
    def representative(answers: tuple[str, ...]) -> str:
        samples = [Sample(sample_index=j, text=a, answer=a) for j, a in enumerate(answers)]
        return str(cluster_samples(samples, judge, None)[0].representative)

    grid = pr._grid(suite_dir)
    runs = grid[(grid["policy"] == RULE) & (grid["split"] == "test")]
    decisions = labelled = 0
    cases: dict[tuple[object, ...], Expansion] = {}
    for path in runs["path"]:
        for trajectory in read_trajectories(rr._resolve(suite_dir, str(path))):
            if not any(d.action == Action.RESAMPLE for d in trajectory.interventions):
                continue
            n, found = replay_expansions(trajectory, task, representative)
            decisions += n
            labelled += len(found)
            for e in found:
                cases.setdefault(e.key, e)
    return decisions, labelled, list(cases.values())


def label_report(results: Path, reported: Sequence[pb.Benchmark]) -> list[str]:
    """What an expansion did to correctness, by benchmark and estimator (distinct expansions)."""
    out = [
        "",
        "### What a resample did to correctness (test side of the rule's grid)",
        "",
        "*Resample decisions*: over every knob of the grid; one batch counts once per knob that "
        "arrives at it. *With a label*: decisions at a step that is on track. *Distinct "
        "expansions*: the same question, step, earlier answers, draw and samples count once; the "
        "columns to its right are over these. *Expected error after*: the mean of what the "
        "effect model expected; *share wrong after*: the share of the expanded batches whose "
        "answer the task's local rule calls wrong.",
        "",
        "| benchmark, estimator | resample decisions | with a label | distinct expansions "
        "| questions | wrong before | wrong after | wrong to right | right to wrong "
        "| expected error after | share wrong after |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals: tuple[int, int, list[Expansion]] = (0, 0, [])
    for bench in reported:
        for suffix, label in pb.ESTIMATORS:
            decisions, labelled, cases = suite_expansions(results / bench.pattern.format(suffix))
            out.append(label_row(f"{bench.label}, {label}", decisions, labelled, cases))
            # an expansion is distinct within its suite: the suites of a benchmark share samples
            # but not decisions, so the total adds the suites' counts
            # the first part of a key is its question: here a question of a benchmark
            cases = [replace(c, key=((bench.label, c.key[0]), suffix, *c.key[1:])) for c in cases]
            totals = (totals[0] + decisions, totals[1] + labelled, totals[2] + cases)
    out.append(label_row("all benchmarks and estimators", *totals))
    return out


def future_report(results: Path, reported: Sequence[pb.Benchmark]) -> list[str]:
    """The stored future factor of a first hop beside the survival of the labelled second hops."""
    out = [
        "",
        "### The future factor of the first hop on the two-hop benchmarks",
        "",
        "*Stored*: what the fit writes and the rule uses; a second hop of the fit that is off "
        "track has no label and counts there as certain to survive. *Over labelled second hops*: "
        "one minus the mean estimated error of the second hops that have a label.",
        "",
        "| benchmark, estimator | second hops of the fit with a label | stored "
        "| over labelled second hops |",
        "|---|---|---|---|",
    ]
    for bench in reported:
        for suffix, label in pb.ESTIMATORS:
            path = results / bench.pattern.format(suffix) / "artifacts" / "seed0.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            stored, mean = (
                artifact.get("future_factor", {}),
                artifact.get("eps_mean_by_position", {}),
            )
            if set(stored) != {"2:1", "2:2"} or "2:2" not in mean:
                continue  # chains of other lengths
            fit = int(artifact["n_trajectories"])  # every first hop has a label
            labelled = int(artifact["isotonic_K"]["n"]) - fit
            out.append(
                f"| {bench.label}, {label} | {labelled} of {fit} | {stored['2:1']:.5f} | "
                f"{1 - mean['2:2']:.5f} |"
            )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--suites",
        choices=sorted(pb.GENERATIONS),
        default=pb.DEFAULT_GENERATION,
        help="the generation of the suites: after the repair of the effect model, or before",
    )
    parser.add_argument(
        "--resamples",
        action="store_true",
        help="also read the rule's grids: what it expected of a resample against what was seen",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="also replay the rule's test grid: what each distinct expansion did to correctness",
    )
    parser.add_argument(
        "--future",
        action="store_true",
        help="also print the stored future factor beside the survival of labelled second hops",
    )
    args = parser.parse_args(argv)
    reported = pb.benchmarks(args.suites)
    cells: list[Cell] = []
    missing = 0
    for bench in reported:
        for suffix, label in pb.ESTIMATORS:
            suite = args.results / bench.pattern.format(suffix)
            for target in bench.targets:
                sel, actions = selection_actions(suite, target)
                if not actions or sel.knob is None or sel.test_error is None:
                    missing += 1
                    continue
                cells.append(
                    Cell(
                        bench.label,
                        label,
                        target,
                        sel.knob,
                        sel.test_error,
                        float(sel.test_cost or 0.0),
                        actions,
                    )
                )
    out = [
        f"### The value-of-intervention rule's actions at its selections (test runs, "
        f"{args.suites} suites)",
        "",
        "| benchmark | estimator | target | knob | held-out error | cost | "
        + " | ".join(label for _, label in ACTIONS)
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(ACTIONS),
    ]
    for c in cells:
        out.append(
            f"| {c.benchmark} | {c.estimator} | {c.target:g} | {c.knob:g} | {c.error:.3f} | "
            f"{c.cost:.2f} | " + " | ".join(f"{c.actions[name]:.3f}" for name, _ in ACTIONS) + " |"
        )
    out += [
        "",
        f"Selections: {len(cells)}; cells without a selection: {missing}.",
        *summary(cells),
        f"Selections that escalate at all: {sum(c.actions['escalate'] > 0 for c in cells)}.",
    ]
    if args.resamples:
        out += resample_report(args.results, reported)
    if args.labels:
        out += label_report(args.results, reported)
    if args.future:
        out += future_report(args.results, reported)
    print("\n".join(out))


if __name__ == "__main__":
    main()
