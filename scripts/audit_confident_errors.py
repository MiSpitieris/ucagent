"""Evidence for a reading of the confident errors of the MuSiQue confirmatory run (no model calls).

Usage::

    uv run python scripts/audit_confident_errors.py            # write the evidence files
    uv run python scripts/audit_confident_errors.py --count \\
        --table docs/figures/preprint/table_confident_errors.tex
    uv run python scripts/audit_confident_errors.py --accepted  # the other side: see below

Reads the corrected confirmatory suites (``results/musique_confirm2d_*``) and writes, under
``docs/reviews/musique_confident_errors/``:

- ``questions.csv``, one row per question, judged on its final answer:
  - set ``floor (calib)``: the calibration questions that the entropy gate leaves wrong at its
    floor point, the knob with the fewest calibration errors among those that check fewer than
    all steps (``floor_point`` of ``scripts/preprint_benchmarks.py``). Their number is the
    confident-error floor of the cross-benchmark table (20 on MuSiQue).
  - set ``residual at selection 0.1 (test)``: the test questions that the confidence-weighted
    gate leaves wrong at its Learn-then-Test selection for 0.1.
- ``hops.csv``, one row per hop: the labelled hops of the uncontrolled test run on which all
  five samples agree (``u = 0`` under entropy) and that the label calls wrong. They are the
  wrong hops of the AUROC within unanimous steps reported for the confirmatory run.

Every row has the sub-questions, the committed answers, the samples with their stated
confidences, the gold answers with the aliases of the final answer, the labels and the
supporting paragraphs, and two columns for the reader: ``verdict`` and ``note``. The verdicts
are those of the 2WikiMultiHopQA audit (``docs/reviews/twowiki_audit/README.md``):
``model right``, ``gold right`` (a model error), ``both acceptable``, ``neither`` and
``unanswerable``, and one more, ``ambiguous``: the question admits the model's reading as well
as the gold's, and the passages support both. Writing the files again keeps every verdict and
note already filled in.

``--accepted`` lists the other side, which the reading does not cover: the answers that the
label rule accepts at the confidence-weighted gate's selection for 0.1 although they are not
the gold or one of its aliases. The rule also accepts an answer that contains the gold or is
contained in it, so a wrong answer can pass (a state for a county of that state). The list
is a first set for a person to read; nothing in it is adjudicated.

``--count`` reads the verdicts and prints, per set, the count of each verdict. For the floor it
prints how many questions would be right on a fair reading (``model right`` or ``both
acceptable``), and the floor that leaves against the Learn-then-Test allowance. For the hops it
prints the stated confidence by verdict and the AUROC of the confidence-weighted agreement
score within the unanimous hops, against every hop the label calls wrong and against the model
errors alone. With ``--table`` it writes the verdict counts as a LaTeX table.

The verdicts are a reading, not a measurement: they say what one reader judged each case to be.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.metrics import roc_auc_score

import preprint_benchmarks as pb
import preprint_results as pr
import real_results as rr
from ucagent.experiments.io import read_trajectories

SUITES = {
    "entropy": Path("results/musique_confirm2d_entropy"),
    "confagree": Path("results/musique_confirm2d_confagree"),
}
OUT = Path("docs/reviews/musique_confident_errors")
VERDICTS = (
    "model right",
    "gold right",
    "both acceptable",
    "ambiguous",
    "neither",
    "unanswerable",
)
FAIR_RIGHT = ("model right", "both acceptable")  # the answer is right on a fair reading
FLOOR_SET = "floor (calib)"
RESIDUAL_SET = "residual at selection 0.1 (test)"
HOP_SET = "unanimous and wrong (test)"
TARGET = 0.1

HOP_FIELDS = [
    "set",
    "instance_id",
    "hop",
    "question",
    "sub_question",
    "previous_answer",
    "model_answer",
    "samples",
    "confidences",
    "u_confagree",
    "gold",
    "aliases",
    "label",
    "final_correct",
    "passage_title",
    "passage",
    "verdict",
    "note",
]
QUESTION_FIELDS = [
    "set",
    "instance_id",
    "question",
    "final_answer",
    "gold_final",
    "aliases",
    *(
        f"hop{h}_{name}"
        for h in (1, 2)
        for name in (
            "sub_question",
            "answer",
            "source",
            "samples",
            "confidences",
            "gold",
            "label",
            "passage_title",
            "passage",
        )
    ),
    "verdict",
    "note",
]


# --- run records --------------------------------------------------------------------------------


def _run(suite_dir: Path, policy: str, split: str, knob: float | None = None) -> list[Any]:
    """The trajectories of one run of a suite."""
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split)]
    if knob is not None:
        rows = rows[(rows["knob"] - knob).abs() < 1e-9]
    if rows.empty:
        raise ValueError(f"{suite_dir.name}: no {policy} run on {split} at knob {knob}")
    path = rr._resolve(suite_dir, str(rows["path"].iloc[0]))
    return list(read_trajectories(str(path)))


def _task() -> Any:
    task = rr._task_of(SUITES["entropy"] / "suite.yaml", key="base")
    if task is None:
        raise SystemExit("the MuSiQue data files are needed (data/musique/)")
    return task


def _support(inst: Any, hop: int) -> Any:
    h = inst.hops[hop - 1]
    return next(p for p in inst.paragraphs if p.idx == h.support_idx)


def _confidences(trajectory: Any, key: str) -> str:
    return " | ".join(
        "-" if c is None else f"{c:g}" for c in (trajectory.sample_confidences.get(key) or [])
    )


def _final_wrong(task: Any, trajectory: Any) -> bool:
    inst = task.get_instance(trajectory.instance_id)
    final = trajectory.final_answer
    if final is None and trajectory.steps:
        final = trajectory.steps[-1].answer
    return not task.is_final_correct(inst, final)


def floor_questions(task: Any) -> tuple[float, list[Any]]:
    """The entropy gate's floor knob and the calibration questions it leaves wrong there."""
    suite = SUITES["entropy"]
    all_asks, _, _ = pb.every_step(suite, "calib")
    floor = pb.floor_point(pb.grid_points(suite), all_asks)
    if floor is None:
        raise ValueError("every knob of the entropy gate checks every step")
    trajs = _run(suite, "per_step_eps", "calib", floor.knob)
    wrong = [t for t in trajs if _final_wrong(task, t)]
    if len(wrong) != floor.errors:
        raise RuntimeError(f"floor run has {len(wrong)} wrong questions, the grid {floor.errors}")
    return floor.knob, wrong


def residual_questions(task: Any) -> tuple[float, list[Any]]:
    """The confidence-weighted gate's selection for 0.1 and the test questions it leaves wrong."""
    suite = SUITES["confagree"]
    sel = pr.select(suite, "per_step_eps", TARGET)
    if not sel.feasible or sel.knob is None:
        raise ValueError("the confidence-weighted gate has no selection for 0.1")
    trajs = _run(suite, "per_step_eps", "test", sel.knob)
    return sel.knob, [t for t in trajs if _final_wrong(task, t)]


def accepted_only_by_containment(task: Any, trajectory: Any) -> bool:
    """A final answer that the label rule accepts and that is not the gold or one of its aliases.

    The run's rule also accepts an answer that contains the gold, or that the gold contains, as
    whole words. Equality is the task's own exact match, with the same normalisation.
    """
    inst = task.get_instance(trajectory.instance_id)
    final = trajectory.final_answer
    if final is None or not task.is_final_correct(inst, final):
        return False
    return not task._matches(final, task._truths(inst, inst.n_steps), "exact")


def accepted_report(task: Any) -> list[str]:
    """Accepted answers at the headline selection that only the containment rule accepts.

    The reading of this folder covers answers that the label calls wrong. These are the other
    side: answers called right without being the gold. Most are ordinary variants; a wrong one
    among them is an error that the measured rates do not count. None is adjudicated here.
    """
    suite = SUITES["confagree"]
    sel = pr.select(suite, "per_step_eps", TARGET)
    if not sel.feasible or sel.knob is None:
        raise ValueError("the confidence-weighted gate has no selection for 0.1")
    gate = {t.instance_id: t for t in _run(suite, "per_step_eps", "test", sel.knob)}
    rand_sel = pr.select(suite, "random_ask", TARGET)
    random = (
        {t.instance_id: t for t in _run(suite, "random_ask", "test", rand_sel.knob)}
        if rand_sel.feasible and rand_sel.knob is not None
        else {}
    )
    accepted = [t for t in gate.values() if not _final_wrong(task, t)]
    loose = [t for t in accepted if accepted_only_by_containment(task, t)]
    out = [
        f"### Accepted answers that are not the gold (confidence-weighted gate at {TARGET:g}, "
        f"knob {sel.knob:g}, test questions)",
        "",
        f"Accepted by the label rule: {len(accepted)} of {len(gate)}. Of these, not equal to the "
        f"gold or an alias after normalisation: {len(loose)}. None is adjudicated here.",
        "",
        "| question | asked | gold and aliases | the gate's answer | random asking's answer "
        "at its selection |",
        "|---|---|---|---|---|",
    ]
    for t in sorted(loose, key=lambda t: t.instance_id):
        inst = task.get_instance(t.instance_id)
        golds = " / ".join([inst.hops[-1].answer, *inst.answer_aliases])
        other = random.get(t.instance_id)
        out.append(
            f"| {t.instance_id} | {inst.question} | {golds} | {t.final_answer} | "
            f"{'-' if other is None else other.final_answer} |"
        )
    return out


def unanimous_hops(task: Any) -> tuple[list[tuple[Any, Any]], dict[tuple[str, int], float]]:
    """The labelled unanimous hops of the uncontrolled test run, and their confagree scores.

    Unanimity is read from the entropy suite (``u = 0``: one cluster); the score is the
    confidence-weighted agreement of the same hop in the confagree suite, which saw the same
    samples.
    """
    ent = _run(SUITES["entropy"], "none", "test")
    conf = {t.instance_id: t for t in _run(SUITES["confagree"], "none", "test")}
    hops, score = [], {}
    for t in ent:
        other = {s.step: s for s in conf[t.instance_id].steps}
        for s in t.steps:
            if s.source != "llm" or s.locally_correct is None or s.u != 0.0:
                continue
            o = other[s.step]
            if o.answer != s.answer:
                raise RuntimeError(f"{t.instance_id} hop {s.step}: the suites committed apart")
            hops.append((t, s))
            score[(t.instance_id, s.step)] = float(o.u)
    return hops, score


# --- rows ---------------------------------------------------------------------------------------


def question_row(set_name: str, task: Any, trajectory: Any) -> dict[str, Any]:
    """One question, with both hops as the run committed them."""
    inst = task.get_instance(trajectory.instance_id)
    row: dict[str, Any] = {
        "set": set_name,
        "instance_id": trajectory.instance_id,
        "question": inst.question,
        "final_answer": trajectory.final_answer
        or (trajectory.steps[-1].answer if trajectory.steps else ""),
        "gold_final": inst.answer,
        "aliases": " | ".join(inst.answer_aliases),
        "verdict": "",
        "note": "",
    }
    answers = {s.step: s.answer for s in trajectory.steps}
    by_step = {s.step: s for s in trajectory.steps}
    for hop in (1, 2):
        s = by_step.get(hop)
        para = _support(inst, hop)
        prefix = f"hop{hop}_"
        row[prefix + "sub_question"] = task.sub_question(
            inst, hop, {k: v for k, v in answers.items() if k < hop}
        )
        row[prefix + "answer"] = s.answer if s else ""
        row[prefix + "source"] = s.source if s else ""
        row[prefix + "samples"] = " | ".join(trajectory.samples.get(s.samples_key, [])) if s else ""
        row[prefix + "confidences"] = _confidences(trajectory, s.samples_key) if s else ""
        row[prefix + "gold"] = inst.hops[hop - 1].answer
        row[prefix + "label"] = (
            {True: "correct", False: "wrong", None: "off track"}[s.locally_correct] if s else ""
        )
        row[prefix + "passage_title"] = para.title
        row[prefix + "passage"] = para.text
    return row


def hop_row(task: Any, trajectory: Any, step: Any, score: float) -> dict[str, Any]:
    """One unanimous and wrong hop."""
    inst = task.get_instance(trajectory.instance_id)
    answers = {s.step: s.answer for s in trajectory.steps}
    para = _support(inst, step.step)
    final = step.step == inst.n_steps
    return {
        "set": HOP_SET,
        "instance_id": trajectory.instance_id,
        "hop": step.step,
        "question": inst.question,
        "sub_question": task.sub_question(
            inst, step.step, {k: v for k, v in answers.items() if k < step.step}
        ),
        "previous_answer": step.context,
        "model_answer": step.answer,
        "samples": " | ".join(trajectory.samples.get(step.samples_key, [])),
        "confidences": _confidences(trajectory, step.samples_key),
        "u_confagree": f"{score:.3f}",
        "gold": inst.hops[step.step - 1].answer,
        "aliases": " | ".join(inst.answer_aliases) if final else "",
        "label": "wrong",
        "final_correct": not _final_wrong(task, trajectory),
        "passage_title": para.title,
        "passage": para.text,
        "verdict": "",
        "note": "",
    }


def key_of(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The identity of a row across rewrites: set, question and, for a hop, the hop."""
    return (str(row["set"]), str(row["instance_id"]), str(row.get("hop", "")))


def merge_verdicts(
    new: Sequence[dict[str, Any]], old: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The new rows with the verdicts and notes of the old rows of the same identity."""
    kept = {key_of(r): (r.get("verdict", ""), r.get("note", "")) for r in old}
    out = []
    for row in new:
        verdict, note = kept.get(key_of(row), ("", ""))
        out.append({**row, "verdict": verdict, "note": note})
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV file, or none when it does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write rows with the given columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


# --- counts -------------------------------------------------------------------------------------


def check_verdicts(rows: Iterable[Mapping[str, Any]]) -> None:
    """Every verdict is one of ``VERDICTS`` or empty (not read yet)."""
    bad = sorted({str(r["verdict"]) for r in rows if r["verdict"] and r["verdict"] not in VERDICTS})
    if bad:
        raise SystemExit(f"unknown verdicts {bad}; use one of {list(VERDICTS)}")


def verdict_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    """Verdicts of a set; an empty verdict counts as ``not read``."""
    return Counter(str(r["verdict"]) or "not read" for r in rows)


def fair_floor(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Floor questions read, those right on a fair reading, and the floor that leaves."""
    read = [r for r in rows if r["verdict"]]
    fair = sum(r["verdict"] in FAIR_RIGHT for r in read)
    return len(read), fair, len(rows) - fair


def auroc_two_ways(
    right_scores: Sequence[float], wrong: Sequence[tuple[float, str]]
) -> tuple[float | None, float | None]:
    """AUROC of a score within unanimous hops: every labelled wrong hop, and model errors only.

    ``wrong`` holds (score, verdict) for the hops the label calls wrong; a model error is a hop
    read as ``gold right``. A side with no hop gives None.
    """

    def auroc(positives: Sequence[float]) -> float | None:
        if not positives or not right_scores:
            return None
        y = [1] * len(positives) + [0] * len(right_scores)
        return float(roc_auc_score(y, list(positives) + list(right_scores)))

    return auroc([s for s, _ in wrong]), auroc([s for s, v in wrong if v == "gold right"])


def tex_table(counts: Mapping[str, Counter[str]], sizes: Mapping[str, int]) -> list[str]:
    """Verdict counts by set as the body of a LaTeX table."""
    cols = [*VERDICTS, "not read"]
    shown = [c for c in cols if any(counts[s][c] for s in counts)]
    head = " & ".join(["set", "rows", *shown]) + r" \\"
    lines = [
        "% written by scripts/audit_confident_errors.py; do not edit by hand",
        r"\begin{tabular}{@{}l" + "c" * (len(shown) + 1) + "@{}}",
        r"\toprule",
        head,
        r"\midrule",
    ]
    for set_name in counts:
        cells = [str(counts[set_name][c]) for c in shown]
        lines.append(" & ".join([set_name, str(sizes[set_name]), *cells]) + r" \\")
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


# --- command line -------------------------------------------------------------------------------


def write(out: Path) -> None:
    """Write the evidence files, keeping verdicts already filled in."""
    task = _task()
    floor_knob, floor = floor_questions(task)
    sel_knob, residual = residual_questions(task)
    hops, score = unanimous_hops(task)
    q_rows = [question_row(FLOOR_SET, task, t) for t in floor]
    q_rows += [question_row(RESIDUAL_SET, task, t) for t in residual]
    h_rows = [
        hop_row(task, t, s, score[(t.instance_id, s.step)])
        for t, s in hops
        if s.locally_correct is False
    ]
    q_rows = merge_verdicts(q_rows, read_csv(out / "questions.csv"))
    h_rows = merge_verdicts(h_rows, read_csv(out / "hops.csv"))
    write_csv(out / "questions.csv", q_rows, QUESTION_FIELDS)
    write_csv(out / "hops.csv", h_rows, HOP_FIELDS)
    print(f"floor: entropy gate at knob {floor_knob:g}, {len(floor)} calibration questions wrong")
    print(f"residual: confidence-weighted gate at knob {sel_knob:g}, {len(residual)} wrong")
    print(f"unanimous labelled test hops: {len(hops)}, of which wrong: {len(h_rows)}")
    print(f"wrote {out / 'questions.csv'} and {out / 'hops.csv'}")


def count(out: Path, table: Path | None) -> None:
    """Print the verdict counts (and write the table)."""
    q_rows, h_rows = read_csv(out / "questions.csv"), read_csv(out / "hops.csv")
    if not q_rows or not h_rows:
        raise SystemExit(f"write the evidence files first ({out})")
    check_verdicts([*q_rows, *h_rows])
    by_set: dict[str, list[dict[str, str]]] = {}
    for r in [*q_rows, *h_rows]:
        by_set.setdefault(r["set"], []).append(r)
    counts = {s: verdict_counts(rows) for s, rows in by_set.items()}
    sizes = {s: len(rows) for s, rows in by_set.items()}
    lines = ["### Reading of the confident errors, MuSiQue confirmatory run (counts)", ""]
    for set_name, c in counts.items():
        lines.append(
            f"- {set_name}: {sizes[set_name]} rows; "
            + ", ".join(f"{v} {c[v]}" for v in (*VERDICTS, "not read") if c[v])
            + "."
        )
    floor_rows = by_set.get(FLOOR_SET, [])
    read, fair, left = fair_floor(floor_rows)
    allowed = pb.allowance(TARGET, 200, 0.1)
    lines += [
        "",
        f"Floor: {len(floor_rows)} calibration questions, {read} read; right on a fair reading "
        f"(model right or both acceptable): {fair}; the floor on a fair reading: {left}, "
        f"against an allowance of {allowed} at {TARGET:g}"
        + (" (within it)." if left <= allowed else " (above it)."),
    ]
    res_rows = by_set.get(RESIDUAL_SET, [])
    res_fair = sum(r["verdict"] in FAIR_RIGHT for r in res_rows)
    lines.append(
        f"Residual at the confidence-weighted gate's selection for {TARGET:g}: {len(res_rows)} "
        f"test questions wrong as labelled, {res_fair} of them right on a fair reading."
    )
    task = _task()
    hops, score = unanimous_hops(task)
    verdict_of = {(r["instance_id"], int(r["hop"])): r["verdict"] for r in h_rows}
    right_scores = [score[(t.instance_id, s.step)] for t, s in hops if s.locally_correct]
    wrong = [
        (score[(t.instance_id, s.step)], verdict_of.get((t.instance_id, s.step), ""))
        for t, s in hops
        if s.locally_correct is False
    ]
    if len(wrong) != len(h_rows):
        raise RuntimeError("hops.csv does not match the run records; write the files again")
    all_wrong, errors_only = auroc_two_ways(right_scores, wrong)
    lines += [
        "",
        f"Unanimous labelled test hops: {len(hops)}; {len(wrong)} wrong as labelled.",
        f"- Confidence-weighted agreement score (one minus the mean stated confidence on these "
        f"hops), mean: right {statistics.fmean(right_scores):.3f}"
        + "".join(
            f"; {v} {statistics.fmean(s for s, w in wrong if w == v):.3f} "
            f"({sum(w == v for _, w in wrong)})"
            for v in VERDICTS
            if any(w == v for _, w in wrong)
        )
        + ".",
        f"- AUROC within the unanimous hops: against every hop the label calls wrong "
        f"{_fmt(all_wrong)} ({len(wrong)}); against the model errors alone (gold right) "
        f"{_fmt(errors_only)} ({sum(v == 'gold right' for _, v in wrong)}).",
    ]
    print("\n".join(lines))
    if table is not None:
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("\n".join(tex_table(counts, sizes)) + "\n", encoding="utf-8")
        print(f"\nwrote {table}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--count", action="store_true", help="print the verdict counts")
    parser.add_argument("--table", type=Path, default=None, help="with --count: the LaTeX table")
    parser.add_argument(
        "--accepted",
        action="store_true",
        help="list the accepted answers at the headline selection that are not the gold",
    )
    args = parser.parse_args(argv)
    if args.accepted:
        print("\n".join(accepted_report(_task())))
    elif args.count:
        count(args.out, args.table)
    else:
        write(args.out)


if __name__ == "__main__":
    main()
