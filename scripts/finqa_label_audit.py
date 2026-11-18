"""How lenient is FinQA's label rule about scale, and what would a strict final label change?

Usage::

    uv run python scripts/finqa_label_audit.py results/finqa_confagree \\
        --also results/finqa_entropy results/finqa_verbalized --targets 0.1,0.05

``ucagent.tasks.finqa.numbers_match`` accepts an answer that matches the gold value, 100
times it or a hundredth of it, with an absolute tolerance applied after the scaling. That
admits percent forms ("4.95 %" for 0.0495) and also errors of scale (5000 for 50) and a zero
answer to a small gold value. This script prints, from the run records:

1. The final answers of the uncontrolled runs that are labelled correct only through a scaled
   gold, split into percent forms (the answer carries a % sign, or the match is the gold times
   100 within the relative tolerance and the question asks for a percentage or a rate) and
   the rest, which are errors of scale or zero answers. The rule is mechanical: it classifies,
   it does not adjudicate.
2. How the steps and the facts are worded, for the fit, calibration and test windows:
   questions with a step description that does not single out its number among the gold
   facts (``ucagent.tasks.finqa.description_issue``, by cause); steps whose two arguments are
   described identically, split into two different operands and the same number used twice
   (the latter is well posed); descriptions that show their own number; questions whose
   program uses a magnitude where the report prints a negative; how often the supporting
   table row, which keeps FinQA's wording, is shown beside rows in the adapter's wording and
   how often it is shown a second time in that wording; and questions that occur in both
   calibration and test (same page, question and gold).
3. The Learn-then-Test selections of the gate, the rule and random asking under a final label
   that is strict about scale (a direct match, or a percent form as defined above; exponent
   notation read as a number), with decisions and step labels as made. This is a relabel of
   recorded runs, not a rerun: a corrected rule would also change step labels and later
   prompts.

No model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.control.ltt import calibrate_knob
from ucagent.core.uq.equivalence import NUMERIC_REL_TOL, parse_number
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.suite import configured_knobs
from ucagent.tasks import make_task
from ucagent.tasks.finqa import (
    _decimals,
    _sig_digits,
    _stand_alone,
    _value,
    convert,
    description_issue,
    numbers_match,
    printed_negative,
    program_ops,
    read_number,
    step_match,
    table_head,
)

_EXPONENT = re.compile(r"^-?\d+(?:\.\d+)?[eE][+-]?\d+$")
# the question asks for a percentage or a rate (a % sign in it may belong to an operand)
_ASKS_PERCENT = re.compile(r"\bpercent(?:age)?\b|\brate\b|\bratio\b", re.IGNORECASE)
POLICIES = (("per_step_eps", "gate"), ("marginal", "rule"), ("random_ask", "random asking"))


def _direct(answer: str, gold: float) -> bool:
    """The recorded rule applied to the unscaled gold value only."""
    parsed = parse_number(answer)
    if parsed is None or not math.isfinite(gold):
        return False
    value = parsed[0]
    if abs(value - gold) <= max(NUMERIC_REL_TOL * abs(gold), 5e-4):
        return True
    return _sig_digits(answer) >= 2 and round(gold, _decimals(answer)) == value


def branch(answer: str, gold: float) -> str | None:
    """How the recorded rule accepts ``answer``: direct, x100 or another scaling (None: no)."""
    if not numbers_match(answer, gold):
        return None
    if _direct(answer, gold):
        return "direct"
    parsed = parse_number(answer)
    assert parsed is not None
    if abs(parsed[0] - gold * 100.0) <= NUMERIC_REL_TOL * abs(gold * 100.0):
        return "x100"
    return "scaled, other"  # the hundredth, or the absolute tolerance after scaling


def percent_form(answer: str, gold: float, question: str) -> bool:
    """A scaled acceptance that is a percentage: a % sign, or x100 on a percentage question."""
    if "%" in answer:
        return True
    parsed = parse_number(answer)
    if parsed is None:
        return False
    times_100 = abs(parsed[0] - gold * 100.0) <= NUMERIC_REL_TOL * abs(gold * 100.0)
    return times_100 and bool(_ASKS_PERCENT.search(question))


def strict(answer: str | None, gold: float, question: str) -> bool:
    """A final label that is strict about scale."""
    text = "" if answer is None else str(answer).strip()
    if _EXPONENT.match(text):
        return abs(float(text) - gold) <= max(NUMERIC_REL_TOL * abs(gold), 5e-4)
    if _direct(text, gold):
        return True
    return numbers_match(text, gold) and percent_form(text, gold, question)


def _runs(suite_dir: Path, policy: str, split: str) -> list[tuple[float | None, Path, float]]:
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split)]
    out = []
    for row in rows.itertuples():
        path = Path(str(row.path))
        path = path if path.is_absolute() else suite_dir.parent.parent / path
        knob = None if row.knob != row.knob else float(row.knob)
        out.append((knob, path, float(row.total_cost)))
    return out


def raw_rows(data_dir: Path) -> dict[str, dict[str, Any]]:
    """FinQA's records by id (the task keeps the converted chains only)."""
    out: dict[str, dict[str, Any]] = {}
    for name in ("train.json", "dev.json", "test.json"):
        for row in json.loads((data_dir / name).read_text(encoding="utf-8")):
            out[str(row["id"])] = row
    return out


def gold_facts(row: dict[str, Any]) -> list[str]:
    """The supporting facts of a record, as the adapter reads them."""
    return [str(v).strip() for v in row["qa"].get("gold_inds", {}).values()]


def adapter_row(header: list[str], cells: list[str]) -> str:
    """A table row in the adapter's wording, as ``ucagent.tasks.finqa._facts`` writes it."""
    parts = [
        f"the {header[j]} of {cells[0]} is {cells[j]}"
        for j in range(1, min(len(cells), len(header)))
        if cells[j]
    ]
    return " ; ".join(parts) + " ;" if parts else ""


def wording(inst: Any, row: dict[str, Any]) -> dict[str, Any]:
    """What the audit reads off one question: its descriptions and the facts shown with it."""
    gold = [str(v).strip() for v in row["qa"].get("gold_inds", {}).values()]
    head = table_head(row)
    issues: set[str] = set()
    identical: set[str] = set()
    shows_number = negative = False
    for step, (_, args) in zip(inst.steps, program_ops(row["qa"]["program"]), strict=True):
        for arg, text in zip(args, (step.left, step.right), strict=True):
            if arg.startswith(("#", "const_")):
                continue
            issue = description_issue(arg, gold, head)
            if issue is not None:
                issues.add(issue)
            negative = negative or printed_negative(arg, gold)
            if _stand_alone(arg.strip().rstrip("%")).search(text):
                shows_number = True
        if step.left == step.right and step.left != "{prev}":
            a, b = (_value(x, []) for x in args)
            identical.add("the same number twice" if a == b else "two different operands")
    table = row.get("table") or []
    header = table[0] if table else []
    gold_rows = {int(k.split("_")[1]) for k in row["qa"].get("gold_inds", {}) if "table_" in k}
    other_rows = {
        index
        for index, cells in enumerate(table[1:], start=1)
        if adapter_row(header, cells) in inst.facts
    }
    return {
        "issues": issues,
        "identical": identical,
        "shows_number": shows_number,
        "negative": negative,
        "gold_row": bool(gold_rows),
        "beside": bool(gold_rows and other_rows),
        "repeated": bool(gold_rows & other_rows),
    }


def same_question(inst: Any) -> tuple[str, str, float]:
    """Page, question text and gold: two ids that agree on all three are one question."""
    text = re.sub(r"\W+", " ", inst.question.lower()).strip()
    return inst.qid.rsplit("-", 1)[0], text, round(float(inst.answer), 9)


def opposite_sign(answer: str | None, gold: float) -> bool:
    """The answer has the gold's size (within the relative tolerance) and the other sign."""
    parsed = parse_number("" if answer is None else str(answer))
    if parsed is None or gold == 0:
        return False
    value = parsed[0]
    return (value > 0) != (gold > 0) and abs(abs(value) - abs(gold)) <= NUMERIC_REL_TOL * abs(gold)


ISSUES = (
    'sentence cut at " is "',
    "blanked label fits another number",
    "cell holds several numbers",
    "label occurs with another value",
    "quoted fact fits another number",
)


def scale_part(
    task: Any, suite_dir: Path, out: list[str]
) -> tuple[dict[str, set[str]], dict[str, dict[str, str | None]]]:
    """Part 1 for the first protocol: finals accepted only through a scaled gold."""
    out += [
        "Final answers of the uncontrolled runs labelled correct only through a scaled gold:",
        "",
        "| split | wrong as labelled | accepted only through a scaled gold | percent forms "
        "| errors of scale or zero answers | wrong under the strict label |",
        "|---|---|---|---|---|---|",
    ]
    listing: list[str] = []
    wrong_ids: dict[str, set[str]] = {}
    finals: dict[str, dict[str, str | None]] = {}
    for split in ("calib", "test"):
        ((_, path, _),) = _runs(suite_dir, "none", split)
        trajs = read_trajectories(str(path))
        wrong = sum(not t.final_correct for t in trajs)
        wrong_ids[split] = {t.instance_id for t in trajs if not t.final_correct}
        finals[split] = {t.instance_id: t.final_answer for t in trajs}
        scaled = percent = strict_wrong = 0
        for t in trajs:
            inst = task.get_instance(t.instance_id)
            answer = "" if t.final_answer is None else str(t.final_answer)
            strict_wrong += not strict(answer, inst.answer, inst.question)
            how = branch(answer, inst.answer) if t.final_correct else None
            if how is None or how == "direct":
                continue
            scaled += 1
            is_percent = percent_form(answer, inst.answer, inst.question)
            percent += is_percent
            kind = "percent form" if is_percent else "scale or zero"
            listing.append(
                f"| {t.instance_id} | {kind} | {inst.answer:.6g} | {answer} "
                f"| {inst.question[:90]} |"
            )
        out.append(
            f"| {split} | {wrong} of {len(trajs)} | {scaled} | {percent} | {scaled - percent} "
            f"| {strict_wrong} of {len(trajs)} |"
        )
    out += [
        "",
        "| question | class | gold | answer | asked |",
        "|---|---|---|---|---|",
        *listing,
        "",
    ]

    return wrong_ids, finals


def strict_part(task: Any, args: argparse.Namespace, targets: list[float], out: list[str]) -> None:
    """Part 3 for the first protocol: selections under a final label strict about scale."""
    out += [
        "Learn-then-Test selections under the recorded final label and under the strict one "
        "(decisions and step labels as made; held-out error at cost):",
        "",
        "| suite | controller | target | as recorded | strict final label |",
        "|---|---|---|---|---|",
    ]
    for suite_dir in (args.suite_dir, *args.also):
        delta = float(pr._base(suite_dir).get("calibration", {}).get("delta", 0.1))
        for policy, name in POLICIES:
            if policy == "random_ask" and suite_dir != args.suite_dir:
                continue  # the same runs in every suite
            table: dict[float, dict[str, tuple[int, int, int, float]]] = {}
            for split in ("calib", "test"):
                for knob, path, cost in _runs(suite_dir, policy, split):
                    if knob is None:
                        continue
                    trajs = read_trajectories(str(path))
                    recorded = sum(not t.final_correct for t in trajs)
                    relabelled = sum(
                        not strict(t.final_answer, inst.answer, inst.question)
                        for t in trajs
                        for inst in [task.get_instance(t.instance_id)]
                    )
                    table.setdefault(knob, {})[split] = (recorded, relabelled, len(trajs), cost)
            for alpha in targets:
                cells = []
                for column in (0, 1):
                    points = [(k, v["calib"][2], v["calib"][column]) for k, v in table.items()]
                    knob_star = calibrate_knob(
                        policy,
                        points,
                        alpha=alpha,
                        delta=delta,
                        grid=configured_knobs(suite_dir, policy),
                    ).knob_star
                    if knob_star is None:
                        cells.append("not certified")
                        continue
                    errors, _, n, cost = (
                        table[knob_star]["test"][column],
                        None,
                        table[knob_star]["test"][2],
                        table[knob_star]["test"][3],
                    )
                    calib = table[knob_star]["calib"][column]
                    cells.append(
                        f"{errors / n:.3f} at {cost:.2f} (knob {knob_star:g}, {calib} "
                        "calibration errors)"
                    )
                out.append(f"| {suite_dir.name} | {name} | {alpha:g} | {cells[0]} | {cells[1]} |")


def forms_part(
    task: Any, suite_dir: Path, out: list[str]
) -> tuple[dict[str, set[str]], dict[str, dict[str, str | None]]]:
    """Part 1 for the second protocol: which accepted answers use a percentage form."""
    out += [
        "Uncontrolled runs under the second protocol. A final answer is accepted as the gold "
        "itself or, for a question that FinQA answers with a percentage, in the other form of "
        "the percentage; an answer with a percent sign may be the percent number or the "
        "fraction:",
        "",
        "| split | wrong | percentage questions | finals accepted in the other form of the "
        "percentage | of them with a percent sign | intermediate answers accepted through "
        "their percent sign | wrong answers the first protocol would have accepted |",
        "|---|---|---|---|---|---|---|",
    ]
    wrong_ids: dict[str, set[str]] = {}
    finals: dict[str, dict[str, str | None]] = {}
    for split in ("calib", "test"):
        ((_, path, _),) = _runs(suite_dir, "none", split)
        trajs = read_trajectories(str(path))
        wrong_ids[split] = {t.instance_id for t in trajs if not t.final_correct}
        finals[split] = {t.instance_id: t.final_answer for t in trajs}
        other = signed = middle = stricter = 0
        for t in trajs:
            inst = task.get_instance(t.instance_id)
            answer = "" if t.final_answer is None else str(t.final_answer)
            written = read_number(answer)
            if t.final_correct and written is not None and not _direct2(answer, inst.answer):
                other += 1
                signed += written.percent
            stricter += numbers_match(answer, inst.answer) and not t.final_correct
            for step in t.steps[:-1]:
                w = read_number(step.answer)
                if step.locally_correct and w is not None and w.percent:
                    middle += not _direct2(step.answer, inst.steps[step.step - 1].gold)
        percentage = sum(task.get_instance(t.instance_id).fraction is not None for t in trajs)
        out.append(
            f"| {split} | {len(wrong_ids[split])} of {len(trajs)} | {percentage} | {other} "
            f"| {signed} | {middle} | {stricter} |"
        )
    out.append("")
    return wrong_ids, finals


def _direct2(answer: str, gold: float) -> bool:
    """The number as written is the gold itself (no percent reading)."""
    written = read_number(answer)
    return written is not None and step_match(answer.replace("%", ""), gold)


def cluster_part(task: Any, suite_dir: Path, out: list[str]) -> None:
    """Part 4: steps on which the clustering judge and the label rule disagree."""
    out += [
        "",
        "Clusters against labels in the uncontrolled runs (on-track steps; the samples of a "
        "step are clustered by the estimator's judge and labelled by the task's rule):",
        "",
        "| split | steps | a cluster holds a right and a wrong answer | right answers lie in "
        "more than one cluster |",
        "|---|---|---|---|",
    ]
    for split in ("calib", "test"):
        ((_, path, _),) = _runs(suite_dir, "none", split)
        steps = mixed = spread = 0
        for t in read_trajectories(str(path)):
            inst = task.get_instance(t.instance_id)
            for step in t.steps:
                answers = t.samples.get(step.samples_key, [])
                if not answers or not task.has_local_truth(inst, step.step, step.context):
                    continue
                steps += 1
                right = [task.is_locally_correct(inst, step.step, step.context, a) for a in answers]
                kinds = [
                    {right[j] for j in cluster.members if j < len(right)}
                    for cluster in step.clusters
                ]
                mixed += any(len(k) == 2 for k in kinds)
                spread += sum(True in k for k in kinds) > 1
        out.append(f"| {split} | {steps} | {mixed} | {spread} |")
    out.append("")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--also", type=Path, nargs="*", default=[], help="more suites for part 3")
    parser.add_argument("--targets", default="0.1,0.05")
    args = parser.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    task: Any = make_task(pr._base(args.suite_dir)["task"])
    protocol = int(getattr(task, "protocol", 1))
    out = [f"### FinQA label audit ({args.suite_dir.name}, protocol {protocol})", ""]
    part = scale_part if protocol == 1 else forms_part
    wrong_ids, finals = part(task, args.suite_dir, out)

    # 2. how the steps and the facts are worded
    rows = raw_rows(Path(task.data_dir))
    read: dict[str, dict[str, dict[str, Any]]] = {}
    windows: dict[str, list[Any]] = {}
    for split in ("train", "calib", "test"):
        windows[split] = list(task.instances(split))
        read[split] = {i.instance_id: wording(i, rows[i.qid]) for i in windows[split]}

    def n_wrong(split: str, ids: set[str]) -> str:
        return "" if split == "train" else f"{len(ids & wrong_ids[split])}"

    out += [
        "Questions with a step description that does not single out its number among the gold "
        "facts, by cause (a question can have several), and how many of them are wrong without "
        "checks:",
        "",
        "| split | questions | with such a description | " + " | ".join(ISSUES) + " | wrong |",
        "|---|---|---|" + "---|" * (len(ISSUES) + 1),
    ]
    for split, per in read.items():
        flagged = {i for i, r in per.items() if r["issues"]}
        by = Counter(issue for r in per.values() for issue in r["issues"])
        cells = " | ".join(str(by[issue]) for issue in ISSUES)
        out.append(
            f"| {split} | {len(per)} | {len(flagged)} | {cells} | {n_wrong(split, flagged)} |"
        )
    usable = [row for row in rows.values() if convert(row) is not None]
    flagged_all = sum(
        any(
            description_issue(arg, gold, table_head(row)) is not None
            for _, operands in program_ops(row["qa"]["program"])
            for arg in operands
            if not arg.startswith(("#", "const_"))
        )
        for row in usable
        for gold in [[str(v).strip() for v in row["qa"].get("gold_inds", {}).values()]]
    )
    out += [
        "",
        f"Over all three FinQA files the screen flags {flagged_all} of the {len(usable)} usable "
        "questions.",
    ]
    out += [
        "",
        "Other properties of the wording (questions; wrong without checks in brackets):",
        "",
        "| split | a step with two identical descriptions | of these: two different operands "
        "| of these: the same number twice | a description shows its own number "
        "| a magnitude where the report prints a negative | of these: final answer of the right "
        "size and the opposite sign |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, per in read.items():

        def cell(ids: set[str], split: str = split) -> str:
            tail = "" if split == "train" else f" ({n_wrong(split, ids)})"
            return f"{len(ids)}{tail}"

        identical = {i for i, r in per.items() if r["identical"]}
        differ = {i for i, r in per.items() if "two different operands" in r["identical"]}
        shows = {i for i, r in per.items() if r["shows_number"]}
        negative = {i for i, r in per.items() if r["negative"]}
        by_sign = ""
        if split != "train":
            golds = {i.instance_id: float(i.answer) for i in windows[split]}
            by_sign = str(sum(opposite_sign(finals[split][i], golds[i]) for i in negative))
        out.append(
            f"| {split} | {cell(identical)} | {cell(differ)} | {cell(identical - differ)} "
            f"| {cell(shows)} | {cell(negative)} | {by_sign} |"
        )
    out += [
        "",
        "The supporting table row keeps FinQA's wording (the row of the column); the other rows "
        "of the table are in the adapter's wording (the column of the row):",
        "",
        "| split | questions | with a supporting table row | shown beside other rows of the "
        "table | the supporting row shown a second time |",
        "|---|---|---|---|---|",
    ]
    for split, per in read.items():
        values = list(per.values())
        out.append(
            f"| {split} | {len(values)} | {sum(r['gold_row'] for r in values)} "
            f"| {sum(r['beside'] for r in values)} | {sum(r['repeated'] for r in values)} |"
        )
    seen = Counter(same_question(i) for i in windows["calib"])
    both = [i.instance_id for i in windows["test"] if seen[same_question(i)]]
    out += [
        "",
        f"Questions of the test window that are also in the calibration window (same page, "
        f"question and gold under another id): {len(both)}"
        + (f" ({', '.join(both)})" if both else ""),
        "",
        "Questions of the calibration and test windows with a flagged description:",
        "",
        "| question | cause | wrong without checks |",
        "|---|---|---|",
    ]
    for split in ("calib", "test"):
        for iid, r in read[split].items():
            if r["issues"]:
                wrong = "yes" if iid in wrong_ids[split] else "no"
                out.append(f"| {iid} | {'; '.join(sorted(r['issues']))} | {wrong} |")
    out.append("")

    if protocol == 2:
        given = sum(
            any(
                text == arg.strip()
                for step, (_, operands) in zip(
                    i.steps, program_ops(rows[i.qid]["qa"]["program"]), strict=True
                )
                for arg, text in zip(operands, (step.left, step.right), strict=True)
                if not arg.startswith(("#", "const_"))
                and description_issue(arg, gold_facts(rows[i.qid]), table_head(rows[i.qid]))
            )
            for split in windows
            for i in windows[split]
        )
        out += [
            "Under the second protocol every description flagged above is replaced by the "
            "quoted fact with the number blanked, or by the number itself where two facts "
            f"differ in nothing else ({given} questions over the three windows).",
            "",
        ]
    if protocol == 1:
        strict_part(task, args, targets, out)
    cluster_part(task, args.suite_dir, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
