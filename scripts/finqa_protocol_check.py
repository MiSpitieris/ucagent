"""What FinQA's second protocol changes, measured before any run of it.

Usage::

    uv run python scripts/finqa_protocol_check.py results/finqa_confagree
    uv run python scripts/finqa_protocol_check.py results/finqa_confagree \\
        --second results/finqa2_confagree

The suite directory is a run of the first protocol; its task block, with ``protocol: 2``
added, gives the second. The script prints, without model calls:

1. Evidence for the rule about percentages, from the train file only (the fit pool; the
   rule was not tuned on the calibration and test questions): how many questions FinQA
   itself answers with a percentage and which form the program's result has; how many of
   FinQA's own answer strings each protocol's final rule accepts; and, for every percentage
   question, whether the right forms of the answer are accepted and forms off by a factor of
   100 rejected.
2. For the fit, calibration and test windows of the suite: that both protocols load the same
   questions with the same facts and gold values in the same order; how many questions get
   another description and how many first prompts differ; how many questions are percentage
   questions; and how many of the simulated reviewer's answers each protocol's rule rejects
   along the gold chain.
3. The recorded uncontrolled runs labelled again by the second protocol, with the answers
   and decisions as made: final errors, and intermediate steps labelled correct. This is a
   relabel of the first run, not a run of the second protocol.
4. With ``--second``, a suite of the second protocol on the same questions: the uncontrolled
   runs of both, apart for the questions whose step descriptions the second protocol replaces
   and for the others (how many final answers are the same in both runs, and how many are
   wrong in each); and, for the questions with replaced descriptions that the second suite
   gets wrong without checks, how many get a check and how many stay wrong at the
   Learn-then-Test selections of the gate and the rule (``--targets``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.llm.cache import canonical_messages
from ucagent.experiments.io import read_trajectories
from ucagent.tasks import make_task
from ucagent.tasks.finqa import (
    OPS,
    convert,
    execute,
    final_match,
    numbers_match,
    percentage_fraction,
    program_ops,
    read_number,
)

SPLITS = ("train", "calib", "test")


def train_evidence(data_dir: Path) -> list[str]:
    """Part 1: the percentage rule against FinQA's own answers, on the train file."""
    rows = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    c: Counter[str] = Counter()
    for row in rows:
        chain = convert(row)
        if chain is None:
            continue
        gold = float(chain["answer"])
        text = str(row["qa"].get("answer", "")).strip()
        fraction = percentage_fraction(text, gold, program_ops(row["qa"]["program"]))
        c["usable"] += 1
        if fraction is not None:
            c["percentage"] += 1
            c["result is the fraction" if fraction == gold else "result is the percent number"] += 1
            if fraction != 0:
                percent = 100.0 * fraction
                right = (f"{percent:.4f}%", f"{percent:.4f}", f"{fraction:.6f}")
                wrong = (
                    f"{100 * percent:.4f}%",  # a hundred times too large, with a sign
                    f"{100 * percent:.4f}",  # and without
                    f"{fraction / 100:.8f}",  # a hundred times too small
                    f"{fraction:.6f}%",  # the fraction with a percent sign
                )
                c["right forms"] += len(right)
                c["right forms accepted"] += sum(final_match(x, gold, fraction) for x in right)
                c["wrong forms"] += len(wrong)
                c["wrong forms accepted"] += sum(final_match(x, gold, fraction) for x in wrong)
        if read_number(text) is None:
            c["no number in FinQA's answer"] += 1
            continue
        old, new = numbers_match(text, gold), final_match(text, gold, fraction)
        c["own answers accepted, protocol 1"] += old
        c["own answers accepted, protocol 2"] += new
        c["own answers accepted by 1 only"] += old and not new
        c["own answers accepted by 2 only"] += new and not old
    with_number = c["usable"] - c["no number in FinQA's answer"]
    return [
        "Evidence from the train file (the fit pool) for the rule about percentages:",
        "",
        f"- Usable questions: {c['usable']}. FinQA answers {c['percentage']} of them with a "
        f"percentage; the program's result is the fraction for {c['result is the fraction']} "
        f"and the percent number for {c['result is the percent number']}.",
        f"- FinQA's own answer strings read as answers ({with_number} hold a number): the first "
        f"protocol's rule accepts {c['own answers accepted, protocol 1']}, the second's "
        f"{c['own answers accepted, protocol 2']} ({c['own answers accepted by 1 only']} "
        f"accepted by the first only, {c['own answers accepted by 2 only']} by the second only).",
        f"- For every percentage question with a gold other than zero: of {c['right forms']} "
        "right forms of the answer (the percent number with and without a sign, the fraction) "
        f"the second protocol accepts {c['right forms accepted']}; of {c['wrong forms']} forms "
        "off by a factor of 100 (too large with and without a sign, too small, the fraction "
        f"with a sign) it accepts {c['wrong forms accepted']}.",
        "",
    ]


def exponent_skips(data_dir: Path) -> dict[str, int]:
    """Linear programs that neither protocol loads only because Python writes their result
    with an exponent, which the admission test's parser reads as the mantissa.

    Both protocols keep that test as it is: reading the exponent there would admit these
    questions, change the list that is shuffled, and so move every question window.
    """
    out = {}
    for name in ("train", "dev", "test"):
        rows = json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))
        n = 0
        for row in rows:
            if convert(row) is not None:
                continue
            ops = program_ops(row["qa"]["program"])
            if not 2 <= len(ops) <= 4 or any(op not in OPS or len(a) != 2 for op, a in ops):
                continue
            if any(
                [a for a in args if a.startswith("#")] != ([f"#{k - 1}"] if k else [])
                for k, (_, args) in enumerate(ops)
            ):
                continue
            try:
                results = execute(ops)
                expected = float(row["qa"]["exe_ans"])
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            if not all(math.isfinite(r) for r in results) or "e" not in f"{results[-1]}":
                continue
            n += numbers_match(format(results[-1], ".12f"), expected)
        out[name] = n
    return out


def gold_chain_rejections(task: Any, instances: list[Any]) -> int:
    """Answers of the simulated reviewer along the gold chain that the task's rule rejects."""
    rejected = 0
    for inst in instances:
        previous = ""
        for step in range(1, inst.n_steps + 1):
            answer = task.local_truth(inst, step, previous)
            if step < inst.n_steps:
                rejected += not task.has_local_truth(inst, step + 1, answer)
            else:
                rejected += not task.is_final_correct(inst, answer)
            previous = answer
    return rejected


def windows(one: Any, two: Any) -> list[str]:
    """Part 2: the windows under both protocols."""
    out = [
        "The windows under both protocols:",
        "",
        "| split | questions | same questions, facts and gold values in the same order "
        "| questions with another description | step questions that change | first prompts "
        "that differ | percentage questions | reviewer answers rejected, protocol 1 | "
        "protocol 2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        a, b = list(one.instances(split)), list(two.instances(split))

        def content(insts: list[Any]) -> list[Any]:
            return [
                (i.instance_id, i.qid, i.question, i.facts, [s.gold for s in i.steps], i.answer)
                for i in insts
            ]

        described = sum(
            [(s.left, s.right) for s in x.steps] != [(s.left, s.right) for s in y.steps]
            for x, y in zip(a, b, strict=True)
        )
        steps = sum(x.n_steps for x in a)
        step_texts = sum(
            (s.left, s.right) != (t.left, t.right)
            for x, y in zip(a, b, strict=True)
            for s, t in zip(x.steps, y.steps, strict=True)
        )
        prompts = sum(
            canonical_messages(one.step_prompt(x, 1, [], 0))
            != canonical_messages(two.step_prompt(y, 1, [], 0))
            for x, y in zip(a, b, strict=True)
        )
        percentage = sum(y.fraction is not None for y in b)
        out.append(
            f"| {split} | {len(a)} | {'yes' if content(a) == content(b) else 'NO'} | {described} "
            f"| {step_texts} of {steps} | {prompts} | {percentage} "
            f"| {gold_chain_rejections(one, a)} | {gold_chain_rejections(two, b)} |"
        )
    skips = exponent_skips(Path(one.data_dir))
    out += [
        "",
        "Questions that neither protocol loads only because the program's result is written "
        "with an exponent (the admission test is the same under both, so the windows are too): "
        + ", ".join(f"{n} in {name}.json" for name, n in skips.items())
        + ".",
    ]
    return [*out, ""]


def _run(suite_dir: Path, split: str, policy: str = "none", knob: float | None = None) -> Any:
    """The trajectories of one run of a suite (the uncontrolled one by default), by question."""
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == split)]
    if knob is not None:
        rows = rows[(rows["knob"] - knob).abs() < 1e-9]
    path = Path(str(rows["path"].iloc[0]))
    path = path if path.is_absolute() else suite_dir.parent.parent / path
    return {t.instance_id: t for t in read_trajectories(str(path))}


def _replaced(one: Any, two: Any, split: str) -> set[str]:
    """The questions of a split whose step descriptions the second protocol replaces."""
    return {
        x.instance_id
        for x, y in zip(one.instances(split), two.instances(split), strict=True)
        if [(s.left, s.right) for s in x.steps] != [(s.left, s.right) for s in y.steps]
    }


def relabel(suite_dir: Path, one: Any, two: Any) -> list[str]:
    """Part 3: the recorded uncontrolled runs under the second protocol's labels."""
    out = [
        "The recorded uncontrolled runs labelled by the second protocol (answers and decisions "
        "as made; a relabel, not a run):",
        "",
        "| split | final answers wrong, as recorded | under the second protocol | intermediate "
        "model steps | labelled correct, as recorded | under the second protocol |",
        "|---|---|---|---|---|---|",
    ]
    moved: list[str] = []
    replaced: list[str] = []
    for split in ("calib", "test"):
        trajs = list(_run(suite_dir, split).values())
        recorded = sum(not t.final_correct for t in trajs)
        second = sum(
            not two.is_final_correct(two.get_instance(t.instance_id), t.final_answer) for t in trajs
        )
        steps = old_ok = new_ok = 0
        touched = _replaced(one, two, split)
        wrong_then = sum(not t.final_correct for t in trajs if t.instance_id in touched)
        wrong_now = sum(
            not two.is_final_correct(two.get_instance(t.instance_id), t.final_answer)
            for t in trajs
            if t.instance_id in touched
        )
        replaced.append(
            f"{split}: {len(touched)} questions, {wrong_then} wrong as recorded, {wrong_now} by "
            "the second protocol's label"
        )
        for t in trajs:
            inst = two.get_instance(t.instance_id)
            if t.final_correct != two.is_final_correct(inst, t.final_answer):
                moved.append(
                    f"| {t.instance_id} | final | {t.final_answer} | {inst.answer:.6g} "
                    f"| {'right' if t.final_correct else 'wrong'} |"
                )
            for s in t.steps:
                if s.step == t.n_steps or s.source != "llm":
                    continue
                steps += 1
                old_ok += bool(s.locally_correct)
                now = two.is_locally_correct(inst, s.step, s.context, s.answer)
                new_ok += now
                if bool(s.locally_correct) != now:
                    moved.append(
                        f"| {t.instance_id} | step {s.step} | {s.answer} "
                        f"| {inst.steps[s.step - 1].gold:.6g} "
                        f"| {'right' if s.locally_correct else 'wrong'} |"
                    )
        out.append(
            f"| {split} | {recorded} of {len(trajs)} | {second} of {len(trajs)} | {steps} "
            f"| {old_ok} | {new_ok} |"
        )
    out += [
        "",
        "Answers whose label moves:",
        "",
        "| question | answer to | answer | gold | as recorded |",
        "|---|---|---|---|---|",
        *moved,
        "",
        "Questions whose step descriptions the second protocol replaces, in these recorded runs "
        "(answers as given to the first protocol's descriptions): " + "; ".join(replaced) + ".",
    ]
    return [*out, ""]


def both_runs(
    first_dir: Path, second_dir: Path, one: Any, two: Any, targets: list[float]
) -> list[str]:
    """Part 4: the uncontrolled runs of both protocols, by whether a question's step
    descriptions are replaced, with every answer labelled by the second protocol; and what
    the second suite's certified controllers do with the errors on the replaced ones."""
    out = [
        f"The uncontrolled runs of {first_dir.name} (first protocol) and {second_dir.name} "
        "(second protocol), by whether the second protocol replaces the question's step "
        "descriptions. Final answers of both runs labelled by the second protocol:",
        "",
        "| split | step descriptions | questions | same final answer in both runs | wrong, "
        "first run | wrong, second run |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("calib", "test"):
        first, second = _run(first_dir, split), _run(second_dir, split)
        if set(first) != set(second):
            raise SystemExit(f"{split}: the two suites do not hold the same questions")
        touched = _replaced(one, two, split)
        for name, ids in (("replaced", touched), ("unchanged", set(first) - touched)):
            same = sum(first[i].final_answer == second[i].final_answer for i in ids)
            wrong = [
                sum(not two.is_final_correct(two.get_instance(i), run[i].final_answer) for i in ids)
                for run in (first, second)
            ]
            out.append(f"| {split} | {name} | {len(ids)} | {same} | {wrong[0]} | {wrong[1]} |")
    out += [
        "",
        f"The questions with replaced descriptions that {second_dir.name} gets wrong without "
        "checks, at the Learn-then-Test selections of its gate and its rule:",
        "",
        "| split | wrong without checks | selection | of them with a step checked | still wrong |",
        "|---|---|---|---|---|",
    ]
    for split in ("calib", "test"):
        base = _run(second_dir, split)
        wrong_ids = sorted(i for i in _replaced(one, two, split) if not base[i].final_correct)
        for policy, name in (("per_step_eps", "gate"), ("marginal", "rule")):
            for alpha in targets:
                sel = pr.select(second_dir, policy, alpha)
                if not sel.feasible or sel.knob is None:
                    continue
                run = _run(second_dir, split, policy, sel.knob)
                checked = sum(any(s.source == "human" for s in run[i].steps) for i in wrong_ids)
                still = sum(not run[i].final_correct for i in wrong_ids)
                out.append(
                    f"| {split} | {len(wrong_ids)} | {name} certified at {alpha:g} "
                    f"(knob {sel.knob:g}) | {checked} | {still} |"
                )
    return [*out, ""]


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path, help="a suite of the first protocol")
    parser.add_argument(
        "--second", type=Path, default=None, help="a suite of the second protocol, same questions"
    )
    parser.add_argument("--targets", default="0.1,0.05", help="targets read with --second")
    args = parser.parse_args()
    block = dict(pr._base(args.suite_dir)["task"])
    if int(block.get("protocol", 1)) != 1:
        raise SystemExit(f"{args.suite_dir.name} is not a suite of the first protocol")
    one: Any = make_task(block)
    two: Any = make_task({**block, "protocol": 2})
    out = [f"### FinQA, second protocol against the first ({args.suite_dir.name})", ""]
    out += train_evidence(Path(one.data_dir))
    out += windows(one, two)
    out += relabel(args.suite_dir, one, two)
    if args.second is not None:
        if int(dict(pr._base(args.second)["task"]).get("protocol", 1)) != 2:
            raise SystemExit(f"{args.second.name} is not a suite of the second protocol")
        targets = [float(x) for x in args.targets.split(",")]
        out += both_runs(args.suite_dir, args.second, one, two, targets)
    print("\n".join(out))


if __name__ == "__main__":
    main()
