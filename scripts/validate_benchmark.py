"""Validate every question a set of configs will run, before any model call.

Usage::

    uv run python scripts/validate_benchmark.py configs/twowiki_confagree.yaml \\
        configs/twowiki_split2_confagree.yaml --disjoint configs/twowiki_smoke.yaml

For each config (a suite or a single run) the task is built from its ``task`` block and
every instance of every split window is checked:

- every hop has a non-empty gold answer and its supporting paragraph is among its passages;
- the gold answers label themselves correct (hop 1, hop 2 on track, final answer);
- the rendered sub-questions resolve their references and use no generic fallback;
- calibration and test windows of different configs that should be fresh do not overlap
  (``--disjoint``: configs whose questions must not appear in any other config's
  calibration or test windows, such as the smoke run). Questions are compared by the
  dataset's own id, not by run id: a run id encodes split, index and seed, so one record
  can stand under two of them (code review of 2026-11-11, C06). Question texts that
  recur under another source id are reported as well;
- with ``--flag-free calib test``, no question of those splits has a flag of the data
  filter (``ucagent.tasks.twowiki.question_flags``), as split 3 requires.

Informational: the share of hops whose gold answer appears in its supporting paragraph, and
the share whose support is doubtful (``ucagent.tasks.twowiki.support_flags``). String presence
is not support: the 2WikiMultiHopQA audit of 2026-11-09 found golds that occur in the passage
without being stated by it ("Madison" as a birthplace from a partner's name).
Exits with status 1 on any violation. Makes no model calls.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ucagent.core.records import StepRecord
from ucagent.tasks import make_task
from ucagent.tasks.twowiki import question_flags, support_flags, uses_fallback

TEXTS: dict[Path, dict[str, set[str]]] = {}  # question texts per config and split


def _task_block_and_splits(path: Path) -> tuple[dict[str, Any], list[str]]:
    """The task block and the splits the config runs (a suite: the fit's train split too)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("kind") == "suite":
        return dict(data["base"]["task"]), ["train", *data.get("splits", ["calib", "test"])]
    return dict(data["task"]), [str(data.get("split", "test"))]


def _rec(answer: str) -> StepRecord:
    return StepRecord(step=1, attempt=0, context="", answer=answer, u=0.0, entropy=0.0, eps=0.0)


def question_text(text: str) -> str:
    """A question's text without case, punctuation and spacing."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def check_config(
    path: Path, flag_free: frozenset[str] = frozenset()
) -> tuple[dict[str, set[str]], list[str], Counter[str]]:
    """Source question ids per split, violations and counters for one config.

    The question texts per split are kept in ``TEXTS[path]``, for the report of wordings
    that recur under another source id.
    """
    block, splits = _task_block_and_splits(path)
    task: Any = make_task(block)
    ids: dict[str, set[str]] = {}
    problems: list[str] = []
    counts: Counter[str] = Counter()
    for split in splits:
        if task.sizes.get(split, 0) <= 0:
            continue
        ids[split] = set()
        TEXTS.setdefault(path, {})[split] = set()
        for inst in task.instances(split):
            ids[split].add(str(getattr(inst, "qid", inst.instance_id)))
            TEXTS[path][split].add(question_text(inst.question))
            counts["instances"] += 1
            golds = [h.answer for h in inst.hops]
            if any(not g.strip() for g in golds):
                problems.append(f"{inst.instance_id}: empty gold answer")
                continue
            if split in flag_free:
                flags = question_flags(inst)
                if flags:
                    problems.append(f"{inst.instance_id}: data-filter flags {flags}")
            current = ""
            answers: dict[int, str] = {}
            for i, gold in enumerate(golds, start=1):
                counts["hops"] += 1
                hop = inst.hops[i - 1]
                shown = {p.idx for p in task.passages(inst, i)}
                if hop.support_idx not in shown:
                    problems.append(f"{inst.instance_id} hop {i}: supporting paragraph not shown")
                if not task.is_locally_correct(inst, i, current, gold):
                    problems.append(f"{inst.instance_id} hop {i}: gold {gold!r} labelled wrong")
                question = task.sub_question(inst, i, answers)
                if "#" in question and any(f"#{k}" in question for k in range(1, 10)):
                    problems.append(f"{inst.instance_id} hop {i}: unresolved reference")
                relation = hop.template.split(">>")[-1].strip() if ">>" in hop.template else ""
                if relation:
                    counts[f"relation:{relation}"] += 1
                    if uses_fallback(relation):
                        problems.append(f"{inst.instance_id} hop {i}: no template for {relation!r}")
                support = next(p for p in inst.paragraphs if p.idx == hop.support_idx)
                if gold.lower() in support.text.lower():
                    counts["gold_in_paragraph"] += 1
                if relation and support_flags(relation, gold, support.text):
                    counts["support_flagged"] += 1
                answers[i] = gold
                current = gold
            if not task.is_final_correct(inst, golds[-1]):
                problems.append(f"{inst.instance_id}: final gold labelled wrong")
    return ids, problems, counts


def shared(
    mine: dict[str, set[str]],
    other: dict[str, set[str]],
    my_texts: dict[str, set[str]],
    other_texts: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """Source ids and question texts of ``mine`` among the evaluation questions of ``other``."""
    fresh = other.get("calib", set()) | other.get("test", set())
    fresh_texts = other_texts.get("calib", set()) | other_texts.get("test", set())
    return set().union(*mine.values()) & fresh, set().union(*my_texts.values()) & fresh_texts


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--disjoint", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--flag-free", nargs="*", default=[], help="splits of the configs that must pass the filter"
    )
    args = parser.parse_args()
    all_ids: dict[Path, dict[str, set[str]]] = {}
    failed = False
    for path in [*args.configs, *args.disjoint]:
        free = frozenset(args.flag_free) if path in args.configs else frozenset()
        ids, problems, counts = check_config(path, free)
        all_ids[path] = ids
        hops = counts["hops"] or 1
        print(
            f"{path}: {counts['instances']} instances, {counts['hops']} hops, "
            f"gold found verbatim in its paragraph for {counts['gold_in_paragraph'] / hops:.2f} "
            f"of hops, but its support doubtful (support_flags) for "
            f"{counts['support_flagged'] / hops:.2f}; {len(problems)} problems"
        )
        rel = sorted(
            ((k.split(":", 1)[1], v) for k, v in counts.items() if k.startswith("relation:")),
            key=lambda kv: -kv[1],
        )
        print(
            "  relations:",
            ", ".join(f"{r} {n}" for r, n in rel[:12]),
            "..." if len(rel) > 12 else "",
        )
        for line in problems[:20]:
            print("  PROBLEM", line)
        failed |= bool(problems)
    for path in args.disjoint:
        for other in args.configs:
            clash, same_text = shared(all_ids[path], all_ids[other], TEXTS[path], TEXTS[other])
            if clash:
                print(
                    f"PROBLEM {path} overlaps the calibration or test windows of {other}: "
                    f"{len(clash)} source questions"
                )
                failed = True
            if len(same_text) > len(clash):
                print(
                    f"  note: {len(same_text)} question texts of {path} recur in the calibration "
                    f"or test windows of {other} ({len(clash)} under the same source id)"
                )
    print("validation", "FAILED" if failed else "passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
