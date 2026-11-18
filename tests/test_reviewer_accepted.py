"""The label rule must accept the answers the simulated reviewer writes.

A checked step that is on track is committed with ``task.local_truth``. If the task's own rule
then rejects that answer, as the previous answer of the next step or as the final answer, a
check silently stops working: the next step counts as off track, or a corrected final answer
counts as wrong. The FinQA runs of 2026-11-09 had this defect for values of a million or more
(``docs/EXPERIMENTS.md``, review of 2026-11-11). FinQA's first protocol is kept as recorded,
defect included, so that its numbers reproduce; the second protocol repairs it, and the two
tests at the bottom say both.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ucagent.tasks.finqa import FinqaTask, numbers_match
from ucagent.tasks.musique import MusiqueTask
from ucagent.tasks.twowiki import TwoWikiTask, chain_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def rejected(task: Any, splits: tuple[str, ...] = ("train", "calib", "test")) -> list[str]:
    """Reviewer-written answers of the gold chain that the task's label rule rejects."""
    out = []
    for split in splits:
        for inst in task.instances(split):
            n, previous = task.n_steps(inst), ""
            for step in range(1, n + 1):
                answer = task.local_truth(inst, step, previous)
                if step < n:
                    ok = task.has_local_truth(inst, step + 1, answer)
                else:
                    ok = task.is_final_correct(inst, answer)
                if not ok:
                    out.append(f"{inst.instance_id} step {step}: {answer!r}")
                previous = answer
    return out


def test_musique_rule_accepts_the_reviewer() -> None:
    task = MusiqueTask(
        data_dir=FIXTURES,
        train_file="musique_mini.jsonl",
        dev_file="musique_mini.jsonl",
        n_instances={"train": 3, "calib": 1, "test": 1},
        n_distractors=2,
        local_match="contains",
        final_match="contains",
    )
    assert rejected(task) == []


def test_twowiki_rule_accepts_the_reviewer(tmp_path: Path) -> None:
    raw = [
        json.loads(line)
        for line in (FIXTURES / "twowiki_mini.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with open(tmp_path / "mini.jsonl", "w", encoding="utf-8") as fh:
        for row in chain_rows(raw, exclude_relations=()):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    task = TwoWikiTask(
        data_dir=tmp_path,
        train_file="mini.jsonl",
        dev_file="mini.jsonl",
        n_instances={"train": 3, "calib": 1, "test": 1},
        n_distractors=2,
        local_match="contains",
        final_match="contains",
    )
    assert rejected(task) == []


def test_finqa_rule_accepts_the_reviewer_on_the_fixture(tmp_path: Path) -> None:
    for name in ("train.json", "dev.json", "test.json"):
        shutil.copy(FIXTURES / "finqa_mini.json", tmp_path / name)
    task = FinqaTask(
        data_dir=tmp_path, n_instances={"train": 2, "calib": 1, "test": 1}, n_distractors=2
    )
    assert rejected(task) == []  # the fixture has no value of a million or more


def test_finqa_first_protocol_keeps_the_recorded_defect() -> None:
    gold = 70_000_000.0
    written = f"{gold:.6g}"  # what protocol 1's reviewer writes: '7e+07'
    assert not numbers_match(written, gold)  # read as 7: the defect of the runs of 2026-11-09


def test_finqa_second_protocol_accepts_the_reviewer(tmp_path: Path) -> None:
    for name in ("train.json", "dev.json", "test.json"):
        shutil.copy(FIXTURES / "finqa_protocol.json", tmp_path / name)  # has values of 7e+07
    sizes = {"train": 6, "calib": 3, "test": 3}
    first = FinqaTask(data_dir=tmp_path, n_instances=sizes, n_distractors=2)
    second = FinqaTask(data_dir=tmp_path, n_instances=sizes, n_distractors=2, protocol=2)
    assert rejected(first) != [] and rejected(second) == []
