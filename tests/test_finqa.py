"""FinQA adapter on a three-record fixture (no large files needed)."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from ucagent.api import run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import HEADER_RE, MockChainLLM
from ucagent.core.records import StepRecord
from ucagent.core.uq.equivalence import make_judge
from ucagent.tasks import make_task
from ucagent.tasks.base import make_human
from ucagent.tasks.finqa import FinqaTask, convert, numbers_match

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "finqa_mini.json"


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def task(tmp_path: Path) -> FinqaTask:
    for name in ("train.json", "dev.json", "test.json"):
        shutil.copy(FIXTURE, tmp_path / name)
    return FinqaTask(
        data_dir=tmp_path, n_instances={"train": 2, "calib": 1, "test": 1}, n_distractors=2
    )


def test_convert_keeps_linear_chains_and_executes_them() -> None:
    table_row, text_row, single = _rows()
    chain = convert(table_row)
    assert chain is not None and [s["op"] for s in chain["steps"]] == ["divide", "multiply"]
    assert chain["steps"][0]["gold"] == pytest.approx(60 / 243)
    assert chain["steps"][1]["left"] == "{prev}" and chain["steps"][1]["right"] == "100"
    assert chain["answer"] == pytest.approx(24.69136, rel=1e-4)
    assert convert(text_row) is not None and convert(single) is None
    fan_in = copy.deepcopy(table_row)
    fan_in["qa"]["program"] = "divide(60, 243), divide(#0, #0)"
    assert convert(fan_in) is None
    for row in (table_row, text_row):
        c = convert(row)
        literals = re.findall(r"\((\d[\d.]*),\s*(\d[\d.]*)\)", row["qa"]["program"])
        descs = " ".join(d for s in c["steps"] for d in (s["left"], s["right"]) if d != "{prev}")
        for lit in {x for pair in literals for x in pair}:
            if any(d == lit for s in c["steps"] for d in (s["left"], s["right"])):
                continue  # a bare constant of the program
            assert not re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d.])", descs), lit


def test_numeric_labels_and_judge() -> None:
    assert numbers_match("24.69", 24.69136) and numbers_match("24.7", 24.69136)
    assert numbers_match("24.69%", 0.2469136) and numbers_match("0.2469", 24.69136 / 100)
    assert numbers_match("$1,000", 1000.0) and numbers_match("(5.2)", -5.2)
    assert not numbers_match("30", 24.69136) and not numbers_match("unknown", 1.0)
    judge = make_judge("numeric")
    assert judge.same("24.69", "24.7", None) and judge.same("1,000", "1000", None)
    assert not judge.same("24.69", "26", None) and judge.same("Unknown", "unknown", None)


def test_task_ids_prompt_substitution_and_labels(task: FinqaTask) -> None:
    ids = [i.instance_id for i in task.instances("train")]
    assert ids == ["finqa-train-00000-s0", "finqa-train-00001-s0"]
    assert isinstance(make_task({"name": "finqa", "data_dir": str(task.data_dir)}), FinqaTask)
    inst = next(i for i in (task.get_instance(x) for x in ids) if i.steps[1].right == "100")
    hist = [StepRecord(step=1, attempt=0, context="", answer="0.2469", u=0, entropy=0, eps=0)]
    text = task.step_prompt(inst, 2, hist, attempt=0)[1].content
    assert HEADER_RE.match(text.splitlines()[0])
    assert "Step 2 of 2 (answer this one): What is 0.2469 times 100?" in text
    assert task.has_local_truth(inst, 2, "0.2469") and not task.has_local_truth(inst, 2, "0.5")
    assert task.is_locally_correct(inst, 2, "0.2469", "24.69")
    assert task.is_final_correct(inst, "24.69") and not task.is_final_correct(inst, "2.5")
    assert task.parse_answer("Answer: 24.69\nConfidence: 0.9") == "24.69"


def test_mock_runs_end_to_end(task: FinqaTask) -> None:
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(3.0, 2.0))
    settings = RunSettings(policy="per_step_u", knob=0.0, K=3, K_prime=6)
    ids = [i.instance_id for i in task.instances("train")]
    trajectories = run_instances(task, llm, settings, ids, make_human(task, settings))
    assert len(trajectories) == 2 and all(len(t.steps) == 2 for t in trajectories)
