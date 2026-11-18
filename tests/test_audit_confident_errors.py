"""scripts/audit_confident_errors.py: verdicts kept across rewrites, checked and counted."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_confident_errors.py"
    spec = importlib.util.spec_from_file_location("audit_confident_errors", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rewriting_keeps_the_verdicts_by_set_question_and_hop(mod: ModuleType) -> None:
    old = [
        {"set": "s", "instance_id": "q1", "hop": "1", "verdict": "model right", "note": "a"},
        {"set": "s", "instance_id": "q1", "hop": "2", "verdict": "gold right", "note": "b"},
        {"set": "t", "instance_id": "q1", "hop": "1", "verdict": "neither", "note": "c"},
    ]
    new = [
        {"set": "s", "instance_id": "q1", "hop": 2, "verdict": "", "note": "", "x": 1},
        {"set": "s", "instance_id": "q2", "hop": 1, "verdict": "", "note": "", "x": 2},
    ]
    merged = mod.merge_verdicts(new, old)
    assert [(r["verdict"], r["note"], r["x"]) for r in merged] == [
        ("gold right", "b", 1),
        ("", "", 2),
    ]
    # question-level rows have no hop
    q_old = [{"set": "f", "instance_id": "q9", "verdict": "ambiguous", "note": "n"}]
    q_new = [{"set": "f", "instance_id": "q9", "verdict": "", "note": ""}]
    assert mod.merge_verdicts(q_new, q_old)[0]["verdict"] == "ambiguous"


def test_unknown_verdicts_are_refused_and_empty_ones_count_as_not_read(mod: ModuleType) -> None:
    rows = [{"verdict": "model right"}, {"verdict": ""}, {"verdict": "ambiguous"}]
    mod.check_verdicts(rows)
    assert mod.verdict_counts(rows) == {"model right": 1, "not read": 1, "ambiguous": 1}
    with pytest.raises(SystemExit, match="unknown verdicts"):
        mod.check_verdicts([*rows, {"verdict": "probably fine"}])


def test_the_floor_on_a_fair_reading(mod: ModuleType) -> None:
    rows = [
        {"verdict": "model right"},
        {"verdict": "both acceptable"},
        {"verdict": "ambiguous"},  # not counted as right: the question admits both readings
        {"verdict": "gold right"},
        {"verdict": ""},
    ]
    assert mod.fair_floor(rows) == (4, 2, 3)  # read, right on a fair reading, floor left


def test_auroc_against_all_labelled_errors_and_against_model_errors(mod: ModuleType) -> None:
    right = [0.0, 0.0, 0.1, 0.2]
    wrong = [(0.3, "model right"), (0.15, "gold right"), (0.05, "both acceptable")]
    every, errors = mod.auroc_two_ways(right, wrong)
    # every wrong hop: 0.3 beats all 4, 0.15 beats 3, 0.05 beats 2 of the right scores
    assert every == pytest.approx((4 + 3 + 2) / 12)
    assert errors == pytest.approx(3 / 4)  # the model error alone, 0.15
    assert mod.auroc_two_ways(right, [(0.3, "model right")])[1] is None
    assert mod.auroc_two_ways([], wrong) == (None, None)


def test_the_table_lists_the_verdicts_that_occur(mod: ModuleType) -> None:
    counts = {"a": mod.verdict_counts([{"verdict": "model right"}, {"verdict": "gold right"}])}
    lines = mod.tex_table(counts, {"a": 2})
    assert lines[0].startswith("% written by scripts/audit_confident_errors.py")
    assert "set & rows & model right & gold right \\\\" in lines
    assert "a & 2 & 1 & 1 \\\\" in lines


def test_an_answer_accepted_only_by_containment_is_told_from_the_gold(mod: ModuleType) -> None:
    from types import SimpleNamespace

    class Task:
        """The label rule accepts an answer that holds a gold word; exact match needs equality."""

        def get_instance(self, instance_id: str) -> SimpleNamespace:
            return SimpleNamespace(n_steps=2)

        def _truths(self, inst: object, step: int) -> set[str]:
            return {"adams county", "adams county nebraska"}

        def _matches(self, answer: str, truths: set[str], rule: str | None = None) -> bool:
            if answer.lower() in truths:
                return True
            return rule != "exact" and any(answer.lower() in t.split() for t in truths)

        def is_final_correct(self, inst: object, answer: str | None) -> bool:
            return answer is not None and self._matches(answer, self._truths(inst, 2), "contains")

    def run(final: str | None) -> SimpleNamespace:
        return SimpleNamespace(instance_id="q", final_answer=final)

    task = Task()
    assert mod.accepted_only_by_containment(task, run("Nebraska"))  # a state for a county
    assert not mod.accepted_only_by_containment(task, run("Adams County"))  # the gold itself
    assert not mod.accepted_only_by_containment(task, run("Kansas"))  # rejected: not this list
    assert not mod.accepted_only_by_containment(task, run(None))
