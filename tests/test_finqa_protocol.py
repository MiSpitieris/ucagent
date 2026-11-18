"""FinQA's second protocol: the repaired label rule, reviewer and step descriptions.

Protocol 1 is the adapter as first run and stays as it was, defects included, so that the
recorded numbers reproduce. Protocol 2 repairs the three defects the reviews of 2026-11-11
found (``docs/EXPERIMENTS.md``). The tests state what must differ between the two and, as
important for a rerun that replays cached samples, what must not: the questions loaded, their
order, the facts shown, and the prompts of every question no defect touches.

``fixtures/finqa_protocol.json`` is written by hand in FinQA's format, after cases met in the
data; it is not FinQA data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ucagent.api import run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.cache import canonical_messages
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import StepRecord
from ucagent.core.uq.equivalence import parse_number
from ucagent.tasks import make_task
from ucagent.tasks.base import make_human
from ucagent.tasks.finqa import (
    FinqaTask,
    _other_number_fits,
    _stand_alone,
    description_issue,
    final_match,
    numbers_match,
    percentage_fraction,
    plain_number,
    read_number,
    step_match,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "finqa_protocol.json"
REAL = Path("data/finqa")
SIZES = {"train": 6, "calib": 3, "test": 3}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for name in ("train.json", "dev.json", "test.json"):
        shutil.copy(FIXTURE, tmp_path / name)
    return tmp_path


def tasks(data_dir: Path, **kw) -> tuple[FinqaTask, FinqaTask]:
    base = {"data_dir": data_dir, "n_instances": SIZES, "n_distractors": 2, **kw}
    return FinqaTask(**base), FinqaTask(**base, protocol=2)


def by_page(task: FinqaTask, page: str):
    for inst in task.instances("train"):
        if inst.qid.startswith(page):
            return inst
    raise AssertionError(page)


def rec(step: int, answer: str) -> StepRecord:
    return StepRecord(step=step, attempt=0, context="", answer=answer, u=0.0, entropy=0.0, eps=0.0)


# --- what must not change -----------------------------------------------------------------


def test_both_protocols_load_the_same_questions_in_the_same_order(data_dir: Path) -> None:
    one, two = tasks(data_dir)
    for split in SIZES:
        a, b = list(one.instances(split)), list(two.instances(split))
        assert [(i.instance_id, i.qid, i.question, i.facts, i.answer) for i in a] == [
            (i.instance_id, i.qid, i.question, i.facts, i.answer) for i in b
        ]
        assert [[s.gold for s in i.steps] for i in a] == [[s.gold for s in i.steps] for i in b]
    # a result that Python writes as 2.5e-05 is skipped by both: the admission test keeps the
    # first protocol's parser, or other questions would load and every window would move
    assert not any(i.qid.startswith("FFF/") for i in two.instances("train"))
    assert len(list(two.instances("train"))) == 6


def test_the_admission_test_keeps_the_first_parser_and_rule() -> None:
    assert parse_number("7e+07") == (7.0, False)  # the mantissa: protocol 1's reading
    assert not numbers_match("2.5e-05", 2.5e-05)
    assert numbers_match("5000", 50.0) and numbers_match("0", 0.0424)  # the recorded defects
    assert numbers_match("24.69%", 0.2469136) and not numbers_match("30", 24.69136)


def test_untouched_questions_get_the_same_prompts(data_dir: Path) -> None:
    one, two = tasks(data_dir)
    a, b = by_page(one, "AAA/"), by_page(two, "AAA/")
    assert [(s.left, s.right) for s in a.steps] == [(s.left, s.right) for s in b.steps]
    assert b.fraction is not None and a.fraction is None
    for history in ([], [rec(1, "94")]):
        step = len(history) + 1
        assert canonical_messages(one.step_prompt(a, step, history, 0)) == canonical_messages(
            two.step_prompt(b, step, history, 0)
        )
    # a previous answer written with a percent sign is passed on as written, as before
    assert two.sub_question(b, 2, "1.6%") == one.sub_question(a, 2, "1.6%")
    assert "1.6%" in two.sub_question(b, 2, "1.6%")
    # and a question that a defect touches does change
    assert one.sub_question(by_page(one, "BBB/"), 1, "") != two.sub_question(
        by_page(two, "BBB/"), 1, ""
    )


# --- numbers --------------------------------------------------------------------------------


def test_numbers_are_read_as_written() -> None:
    assert read_number("7e+07") == (7e7, False, 1, -7)
    assert read_number("24.70") == (24.7, False, 4, 2)
    assert read_number(".25") == (0.25, False, 2, 2)
    assert read_number("12.5 %") == (12.5, True, 3, 1)
    assert read_number("0.125 (12.5%)") == (0.125, False, 3, 3)  # the sign is not on the number
    assert read_number("(5.2)") == (-5.2, False, 2, 1) and read_number("(5.2)%")[:2] == (-5.2, True)
    assert read_number("- 5.2")[0] == -5.2
    assert read_number(chr(0x2212) + "5.2")[0] == -5.2  # a Unicode minus sign
    assert read_number("$1,000")[0] == 1000.0 and read_number("3.8 million")[0] == 3.8
    assert read_number("unknown") is None and read_number("") is None


@pytest.mark.parametrize(
    ("answer", "gold", "right"),
    [
        ("50", 50.0, True),
        ("5000", 50.0, False),  # an error of scale that protocol 1 accepts
        ("1100", 11.0, False),
        ("0.565217", 56.521739, False),
        ("0", 0.0423913, False),  # zero for a small value
        ("0", 0.0, True),
        ("0.0", 0.0, True),
        ("0.01", 0.0, False),
        ("24.7", 24.69136, True),  # the gold rounded to the answer's precision
        ("25", 24.69136, True),
        ("30", 24.69136, False),
        ("2", 2.4, False),  # one digit is not a rounding
        ("-134", -134.0, True),
        ("134", -134.0, False),  # the sign counts
        ("(5.2)", -5.2, True),
        ("7e+07", 7e7, True),
        ("70000000", 7e7, True),
        ("56.8%", 0.567568, True),  # a signed percent: the fraction ...
        ("0.8%", 0.8, True),  # ... or the percent number (percentage points)
        ("-1.36%", -1.36, True),
        ("7%", 1.06, False),
        ("unknown", 1.0, False),
    ],
)
def test_the_step_rule(answer: str, gold: float, right: bool) -> None:
    assert step_match(answer, gold) is right


@pytest.mark.parametrize(
    ("answer", "gold", "fraction", "right"),
    [
        # FinQA answers "76.0%", the program gives the fraction
        ("75.98%", 0.760024, 0.760024, True),
        ("76%", 0.760024, 0.760024, True),
        ("76", 0.760024, 0.760024, True),
        ("0.76", 0.760024, 0.760024, True),
        ("7600", 0.760024, 0.760024, False),
        ("0.0076", 0.760024, 0.760024, False),
        ("0.76%", 0.760024, 0.760024, False),  # the fraction with a sign is another number
        ("-4.95%", -0.0495049, -0.0495049, True),
        ("4.95%", -0.0495049, -0.0495049, False),
        ("0", 0.0423913, 0.0423913, False),
        # FinQA answers "96.55%", the program ends with a multiplication by 100
        ("96.55%", 96.5517, 0.965517, True),
        ("96.55", 96.5517, 0.965517, True),
        ("0.9655", 96.5517, 0.965517, True),
        ("9655%", 96.5517, 0.965517, False),
        ("9655", 96.5517, 0.965517, False),
        # not a percentage question: the step rule
        ("50", 50.0, None, True),
        ("5000", 50.0, None, False),
        ("0.5", 50.0, None, False),
    ],
)
def test_the_final_rule(answer: str, gold: float, fraction: float | None, right: bool) -> None:
    assert final_match(answer, gold, fraction) is right


def test_finqas_own_answer_says_whether_the_result_is_a_fraction() -> None:
    divide = [("subtract", ["5829", "5735"]), ("divide", ["#0", "5735"])]
    times = [("divide", ["60", "243"]), ("multiply", ["#0", "const_100"])]
    assert percentage_fraction("1.6%", 0.016391, divide) == 0.016391  # the fraction
    assert percentage_fraction("24.7%", 24.69136, times) == pytest.approx(0.2469136)
    assert (
        percentage_fraction("35%", 0.21635, divide) == 0.21635
    )  # a careless string: still a fraction
    assert (
        percentage_fraction("1.64", 0.016391, divide) == 0.016391
    )  # a percentage without its sign
    assert percentage_fraction("50", 50.0, divide) is None
    assert percentage_fraction("$ 3.8 million", 3.8, divide) is None
    assert percentage_fraction("", 0.5, divide) is None
    assert percentage_fraction("0%", 0.0032, divide) == 0.0032  # nothing to compare: by the program
    assert percentage_fraction("0%", 0.32, times) == pytest.approx(0.0032)


# --- the reviewer ---------------------------------------------------------------------------


def test_the_reviewer_writes_plain_decimals() -> None:
    for value in (0.0, 1.5, -2.25, 24.69136, 123456.0, 999999.0, 0.000123456):
        assert plain_number(value) == f"{value:.6g}"  # as protocol 1 wherever it had no exponent
    assert plain_number(70_000_000.0) == "70000000"
    assert plain_number(56_571_500.4) == "56571500"
    assert plain_number(-70_000_000.0) == "-70000000"
    assert plain_number(0.000025) == "0.000025"


def test_the_rule_accepts_the_reviewer_at_every_size(data_dir: Path) -> None:
    one, two = tasks(data_dir)
    a, b = by_page(one, "CCC/"), by_page(two, "CCC/")
    assert one.local_truth(a, 1, "") == "7e+07" and not one.has_local_truth(a, 2, "7e+07")  # defect
    assert two.local_truth(b, 1, "") == "70000000" and two.has_local_truth(b, 2, "70000000")
    assert two.is_final_correct(b, two.local_truth(b, 2, "70000000"))
    for split in SIZES:
        for inst in two.instances(split):
            previous = ""
            for step in range(1, inst.n_steps + 1):
                answer = two.local_truth(inst, step, previous)
                assert "e" not in answer
                if step < inst.n_steps:
                    assert two.has_local_truth(inst, step + 1, answer), (inst.qid, step, answer)
                else:
                    assert two.is_final_correct(inst, answer), (inst.qid, answer)
                assert two.is_locally_correct(inst, step, previous, answer)
                previous = answer


# --- labels on the fixture --------------------------------------------------------------------


def test_labels_follow_the_protocol(data_dir: Path) -> None:
    one, two = tasks(data_dir)
    a, b = by_page(one, "AAA/"), by_page(two, "AAA/")  # gold 94, then 0.016391 ("1.6%")
    assert one.is_locally_correct(a, 1, "", "9400") and not two.is_locally_correct(b, 1, "", "9400")
    assert one.has_local_truth(a, 2, "0.94") and not two.has_local_truth(b, 2, "0.94")
    for answer in ("0.0164", "1.64", "1.64%", "1.6%"):
        assert two.is_final_correct(b, answer), answer
    for answer in ("164", "0.000164", "0.0164%", "0", None):
        assert not two.is_final_correct(b, answer), answer
    assert one.is_final_correct(a, "0") and one.is_final_correct(a, "0.000164")  # as recorded
    # the last step is labelled as a final answer, so fit labels and final labels agree
    assert two.is_locally_correct(b, 2, "94", "1.64") and not two.is_locally_correct(
        b, 2, "94", "164"
    )
    # percentage points at an intermediate step: the sign is accepted either way
    c = by_page(two, "DDD/2010/page_40.pdf-4")  # 4.51 - 3.15 = 1.36 points
    assert two.is_locally_correct(c, 1, "", "1.36%") and two.is_locally_correct(c, 1, "", "1.36")
    assert not two.is_locally_correct(c, 1, "", "0.0136") and not two.is_locally_correct(
        c, 1, "", "136"
    )


# --- descriptions -----------------------------------------------------------------------------


def test_descriptions_that_failed_are_replaced_and_the_others_kept(data_dir: Path) -> None:
    one, two = tasks(data_dir)
    # two sentences that the cut at " is " had made one description
    old, new = by_page(one, "BBB/").steps[0], by_page(two, "BBB/").steps[0]
    assert old.left == old.right
    assert new.left != new.right and new.left.startswith('the number given in "the maturity date')
    assert "$ ___ million" in new.left and "october 24 , 2018" in new.left
    assert (
        "october 24 , 2017" in new.right
        and "164" not in new.left
        and "30" not in new.right.split("is")[0]
    )
    # a repeated column header: the whole row is quoted, with the operand blanked
    old, new = (
        by_page(one, "DDD/2010/page_40.pdf-3").steps[0],
        by_page(two, "DDD/2010/page_40.pdf-3").steps[0],
    )
    assert old.right == "the treasury of 2010 market value"
    assert "is $ ___ ; the treasury of 2010 percentage" in new.left and "is $ 2068" in new.left
    assert (
        "is $ 2075" in new.right and "is $ ___ ; the treasury of percentage of total" in new.right
    )
    # two facts that differ in nothing but the number: the number is given
    old, new = by_page(one, "EEE/").steps[0], by_page(two, "EEE/").steps[0]
    assert old.left == old.right and (new.left, new.right) == ("1390", "15244")
    # the same number used twice keeps its one description (the step is well posed)
    old, new = by_page(one, "CCC/").steps[0], by_page(two, "CCC/").steps[0]
    assert old.left == old.right == new.left == new.right
    # and no description of protocol 2 states a number that it should ask for, except by rule
    assert "the plan paid" not in by_page(two, "EEE/").steps[0].left


def test_every_description_of_the_second_protocol_singles_out_its_number(data_dir: Path) -> None:
    check_descriptions(tasks(data_dir)[1], SIZES)


def check_descriptions(task: FinqaTask, sizes: dict[str, int]) -> int:
    """Every literal of every step: kept where the screen finds nothing, unambiguous else."""
    changed = 0
    for split in sizes:
        rows = {row["id"]: row for row in task._candidates(split)}
        for inst in task.instances(split):
            row = rows[inst.qid]
            gold, head = row["gold_facts"], row["head"]
            touched = False
            for step, (_, args) in zip(inst.steps, row["ops"], strict=True):
                texts = {}
                for arg, text in zip(args, (step.left, step.right), strict=True):
                    if arg.startswith(("#", "const_")):
                        continue
                    texts[arg] = text
                    if description_issue(arg, gold, head) is None:
                        continue
                    touched = True
                    if text == arg.strip():
                        continue  # the number is given: nothing else tells the facts apart
                    assert text.startswith('the number given in "'), (inst.qid, text)
                    mine = {abs(float(arg.strip().rstrip("%")))}
                    assert not _other_number_fits(
                        text[len('the number given in "') : -1], gold, mine
                    )
                    assert not _stand_alone(arg.strip().rstrip("%").lstrip("-")).search(text)
                if len(texts) == 2:
                    (a, ta), (b, tb) = texts.items()
                    if abs(float(a.rstrip("%")) - float(b.rstrip("%"))) > 1e-12:
                        assert ta != tb, (inst.qid, ta)  # two different operands, two descriptions
            changed += touched
    return changed


# --- end to end with the mock -----------------------------------------------------------------


def test_the_second_protocol_runs_end_to_end_with_the_mock(data_dir: Path) -> None:
    task = tasks(data_dir)[1]
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(1.0, 500.0))  # hardly ever wrong
    settings = RunSettings(policy="none", K=3, K_prime=3)
    ids = [i.instance_id for i in task.instances("train")]
    trajectories = run_instances(task, llm, settings, ids, make_human(task, settings))
    assert len(trajectories) == SIZES["train"]
    correct = {t.instance_id for t in trajectories if t.final_correct}
    assert len(correct) >= SIZES["train"] - 1
    assert by_page(task, "CCC/").instance_id in correct  # the values of a million or more


def test_options_are_validated(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="protocol must be 1"):
        FinqaTask(data_dir=data_dir, protocol=3)
    built = make_task(
        {"name": "finqa", "data_dir": str(data_dir), "n_instances": SIZES, "protocol": 2}
    )
    assert built.protocol == 2 and FinqaTask(data_dir=data_dir).protocol == 1


# --- the real files, when they are there --------------------------------------------------------


@pytest.mark.skipif(not (REAL / "train.json").exists(), reason="FinQA files not downloaded")
def test_on_the_real_windows_the_second_protocol_changes_only_what_it_should() -> None:
    sizes = {"train": 600, "calib": 200, "test": 200}
    one = FinqaTask(data_dir=REAL, n_instances=sizes)
    two = FinqaTask(data_dir=REAL, n_instances=sizes, protocol=2)
    changed = {}
    for split in sizes:
        a, b = list(one.instances(split)), list(two.instances(split))
        assert [(i.instance_id, i.qid, i.facts, i.answer) for i in a] == [
            (i.instance_id, i.qid, i.facts, i.answer) for i in b
        ]
        differ = [
            x.instance_id
            for x, y in zip(a, b, strict=True)
            if [(s.left, s.right) for s in x.steps] != [(s.left, s.right) for s in y.steps]
        ]
        changed[split] = len(differ)
        for x, y in zip(a, b, strict=True):
            if x.instance_id not in differ:  # untouched: the first prompt is the same text
                assert canonical_messages(one.step_prompt(x, 1, [], 0)) == canonical_messages(
                    two.step_prompt(y, 1, [], 0)
                )
    assert changed == {"train": 21, "calib": 8, "test": 9}
    assert check_descriptions(two, sizes) == 21 + 8 + 9
    assert [sum(i.fraction is not None for i in two.instances(s)) for s in sizes] == [424, 160, 136]
    for split in sizes:  # the reviewer is accepted at every step of every question
        for inst in two.instances(split):
            previous = ""
            for step in range(1, inst.n_steps + 1):
                answer = two.local_truth(inst, step, previous)
                assert two.is_locally_correct(inst, step, previous, answer), (inst.qid, step)
                previous = answer
