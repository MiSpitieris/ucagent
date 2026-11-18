"""How FinQA's steps and facts are worded: the screen behind the label audit.

``describe`` names an operand without stating it. The review of 2026-11-11 found descriptions
that do not single out their number (``docs/EXPERIMENTS.md``); ``description_issue`` says why.
The facts below are written by hand in FinQA's format, after cases met in the data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ucagent.tasks.finqa import (
    _facts,
    _without_head,
    describe,
    description_issue,
    is_table_row,
    printed_negative,
    program_ops,
    table_head,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "finqa_mini.json"
REAL = Path("data/finqa/train.json")


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "finqa_label_audit.py"
    spec = importlib.util.spec_from_file_location("finqa_label_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_plain_table_cell_and_a_quoted_sentence_single_out_their_number() -> None:
    row = "( in millions ) the revenue of 2018 is $ 5829 ; the revenue of 2017 is $ 5735 ;"
    sentence = "the company repaid $ 300 million of notes in 2018 and $ 164 million in 2017 ."
    assert is_table_row(row) and not is_table_row(sentence)
    assert description_issue("5829", [row], "( in millions )") is None
    assert description_issue("5735", [row], "( in millions )") is None
    assert describe("300", [sentence]).startswith('the number given in "')
    assert description_issue("300", [sentence]) is None
    assert description_issue("const_100", [row]) is None
    assert description_issue("42", [row]) is None  # not among the facts: the literal is shown


def test_a_sentence_cut_at_is_loses_what_tells_two_operands_apart() -> None:
    facts = [
        "the maturity date for $ 30 million of the senior credit facility is october 24 , 2017 .",
        "the maturity date for $ 164 million of the senior credit facility is october 24 , 2018 .",
    ]
    assert describe("30", facts) == describe("164", facts)  # the defect
    assert description_issue("30", facts) == 'sentence cut at " is "'
    assert description_issue("164", facts) == 'sentence cut at " is "'


def test_a_repeated_column_header_gives_two_cells_one_label() -> None:
    head = "( in millions of u.s . dollars except for percentages )"
    row = (
        f"{head} the treasury of 2010 market value is $ 2075 ; the treasury of 2010 percentage "
        "of total is 4% ( 4 % ) ; the treasury of 2010 market value is $ 2068 ; the treasury of "
        "percentageof total is 5% ( 5 % ) ;"
    )
    assert description_issue("2075", [row], head) == "label occurs with another value"
    assert description_issue("2068", [row], head) == "label occurs with another value"
    assert description_issue("4%", [row], head) is None  # one label, one value
    # FinQA respaces the head cell ("31," becomes "31 ,"); it is still recognised
    assert _without_head(
        "years ended december 31 , the revenue of 2010", "years ended december 31,"
    ) == ("the revenue of 2010")
    assert _without_head("the revenue of 2010", "") == "the revenue of 2010"


def test_cells_with_several_numbers_and_header_rows_that_are_data() -> None:
    two = "the medway power limited of sales proceeds ( in millions ) is a347/$ 78 ;"
    assert description_issue("78", [two]) == "cell holds several numbers"
    span = "the volatility of 2017 is 22% ( 22 % ) - 27% ( 27 % ) ;"
    assert description_issue("22%", [span]) == "cell holds several numbers"
    data_header = (
        "2001 first quarter the 2001 first quarter of high $ 60.15 is high $ 60.15 ; the 2001 "
        "first quarter of low $ 41.30 is low $ 41.30 ; the 2001 first quarter of low $ 34.25 is "
        "low $ 34.25 ;"
    )
    assert (
        description_issue("34.25", [data_header], "2001 first quarter")
        == "blanked label fits another number"
    )
    assert description_issue("60.15", [data_header], "2001 first quarter") is None


def test_two_facts_that_differ_only_in_the_number() -> None:
    facts = [
        "the plan paid $ 1390 thousand in the period .",
        "the plan paid $ 15244 thousand in the period .",
    ]
    assert description_issue("1390", facts) == "quoted fact fits another number"


def test_a_magnitude_where_the_report_prints_a_negative() -> None:
    row = "the net loss of 2018 is -2547 ( 2547 ) ; the net loss of 2017 is $ 2100 ;"
    assert printed_negative("2547", [row])
    assert not printed_negative("2100", [row])
    assert not printed_negative("-2547", [row]) and not printed_negative("const_100", [row])
    both = [row, "the loss of 2547 was larger than expected ."]
    assert not printed_negative("2547", both)  # also printed as a plain number


def test_program_ops_and_the_table_head() -> None:
    assert program_ops("subtract(5829, 5735), divide(#0, 5735)") == [
        ("subtract", ["5829", "5735"]),
        ("divide", ["#0", "5735"]),
    ]
    assert (
        table_head({"table": [["( in millions )", "2018"], ["revenue", "5829"]]})
        == "( in millions )"
    )
    assert table_head({"table": []}) == "" and table_head({}) == ""


def test_the_audit_rebuilds_the_adapter_rows_as_the_task_writes_them(audit: ModuleType) -> None:
    for row in json.loads(FIXTURE.read_text(encoding="utf-8")):
        table = row["table"]
        rebuilt = {audit.adapter_row(table[0], cells) for cells in table[1:]} - {""}
        assert rebuilt and rebuilt <= set(_facts(row)[1])
        assert all(is_table_row(text) for text in rebuilt)


def test_same_question_keys_page_text_and_gold(audit: ModuleType) -> None:
    class Inst:
        def __init__(self, qid: str, question: str, answer: float) -> None:
            self.qid, self.question, self.answer = qid, question, answer

    a = Inst("IPG/2006/page_77.pdf-1", "What percent increase?", 1.0756)
    b = Inst("IPG/2006/page_77.pdf-2", "what percent  increase", 1.0756)
    c = Inst("IPG/2007/page_77.pdf-2", "what percent increase", 1.0756)
    assert audit.same_question(a) == audit.same_question(b) != audit.same_question(c)


@pytest.mark.skipif(not REAL.exists(), reason="FinQA files not downloaded")
def test_table_rows_and_sentences_are_told_apart_on_the_real_file() -> None:
    rows = json.loads(REAL.read_text(encoding="utf-8"))
    for row in rows[:400]:
        for key, fact in row["qa"].get("gold_inds", {}).items():
            assert is_table_row(str(fact)) == key.startswith("table_"), (row["id"], key)
