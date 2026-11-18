"""Print-only counts of the two 2WikiMultiHopQA scripts (no data files, no model calls)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(qid: str, rel1: str, rel2: str) -> dict:
    """A raw record whose two evidence triples form a chain a -> b -> c."""
    return {
        "_id": qid,
        "type": "compositional",
        "question": "q",
        "answer": "c",
        "evidences": [["a", rel1, "b"], ["b", rel2, "c"]],
        "supporting_facts": [["A", 0], ["B", 0]],
        "context": [["A", ["a is linked to b."]], ["B", ["b is linked to c."]]],
    }


def test_the_exclusion_of_relations_is_counted_by_relation() -> None:
    mod = load("prepare_twowiki")
    records = [
        _record("1", "director", "place of birth"),
        _record("2", "spouse", "place of birth"),
        _record("3", "director", "country of citizenship"),
        _record("4", "spouse", "country of citizenship"),
        _record("5", "director", "date of death"),
    ]
    lines = mod.exclusion_counts(records, skip=["5"])
    assert lines[0] == "chain questions before the exclusion of relations: 4"
    assert lines[1].startswith("left out by it: 3 (")
    assert lines[2] == "kept: 1"
    assert "spouse 2" in lines[3] and "country of citizenship 2" in lines[3]


def test_the_filter_checks_a_hop_only_when_it_has_keywords_for_the_relation() -> None:
    mod = load("make_twowiki_split3")

    def question(*relations: str) -> SimpleNamespace:
        return SimpleNamespace(hops=[SimpleNamespace(template=f"X >> {r}") for r in relations])

    assert mod.checked_hops(question("director", "place of birth")) == 1
    assert mod.checked_hops(question("mother", "date of death")) == 2
    assert mod.checked_hops(question("director", "performer")) == 0

    task = SimpleNamespace(
        instance_id=lambda split, index: (split, index),
        get_instance=lambda key: [
            question("mother", "date of death"),
            question("director", "place of birth"),
            question("director", "performer"),
        ][key[1]],
    )
    (line,) = mod.coverage(task, {"calib": [0, 1, 2]})
    assert line == (
        "calib: 3 kept questions; relation checked by keyword on both hops 1, on one hop 1, "
        "on neither 1"
    )
