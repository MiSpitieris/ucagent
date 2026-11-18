"""scripts/robustness.py: clusters of questions and the design effect (no run records)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "robustness.py"
    spec = importlib.util.spec_from_file_location("robustness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clusters_join_questions_through_shared_keys(mod: ModuleType) -> None:
    groups = mod.clusters({"a": [1], "b": [1, 2], "c": [2], "d": [3], "e": []})
    assert sorted(sorted(g) for g in groups) == [["a", "b", "c"], ["d"], ["e"]]


def test_design_effect_is_one_without_clusters_and_above_with_correlated_errors(
    mod: ModuleType,
) -> None:
    ids = [f"q{k}" for k in range(40)]
    errors = {i: float(k % 4 == 0) for k, i in enumerate(ids)}  # 10 of 40 wrong
    alone, _ = mod.design_effect(errors, [[i] for i in ids])
    assert alone == pytest.approx(1.0)
    # pair every wrong question with another wrong one, every right one with a right one
    wrong = [i for i in ids if errors[i]]
    right = [i for i in ids if not errors[i]]
    pairs = [wrong[k : k + 2] for k in range(0, 10, 2)] + [
        right[k : k + 2] for k in range(0, 30, 2)
    ]
    together, rho = mod.design_effect(errors, pairs)
    assert together == pytest.approx(2.0) and rho == pytest.approx(1.0)


def test_selection_uses_only_the_chosen_questions(mod: ModuleType) -> None:
    # two knobs on 200 questions: the careful one leaves 5 wrong, the loose one 30
    errors = {0.1: np.array([1.0] * 5 + [0.0] * 195), 0.5: np.array([1.0] * 30 + [0.0] * 170)}
    everyone = np.arange(200)
    assert mod.select(errors, everyone, alpha=0.2, delta=0.1) == 0.5
    assert mod.select(errors, everyone, alpha=0.1, delta=0.1) == 0.1
    assert mod.select(errors, np.arange(40), alpha=0.1, delta=0.1) is None  # 5 wrong of 40


def test_shares_are_never_rounded_to_none_or_all(mod: ModuleType) -> None:
    hits = np.zeros(400, dtype=bool)
    assert mod.count(hits) == "0/400" and mod.percent(hits) == "0\\,\\%"
    hits[0] = True  # 1 of 400 is 0.25 %, which is not none
    assert mod.count(hits) == "1/400" and mod.percent(hits) == "$<$1\\,\\%"
    assert mod.percent(~hits) == "$>$99\\,\\%"  # 399 of 400 is not all
    assert mod.percent(np.ones(400, dtype=bool)) == "100\\,\\%"
    hits[:164] = True
    assert mod.percent(hits) == "41\\,\\%"


def test_the_generations_of_suites_differ_in_the_prefix_and_keep_the_first_run(
    mod: ModuleType,
) -> None:
    repaired, recorded = mod.benchmarks("repaired"), mod.benchmarks("recorded")
    assert [p for _, p, _ in recorded][:2] == ["musique_confirm2d_{}", "twowiki_{}"]
    assert [p for _, p, _ in repaired][:2] == ["fx_musique_confirm2d_{}", "fx_twowiki_{}"]
    assert repaired[-1] == recorded[-1] == mod.FIRST_RUNS[0]  # the first FinQA run was not replayed


def test_dependence_lines_count_shared_keys_within_and_across_the_splits(mod: ModuleType) -> None:
    keys = {
        "train": {"t1": ["x"], "t2": ["y"]},
        "calib": {"c1": ["a"], "c2": ["a", "b"], "c3": ["x"], "c4": ["k"]},
        "test": {"e1": ["b"], "e2": ["y"], "e3": ["m"]},
    }
    first, second = mod.dependence_lines(keys, mod.clusters(keys["calib"]))
    assert first.startswith("- Calibration half: 4 questions in 3 clusters; 2 questions share")
    assert "(1 clusters of more than one, the largest of 2)" in first
    assert second.startswith(
        "- 1 calibration questions share a key with a test question, and 1 test questions"
    )
    assert "1 calibration and 1 test questions share one with the 2 questions of the fit" in second


def test_development_overlap_counts_questions_that_share_a_key_with_an_earlier_run(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    earlier = {"a.yaml": {"p", "q"}, "b.yaml": {"r"}}
    monkeypatch.setattr(mod, "window_keys", lambda config, keyfn: earlier[config.name])
    keys = {
        "calib": {"c1": ["p", "z"], "c2": ["z"], "c3": ["r"]},
        "test": {"e1": ["q"], "e2": ["w"]},
    }
    line = mod.development_overlap(keys, lambda i: [], [Path("a.yaml"), Path("b.yaml")])
    assert line.startswith("- Earlier runs (2 configurations): 2 of the 3 calibration questions")
    assert "1 of the 2 test questions" in line
