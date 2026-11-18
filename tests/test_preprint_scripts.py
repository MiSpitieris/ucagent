"""Two reporting scripts of the preprint: re-pricing and the figure's selection markers.

Both had a defect found by the code review of 2026-11-11 (``docs/EXPERIMENTS.md``): the
re-pricing compared a selection with the longest question instead of the mean length, and the
figure marked the test knob nearest to a selection when the selected knob had no test run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def load(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_review_costs_the_mean_length_not_the_longest_question() -> None:
    mod = load("price_sensitivity")
    assert mod.direct_review_checks([2, 2, 2]) == 2.0  # fixed length: unchanged
    assert mod.direct_review_checks([2, 2, 4]) == pytest.approx(8 / 3)
    # FinQA, confidence-weighted gate at 0.1: 0.325 checks and 10.6 calls per question
    assert mod.break_even(2.12, 0.325, 10.6) == pytest.approx(0.1693, abs=1e-4)
    assert mod.break_even(4, 0.325, 10.6) == pytest.approx(0.3467, abs=1e-4)  # the old baseline
    assert mod.break_even(2.0, 0.5, 0.0) != mod.break_even(2.0, 0.5, 0.0)  # no calls: nan


def test_a_selection_is_marked_only_at_its_own_test_run() -> None:
    mod = load("preprint_figures")
    curve = pd.DataFrame({"knob": [0.1, 0.3], "total_cost": [1.0, 0.4], "final_error": [0.40, 0.1]})
    assert mod.selected_point(curve, 0.3, "suite") == (0.4, 0.1)
    with pytest.raises(SystemExit, match=r"no test run at the selected knob 0\.2"):
        mod.selected_point(curve, 0.2, "suite")  # the nearest knob is not a substitute


def test_a_certified_point_is_distinct_by_its_questions_controller_and_outcome(tmp_path) -> None:
    mod = load("preprint_results")
    import yaml

    for name, task in (("a", {"name": "t", "offset": 0}), ("b", {"name": "t", "offset": 0})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "suite.yaml").write_text(
            yaml.safe_dump({"suite_id": name, "base": {"task": task}}), encoding="utf-8"
        )
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "suite.yaml").write_text(
        yaml.safe_dump({"suite_id": "c", "base": {"task": {"name": "t", "offset": 200}}}),
        encoding="utf-8",
    )

    def sel(knob: float, cost: float) -> object:
        return mod.Selection("random_ask", 0.1, True, knob, 12, 200, 0.07, cost, 1.4)

    same = mod.selection_key(tmp_path / "a", sel(0.3, 1.53))
    assert same == mod.selection_key(
        tmp_path / "b", sel(0.3, 1.53)
    )  # a replay, or another estimator
    assert same != mod.selection_key(tmp_path / "c", sel(0.3, 1.53))  # other questions
    assert same != mod.selection_key(tmp_path / "a", sel(0.4, 1.34))  # another outcome


def test_the_census_counts_replays_apart_from_the_suites_as_first_run() -> None:
    mod = load("preprint_results")
    rows = [
        ("finqa2_verbalized", 12, 2),
        ("fx_finqa2_verbalized", 12, 2),
        ("twowiki_abstain", 1, 0),
    ]
    line = mod.census_split(rows)
    assert "2 suites as first run, 13 certified points, 2 above the target" in line
    assert "1 suites replays with the repaired effect model (`fx_*`), 12 certified points" in line
