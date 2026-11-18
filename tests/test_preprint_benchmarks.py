"""scripts/preprint_benchmarks.py: the allowance, the floor and the table cells (no run records)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "preprint_benchmarks.py"
    spec = importlib.util.spec_from_file_location("preprint_benchmarks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_allowance_matches_the_learn_then_test_thresholds(mod: ModuleType) -> None:
    # 200 calibration questions, delta 0.1: the numbers quoted throughout the experiments
    assert [mod.allowance(a, 200, 0.1) for a in (0.2, 0.1, 0.05, 0.02)] == [32, 14, 5, 1]
    assert mod.allowance(0.001, 10, 0.1) == -1  # no error count is rare enough


def test_floor_is_the_fewest_errors_short_of_checking_every_step(mod: ModuleType) -> None:
    point = mod.GridPoint
    grid = [point(0.02, 0, 2.0), point(0.08, 20, 0.4), point(0.38, 20, 0.3), point(0.4, 38, 0.12)]
    floor = mod.floor_point(grid, all_steps=2.0)
    assert floor == point(0.38, 20, 0.3)  # a tie in errors goes to the fewer checks
    assert mod.floor_point([point(0.02, 0, 2.0)], all_steps=2.0) is None
    # questions of different lengths: 2.08 checks per question is every step
    assert mod.floor_point([point(0.02, 4, 2.08), point(0.06, 9, 0.52)], 2.08) == point(
        0.06, 9, 0.52
    )


def test_cells_mark_every_step_and_points_above_target(mod: ModuleType) -> None:
    cell = mod.Cell
    assert mod.tex_cell(cell(False)) == "--"
    assert mod.tex_cell(cell(True, 0.085, 0.66)) == "0.085 at 0.66"
    assert mod.tex_cell(cell(True, 0.0, 2.1, all_steps=True)) == r"\textit{0.000 at 2.10}"
    assert mod.tex_cell(cell(True, 0.025, 0.16, above=True)).endswith(r"$^{\dagger}$")
    assert mod.md_cell(cell(True, 0.04, 0.12, above=True)) == "0.040 at 0.12 (above target)"
    assert mod.md_cell(cell(False)) == "not certified"
    # nothing certified within a grid that cannot check every step is marked as such
    assert mod.tex_cell(cell(False, no_anchor=True)) == "--$^{g}$"
    assert mod.md_cell(cell(False, no_anchor=True)).endswith("(no grid point checks every step)")


def test_gate_against_rule_counts_cost_and_the_errors_of_the_cheaper(mod: ModuleType) -> None:
    cell = mod.Cell
    pairs = [
        (cell(True, 0.085, 0.66), cell(True, 0.070, 0.71)),  # gate cheaper, more errors
        (cell(True, 0.040, 1.03), cell(True, 0.070, 0.74)),  # rule cheaper, more errors
        (cell(True, 0.055, 0.24), cell(True, 0.060, 0.33)),  # gate cheaper, no more errors
        (cell(True, 0.040, 1.35), cell(True, 0.040, 1.354)),  # printed alike: a tie
        (cell(True, 0.0, 2.1, all_steps=True), cell(False)),  # the rule selected nothing
    ]
    assert mod.compare(pairs) == {
        "both": 4,
        "gate_cheaper": 2,
        "rule_cheaper": 1,
        "tie": 1,
        "gate_more_errors": 1,
        "rule_more_errors": 1,
    }
    assert "gate cheaper in 2" in mod.compare_md("x", mod.compare(pairs))


def test_a_pair_of_selections_that_serves_several_targets_counts_once(mod: ModuleType) -> None:
    cell = mod.Cell
    same = (cell(True, 0.040, 0.12), cell(True, 0.045, 0.14))
    rows = [
        ("confagree", *same),  # target 0.2
        ("confagree", *same),  # target 0.1: the same two selections
        ("confagree", cell(True, 0.040, 0.12, above=True), cell(True, 0.045, 0.14, above=True)),
        ("entropy", *same),  # another estimator is another pair
        ("confagree", cell(True, 0.040, 0.12), cell(True, 0.025, 0.16)),  # the rule moved
    ]
    kept = mod.distinct_pairs(rows)
    assert [k[0] for k in kept] == ["confagree", "entropy", "confagree"]
    assert kept[0][1:] == same


def test_a_handover_is_never_shown_as_checking_every_step(mod: ModuleType) -> None:
    # the final gate at 0.1 on MuSiQue costs more than checking every step but checks nothing
    assert not mod.every_step_by_checks(0.0, all_checks=2.0)
    assert mod.every_step_by_checks(2.0, all_checks=2.0)  # a gate at its most conservative knob
    assert mod.every_step_by_checks(2.08, all_checks=2.08)  # questions of different lengths
    assert not mod.every_step_by_checks(0.555, all_checks=2.0)
    final_gate = mod.Cell(True, 0.055, 4.15, all_steps=mod.every_step_by_checks(0.0, 2.0))
    assert mod.tex_cell(final_gate) == "0.055 at 4.15"  # not in italics


def test_the_comparison_of_controllers_has_short_heads_and_plain_rows(mod: ModuleType) -> None:
    assert mod.short_label("MuSiQue") == "MuSiQue"
    assert mod.short_label("2WikiMultiHopQA, split 1") == "2Wiki 1"
    assert mod.short_label("2WikiMultiHopQA, split 3 (filtered)") == "2Wiki 3"
    cells = [mod.tex_cell(mod.Cell(False, no_anchor=True)), "0.050 at 0.62"]
    assert mod.baseline_row("KnowNo-style gate", cells) == (
        r"KnowNo-style gate & --$^{g}$ & 0.050 at 0.62 \\"
    )


def test_the_two_generations_of_suites_differ_only_in_the_prefix(mod: ModuleType) -> None:
    repaired, recorded = mod.benchmarks("repaired"), mod.benchmarks("recorded")
    assert [b.label for b in repaired] == [b.label for b in recorded]
    assert recorded[0].pattern == "musique_confirm2d_{}"
    assert all(a.pattern == "fx_" + b.pattern for a, b in zip(repaired, recorded, strict=True))
    assert mod.DEFAULT_GENERATION == "repaired"  # the tables come from the replay
    assert mod.benchmarks(mod.DEFAULT_GENERATION) == mod.BENCHMARKS
    # the abstention suites were not replayed: both generations read the recorded one
    assert repaired[1].abstain == recorded[1].abstain == "twowiki_abstain"
    assert mod.FIRST_RUNS[0].abstain == "finqa_abstain"


def test_the_chance_of_selection_at_the_target_and_at_half_of_it(mod: ModuleType) -> None:
    # at its target a pipeline is selected with probability at most delta
    for alpha in (0.2, 0.1, 0.05, 0.02):
        assert mod.selection_chance(alpha, 200, 0.1, alpha) <= 0.1
    assert mod.selection_chance(0.1, 200, 0.1, 0.1) == pytest.approx(0.0929, abs=5e-4)
    # at half the target the loose targets are selected almost surely, the strict ones are not
    assert mod.selection_chance(0.2, 200, 0.1, 0.1) > 0.99
    assert mod.selection_chance(0.02, 200, 0.1, 0.01) < 0.5


def test_how_far_a_change_moved_the_selections_is_read_from_the_printed_numbers(
    mod: ModuleType,
) -> None:
    cell = mod.Cell
    rows = [
        ("a", cell(True, 0.065, 0.51), cell(True, 0.050, 0.63)),  # cost by 0.12, error by 0.015
        ("b", cell(True, 0.155, 0.20), cell(True, 0.105, 0.26)),  # cost by 0.06, error by 0.050
        ("c", cell(True, 0.040, 1.35), cell(True, 0.040, 1.354)),  # prints alike: not a change
        ("d", cell(False), cell(True, 0.020, 0.43)),  # selected on one side only
        ("e", cell(False), cell(False)),
    ]
    m = mod.moves(rows)
    assert (m.cells, m.both, m.one, m.changed) == (5, 3, 1, 2)
    assert m.cost == pytest.approx(0.12) and m.cost_where.startswith("a: 0.065 at 0.51 before")
    assert m.error == pytest.approx(0.050) and m.error_where.startswith("b: ")
    line = mod.moves_md("rule", m)
    assert "prints differently in 2 of the 3" in line and "largest move of the cost 0.12" in line
    # nothing moved: the line says so and names no cell
    same = mod.moves([("a", cell(True, 0.085, 0.66), cell(True, 0.085, 0.66))])
    assert same.changed == 0 and mod.moves_md("gate", same).endswith("differently in 0 of the 1.")


def test_the_comparison_of_generations_is_silent_without_both_on_disk(
    mod: ModuleType, tmp_path: Path
) -> None:
    assert mod.generation_moves(tmp_path) == []
