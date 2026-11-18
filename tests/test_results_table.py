"""scripts/results_table.py: the Learn-then-Test table of a suite as a LaTeX table body."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "results_table.py"
    spec = importlib.util.spec_from_file_location("results_table", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_tex_table_averages_the_feasible_seeds_and_names_the_controllers(
    tmp_path: Path, mod: ModuleType
) -> None:
    pd.DataFrame(
        [
            {"policy": "none", "split": "test", "final_error": 0.20, "total_cost": 0.5},
            {"policy": "none", "split": "test", "final_error": 0.16, "total_cost": 0.5},
            {"policy": "none", "split": "calib", "final_error": 0.90, "total_cost": 9.0},
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)
    star = ("test_error_at_star", "test_cost_at_star", "test_interventions_at_star")

    def row(policy: str, feasible: bool, *values: float) -> dict[str, object]:
        return {"policy": policy, "feasible": feasible, **dict(zip(star, values, strict=True))}

    pd.DataFrame(
        [
            row("marginal", True, 0.02, 1.20, 0.90),
            row("marginal", True, 0.04, 1.00, 0.70),
            row("marginal", False, 0.90, 9.00, 9.00),  # an infeasible seed is not averaged
            row("knowno", False, 0.0, 0.0, 0.0),
        ]
    ).to_csv(tmp_path / "ltt.csv", index=False)
    lines = mod.ltt_tex(tmp_path).splitlines()
    assert lines[0].startswith("% written by scripts/results_table.py")
    # the uncontrolled agent: the mean over the test runs only
    assert r"none (uncontrolled) & -- & 0.180 & 0.50 & 0.00 \\" in lines
    assert r"value-of-intervention rule & 2/3 & 0.030 & 1.10 & 0.80 \\" in lines
    assert r"KnowNo-style gate & 0/1 & -- & -- & -- \\" in lines
    assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]
    # a controller that the suite did not run has no row
    assert not any(line.startswith("calibrated gate") for line in lines)
