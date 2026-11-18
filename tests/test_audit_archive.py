"""The archive audit: a finished suite matches its configuration, and tampering is found."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from ucagent.experiments.config import ExperimentConfig, GridConfig
from ucagent.experiments.suite import SuiteConfig, run_suite


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_archive.py"
    spec = importlib.util.spec_from_file_location("audit_archive", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_a_finished_suite_matches_and_tampering_is_found(tmp_path: Path, mod: ModuleType) -> None:
    suite = SuiteConfig(
        suite_id="tiny",
        base=ExperimentConfig(
            run_id="tiny",
            task={
                "n_steps": 2,
                "ambiguity": 0.5,
                "n_instances": {"train": 6, "calib": 6, "test": 6},
            },
            llm={"K": 2, "K_prime": 4},
            limits={"max_concurrency": 1},
        ),
        policies={
            "none": None,
            "per_step_eps": GridConfig(kind="linear", min=0.1, max=0.5, points=2),
        },
        splits=["calib", "test"],
    )
    root = run_suite(suite, results_dir=tmp_path / "results", workers=1)
    audit = mod.audit_suite(root, None)
    assert audit.runs == 6 and audit.trajectories == 36
    assert audit.problems == 0 and audit.config == "no tracked config"
    assert audit.artifact == "task hash same, fit questions same"
    n, gaps = mod.selection_gaps(root, (0.5,))
    assert n >= 1 and gaps == []

    # a label that the task does not reproduce
    manifest = pd.read_csv(root / "manifest.csv")
    path = Path(manifest["path"].iloc[0])
    frame = pd.read_parquet(path)
    frame.loc[0, "final_correct"] = not bool(frame.loc[0, "final_correct"])
    frame.to_parquet(path, index=False)
    assert mod.audit_suite(root, None).labels_differ == 1

    # a run that holds other questions than its window
    frame.loc[0, "instance_id"] = frame.loc[1, "instance_id"]
    frame.to_parquet(path, index=False)
    assert mod.audit_suite(root, None).other_questions == 1

    # a selection whose test run is missing
    gate_test = (manifest["policy"] == "per_step_eps") & (manifest["split"] == "test")
    manifest[~gate_test].to_csv(root / "manifest.csv", index=False)
    n, gaps = mod.selection_gaps(root, (0.5,))
    assert n >= 1 and len(gaps) == n and "no test run at knob" in gaps[0]


def test_the_default_suites_cover_both_generations_and_the_abstention_runs_once(
    tmp_path: Path, mod: ModuleType
) -> None:
    names = [
        "twowiki_confagree",  # as recorded
        "fx_twowiki_confagree",  # as replayed with the repaired effect model
        "twowiki_abstain",  # not replayed: read once, from the recorded generation
        "finqa_entropy",  # a run that a corrected run has replaced
        "musique_small_gptoss",  # a development run: not in the tables
    ]
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.csv").write_text("path\n", encoding="utf-8")
    (tmp_path / "fx_twowiki_entropy").mkdir()  # no manifest: an unfinished suite is left out
    found = [p.name for p in mod.default_suites(tmp_path)]
    assert found == [
        "twowiki_confagree",
        "twowiki_abstain",
        "fx_twowiki_confagree",
        "finqa_entropy",
    ]


def test_a_configured_run_missing_from_the_manifest_is_reported(
    tmp_path: Path, mod: ModuleType
) -> None:
    suite = SuiteConfig(
        suite_id="tiny",
        base=ExperimentConfig(
            run_id="tiny",
            task={
                "n_steps": 2,
                "ambiguity": 0.5,
                "n_instances": {"train": 6, "calib": 6, "test": 6},
            },
            llm={"K": 2, "K_prime": 4},
            control={"alpha": 0.95},
            limits={"max_concurrency": 1},
        ),
        policies={
            "none": None,
            "per_step_eps": GridConfig(kind="linear", min=0.1, max=0.5, points=3),
        },
        splits=["calib", "test"],
    )
    root = run_suite(suite, results_dir=tmp_path / "results", workers=1)
    assert mod.grid_rows(root) == (6, [])  # three knobs on two splits, one seed
    n_whole, _ = mod.selection_gaps(root, (0.95,))

    manifest = pd.read_csv(root / "manifest.csv")
    gone = (
        (manifest["policy"] == "per_step_eps")
        & (manifest["split"] == "calib")
        & ((manifest["knob"] - 0.3).abs() < 1e-9)
    )
    manifest[~gone].to_csv(root / "manifest.csv", index=False)
    n, missing = mod.grid_rows(root)
    assert n == 6 and missing == ["tiny per_step_eps calib seed 0: knob 0.3"]
    # the selection is still made, at the knob before the gap, and it has its test run
    n_holed, gaps = mod.selection_gaps(root, (0.95,))
    assert (n_holed, gaps) == (n_whole, [])
