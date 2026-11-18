import pytest
import yaml

from ucagent.experiments.config import ExperimentConfig, GridConfig
from ucagent.experiments.figures import fig2
from ucagent.experiments.suite import (
    SuiteConfig,
    grid_values,
    load_manifest,
    resolve_knob,
    run_suite,
)


def _suite_cfg(tmp_path) -> SuiteConfig:
    data = {
        "suite_id": "tiny",
        "base": {
            "run_id": "tiny",
            "task": {
                "n_steps": 3,
                "ambiguity": 0.5,
                "n_instances": {"train": 12, "calib": 10, "test": 10},
            },
            "llm": {"K": 3, "K_prime": 6},
            "control": {"alpha": 0.2},
            "limits": {"max_concurrency": 2},
        },
        "policies": {
            "none": None,
            "per_step_eps": {"kind": "log", "min": 0.01, "max": 0.5, "points": 2},
            "marginal": {"kind": "log", "min": 0.01, "max": 1.0, "points": 2},
        },
        "seeds": [0],
        "splits": ["test"],
        "extra_points": [{"policy": "per_step_eps", "label": "bonferroni", "knob": "alpha / n"}],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data))
    return SuiteConfig.from_yaml(path)


def test_grid_and_knob_helpers() -> None:
    assert grid_values(GridConfig(kind="linear", min=0, max=1, points=3)) == [0.0, 0.5, 1.0]
    log = grid_values(GridConfig(kind="log", min=0.01, max=1.0, points=3))
    assert (
        log[0] == pytest.approx(0.01)
        and log[1] == pytest.approx(0.1)
        and log[2] == pytest.approx(1.0)
    )
    cfg = ExperimentConfig(run_id="x", control={"alpha": 0.3}, task={"n_steps": 3})
    assert resolve_knob("alpha / n", cfg) == pytest.approx(0.1)
    assert resolve_knob(0.2, cfg) == 0.2


def test_run_suite_manifest_and_fig2(tmp_path) -> None:
    suite = _suite_cfg(tmp_path)
    root = run_suite(suite, workers=2, results_dir=tmp_path / "results")
    manifest = load_manifest(root)
    assert len(manifest) == 1 + 2 + 2 + 1
    assert set(manifest["policy"]) == {"none", "per_step_eps", "marginal"}
    assert (root / "artifacts" / "seed0.json").exists()
    assert all((tmp_path / "results" / "tiny" / "runs").glob("*"))
    assert "final_error" in manifest.columns and manifest["n"].eq(10).all()
    bonf = manifest[manifest["label"] == "bonferroni"]
    assert len(bonf) == 1 and bonf["knob"].iloc[0] == pytest.approx(0.2 / 3)
    paths = fig2(root)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert "mock LLM" in (root / "figures" / "fig2_error_vs_cost.caption.txt").read_text()
    # resume: a second call reruns nothing and keeps the manifest
    root2 = run_suite(suite, workers=2, results_dir=tmp_path / "results")
    assert len(load_manifest(root2)) == len(manifest)
    # figures come only from files: deleting the manifest breaks them
    (root / "manifest.csv").unlink()
    with pytest.raises(FileNotFoundError):
        fig2(root)


def test_calibrate_suite_and_risk_plot(tmp_path) -> None:
    from ucagent.experiments.figures import plot_risk_vs_knob
    from ucagent.experiments.suite import calibrate_suite, selected_knobs

    suite = _suite_cfg(tmp_path)
    suite = suite.model_copy(update={"splits": ["calib", "test"]})
    root = run_suite(suite, workers=2, results_dir=tmp_path / "results")
    table = calibrate_suite(root)
    assert set(table["policy"]) == {"per_step_eps", "marginal"}
    assert {"feasible", "knob_star", "min_calib_risk"} <= set(table.columns)
    picks = selected_knobs(root)
    assert set(picks) <= {"per_step_eps", "marginal"}
    paths = plot_risk_vs_knob(root)
    assert all(p.exists() for p in paths)
    paths2 = fig2(root)
    assert all(p.exists() for p in paths2)


def test_variants_add_runs_and_a_manifest_column(tmp_path) -> None:
    from ucagent.experiments.suite import Variant

    suite = _suite_cfg(tmp_path).model_copy(
        update={
            "variants": [
                Variant(),
                Variant(label="ones", overrides={"risk": {"influence": "ones"}}),
            ]
        }
    )
    root = run_suite(suite, workers=2, results_dir=tmp_path / "results")
    manifest = load_manifest(root)
    assert set(manifest["variant"]) == {"main", "ones"}
    assert len(manifest) == 2 * 6
    assert any(p.name.startswith("ones__") for p in (root / "runs").iterdir())


def test_a_missing_calibration_run_ends_the_selection_before_it(tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    from ucagent.experiments.suite import calibrate_suite, configured_knobs

    suite = _suite_cfg(tmp_path).model_copy(
        update={
            "splits": ["calib", "test"],
            "extra_points": [],
            # a target that every knob meets on ten questions, so that the walk reaches the end
            "base": _suite_cfg(tmp_path).base.with_updates(control={"alpha": 0.95}),
            "policies": {
                "none": None,
                "per_step_eps": GridConfig(kind="linear", min=0.1, max=0.5, points=3),
            },
        }
    )
    root = run_suite(suite, workers=2, results_dir=tmp_path / "results")
    knobs = configured_knobs(root, "per_step_eps")
    assert knobs == pytest.approx([0.1, 0.3, 0.5]) and configured_knobs(root, "none") is None
    whole = calibrate_suite(root)
    assert float(whole["knob_star"].iloc[0]) == pytest.approx(0.5)
    assert int(whole["missing_calib_knobs"].iloc[0]) == 0

    # the calibration run at the middle knob disappears from the manifest
    manifest = pd.read_csv(root / "manifest.csv")
    gone = (
        (manifest["policy"] == "per_step_eps")
        & (manifest["split"] == "calib")
        & ((manifest["knob"] - 0.3).abs() < 1e-9)
    )
    assert int(gone.sum()) == 1
    manifest[~gone].to_csv(root / "manifest.csv", index=False)
    with caplog.at_level(logging.WARNING, logger="ucagent"):
        holed = calibrate_suite(root)
    assert float(holed["knob_star"].iloc[0]) == pytest.approx(0.1)  # not 0.5, beyond the gap
    assert int(holed["missing_calib_knobs"].iloc[0]) == 1
    assert "has no calib run at 1 configured knob(s)" in caplog.text
