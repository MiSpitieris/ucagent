"""M5 acceptance on the mock: monotone risk in the knob; held-out coverage across seeds."""

import itertools
import json
import math

import pytest
import yaml

from ucagent.experiments.suite import SuiteConfig, calibrate_suite, load_manifest, run_suite

pytestmark = pytest.mark.slow


def _suite(
    tmp_path, suite_id, policies, *, seeds, n_steps, K, K_prime, n_inst, alpha, ambiguity=0.5
):
    data = {
        "suite_id": suite_id,
        "base": {
            "run_id": suite_id,
            "task": {
                "n_steps": n_steps,
                "ambiguity": ambiguity,
                "distractors": 3,
                "n_instances": {"train": n_inst, "calib": n_inst, "test": n_inst},
            },
            "llm": {"K": K, "K_prime": K_prime},
            "uq": {"estimator": "semantic_entropy"},
            "control": {"alpha": alpha, "max_resamples": 1, "max_backtracks": 1},
            "calibration": {"delta": 0.1, "pvalue": "binomial"},
            "limits": {"max_concurrency": 8},
        },
        "policies": policies,
        "seeds": seeds,
        "splits": ["calib", "test"],
    }
    path = tmp_path / f"{suite_id}.yaml"
    path.write_text(yaml.safe_dump(data))
    return SuiteConfig.from_yaml(path)


def test_average_risk_is_monotone_in_the_knob(tmp_path) -> None:
    policies = {
        "per_step_eps": {"kind": "log", "min": 0.0005, "max": 0.5, "points": 6},
        "marginal": {"kind": "log", "min": 0.002, "max": 1.0, "points": 6},
    }
    suite = _suite(
        tmp_path, "m5mono", policies, seeds=[0], n_steps=5, K=10, K_prime=20, n_inst=200, alpha=0.1
    )
    root = run_suite(suite, workers=8, results_dir=tmp_path / "r")
    manifest = load_manifest(root)
    for policy in policies:
        sub = manifest[(manifest["policy"] == policy) & (manifest["split"] == "calib")].sort_values(
            "knob"
        )
        risk = sub["final_error"].tolist()
        for a, b in itertools.pairwise(risk):
            assert b >= a - 0.05, (policy, risk)
        assert risk[-1] - risk[0] >= 0.05, (policy, risk)


def test_ltt_coverage_across_seeds(tmp_path) -> None:
    policies = {"per_step_eps": {"kind": "log", "min": 0.001, "max": 0.5, "points": 6}}
    seeds = list(range(12))
    n_test, alpha = 60, 0.2
    suite = _suite(
        tmp_path,
        "m5cov",
        policies,
        seeds=seeds,
        n_steps=3,
        K=5,
        K_prime=10,
        n_inst=n_test,
        alpha=alpha,
    )
    root = run_suite(suite, workers=8, results_dir=tmp_path / "r")
    table = calibrate_suite(root)
    assert (root / "ltt.json").exists() and (root / "coverage.json").exists()
    feasible = table[table["feasible"]]
    assert len(feasible) >= 8, table
    slack = 2 * math.sqrt(alpha * (1 - alpha) / n_test)
    ok = (feasible["test_error_at_star"] <= alpha + slack).mean()
    assert ok >= 0.9, feasible[["seed", "knob_star", "calib_risk_at_star", "test_error_at_star"]]
    cov = json.loads((root / "coverage.json").read_text())["per_step_eps"]
    assert cov["seeds"] == 12 and cov["feasible"] == len(feasible)
    assert (feasible["calib_risk_at_star"] <= alpha).all()
