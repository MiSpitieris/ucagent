"""An estimator registered by a plugin module runs through the fit and through a suite.

The fit used to look its estimator up before any plugin module was imported, so a plugin
estimator failed in ``ucagent fit`` and in a fresh suite, while a plugin policy or task worked
(the review of 2026-11-17). Tests that register an estimator in the test process itself hide
this, so each test here first makes the parent process forget the plugin.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest
import yaml

from tests.helpers.configs import small_config
from ucagent.api import fit_calibrators
from ucagent.core.uq.base import ESTIMATORS
from ucagent.experiments.suite import SuiteConfig, load_manifest, run_suite

PLUGIN = "tests.helpers.plugin_estimator"
NAME = "plugin_disagreement"


@pytest.fixture
def forgotten_plugin() -> Iterator[None]:
    """The parent process as a fresh one would be: the plugin not imported, its name unknown."""

    def forget() -> None:
        sys.modules.pop(PLUGIN, None)
        ESTIMATORS.pop(NAME, None)

    forget()
    yield
    forget()


def test_the_fit_imports_a_plugin_estimator_before_it_looks_it_up(forgotten_plugin: None) -> None:
    config = small_config(
        plugins=[PLUGIN], uq={"estimator": NAME}, llm={"backend": "mock", "K": 3, "K_prime": 6}
    )
    assert NAME not in ESTIMATORS
    artifact = fit_calibrators(config, workers=1)
    assert artifact.estimator == NAME and artifact.resample_map is not None


def test_a_fresh_suite_runs_a_plugin_estimator_with_two_workers(
    forgotten_plugin: None, tmp_path
) -> None:
    data = {
        "suite_id": "plugin_estimator",
        "base": {
            "run_id": "plugin_estimator",
            "plugins": [PLUGIN],
            "task": {
                "n_steps": 2,
                "ambiguity": 0.5,
                "n_instances": {"train": 12, "calib": 8, "test": 8},
            },
            "llm": {"K": 3, "K_prime": 6},
            "uq": {"estimator": NAME},
            "control": {"alpha": 0.2},
            "limits": {"max_concurrency": 2},
        },
        "policies": {
            "none": None,
            "marginal": {"kind": "log", "min": 0.05, "max": 0.5, "points": 2},
        },
        "seeds": [0],
        "splits": ["test"],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert NAME not in ESTIMATORS
    root = run_suite(SuiteConfig.from_yaml(path), workers=2, results_dir=tmp_path / "results")
    manifest = load_manifest(root)
    assert sorted(set(manifest["policy"])) == ["marginal", "none"]
    assert len(manifest) == 3  # one uncontrolled run and two knobs of the rule
