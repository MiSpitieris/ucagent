"""Regression tests for the implementation review of 2026-11-08.

The review and its reproduction scripts were committed in ``4dc2c45`` and removed from
the tree before the public release (they remain in the git history). Each test is the
review's reproduction turned
into the statement of the intended behaviour: R1-R5, R10, R11 and R13 are
repaired; R6-R9 and R12 are deferred and have no test here.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.helpers.configs import small_config
from ucagent.api import calibrate_knob, fit_calibrators, observations_from, run_instances
from ucagent.core.control.ltt import binomial_pvalue, hoeffding_pvalue
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import Sample
from ucagent.core.uq.base import StepContext, make_estimator
from ucagent.core.uq.calibration import (
    CalibrationArtifact,
    IsotonicData,
    ResampleTableData,
    StepObservation,
    fit_artifact,
)
from ucagent.core.uq.equivalence import NormalizedExactMatch
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import Worker
from ucagent.experiments.suite import ResumeMismatch, SuiteConfig, _run_dir, run_suite
from ucagent.tasks.base import make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig
from ucagent.ui import bootstrap


def test_r2_resample_table_is_the_expansion_experiment() -> None:
    """The table equals the conditional error of the batch expanded from K to K' draws."""
    estimator = make_estimator("disagreement", NormalizedExactMatch())
    context = StepContext(instance_id="x", step=1, attempt=0, current="", n_steps=1)
    rows: list[StepObservation] = []
    totals: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for draws in product([True, False], repeat=4):  # binary answers, P(correct) = 0.7
        samples = [
            Sample(sample_index=i, text=str(x), answer="correct" if x else "wrong")
            for i, x in enumerate(draws)
        ]
        at_k = estimator.estimate(samples[:2], context)
        at_kp = estimator.estimate(samples, context)
        weight = 7 ** sum(draws) * 3 ** (4 - sum(draws))
        wrong = at_kp.committed != "correct"
        totals[at_k.u][0] += weight
        totals[at_k.u][1] += weight * wrong
        rows.extend(
            [
                StepObservation(
                    n_steps=1,
                    step=1,
                    u_K=at_k.u,
                    wrong_K=at_k.committed != "correct",
                    u_K_prime=at_kp.u,
                    wrong_K_prime=wrong,
                )
            ]
            * weight
        )
    fitted = fit_artifact(
        rows,
        estimator="disagreement",
        equivalence="normalized_exact",
        K=2,
        K_prime=4,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=10000,
    )
    # bucket 0 holds u = 0 (unanimous at K), bucket 2 holds u = 0.5 (split at K)
    assert fitted.resample_table.ehat["0"] == pytest.approx(totals[0.0][1] / totals[0.0][0])
    assert fitted.resample_table.ehat["2"] == pytest.approx(totals[0.5][1] / totals[0.5][0])


def test_r3_verbalized_fit_sees_the_runtime_confidences() -> None:
    cfg = small_config(
        task={"n_steps": 1, "ambiguity": 0.0, "n_instances": {"train": 5, "calib": 1, "test": 1}},
        llm={"K": 2, "K_prime": 4, "verbalized": True, "verbalized_sigma": 0.0},
        uq={"estimator": "verbalized"},
    )
    fit_cfg = cfg.with_updates(llm={"K": 4}).model_copy(update={"split": "train"})
    worker = Worker.from_config(fit_cfg)
    trajectories = [worker.run_instance(x.instance_id) for x in worker.task.instances("train")]
    assert all(t.sample_confidences for t in trajectories)
    observations = observations_from(cfg, trajectories)
    for t, obs in zip(trajectories, observations, strict=True):
        assert obs[0].u_K_prime == pytest.approx(
            t.steps[0].u
        )  # the same feature at fit and run time
        assert t.steps[0].extra["n_with_confidence"] == 4.0


def test_r3_fit_refuses_a_confidence_estimator_without_confidences() -> None:
    cfg = small_config(
        task={"n_steps": 1, "ambiguity": 0.0, "n_instances": {"train": 3, "calib": 1, "test": 1}},
        llm={"K": 1, "K_prime": 2, "verbalized": False},
        uq={"estimator": "verbalized"},
    )
    with pytest.raises(ValueError, match="reads stated confidences"):
        fit_calibrators(cfg, workers=1)


def _foreign_artifact() -> CalibrationArtifact:
    return CalibrationArtifact(
        estimator="disagreement",
        equivalence="normalized_exact",
        K=1,
        K_prime=2,
        task_config_hash="synthetic-hash",
        config_hash="mock-hash",
        data_hash="train-hash",
        isotonic_K=IsotonicData(x=[0], y=[0.99]),
        isotonic_K_prime=IsotonicData(x=[0], y=[0.01]),
        resample_table=ResampleTableData(),
    )


def test_r4_foreign_artifact_is_rejected(tmp_path: Path) -> None:
    cfg = small_config(llm={"K": 1, "K_prime": 2})
    changed = cfg.with_updates(
        llm={"backend": "ollama:different-model", "temperature": 0.9},
        uq={"equivalence": "normalized_contains"},
        task={"name": "different_task", "n_steps": 7},
    )
    with pytest.raises(ValueError):
        _foreign_artifact().to_tables(changed)
    with pytest.raises(ValueError, match="task config"):  # same judge, other task
        _foreign_artifact().to_tables(cfg.with_updates(task={"n_steps": 7}))
    path = tmp_path / "suite_small" / "artifacts" / "seed0.json"
    _foreign_artifact().save(path)
    resolved, note = bootstrap.resolve_artifact(changed, "auto", tmp_path)
    assert resolved is None and "identity tables" in note


def test_r5_resume_refuses_a_changed_configuration(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        run_id="base",
        task={"n_steps": 1, "ambiguity": 0, "n_instances": {"train": 2, "calib": 2, "test": 2}},
        llm={"K": 1, "K_prime": 2},
        limits={"max_concurrency": 1},
    )
    suite = SuiteConfig(
        suite_id="same", base=cfg, policies={"none": None}, splits=["test"], fit_artifact=False
    )
    run_suite(suite, results_dir=tmp_path, workers=1)
    run_suite(suite, results_dir=tmp_path, workers=1)  # the same request resumes quietly
    changed = suite.model_copy(
        update={
            "base": cfg.with_updates(
                task={
                    "n_steps": 2,
                    "ambiguity": 1,
                    "n_instances": {"train": 5, "calib": 5, "test": 5},
                }
            )
        }
    )
    with pytest.raises(ResumeMismatch, match="different configuration"):
        run_suite(changed, results_dir=tmp_path, workers=1)
    assert run_suite(changed, results_dir=tmp_path, workers=1, resume=False)


def test_r5_distinct_knobs_get_distinct_run_directories(tmp_path: Path) -> None:
    a = _run_dir(tmp_path, "marginal", 0.12345611, "test", 0, "grid")
    b = _run_dir(tmp_path, "marginal", 0.12345612, "test", 0, "grid")
    assert a != b
    # knobs that six significant digits render exactly keep their short names
    assert _run_dir(tmp_path, "marginal", 0.08, "test", 0, "grid").name.endswith("k0.08__test__s0")


def test_r10_repeated_instance_ids_are_isolated() -> None:
    task = SyntheticChainTask(SyntheticConfig(n_steps=2, n_instances={"test": 1}))
    instance_id = next(iter(task.instances("test"))).instance_id
    settings = RunSettings(policy="none", K=2, K_prime=4)
    first, second = run_instances(
        task,
        MockChainLLM(task=task, seed=0),
        settings,
        [instance_id, instance_id],
        make_human(task, settings),
    )
    assert first.llm_calls == second.llm_calls == 4
    assert first.steps == second.steps and first.total_cost == second.total_cost
    assert first.thread_id != second.thread_id


def test_r13_invalid_counts_and_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="errors must lie"):
        calibrate_knob("p", [(0.1, 100, -1)], alpha=0.1, delta=0.1)
    with pytest.raises(ValueError, match="errors must lie"):
        binomial_pvalue(5, 3, 0.1)
    with pytest.raises(ValueError, match="alpha"):
        hoeffding_pvalue(1, 10, 1.5)
    with pytest.raises(ValueError, match="delta"):
        calibrate_knob("p", [(0.1, 100, 1)], alpha=0.1, delta=0.0)
    with pytest.raises(ValueError, match="distinct"):
        calibrate_knob("p", [(0.1, 100, 1), (0.1, 100, 2)], alpha=0.1, delta=0.1)
    for bad in (
        {"llm": {"K": 5, "K_prime": 4}},
        {"llm": {"K": 0}},
        {"control": {"alpha": 1.5}},
        {"control": {"max_resamples": -1}},
        {"control": {"costs": {"ask_human": -1.0}}},
        {"oracle": {"p_h": -0.1}},
        {"calibration": {"delta": 1.0}},
        {"task": {"n_steps": 0}},
        {"task": {"n_instances": {"train": -1}}},
        {"limits": {"max_concurrency": 0}},
    ):
        with pytest.raises(ValidationError):
            ExperimentConfig(run_id="x", **bad)
    with pytest.raises(ValidationError):
        RunSettings(K=5, K_prime=4)
    assert RunSettings(K=5, K_prime=5).max_resamples == 1
