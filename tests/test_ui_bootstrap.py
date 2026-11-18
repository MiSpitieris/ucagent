"""Building the run from the page's choices (no Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.configs import small_config
from ucagent.core.control.policies import DEFAULT_KNOB
from ucagent.experiments.runner import Worker
from ucagent.ui import bootstrap
from ucagent.ui.bootstrap import UiChoices

SUITE_ARTIFACT = Path("results/suite_small/artifacts/seed0.json")


def test_defaults_from_config_and_build_config(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    cfg = small_config(
        llm={"backend": "ollama:llama3.1:8b", "K": 4}, control={"policy": "marginal"}
    )
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    defaults = bootstrap.defaults_from_config(bootstrap.base_config(str(path)))
    assert defaults["backend"] == "ollama" and defaults["ollama_model"] == "llama3.1:8b"
    assert defaults["k"] == 4 and defaults["n_steps"] == 3 and defaults["ambiguity"] == 0.5
    assert defaults["knob"] == DEFAULT_KNOB["marginal"]
    choices = UiChoices(config_path=str(path), backend="mock", policy="per_step_u", knob=0.3, K=2)
    built = bootstrap.build_config(choices)
    assert built.llm.backend == "mock" and built.llm.K == 2 and built.task.n_steps == 5
    assert built.control.policy == "per_step_u" and built.control.knob == 0.3
    assert built.run_id == "ui" and built.plugins == ["tests.helpers.policies"]


def test_resolve_artifact_none_and_no_match(tmp_path: Path) -> None:
    cfg = bootstrap.build_config(UiChoices(estimator="disagreement", K=3, K_prime=6))
    path, note = bootstrap.resolve_artifact(cfg, "none", tmp_path)
    assert path is None and "identity tables" in note
    path, note = bootstrap.resolve_artifact(cfg, "auto", tmp_path)
    assert path is None and "no fitted artifact matches disagreement, K=3, K'=6" in note


@pytest.mark.skipif(not SUITE_ARTIFACT.exists(), reason="needs results/suite_small (make figures)")
def test_resolve_artifact_picks_a_compatible_one() -> None:
    import yaml

    cfg = bootstrap.build_config(UiChoices(estimator="semantic_entropy", K=10, K_prime=20))
    # same estimator and sample counts, but the page's task block differs from the suite's:
    # a foreign artifact is no longer applied (review finding R4)
    path, note = bootstrap.resolve_artifact(cfg, "auto", "results")
    assert path is None and "identity tables" in note
    suite_task = yaml.safe_load((SUITE_ARTIFACT.parents[1] / "suite.yaml").read_text())["base"][
        "task"
    ]
    same_task = cfg.with_updates(task=suite_task, uq={"equivalence": "normalized_exact"})
    path, note = bootstrap.resolve_artifact(same_task, "auto", "results")
    assert path == str(SUITE_ARTIFACT) and "calibration artifact" in note
    cfg = bootstrap.build_config(UiChoices(estimator="semantic_entropy", K=5, K_prime=10))
    path, _ = bootstrap.resolve_artifact(cfg, "auto", "results")
    assert path is None


def test_resolve_knob_sources(tmp_path: Path) -> None:
    assert bootstrap.resolve_knob(UiChoices(policy="none")) == (None, "policy none has no knob")
    value, note = bootstrap.resolve_knob(UiChoices(policy="marginal", knob=None))
    assert value == DEFAULT_KNOB["marginal"] and "default" in note
    value, note = bootstrap.resolve_knob(UiChoices(policy="marginal", knob=0.05))
    assert value == 0.05 and "sidebar" in note
    choices = UiChoices(policy="marginal", knob=0.05, use_ltt_knob=True, results_dir=str(tmp_path))
    assert bootstrap.resolve_knob(choices)[0] == 0.05  # no table: falls back to the sidebar knob
    suite = tmp_path / bootstrap.KNOB_SUITE
    suite.mkdir()
    pd.DataFrame(
        {
            "policy": ["marginal", "marginal", "marginal", "final_u"],
            "seed": [0, 1, 2, 0],
            "feasible": [True, True, False, True],
            "knob_star": [0.02, 0.04, 9.0, 0.5],
            "alpha": [0.1, 0.1, 0.1, 0.1],
        }
    ).to_csv(suite / "ltt.csv", index=False)
    # two feasible seeds certified different knobs: their median is uncertified, so none is imported
    value, note = bootstrap.ltt_knob("marginal", tmp_path)
    assert value is None and "different knobs" in note
    assert bootstrap.resolve_knob(choices)[0] == 0.05  # falls back to the sidebar knob
    pd.DataFrame(
        {
            "policy": ["marginal", "marginal", "marginal"],
            "seed": [0, 1, 2],
            "feasible": [True, True, False],
            "knob_star": [0.02, 0.02, 9.0],
            "alpha": [0.1, 0.1, 0.1],
        }
    ).to_csv(suite / "ltt.csv", index=False)
    value, note = bootstrap.resolve_knob(choices)
    assert value == pytest.approx(0.02) and "certified by all 2 seeds" in note
    assert bootstrap.ltt_knob("budgeted", tmp_path)[0] is None


def test_prepare_run_worker_key_and_instances(tmp_path: Path) -> None:
    choices = UiChoices(policy="marginal", results_dir=str(tmp_path), n_steps=3, K=2, K_prime=4)
    cfg, notes = bootstrap.prepare_run(choices)
    assert cfg.control.knob == DEFAULT_KNOB["marginal"] and cfg.uq.artifact is None
    assert len(notes) == 2
    key = bootstrap.worker_key(cfg)
    assert key != bootstrap.worker_key(cfg.with_updates(llm={"K": 3}))
    assert bootstrap.worker_key(cfg) == key
    worker = bootstrap.build_worker(cfg)
    assert isinstance(worker, Worker) and bootstrap.llm_counters(worker.llm) is None
    inst = bootstrap.instance_at(worker.task, "test", 2)
    assert inst.instance_id == "test-00002-s0" and inst.n_steps == 3
    with pytest.raises(ValueError, match="no instance"):
        bootstrap.instance_at(worker.task, "test", 10**6)
    with pytest.raises(ValueError, match="global oracle"):
        bootstrap.build_worker(cfg.with_updates(oracle={"scope": "global"}))


def test_list_configs_and_ollama_status(tmp_path: Path) -> None:
    (tmp_path / "exp.yaml").write_text("run_id: x\n", encoding="utf-8")
    (tmp_path / "suite.yaml").write_text("kind: suite\nsuite_id: s\n", encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("a: [", encoding="utf-8")
    assert bootstrap.list_configs(tmp_path) == [str(tmp_path / "exp.yaml")]
    assert bootstrap.ollama_status("http://127.0.0.1:9", timeout=0.5) == (False, [])
    bootstrap.ensure_cwd_on_path()
    bootstrap.ensure_cwd_on_path()
