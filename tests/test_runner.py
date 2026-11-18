import pytest
from typer.testing import CliRunner

import tests.helpers.policies  # noqa: F401
from tests.helpers.configs import small_config
from ucagent.cli import app
from ucagent.experiments.io import read_frame, read_trajectories
from ucagent.experiments.runner import run_experiment


def _strip(t):
    return t.model_dump(exclude={"duration_s"})


def test_workers_do_not_change_results(tmp_path) -> None:
    cfg = small_config(task={"n_instances": {"test": 12}}, llm={"K": 3})
    one = run_experiment(cfg, workers=1, write=False)
    three = run_experiment(cfg, workers=3, write=False)
    assert [_strip(t) for t in one.trajectories] == [_strip(t) for t in three.trajectories]
    assert len(one.trajectories) == 12


def test_outputs_and_round_trip(tmp_path) -> None:
    cfg = small_config(task={"n_instances": {"test": 5}})
    res = run_experiment(cfg, workers=1, results_dir=tmp_path)
    out = res.out_dir
    assert out is not None
    for name in ["trajectories.parquet", "config.yaml", "git_sha.txt", "env.txt", "timing.txt"]:
        assert (out / name).exists()
    frame = read_frame(out / "trajectories.parquet")
    assert len(frame) == 5 and "steps_json" in frame.columns and "n_ask_human" in frame.columns
    back = read_trajectories(out / "trajectories.parquet")
    assert [t.model_dump() for t in back] == [t.model_dump() for t in res.trajectories]
    assert all(s.locally_correct is not None for t in back for s in t.steps)


def test_dry_run_writes_nothing(tmp_path) -> None:
    cfg = small_config()
    res = run_experiment(cfg, dry_run=True, results_dir=tmp_path)
    assert res.trajectories == [] and res.out_dir is None and not list(tmp_path.iterdir())


def test_global_oracle_is_refused_outside_tests() -> None:
    cfg = small_config(oracle={"scope": "global"}, control={"policy": "ask_last"})
    with pytest.raises(ValueError, match="global"):
        run_experiment(cfg, workers=1, write=False)
    res = run_experiment(cfg, workers=1, write=False, allow_global_oracle=True)
    assert all(t.final_correct for t in res.trajectories)


def test_cli_run_dry(tmp_path) -> None:
    cfg = small_config()
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    result = CliRunner().invoke(app, ["run", "--config", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    result = CliRunner().invoke(
        app, ["run", "--config", str(path), "--workers", "1", "--results-dir", str(tmp_path / "r")]
    )
    assert result.exit_code == 0, result.output
    assert "trajectories" in result.output and (tmp_path / "r" / "test").exists()


def test_shared_pool_matches_in_process(tmp_path) -> None:
    from ucagent.experiments.runner import make_pool

    cfg = small_config(task={"n_instances": {"test": 6}})
    cfg2 = cfg.with_updates(control={"policy": "ask_last"}).model_copy(update={"run_id": "t2"})
    with make_pool(2, cfg.plugins) as pool:
        a = run_experiment(cfg, write=False, pool=pool)
        b = run_experiment(cfg2, write=False, pool=pool)
    assert [_strip(t) for t in a.trajectories] == [
        _strip(t) for t in run_experiment(cfg, workers=1, write=False).trajectories
    ]
    assert all(t.n_interventions == 1 for t in b.trajectories)


def test_build_llm_forwards_model_kwargs_and_yaml_round_trip(monkeypatch, tmp_path) -> None:
    from ucagent.core.llm import backend
    from ucagent.experiments.config import ExperimentConfig
    from ucagent.experiments.runner import build_llm, build_task

    seen: dict = {}

    def fake(name, temperature=0.7, **kw):
        seen.update(name=name, temperature=temperature, **kw)
        return object()

    monkeypatch.setattr(backend, "get_chat_model", fake)
    extra = {"reasoning": "low", "num_ctx": 8192, "num_predict": 1024}
    cfg = small_config(llm={"backend": "ollama:gpt-oss:20b", "model_kwargs": extra})
    build_llm(cfg, build_task(cfg))
    assert seen["name"] == "ollama:gpt-oss:20b" and seen["model_kwargs"] == extra
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    assert ExperimentConfig.from_yaml(path).llm.model_kwargs == extra
    assert small_config().llm.model_kwargs == {}
