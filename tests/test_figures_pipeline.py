import yaml

from ucagent.experiments.stages import PipelineConfig, run_pipeline


def test_tiny_pipeline_produces_all_figures(tmp_path) -> None:
    base = {
        "run_id": "tiny",
        "task": {
            "n_steps": 3,
            "ambiguity": 0.5,
            "n_instances": {"train": 12, "calib": 10, "test": 10},
        },
        "llm": {"K": 3, "K_prime": 6},
        "risk": {"influence": "counterfactual"},
        "control": {"alpha": 0.2},
        "limits": {"max_concurrency": 4},
    }
    suite = {
        "suite_id": "tiny_suite",
        "base": base,
        "policies": {
            "none": None,
            "per_step_eps": {"kind": "log", "min": 0.01, "max": 0.5, "points": 2},
            "marginal": {"kind": "log", "min": 0.01, "max": 1.0, "points": 2},
            "budgeted": {"kind": "linear", "min": 0.5, "max": 2.0, "points": 2},
        },
        "seeds": [0],
        "splits": ["calib", "test"],
    }
    rec = dict(
        suite,
        suite_id="tiny_rec",
        base={**base, "task": {**base["task"], "reconvergence": [0.5, 0.5, 0.0]}},
    )
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))
    (tmp_path / "rec.yaml").write_text(yaml.safe_dump(rec))
    pipe = {
        "pipeline_id": "tiny_pipe",
        "suite": "suite.yaml",
        "reconverge_suite": "rec.yaml",
        "fig1": {"ns": [1, 2], "n_train": 8, "n_test": 8},
        "out_dir": str(tmp_path / "figs"),
    }
    (tmp_path / "pipe.yaml").write_text(yaml.safe_dump(pipe))
    cfg = PipelineConfig.from_yaml(tmp_path / "pipe.yaml")
    assert cfg.suite.endswith("suite.yaml")
    out = run_pipeline(cfg, workers=4, results_dir=tmp_path / "results")
    names = {p.rsplit("/", 1)[-1] for p in out["figures"]}
    for stem in [
        "fig1_error_vs_n",
        "fig2_error_vs_cost",
        "fig3_intervention_locations",
        "fig4_reliability_certificate",
        "risk_vs_knob",
    ]:
        assert f"{stem}.pdf" in names and f"{stem}.png" in names, names
    assert (tmp_path / "figs" / "timing.txt").exists() and (
        tmp_path / "figs" / "reconverge"
    ).exists()
    assert "total" in out["timings_s"]
    # resume: a second call reuses everything
    out2 = run_pipeline(cfg, workers=4, results_dir=tmp_path / "results")
    assert out2["timings_s"]["suite"] < out["timings_s"]["suite"]
