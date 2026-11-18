"""Experiment stages that are not plain suites, and the figure pipeline.

``run_fig1_stage`` fits an artifact (with counterfactual influence) and runs
the uncontrolled agent for several chain lengths. ``run_pipeline`` chains the
main suite, the reconvergence suite, calibration, the fig1 stage and all
figures into one directory, and records the wall time: this is ``make figures``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ucagent.api import fit_calibrators
from ucagent.experiments import provenance
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.figures import fig1, fig2, fig3, fig4, plot_risk_vs_knob
from ucagent.experiments.metrics import summary
from ucagent.experiments.runner import build_task, make_pool, run_experiment
from ucagent.experiments.suite import SuiteConfig, calibrate_suite, check_run, run_suite

log = logging.getLogger("ucagent")


class Fig1Config(BaseModel):
    """Chain lengths and sizes for the fig1 stage."""

    ns: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 8])
    n_train: int = 100
    n_test: int = 100
    seeds: list[int] = Field(default_factory=lambda: [0])


class PipelineConfig(BaseModel):
    """``kind: pipeline`` YAML: the suites and stages behind the README figures."""

    kind: str = "pipeline"
    pipeline_id: str
    suite: str
    reconverge_suite: str | None = None
    fig1: Fig1Config = Field(default_factory=Fig1Config)
    out_dir: str = "docs/figures"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load from YAML; relative suite paths resolve against the YAML's folder."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls.model_validate(data)
        for name in ("suite", "reconverge_suite"):
            value = getattr(cfg, name)
            if value and not Path(value).is_absolute() and not Path(value).exists():
                setattr(cfg, name, str(path.parent / value))
        return cfg


def run_fig1_stage(
    base: ExperimentConfig,
    fig: Fig1Config,
    out_dir: str | Path,
    *,
    pool: Any = None,
    workers: int | None = None,
    resume: bool = True,
) -> Path:
    """Policy ``none`` at several chain lengths; writes ``fig1_summary.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "fig1_summary.csv"
    rows: list[dict[str, Any]] = []
    for seed in fig.seeds:
        for n in fig.ns:
            run_dir = out_dir / f"n{n}__s{seed}"
            parquet = run_dir / "trajectories.parquet"
            cfg = base.with_updates(
                task={
                    "n_steps": n,
                    "n_instances": {"train": fig.n_train, "calib": 1, "test": fig.n_test},
                    "seed": seed,
                },
                control={"policy": "none", "knob": None},
                risk={"influence": "counterfactual", "propagation": "weighted"},
                uq={"artifact": None},
            ).model_copy(update={"run_id": run_dir.name, "split": "test", "seed": seed})
            artifact_path = run_dir / "artifact.json"
            if not (resume and parquet.exists() and artifact_path.exists()):
                fit_calibrators(cfg, out_path=artifact_path, pool=pool, workers=workers)
                cfg = cfg.with_updates(uq={"artifact": str(artifact_path)})
                result = run_experiment(cfg, results_dir=out_dir, pool=pool, workers=workers)
                trajectories = result.trajectories
            else:
                from ucagent.experiments.io import read_trajectories

                task = build_task(cfg)
                ids = [inst.instance_id for inst in task.instances("test")]
                check_run(
                    run_dir,
                    cfg.with_updates(uq={"artifact": str(artifact_path)}),
                    provenance.record(task, ids),
                )
                trajectories = read_trajectories(parquet)
            stats = summary(trajectories, alpha=cfg.control.alpha)
            rows.append({"n_steps": n, "seed": seed, "backend": cfg.llm.backend, **stats})
            pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def run_pipeline(
    pipeline: PipelineConfig,
    *,
    workers: int | None = None,
    results_dir: str | Path = "results",
    resume: bool = True,
) -> dict[str, Any]:
    """Everything behind ``make figures``; returns a dict of outputs and timings."""
    t0 = time.perf_counter()
    results_dir = Path(results_dir)
    out_dir = Path(pipeline.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = SuiteConfig.from_yaml(pipeline.suite)
    n_workers = workers or min(suite.base.limits.max_concurrency, os.cpu_count() or 1)
    timings: dict[str, float] = {}
    outputs: dict[str, Any] = {}
    pool = make_pool(n_workers, suite.base.plugins) if n_workers > 1 else None
    try:
        t = time.perf_counter()
        suite_dir = run_suite(suite, workers=n_workers, results_dir=results_dir, resume=resume)
        calibrate_suite(suite_dir)
        timings["suite"] = time.perf_counter() - t
        outputs["suite_dir"] = str(suite_dir)
        if pipeline.reconverge_suite:
            t = time.perf_counter()
            rec = SuiteConfig.from_yaml(pipeline.reconverge_suite)
            rec_dir = run_suite(rec, workers=n_workers, results_dir=results_dir, resume=resume)
            calibrate_suite(rec_dir)
            timings["reconverge_suite"] = time.perf_counter() - t
            outputs["reconverge_dir"] = str(rec_dir)
        t = time.perf_counter()
        fig1_csv = run_fig1_stage(
            suite.base,
            pipeline.fig1,
            results_dir / pipeline.pipeline_id / "fig1",
            pool=pool,
            workers=n_workers,
            resume=resume,
        )
        timings["fig1_stage"] = time.perf_counter() - t
    finally:
        if pool is not None:
            pool.shutdown()
    t = time.perf_counter()
    paths = (
        fig1(fig1_csv, out_dir) + fig2(suite_dir, out_dir) + plot_risk_vs_knob(suite_dir, out_dir)
    )
    paths += fig3(suite_dir, out_dir)
    paths += fig4(suite_dir, out_dir)
    if pipeline.reconverge_suite:
        rec_out = out_dir / "reconverge"
        paths += fig2(outputs["reconverge_dir"], rec_out) + fig3(outputs["reconverge_dir"], rec_out)
    timings["figures"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    outputs["figures"] = [str(p) for p in paths]
    outputs["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    outputs["workers"] = n_workers
    (results_dir / pipeline.pipeline_id).mkdir(parents=True, exist_ok=True)
    (results_dir / pipeline.pipeline_id / "pipeline.json").write_text(
        json.dumps(outputs, indent=1), encoding="utf-8"
    )
    (out_dir / "timing.txt").write_text(
        "\n".join(f"{k} {v}" for k, v in outputs["timings_s"].items()) + f"\nworkers {n_workers}\n",
        encoding="utf-8",
    )
    shutil.copy(results_dir / pipeline.pipeline_id / "pipeline.json", out_dir / "pipeline.json")
    log.info("pipeline %s done in %.0fs", pipeline.pipeline_id, timings["total"])
    return outputs
