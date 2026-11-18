"""Suites: one task, several policies, each over its knob grid, several seeds.

A suite writes ``results/<suite_id>/`` with one ``runs/<...>/`` directory per
(policy, knob, split, seed) run, a ``manifest.csv`` listing them, and the
calibration artifact(s). Figures and Learn-then-Test read the manifest only;
nothing is ever typed in by hand.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ucagent.api import fit_calibrators
from ucagent.core.control.ltt import LTTResult, calibrate_knob
from ucagent.core.uq.calibration import CalibrationArtifact, load_artifact
from ucagent.core.utils import stable_digest
from ucagent.experiments import provenance
from ucagent.experiments.config import ExperimentConfig, GridConfig
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.metrics import summary
from ucagent.experiments.runner import build_task, make_pool, run_experiment

log = logging.getLogger("ucagent")


class ExtraPoint(BaseModel):
    """A single labelled operating point, e.g. the Bonferroni gate ``tau = alpha / n``."""

    policy: str
    label: str
    knob: float | str
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Variant(BaseModel):
    """A labelled set of config overrides run for every policy and knob (ablations)."""

    label: str = "main"
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SuiteConfig(BaseModel):
    """A suite definition (``configs/*.yaml`` with ``kind: suite``)."""

    kind: str = "suite"
    suite_id: str
    base: ExperimentConfig
    policies: dict[str, GridConfig | None]
    seeds: list[int] = Field(default_factory=lambda: [0])
    splits: list[str] = Field(default_factory=lambda: ["calib", "test"])
    extra_points: list[ExtraPoint] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=lambda: [Variant()])
    fit_artifact: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> SuiteConfig:
        """Load from YAML."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if "run_id" not in data.get("base", {}):
            data.setdefault("base", {})["run_id"] = data.get("suite_id", "suite")
        return cls.model_validate(data)


def grid_values(grid: GridConfig) -> list[float]:
    """The knob values of a grid, ascending."""
    if grid.points <= 1:
        return [grid.min]
    if grid.kind == "log":
        lo, hi = math.log(grid.min), math.log(grid.max)
        return [math.exp(lo + (hi - lo) * k / (grid.points - 1)) for k in range(grid.points)]
    return [grid.min + (grid.max - grid.min) * k / (grid.points - 1) for k in range(grid.points)]


@functools.lru_cache(maxsize=512)
def _stored_grids(suite_yaml: str) -> dict[str, tuple[float, ...] | None]:
    data = yaml.safe_load(Path(suite_yaml).read_text(encoding="utf-8"))
    return {
        str(policy): None if grid is None else tuple(grid_values(GridConfig.model_validate(grid)))
        for policy, grid in (data.get("policies") or {}).items()
    }


def configured_knobs(suite_dir: str | Path, policy: str) -> list[float] | None:
    """The knobs that a finished suite's stored configuration gives a policy.

    None when the policy has no grid there. Learn-then-Test is walked along this grid
    (``calibrate_knob(..., grid=...)``), so that a calibration run that is missing from
    the manifest ends the walk and cannot be passed over.
    """
    grids = _stored_grids(str(Path(suite_dir) / "suite.yaml"))
    knobs = grids.get(policy)
    return None if knobs is None else list(knobs)


def resolve_knob(value: float | str, config: ExperimentConfig) -> float:
    """``alpha/n`` style expressions for extra points."""
    if isinstance(value, int | float):
        return float(value)
    env = {"alpha": config.control.alpha, "n": config.task.n_steps}
    return float(eval(value, {"__builtins__": {}}, env))


def _run_dir(
    root: Path,
    policy: str,
    knob: float | None,
    split: str,
    seed: int,
    label: str,
    variant: str = "main",
) -> Path:
    if knob is None:
        knob_txt = "none"
    else:
        knob_txt = f"{knob:.6g}"
        if float(knob_txt) != knob:  # six significant digits lose the value: disambiguate
            knob_txt += "-" + stable_digest(repr(float(knob)))[:6]
    prefix = "" if variant == "main" else f"{variant}__"
    return root / "runs" / f"{prefix}{policy}__{label or 'grid'}__k{knob_txt}__{split}__s{seed}"


class ResumeMismatch(RuntimeError):
    """An existing output was produced under a different configuration or from other inputs.

    Review finding R5 (configuration) and finding C04 of the code review of 2026-11-11
    (inputs: the files the task reads, the questions as it presents them, the code).
    """


Inputs = dict[tuple[str, str], dict[str, Any]]


def _check_resume(
    stored: dict[str, Any], requested: dict[str, Any], what: str, where: Path
) -> None:
    if stored != requested:
        changed = sorted(
            k for k in set(stored) | set(requested) if stored.get(k) != requested.get(k)
        )
        raise ResumeMismatch(
            f"{what} at {where} was produced under a different configuration "
            f"(differs in {changed}); use a new suite_id or --no-resume"
        )


def current_inputs(cache: Inputs, config: ExperimentConfig, split: str) -> dict[str, Any]:
    """The inputs a run of ``config`` on ``split`` has today (worked out once per task)."""
    key = (stable_digest([config.task.model_dump(), config.plugins]), split)
    if key not in cache:
        task = build_task(config)
        ids = [inst.instance_id for inst in task.instances(split)]
        cache[key] = provenance.record(task, ids)
    return cache[key]


def check_run(out: Path, config: ExperimentConfig, current: dict[str, Any]) -> None:
    """Refuse to reuse the finished run at ``out`` unless its config and inputs are today's.

    A run without ``config.yaml`` or without a record of its inputs is refused as well:
    what cannot be compared is not assumed to match.
    """
    stored_cfg = out / "config.yaml"
    if not stored_cfg.exists():
        raise ResumeMismatch(
            f"run at {out} has results but no config.yaml; use a new suite_id or --no-resume"
        )
    stored = yaml.safe_load(stored_cfg.read_text(encoding="utf-8")) or {}
    _check_resume(stored, config.model_dump(mode="json"), "run", out)
    changed = provenance.differences(provenance.read(out), current)
    if changed:
        raise ResumeMismatch(
            f"run at {out} was computed from other inputs ({'; '.join(changed)}); "
            "use a new suite_id or --no-resume"
        )


def run_suite(
    suite: SuiteConfig,
    *,
    workers: int | None = None,
    results_dir: str | Path = "results",
    resume: bool = True,
) -> Path:
    """Run every (policy, knob, split, seed) and write the manifest; returns the suite dir.

    With ``resume`` the suite refuses to continue from outputs of a different
    configuration or of other inputs: ``suite.yaml``, each reused artifact and each
    reused run are compared with what is requested, and each run's record of its
    inputs (data files, questions, code) with today's, before anything is reused.
    """
    root = Path(results_dir) / suite.suite_id
    root.mkdir(parents=True, exist_ok=True)
    requested = suite.model_dump(mode="json")
    stored_path = root / "suite.yaml"
    if resume and stored_path.exists():
        stored = yaml.safe_load(stored_path.read_text(encoding="utf-8")) or {}
        _check_resume(stored, requested, "suite", stored_path)
    stored_path.write_text(yaml.safe_dump(requested, sort_keys=False), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_workers = workers or min(suite.base.limits.max_concurrency, os.cpu_count() or 1)
    pool = make_pool(n_workers, suite.base.plugins) if n_workers > 1 else None
    try:
        _run_points(suite, root, rows, pool, resume, {})
    finally:
        if pool is not None:
            pool.shutdown()
    (root / "timing.txt").write_text(
        f"runs {len(rows)}\nseconds {time.perf_counter() - t0:.1f}\nworkers {n_workers}\n",
        encoding="utf-8",
    )
    return root


def _run_points(
    suite: SuiteConfig,
    root: Path,
    rows: list[dict[str, Any]],
    pool: Any,
    resume: bool,
    inputs: Inputs,
) -> None:
    for seed in suite.seeds:
        base = suite.base.model_copy(update={"seed": seed})
        base = base.with_updates(task={"seed": seed})
        artifact_path: str | None = None
        if suite.fit_artifact:
            artifact_path = str(root / "artifacts" / f"seed{seed}.json")
            if resume and Path(artifact_path).exists():
                artifact = load_artifact(artifact_path)
                artifact.check_compatible(
                    base.uq.estimator,
                    base.llm.K,
                    base.llm.K_prime,
                    base.uq.equivalence,
                    stable_digest(base.task.model_dump()),
                )
                changed = provenance.differences(
                    artifact.inputs or None, current_inputs(inputs, base, "train")
                )
                if changed:
                    raise ResumeMismatch(
                        f"artifact at {artifact_path} was fitted on other inputs "
                        f"({'; '.join(changed)}); use a new suite_id or --no-resume"
                    )
            else:
                log.info("suite %s: fitting artifact for seed %d", suite.suite_id, seed)
                fit_calibrators(base, out_path=artifact_path, pool=pool)
            base = base.with_updates(uq={"artifact": artifact_path})
        points: list[tuple[str, float | None, str, dict[str, dict[str, Any]]]] = []
        for policy, grid in suite.policies.items():
            if grid is None:
                points.append((policy, None, "grid", {}))
            else:
                points.extend((policy, k, "grid", {}) for k in grid_values(grid))
        for extra in suite.extra_points:
            points.append(
                (extra.policy, resolve_knob(extra.knob, base), extra.label, extra.overrides)
            )
        for variant in suite.variants:
            _run_variant(
                suite, root, rows, pool, resume, base, seed, artifact_path, points, variant, inputs
            )


def _run_variant(
    suite: SuiteConfig,
    root: Path,
    rows: list[dict[str, Any]],
    pool: Any,
    resume: bool,
    base: ExperimentConfig,
    seed: int,
    artifact_path: str | None,
    points: list[tuple[str, float | None, str, dict[str, dict[str, Any]]]],
    variant: Variant,
    inputs: Inputs,
) -> None:
    for policy, knob, label, overrides in points:
        for split in suite.splits:
            out = _run_dir(root, policy, knob, split, seed, label, variant.label)
            cfg = base.with_updates(**variant.overrides)
            cfg = cfg.with_updates(control={"policy": policy, "knob": knob}, **overrides)
            cfg = cfg.model_copy(update={"run_id": out.name, "split": split})
            parquet = out / "trajectories.parquet"
            if resume and parquet.exists():
                check_run(out, cfg, current_inputs(inputs, cfg, split))
                trajectories = read_trajectories(parquet)
            else:
                result = run_experiment(cfg, results_dir=out.parent, pool=pool)
                trajectories = result.trajectories
            stats = summary(trajectories, alpha=cfg.control.alpha)
            rows.append(
                {
                    "policy": policy,
                    "knob": knob,
                    "label": label,
                    "variant": variant.label,
                    "split": split,
                    "seed": seed,
                    "path": str(parquet),
                    "artifact": artifact_path,
                    "alpha": cfg.control.alpha,
                    "n_steps": cfg.task.n_steps,
                    "backend": cfg.llm.backend,
                    **stats,
                }
            )
            pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def load_manifest(suite_dir: str | Path) -> pd.DataFrame:
    """The suite manifest with one row per run."""
    return pd.read_csv(Path(suite_dir) / "manifest.csv")


def load_artifact_for(suite_dir: str | Path, seed: int) -> CalibrationArtifact:
    """The artifact fitted for a seed."""
    from ucagent.core.uq.calibration import load_artifact

    return load_artifact(Path(suite_dir) / "artifacts" / f"seed{seed}.json")


def calibrate_suite(
    suite_dir: str | Path, *, calib_split: str = "calib", test_split: str = "test"
) -> pd.DataFrame:
    """Learn-then-Test per (policy, seed) from the manifest; writes ltt.json/ltt.csv/coverage.json.

    Returns one row per (policy, seed) with the selected knob, the calibration
    risk at it, and the held-out error and cost at the same knob.
    """
    suite_dir = Path(suite_dir)
    manifest = load_manifest(suite_dir)
    suite = SuiteConfig.model_validate(
        yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    )
    cal = suite.base.calibration
    grid_rows = manifest[manifest["label"] == "grid"]
    if "variant" in grid_rows.columns:
        grid_rows = grid_rows[grid_rows["variant"] == "main"]
    results: list[LTTResult] = []
    out_rows: list[dict[str, Any]] = []
    for key, group in grid_rows.groupby(["policy", "seed"]):
        policy, seed = str(cast(Any, key)[0]), int(cast(Any, key)[1])
        c = group[group["split"] == calib_split].dropna(subset=["knob"])
        if c.empty:
            continue
        knobs = c["knob"].astype(float).tolist()
        sizes = c["n"].astype(int).tolist()
        errs = c["final_error"].astype(float).tolist()
        points = [
            (float(k), int(n_c), round(float(e) * int(n_c)))
            for k, n_c, e in zip(knobs, sizes, errs, strict=True)
        ]
        grid_cfg = suite.policies.get(policy)
        res = calibrate_knob(
            policy,
            points,
            alpha=float(group["alpha"].iloc[0]),
            delta=cal.delta,
            pvalue=cal.pvalue,
            seed=seed,
            grid=None if grid_cfg is None else grid_values(grid_cfg),
        )
        if res.missing_knobs:
            log.warning(
                "%s: %s (seed %d) has no %s run at %d configured knob(s), the first at "
                "%g; nothing at or beyond it can be selected",
                suite_dir.name,
                policy,
                seed,
                calib_split,
                len(res.missing_knobs),
                min(res.missing_knobs),
            )
        results.append(res)
        row: dict[str, Any] = {
            "policy": policy,
            "seed": seed,
            "feasible": res.feasible,
            "knob_star": res.knob_star,
            "min_calib_risk": res.min_risk_hat,
            "alpha": res.alpha,
            "missing_calib_knobs": len(res.missing_knobs),
        }
        if res.selected is not None:
            row["calib_risk_at_star"] = res.selected.risk_hat
            t = group[(group["split"] == test_split) & (group["knob"] == res.knob_star)]
            if not t.empty:
                row["test_error_at_star"] = float(t["final_error"].iloc[0])
                row["test_cost_at_star"] = float(t["total_cost"].iloc[0])
                row["test_interventions_at_star"] = float(t["interventions_total"].iloc[0])
        out_rows.append(row)
    table = pd.DataFrame(out_rows)
    (suite_dir / "ltt.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=1), encoding="utf-8"
    )
    table.to_csv(suite_dir / "ltt.csv", index=False)
    coverage: dict[str, Any] = {}
    for policy_key, group in table.groupby("policy"):
        policy = str(policy_key)
        feasible = group[group["feasible"]]
        with_test = (
            feasible.dropna(subset=["test_error_at_star"])
            if "test_error_at_star" in feasible
            else feasible.iloc[0:0]
        )
        coverage[str(policy)] = {
            "seeds": len(group),
            "feasible": len(feasible),
            "guarantee_coverage": (
                float((with_test["test_error_at_star"] <= with_test["alpha"]).mean())
                if len(with_test)
                else None
            ),
        }
    (suite_dir / "coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")
    return table


def selected_knobs(suite_dir: str | Path) -> dict[str, float]:
    """Median selected knob per policy over seeds (for figure markers), if calibrated."""
    path = Path(suite_dir) / "ltt.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    table = table[table["feasible"]]
    if table.empty:
        return {}
    return {str(p): float(v) for p, v in table.groupby("policy")["knob_star"].median().items()}
