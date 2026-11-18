"""Run a split of a task through the graph and write ``results/<run_id>/``.

Instances run concurrently in a ``forkserver`` process pool (one compiled
graph and one in-memory checkpointer per worker; the mock and the task are
stateless and hash-seeded, so results are identical for any number of
workers). Workers import ``config.plugins`` first, which is how custom
estimators, policies or tasks registered outside the package become available;
the parent process imports them too, before it first looks a name up
(``import_plugins``). Each
trajectory becomes one parquet row; the per-step records and every decision
with its reason are the audit log.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ucagent.core.graph.build import Graph, build_graph
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.runtime import RunSettings, init_state, make_config
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.protocols import HumanInterface, Task
from ucagent.core.records import StepRecord, Trajectory
from ucagent.core.tables import Tables
from ucagent.core.uq.normalize import normalize_answer
from ucagent.experiments import provenance
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.io import write_trajectories
from ucagent.tasks import make_task
from ucagent.tasks.base import make_human

log = logging.getLogger("ucagent")


@dataclass
class RunResult:
    """What a run produced."""

    run_id: str
    out_dir: Path | None
    trajectories: list[Trajectory]
    seconds: float


def import_plugins(modules: Iterable[str]) -> None:
    """Import the plugin modules of a configuration, which register their components.

    Every entry point that looks up an estimator, a calibrator, a policy or a task by name
    calls this first: the fit used to look its estimator up before any plugin was imported,
    so a plugin estimator failed in ``ucagent fit`` and in a fresh suite.
    """
    for module in modules:
        importlib.import_module(module)


def build_task(config: ExperimentConfig) -> Task:
    """Instantiate the task named in the config (registry in ``ucagent.tasks``)."""
    import_plugins(config.plugins)
    return make_task(config.task.model_dump())


def build_llm(config: ExperimentConfig, task: Task) -> BaseChatModel:
    """Instantiate the chat model (mock or, from M4, a cached real backend)."""
    if config.llm.backend == "mock":
        a, b = config.llm.difficulty_beta
        return MockChainLLM(
            task=task,
            seed=config.llm.mock_seed if config.llm.mock_seed is not None else config.seed,
            difficulty_beta=(a, b),
            clean_factor=config.llm.clean_factor,
            verbalized=config.llm.verbalized,
            verbalized_sigma=config.llm.verbalized_sigma,
        )
    from ucagent.config import get_settings
    from ucagent.core.llm.backend import get_chat_model

    env = get_settings()
    return get_chat_model(
        config.llm.backend,
        temperature=config.llm.temperature,
        cache_path=env.cache_path,
        max_calls=min(config.limits.max_llm_calls, env.max_llm_calls),
        max_usd=env.max_usd,
        model_kwargs=config.llm.model_kwargs,
    )


def build_tables(config: ExperimentConfig) -> Tables:
    """Tables from the artifact (if any) with the influence variant of ``risk.influence``.

    ``ones`` and ``position_prior[:decreasing|increasing]`` need no artifact;
    ``counterfactual`` takes the influence fitted into the artifact and falls
    back to ones (with a warning) when the artifact has none.
    """
    from ucagent.core.risk.influence import ones, position_prior

    if config.uq.artifact is None:
        tables = Tables()
    else:
        from ucagent.core.uq.calibration import load_artifact

        tables = load_artifact(config.uq.artifact).to_tables(config)
    n = config.task.n_steps
    kind = config.risk.influence
    if kind == "ones":
        tables.influence = ones(n)
    elif kind.startswith("position_prior"):
        shape = kind.split(":", 1)[1] if ":" in kind else "decreasing"
        if shape not in ("decreasing", "increasing"):
            raise ValueError(f"unknown position prior shape {shape!r}")
        tables.influence = position_prior(n, shape)  # type: ignore[arg-type]
    elif kind == "counterfactual":
        if not tables.influence:
            if config.uq.artifact is not None:
                log.warning("risk.influence=counterfactual but the artifact has none; using ones")
            tables.influence = ones(n)
    else:
        raise ValueError(f"unknown influence {kind!r}")
    return tables


@dataclass
class Worker:
    """Everything one worker needs to run instances."""

    config: ExperimentConfig
    task: Task
    llm: BaseChatModel
    graph: Graph
    tables: Tables
    settings: RunSettings
    human: HumanInterface

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Worker:
        """Build the task, model, graph and simulated human from a config."""
        task = build_task(config)  # imports the plugins first
        llm = build_llm(config, task)
        settings = config.to_run_settings()
        return cls(
            config=config,
            task=task,
            llm=llm,
            graph=build_graph(task, llm),
            tables=build_tables(config),
            settings=settings,
            human=make_human(task, settings),
        )

    def run_instance(self, instance_id: str) -> Trajectory:
        """Run one instance to completion and label it for analysis."""
        instance = self.task.get_instance(instance_id)
        thread_id = f"{self.config.run_id}:{instance_id}:{self.settings.seed}"
        config = make_config(thread_id, self.settings, self.tables, instance.n_steps)
        t0 = time.perf_counter()
        final = run_with_human(self.graph, init_state(instance, self.settings), config, self.human)
        seconds = time.perf_counter() - t0
        checkpointer = self.graph.checkpointer
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(thread_id)
        return self.to_trajectory(final, instance_id, thread_id, seconds)

    def counterfactuals(self, instance_id: str, method: str = "fork") -> list[dict[str, Any]]:
        """Counterfactual outcomes of one instance for influence estimation."""
        from ucagent.core.risk.influence import (
            counterfactuals_by_forking,
            counterfactuals_by_replay,
        )

        fn = counterfactuals_by_forking if method == "fork" else counterfactuals_by_replay
        instance = self.task.get_instance(instance_id)
        outcomes = fn(self.graph, self.task, instance, self.settings, self.tables, self.human)
        return [
            {
                "step": o.step,
                "alternative": o.alternative,
                "final_wrong": o.final_wrong,
                "kind": o.kind,
            }
            for o in outcomes
        ]

    def to_trajectory(
        self, final: dict[str, Any], instance_id: str, thread_id: str, seconds: float
    ) -> Trajectory:
        """Convert the final graph state into the audit record."""
        return trajectory_from_state(
            self.task, self.settings, self.config.run_id, final, instance_id, thread_id, seconds
        )


def trajectory_from_state(
    task: Task,
    settings: RunSettings,
    run_id: str,
    final: dict[str, Any],
    instance_id: str,
    thread_id: str = "",
    seconds: float = 0.0,
) -> Trajectory:
    """Label a finished graph state for analysis and pack it as a :class:`Trajectory`."""
    instance = task.get_instance(instance_id)
    has_local_truth = getattr(task, "has_local_truth", None)
    steps: list[StepRecord] = []
    for rec in final["history"]:
        # No local truth (an off-track hop on a benchmark): the label is missing, not wrong.
        unlabelled = has_local_truth is not None and not has_local_truth(
            instance, rec.step, rec.context
        )
        steps.append(
            rec.model_copy(
                update={
                    "locally_correct": None
                    if unlabelled
                    else task.is_locally_correct(instance, rec.step, rec.context, rec.answer),
                    "globally_correct": (
                        task.is_final_correct(instance, rec.answer)
                        if rec.step == instance.n_steps
                        else normalize_answer(rec.answer)
                        == normalize_answer(task.global_truth(instance, rec.step))
                    ),
                }
            )
        )
    naive = 1.0
    weighted = 1.0
    for rec in steps:
        naive *= 1.0 - rec.eps
        weighted *= 1.0 - rec.w * rec.eps
    s = settings
    return Trajectory(
        run_id=run_id,
        instance_id=instance_id,
        split=instance.split,
        seed=s.seed,
        n_steps=instance.n_steps,
        policy=s.policy,
        knob=s.knob,
        estimator=s.estimator,
        K=s.K,
        K_prime=s.K_prime,
        steps=steps,
        interventions=list(final["interventions"]),
        final_answer=final.get("final_answer"),
        final_correct=bool(final.get("final_correct")),
        escalated=bool(final.get("escalated")),
        total_cost=float(final.get("total_cost", 0.0)),
        fixed_cost=float(final.get("fixed_cost", 0.0)),
        llm_calls=int(final.get("llm_calls", 0)),
        u_final=steps[-1].u if steps else None,
        risk_naive_final=1.0 - naive,
        risk_weighted_final=1.0 - weighted,
        survival_final=float(final.get("survival", 1.0)),
        duration_s=seconds,
        thread_id=thread_id,
        samples={
            key: [smp.answer for smp in sorted(group, key=lambda x: x.sample_index)]
            for key, group in final.get("samples", {}).items()
        }
        if s.record_samples
        else {},
        sample_confidences={
            key: [smp.confidence for smp in sorted(group, key=lambda x: x.sample_index)]
            for key, group in final.get("samples", {}).items()
        }
        if s.record_samples
        else {},
    )


_WORKERS: dict[str, Worker] = {}


def _init_worker(plugins: list[str]) -> None:
    import_plugins(plugins)


def _run_in_worker(args: tuple[str, str]) -> dict[str, Any]:
    config_json, instance_id = args
    worker = _WORKERS.get(config_json)
    if worker is None:
        if len(_WORKERS) >= 8:
            _WORKERS.clear()
        worker = Worker.from_config(ExperimentConfig.model_validate(json.loads(config_json)))
        _WORKERS[config_json] = worker
    return worker.run_instance(instance_id).model_dump()


def _counterfactuals_in_worker(args: tuple[str, str, str]) -> list[dict[str, Any]]:
    config_json, instance_id, method = args
    worker = _WORKERS.get(config_json)
    if worker is None:
        if len(_WORKERS) >= 8:
            _WORKERS.clear()
        worker = Worker.from_config(ExperimentConfig.model_validate(json.loads(config_json)))
        _WORKERS[config_json] = worker
    return worker.counterfactuals(instance_id, method)


def run_influence(
    config: ExperimentConfig,
    *,
    split: str = "train",
    workers: int | None = None,
    pool: ProcessPoolExecutor | None = None,
    method: str = "fork",
    instance_ids: Iterable[str] | None = None,
    estimator: str = "paired",
) -> dict[tuple[int, int], float]:
    """Counterfactual influence ``w_i`` per ``(n, i)`` from a split (policy ``none``).

    ``estimator`` is ``paired`` (fatality probability from wrong and gold forks)
    or ``absolute`` (failure rate of the wrong forks); see ``core.risk.influence``.
    """
    from ucagent.core.risk.influence import CounterfactualOutcome, estimate_influence

    cfg = config.with_updates(control={"policy": "none", "knob": None})
    task = build_task(cfg)
    ids = (
        list(instance_ids)
        if instance_ids is not None
        else [inst.instance_id for inst in task.instances(split)]
    )
    workers = workers or min(cfg.limits.max_concurrency, os.cpu_count() or 1)
    config_json = json.dumps(cfg.model_dump(mode="json"))
    args = [(config_json, i, method) for i in ids]
    if pool is not None:
        raw = list(pool.map(_counterfactuals_in_worker, args, chunksize=2))
    elif workers <= 1 or len(ids) <= 1:
        worker = Worker.from_config(cfg)
        raw = [worker.counterfactuals(i, method) for i in ids]
    else:
        with make_pool(workers, cfg.plugins) as own_pool:
            raw = list(own_pool.map(_counterfactuals_in_worker, args, chunksize=2))
    outcomes = [
        (
            task.get_instance(instance_id).n_steps,
            [
                CounterfactualOutcome(
                    r["step"], r["alternative"], r["final_wrong"], r.get("kind", "wrong")
                )
                for r in rows
            ],
        )
        for instance_id, rows in zip(ids, raw, strict=True)
    ]
    if estimator not in ("paired", "absolute"):
        raise ValueError(f"unknown influence estimator {estimator!r}")
    return estimate_influence(outcomes, estimator=estimator)  # type: ignore[arg-type]


def make_pool(workers: int, plugins: list[str] | None = None) -> ProcessPoolExecutor:
    """A forkserver pool whose workers have imported ``plugins``; reuse it across runs."""
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("forkserver"),
        initializer=_init_worker,
        initargs=(list(plugins or []),),
    )


def estimated_llm_calls(config: ExperimentConfig, n_instances: int) -> int:
    """Upper-ish estimate of LLM calls.

    Every step may be expanded to ``K'`` samples once (a resample, or a
    backtrack's redo of its target), and a backtrack may redo every step once
    more.
    """
    expands = config.control.max_resamples > 0 or config.control.max_backtracks > 0
    per_step = config.llm.K_prime if expands else config.llm.K
    return n_instances * config.task.n_steps * per_step * (1 + config.control.max_backtracks)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def model_digest(backend: str) -> str | None:
    """The server's digest of an Ollama model (None for other backends or no answer).

    A tag such as ``gpt-oss:20b`` can come to name other weights; the digest says which ones
    served a run. One read of the server's list of models, no model call.
    """
    if not backend.startswith("ollama:"):
        return None
    name = backend.split(":", 1)[1]
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as reply:
            models = json.loads(reply.read().decode("utf-8")).get("models", [])
    except (OSError, ValueError):
        return None
    for model in models:
        if model.get("name") == name or model.get("model") == name:
            return str(model.get("digest") or "") or None
    return None


def _env_text(backend: str = "mock") -> str:
    names = ["langgraph", "langchain-core", "langchain", "pydantic", "numpy", "pandas", "pyarrow"]
    lines = [f"python {sys.version.split()[0]} ({platform.platform()})"]
    for name in names:
        try:
            lines.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name} (not installed)")
    if backend.startswith("ollama:"):
        lines.append(f"model {backend} digest {model_digest(backend) or 'unknown'}")
    return "\n".join(lines) + "\n"


def run_experiment(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    workers: int | None = None,
    instance_ids: Iterable[str] | None = None,
    dry_run: bool = False,
    allow_global_oracle: bool = False,
    results_dir: str | Path | None = None,
    write: bool = True,
    pool: ProcessPoolExecutor | None = None,
) -> RunResult:
    """Run every instance of a split; see the module docstring.

    Pass ``pool`` (from :func:`make_pool`) to reuse worker processes across
    runs; otherwise a pool is created for this run when ``workers > 1``.
    """
    if config.oracle.scope == "global" and not allow_global_oracle:
        raise ValueError(
            "oracle.scope='global' makes 'ask at the last step' trivially optimal "
            "(instructions Section 2.1); it is only allowed in tests"
        )
    split = split or config.split
    task = build_task(config)
    ids = (
        list(instance_ids)
        if instance_ids is not None
        else [inst.instance_id for inst in task.instances(split)]
    )
    workers = workers or min(config.limits.max_concurrency, os.cpu_count() or 1)
    if dry_run:
        log.info(
            "dry run: run_id=%s split=%s instances=%d n=%d K=%d policy=%s backend=%s "
            "estimated_llm_calls<=%d workers=%d model_kwargs=%s",
            config.run_id,
            split,
            len(ids),
            config.task.n_steps,
            config.llm.K,
            config.control.policy,
            config.llm.backend,
            estimated_llm_calls(config, len(ids)),
            workers,
            config.llm.model_kwargs,
        )
        return RunResult(config.run_id, None, [], 0.0)
    t0 = time.perf_counter()
    config_json = json.dumps(config.model_dump(mode="json"))
    if pool is not None:
        rows = list(pool.map(_run_in_worker, [(config_json, i) for i in ids], chunksize=4))
        trajectories = [Trajectory.model_validate(r) for r in rows]
    elif workers <= 1 or len(ids) <= 1:
        worker = Worker.from_config(config)
        trajectories = [worker.run_instance(i) for i in ids]
    else:
        with make_pool(workers, config.plugins) as own_pool:
            rows = list(own_pool.map(_run_in_worker, [(config_json, i) for i in ids], chunksize=4))
        trajectories = [Trajectory.model_validate(r) for r in rows]
    seconds = time.perf_counter() - t0
    out_dir: Path | None = None
    if write:
        out_dir = Path(results_dir or "results") / config.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")
        provenance.write(out_dir, provenance.record(task, ids))
        (out_dir / "git_sha.txt").write_text(_git_sha() + "\n", encoding="utf-8")
        (out_dir / "env.txt").write_text(_env_text(config.llm.backend), encoding="utf-8")
        rate = len(trajectories) / seconds if seconds > 0 else float("inf")
        (out_dir / "timing.txt").write_text(
            f"trajectories {len(trajectories)}\nseconds {seconds:.2f}\n"
            f"workers {workers}\ntrajectories_per_second {rate:.2f}\n",
            encoding="utf-8",
        )
        # last: a resumed suite takes the presence of this file as "the run is finished"
        write_trajectories(out_dir / "trajectories.parquet", trajectories)
    log.info(
        "run %s: %d trajectories in %.1fs (%d workers)",
        config.run_id,
        len(trajectories),
        seconds,
        workers,
    )
    return RunResult(config.run_id, out_dir, trajectories, seconds)
