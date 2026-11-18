"""Adopter-facing entry points.

- :func:`fit_calibrators` fits the calibration artifact on the train split.
- :func:`run` runs instances of a task through the controller.
- :func:`calibrate_knob` (milestone M5) selects the knob with Learn-then-Test.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ucagent.core.control.ltt import LTTResult, PValueKind
from ucagent.core.control.ltt import calibrate_knob as _calibrate_knob
from ucagent.core.graph.build import build_graph
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.runtime import RunSettings, init_state, make_config
from ucagent.core.protocols import HumanInterface, Task
from ucagent.core.records import Cluster, Sample, Trajectory
from ucagent.core.tables import Tables
from ucagent.core.uq.base import RECORDED_FIELDS, StepContext, estimator_reads, make_estimator
from ucagent.core.uq.calibration import CalibrationArtifact, StepObservation, fit_artifact
from ucagent.core.uq.equivalence import make_judge
from ucagent.core.uq.features import CONFIDENCE_FEATURES, step_features
from ucagent.core.utils import stable_digest
from ucagent.experiments import provenance
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import (
    RunResult,
    build_llm,
    build_task,
    import_plugins,
    run_experiment,
    run_influence,
    trajectory_from_state,
)


def run(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    workers: int | None = None,
    **kwargs: object,
) -> RunResult:
    """Run a split under the config's policy; see :func:`run_experiment`."""
    return run_experiment(config, split=split, workers=workers, **kwargs)  # type: ignore[arg-type]


def run_instances(
    task: Task,
    llm: Any,
    settings: RunSettings,
    instance_ids: Sequence[str],
    human: HumanInterface,
    *,
    tables: Tables | None = None,
    run_id: str = "adhoc",
) -> list[Trajectory]:
    """Run instances of any :class:`Task` in-process under ``settings`` (adopter entry point).

    ``llm`` is any LangChain chat model (the mock, or a cached real model from
    :func:`ucagent.core.llm.backend.get_chat_model`); ``human`` answers step
    checks and escalations; ``tables`` come from a fitted artifact
    (``CalibrationArtifact.to_tables()``) or default to identity.
    """
    graph = build_graph(task, llm)
    tables = tables or Tables()
    out: list[Trajectory] = []
    checkpointer = graph.checkpointer
    for position, instance_id in enumerate(instance_ids):
        instance = task.get_instance(instance_id)
        # one thread per execution (a repeated id must not inherit the earlier run's state)
        thread_id = f"{run_id}:{instance_id}:{settings.seed}:{position}"
        config = make_config(thread_id, settings, tables, instance.n_steps)
        try:
            final = run_with_human(graph, init_state(instance, settings), config, human)
            out.append(trajectory_from_state(task, settings, run_id, final, instance_id, thread_id))
        finally:
            if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
                checkpointer.delete_thread(thread_id)
    return out


def _correct_share(
    task: Task, instance: Any, step: int, context: str, clusters: list[Cluster]
) -> float:
    """Proportion of the locally correct cluster (0 when no cluster is correct)."""
    shares = [
        c.proportion
        for c in clusters
        if task.is_locally_correct(instance, step, context, c.representative)
    ]
    return max(shares, default=0.0)


def observations_from(
    config: ExperimentConfig, trajectories: list[Trajectory]
) -> list[list[StepObservation]]:
    """Re-estimate every step at ``K`` and ``K'`` from the recorded ``K'`` samples.

    Because samples are hash-seeded by ``sample_index`` (mock) or cached by it
    (real models), the first ``K`` of the ``K'`` samples are exactly what a run
    with ``K`` samples would have drawn in the same context, and the ``K'``
    batch is exactly that run followed by one resample (which expands the batch
    under the same prompt). The stated confidences travel with the answers, so
    an estimator that reads them is fitted on what it sees at run time.
    """
    task = build_task(config)
    judge = make_judge(config.uq.equivalence, llm=build_llm(config, task))
    estimator = make_estimator(config.uq.estimator, judge)
    k, kp = config.llm.K, config.llm.K_prime
    has_local_truth = getattr(task, "has_local_truth", None)
    out: list[list[StepObservation]] = []
    for t in trajectories:
        instance = task.get_instance(t.instance_id)
        rows: list[StepObservation] = []
        for s in t.steps:
            if s.source != "llm":
                continue
            if has_local_truth is not None and not has_local_truth(instance, s.step, s.context):
                continue  # off-track step on a task without local truth there: no label
            answers = t.samples.get(s.samples_key, [])
            if len(answers) < kp:
                raise ValueError(
                    f"{t.instance_id} step {s.step}: need {kp} samples, got {len(answers)}"
                )
            confidences = t.sample_confidences.get(s.samples_key) or [None] * len(answers)
            samples = [
                Sample(sample_index=j, text=a, answer=a, confidence=c)
                for j, (a, c) in enumerate(zip(answers, confidences, strict=True))
            ]
            ctx = StepContext(
                instance_id=t.instance_id,
                step=s.step,
                attempt=s.attempt,
                current=s.context,
                n_steps=t.n_steps,
            )
            est_k = estimator.estimate(samples[:k], ctx)
            est_kp = estimator.estimate(samples[:kp], ctx)
            x_k = step_features(est_k.u, samples[:k], est_k.clusters, s.step, t.n_steps)
            x_kp = step_features(est_kp.u, samples[:kp], est_kp.clusters, s.step, t.n_steps)

            rows.append(
                StepObservation(
                    n_steps=t.n_steps,
                    step=s.step,
                    u_K=est_k.u,
                    wrong_K=not task.is_locally_correct(
                        instance, s.step, s.context, est_k.committed
                    ),
                    u_K_prime=est_kp.u,
                    wrong_K_prime=not task.is_locally_correct(
                        instance, s.step, s.context, est_kp.committed
                    ),
                    p_true_K=_correct_share(task, instance, s.step, s.context, est_k.clusters),
                    p_true_K_prime=_correct_share(
                        task, instance, s.step, s.context, est_kp.clusters
                    ),
                    x_K=x_k,
                    x_K_prime=x_kp,
                )
            )
        out.append(rows)
    return out


def fit_calibrators(
    config: ExperimentConfig,
    *,
    workers: int | None = None,
    out_path: str | Path | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> CalibrationArtifact:
    """Fit the artifact on the ``train`` split under policy ``none`` with ``K'`` samples."""
    import_plugins(config.plugins)  # before the estimator is looked up by name
    reads = estimator_reads(config.uq.estimator)
    if reads - RECORDED_FIELDS:
        raise ValueError(
            f"estimator {config.uq.estimator!r} reads {sorted(reads - RECORDED_FIELDS)}, which a "
            f"run does not record (recorded: {sorted(RECORDED_FIELDS)}): the fit would estimate "
            "from other inputs than the run"
        )
    fit_cfg = config.with_updates(
        control={"policy": "none", "knob": None},
        llm={"K": config.llm.K_prime},
        uq={"artifact": None},
    ).model_copy(update={"split": "train", "run_id": f"{config.run_id}__fit"})
    result = run_experiment(fit_cfg, workers=workers, write=False, pool=pool)
    reads_confidence = "confidence" in reads or bool(CONFIDENCE_FEATURES & set(config.uq.features))
    if reads_confidence and not any(
        c is not None
        for t in result.trajectories
        for confs in t.sample_confidences.values()
        for c in confs
    ):
        raise ValueError(
            f"estimator {config.uq.estimator!r} with features {config.uq.features} reads stated "
            "confidences, but no sample of the train run states one: the fit would silently "
            "calibrate disagreement instead (check the prompt asks for 'Confidence: <number>')"
        )
    per_traj = observations_from(config, result.trajectories)
    flat = [o for rows in per_traj for o in rows]
    influence: dict[tuple[int, int], float] | None = None
    if config.risk.influence == "counterfactual":
        inf_cfg = config.with_updates(uq={"artifact": None}).model_copy(
            update={"split": "train", "run_id": f"{config.run_id}__influence"}
        )
        influence = run_influence(
            inf_cfg,
            split="train",
            workers=workers,
            pool=pool,
            estimator=config.risk.influence_estimator,
        )
    artifact = fit_artifact(
        flat,
        estimator=config.uq.estimator,
        equivalence=config.uq.equivalence,
        K=config.llm.K,
        K_prime=config.llm.K_prime,
        task_config_hash=stable_digest(config.task.model_dump()),
        config_hash=stable_digest(config.model_dump(mode="json")),
        data_hash=stable_digest([t.instance_id for t in result.trajectories]),
        inputs=provenance.record(build_task(config), [t.instance_id for t in result.trajectories]),
        n_trajectories=len(result.trajectories),
        trajectories=per_traj,
        influence=influence,
        influence_method=(
            f"counterfactual:{config.risk.influence_estimator}" if influence else "ones"
        ),
        calibrator=config.uq.calibrator,
        features=config.uq.features,
    )
    if out_path is not None:
        artifact.save(out_path)
    return artifact


def calibrate_knob(
    policy: str,
    points: Sequence[tuple[float, int, int]],
    *,
    alpha: float,
    delta: float = 0.1,
    pvalue: PValueKind = "binomial",
) -> LTTResult:
    """Learn-then-Test over ``[(knob, n_calib, errors), ...]``; see ``core.control.ltt``."""
    return _calibrate_knob(policy, points, alpha=alpha, delta=delta, pvalue=pvalue)
