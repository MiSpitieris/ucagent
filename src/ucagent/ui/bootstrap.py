"""Build the run from the page's choices: config, worker, artifact and knob lookups.

No Streamlit import here, so everything is testable without the page.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel

from ucagent.core.control.policies import DEFAULT_KNOB
from ucagent.core.protocols import Instance, Task
from ucagent.core.uq.calibration import load_artifact
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import Worker

OLLAMA_DEFAULT_MODEL = "llama3.1:8b"
KNOB_SUITE = "suite_small"  # where Learn-then-Test knobs are looked up (results/<suite>/ltt.csv)


class UiChoices(BaseModel):
    """Everything the sidebar decides (plain data, so it can be hashed and tested)."""

    config_path: str | None = None
    backend: str = "mock"  # "mock" or "<provider>:<model>", e.g. "ollama:llama3.1:8b"
    policy: str = "marginal"
    knob: float | None = None  # None: the policy's default
    use_ltt_knob: bool = False  # prefer the Learn-then-Test knob from results/<KNOB_SUITE>
    K: int = 5
    K_prime: int = 10
    alpha: float = 0.1
    estimator: str = "semantic_entropy"
    artifact_mode: str = "auto"  # "auto" (first compatible fitted artifact) or "none"
    n_steps: int = 5
    ambiguity: float = 0.5
    split: str = "test"
    instance_index: int = 0
    seed: int = 0
    supervise: bool = True
    human_mode: str = "me"  # "me" (answer in the page) or "oracle" (the simulated human)
    p_h: float = 0.0
    results_dir: str = "results"


def ensure_cwd_on_path() -> None:
    """Make ``plugins: [examples.toy_task]``-style modules importable under Streamlit."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def list_configs(directory: str | Path = "configs") -> list[str]:
    """Experiment YAML files in ``directory`` (suites and pipelines skipped), sorted."""
    out: list[str] = []
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and data.get("kind", "experiment") == "experiment":
            out.append(str(path))
    return out


def base_config(path: str | None) -> ExperimentConfig:
    """The config a file defines, or the package defaults."""
    if path:
        return ExperimentConfig.from_yaml(path)
    return ExperimentConfig(run_id="ui")


def defaults_from_config(cfg: ExperimentConfig) -> dict[str, Any]:
    """Sidebar values a config file implies (keys are the page's widget keys)."""
    backend = cfg.llm.backend
    knob = cfg.control.knob
    out: dict[str, Any] = {
        "backend": "mock" if backend == "mock" else "ollama",
        "policy": cfg.control.policy,
        "knob": float(knob if knob is not None else DEFAULT_KNOB.get(cfg.control.policy, 0.0)),
        "k": int(cfg.llm.K),
        "k_prime": int(cfg.llm.K_prime),
        "alpha": float(cfg.control.alpha),
        "estimator": cfg.uq.estimator,
        "n_steps": int(cfg.task.n_steps),
        "seed": int(cfg.seed),
        "split": cfg.split,
        "p_h": float(cfg.oracle.p_h),
    }
    if backend.startswith("ollama:"):
        out["ollama_model"] = backend.split(":", 1)[1]
    extra = cfg.task.model_extra or {}
    if "ambiguity" in extra:
        out["ambiguity"] = float(extra["ambiguity"])
    return out


def build_config(choices: UiChoices) -> ExperimentConfig:
    """The experiment config for the choices (the artifact is resolved by ``prepare_run``)."""
    cfg = base_config(choices.config_path)
    task: dict[str, Any] = {"seed": choices.seed, "n_steps": choices.n_steps}
    if cfg.task.name == "synthetic_chain":
        task["ambiguity"] = choices.ambiguity
    llm: dict[str, Any] = {"backend": choices.backend, "K": choices.K, "K_prime": choices.K_prime}
    if choices.backend != cfg.llm.backend:
        llm["model_kwargs"] = {}  # the file's provider options (e.g. a thinking mode) may not fit
    cfg = cfg.with_updates(
        task=task,
        llm=llm,
        uq={"estimator": choices.estimator, "artifact": None},
        control={"policy": choices.policy, "knob": choices.knob, "alpha": choices.alpha},
        oracle={"scope": "local", "p_h": choices.p_h},
    )
    return cfg.model_copy(update={"run_id": "ui", "split": choices.split, "seed": choices.seed})


def artifact_candidates(cfg: ExperimentConfig, results_dir: str | Path) -> list[Path]:
    """Where fitted artifacts usually live, most specific first."""
    root = Path(results_dir)
    n, seed = cfg.task.n_steps, cfg.seed
    return [
        root / KNOB_SUITE / "artifacts" / f"seed{seed}.json",
        root / KNOB_SUITE / "artifacts" / "seed0.json",
        root / "figures_small" / "fig1" / f"n{n}__s0" / "artifact.json",
        root / "synthetic_reconverge" / "artifacts" / "seed0.json",
    ]


def resolve_artifact(
    cfg: ExperimentConfig,
    mode: str,
    results_dir: str | Path,
    file_artifact: str | None = None,
) -> tuple[str | None, str]:
    """The artifact path to use (or None for identity tables) and a note for the page."""
    need = f"{cfg.uq.estimator}, K={cfg.llm.K}, K'={cfg.llm.K_prime}"
    if mode == "none":
        return None, f"identity tables (eps = u, w = G = 1): no artifact requested ({need})"
    candidates = ([Path(file_artifact)] if file_artifact else []) + artifact_candidates(
        cfg, results_dir
    )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            # same estimator, sample counts, judge and task block as the run (review R4)
            load_artifact(path).to_tables(cfg)
        except ValueError:
            continue
        return str(path), f"calibration artifact {path} ({need}, same task and judge)"
    return None, (
        f"identity tables (eps = u, w = G = 1): no fitted artifact matches {need} on this "
        "task with this equivalence judge; fit one with `ucagent fit`"
    )


def ltt_knob(
    policy: str, results_dir: str | Path, suite: str = KNOB_SUITE
) -> tuple[float | None, str]:
    """The Learn-then-Test knob of ``policy`` from a suite's ``ltt.csv``.

    Only a knob that every feasible seed certified is returned; a median of
    different certified knobs is itself uncertified (review finding R4), so the
    page then falls back to the sidebar knob with a note.
    """
    path = Path(results_dir) / suite / "ltt.csv"
    if not path.exists():
        return (
            None,
            f"no Learn-then-Test table at {path} (run `make figures` or `ucagent calibrate`)",
        )
    table = pd.read_csv(path)
    rows = table[(table["policy"] == policy) & table["feasible"].astype(bool)]
    if rows.empty:
        return None, f"no feasible Learn-then-Test knob for {policy} in {path}"
    alpha = float(rows["alpha"].iloc[0]) if "alpha" in rows.columns else float("nan")
    knobs = sorted({float(k) for k in rows["knob_star"]})
    if len(knobs) != 1:
        return None, (
            f"the seeds of {suite} certified different knobs for {policy} ({knobs}); no single "
            "certified knob exists, so none is imported"
        )
    value = knobs[0]
    return value, (
        f"Learn-then-Test knob {value:.4g} for {policy}: certified by all {len(rows)} seeds of "
        f"{suite} (alpha {alpha:g}, mock LLM); valid for that suite's task and estimator"
    )


def resolve_knob(choices: UiChoices) -> tuple[float | None, str]:
    """The knob to run with and a note saying where it came from."""
    if choices.policy == "none":
        return None, "policy none has no knob"
    if choices.use_ltt_knob:
        value, note = ltt_knob(choices.policy, choices.results_dir)
        if value is not None:
            return value, note
    if choices.knob is not None:
        return float(choices.knob), f"knob {choices.knob:g} (sidebar)"
    default = DEFAULT_KNOB.get(choices.policy)
    return default, f"default knob {default} for {choices.policy}"


def prepare_run(choices: UiChoices) -> tuple[ExperimentConfig, list[str]]:
    """Resolve knob and artifact, then build the config; returns it with the notes."""
    file_cfg = base_config(choices.config_path)
    knob, knob_note = resolve_knob(choices)
    cfg = build_config(choices.model_copy(update={"knob": knob}))
    artifact, artifact_note = resolve_artifact(
        cfg, choices.artifact_mode, choices.results_dir, file_cfg.uq.artifact
    )
    if artifact is not None:
        cfg = cfg.with_updates(uq={"artifact": artifact})
    return cfg, [knob_note, artifact_note]


def worker_key(cfg: ExperimentConfig) -> str:
    """Fingerprint of a config: the page rebuilds its ``Worker`` only when this changes."""
    return hashlib.sha1(cfg.model_dump_json().encode("utf-8")).hexdigest()[:12]


def build_worker(cfg: ExperimentConfig) -> Worker:
    """Task, model, graph, tables and simulated human for a config (one process)."""
    if cfg.oracle.scope == "global":
        raise ValueError("the global oracle is for tests only; use oracle.scope: local")
    return Worker.from_config(cfg)


def instance_at(task: Task, split: str, index: int) -> Instance:
    """The ``index``-th instance of ``split``."""
    try:
        return next(islice(task.instances(split), index, index + 1))
    except StopIteration as exc:
        raise ValueError(f"split {split!r} has no instance number {index}") from exc


def ollama_status(base_url: str | None = None, timeout: float = 2.0) -> tuple[bool, list[str]]:
    """Whether an Ollama server answers and which models it lists."""
    url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False, []
    models = [str(m.get("name", "")) for m in data.get("models", [])]
    return True, sorted(m for m in models if m)


def llm_counters(llm: Any) -> dict[str, Any] | None:
    """Real-call counter, cap and cache facts of a cached model; None for the mock."""
    store = getattr(llm, "store", None)
    if store is None:
        return None
    out: dict[str, Any] = {
        "real_calls": int(store.total_calls()),
        "cache_rows": len(store),
        "cache_path": str(getattr(store, "path", "")),
    }
    guard = getattr(llm, "guard", None)
    if guard is not None:
        out.update({"max_calls": guard.max_calls, "usd": guard.usd, "max_usd": guard.max_usd})
    return out
