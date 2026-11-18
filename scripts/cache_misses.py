"""How many model calls would a suite need that the response cache does not hold?

Usage::

    uv run python scripts/cache_misses.py configs/finqa2_confagree.yaml
    uv run python scripts/cache_misses.py configs/finqa2_confagree.yaml \\
        --grid per_step_eps marginal

A dry replay, without a single model call and without writing to the cache. The suite's fit
and a few of its runs are executed in this process with a model that answers from the cache
(opened read-only). A request the cache does not hold is counted as a miss and answered with
a placeholder that no cached prompt contains, so every later step of that question misses too.

What the count is: an estimate of the new calls for the stages replayed, biased upwards for a
question once it has missed (a real answer could lead back to a cached prompt) and blind to
the runs that are not replayed. The fit and the uncontrolled runs depend on no artifact and
are counted exactly up to that bias. The controlled runs use the artifact of the dry fit,
whose labels are wrong for the steps that missed, so their decisions, and with them the
expansions they ask for, are approximate. Use the result to decide whether a run fits its
budget; the cost guard of the run remains the limit.

Stages, in order: the fit (train split, ``K'`` samples); the uncontrolled runs; random asking
at knob 0 (every step checked, which sends the reviewer's answers to the next step); the
gate at its most conservative knob; and, with ``--grid POLICY ...``, every knob of those
policies. Misses are counted once across stages.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

import ucagent.api as api
import ucagent.experiments.runner as runner
from ucagent.config import get_settings
from ucagent.core.llm.cache import cache_key, model_tag
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.suite import SuiteConfig, grid_values

PLACEHOLDER = "Answer: 918273645.5\nConfidence: 0.5"
MISSES: set[str] = set()
COUNTS = {"requests": 0, "hits": 0}


class DryModel(BaseChatModel):
    """Answers from the response cache; a miss is counted and answered with a placeholder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: str
    model_name: str
    temperature: float = 0.7
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return f"dry({self.model_name})"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        sample_index = int(kwargs.pop("sample_index", 0))
        tag = model_tag(self.model_name, self.model_kwargs)
        key = cache_key(tag, messages, self.temperature, sample_index)
        COUNTS["requests"] += 1
        row = (
            _connection(self.db)
            .execute("SELECT response FROM responses WHERE key = ?", (key,))
            .fetchone()
        )
        if row is None:
            MISSES.add(key)
            content = PLACEHOLDER
        else:
            COUNTS["hits"] += 1
            content = str(row[0])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


_LOCAL = threading.local()


def _connection(db: str) -> sqlite3.Connection:
    """A read-only connection of the calling thread (the graph runs its nodes in threads)."""
    connections: dict[str, sqlite3.Connection] = _LOCAL.__dict__.setdefault("connections", {})
    if db not in connections:
        connections[db] = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    return connections[db]


def dry_llm(config: ExperimentConfig, task: Any) -> BaseChatModel:
    """Stands in for ``runner.build_llm``: the mock as it is, any real backend from the cache."""
    if config.llm.backend == "mock":
        return _REAL_BUILD_LLM(config, task)
    return DryModel(
        db=str(get_settings().cache_path),
        model_name=config.llm.backend,
        temperature=config.llm.temperature,
        model_kwargs=dict(config.llm.model_kwargs),
    )


_REAL_BUILD_LLM = runner.build_llm


def counter(db: str) -> int:
    """The lifetime count of real calls, read from the cache."""
    row = _connection(db).execute("SELECT value FROM counters WHERE name = 'calls'").fetchone()
    return int(row[0]) if row else 0


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="a suite config")
    parser.add_argument("--grid", nargs="*", default=[], help="policies replayed at every knob")
    parser.add_argument("--splits", default="calib,test")
    args = parser.parse_args()
    suite = SuiteConfig.from_yaml(args.config)
    splits = args.splits.split(",")
    db = str(get_settings().cache_path)
    before = counter(db)
    runner.build_llm = dry_llm  # type: ignore[assignment]
    api.build_llm = dry_llm  # type: ignore[assignment]

    seed = suite.seeds[0]
    base = suite.base.model_copy(update={"seed": seed}).with_updates(task={"seed": seed})
    stages: list[tuple[str, int, int, int, float]] = []

    def stage(name: str, run: Any) -> None:
        requests, known, t0 = COUNTS["requests"], len(MISSES), time.perf_counter()
        run()
        stages.append(
            (
                name,
                COUNTS["requests"] - requests,
                len(MISSES) - known,
                len(MISSES),
                time.perf_counter() - t0,
            )
        )
        print(f"  {name}: {stages[-1][2]} new misses", file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        artifact = str(Path(tmp) / "artifact.json")
        stage(
            "fit (train, K' samples)",
            lambda: api.fit_calibrators(base, out_path=artifact, workers=1),
        )
        fitted = base.with_updates(uq={"artifact": artifact})

        def run(policy: str, knob: float | None, split: str) -> None:
            cfg = fitted.with_updates(control={"policy": policy, "knob": knob})
            cfg = cfg.model_copy(
                update={"run_id": f"dry__{policy}__{knob}__{split}", "split": split}
            )
            runner.run_experiment(cfg, workers=1, write=False)

        probes: list[tuple[str, str, float | None]] = [("uncontrolled", "none", None)]
        if "random_ask" in suite.policies:
            probes.append(("random asking, every step checked (knob 0)", "random_ask", 0.0))
        gate = suite.policies.get("per_step_eps")
        if gate is not None:
            knob = min(grid_values(gate))
            probes.append((f"gate at its most conservative knob ({knob:g})", "per_step_eps", knob))
        for policy in args.grid:
            grid = suite.policies.get(policy)
            if grid is None:
                raise SystemExit(f"{args.config}: no grid for policy {policy!r}")
            probes += [(f"{policy} at knob {k:g}", policy, k) for k in grid_values(grid)]
        for name, policy, knob in probes:
            for split in splits:
                stage(f"{name}, {split}", lambda p=policy, k=knob, s=split: run(p, k, s))

    after = counter(db)
    out = [
        f"### Dry replay of {suite.suite_id} against the cache (no model calls)",
        "",
        "| stage | requests | not in the cache, new | not in the cache, so far | seconds |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {req} | {new} | {total} | {sec:.0f} |"
            for name, req, new, total, sec in stages
        ),
        "",
        f"Requests {COUNTS['requests']}, answered from the cache {COUNTS['hits']}, distinct "
        f"requests not in the cache {len(MISSES)}.",
        f"Call counter before and after the replay: {before} and {after}.",
    ]
    print("\n".join(out))
    if after != before:
        raise SystemExit("the call counter moved during a dry replay")


if __name__ == "__main__":
    main()
