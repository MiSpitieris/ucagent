"""The worked example of docs/ADOPTION.md must keep running end to end."""

import importlib.util
from pathlib import Path
from typing import Any

from ucagent.api import run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import Action
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def _load_toy():
    spec = importlib.util.spec_from_file_location(
        "examples.toy_task", ROOT / "examples" / "toy_task.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


class ReviewQueue:
    """What an adopter implements: a human that checks a step or takes over a case."""

    def __init__(self, task: Any) -> None:
        self.task, self.checked, self.taken = task, 0, 0

    def check_step(self, payload: dict[str, Any]) -> str:
        self.checked += 1
        inst = self.task.get_instance(payload["instance_id"])
        return self.task.local_truth(inst, payload["step"], payload["current"])

    def take_over(self, payload: dict[str, Any]) -> str:
        self.taken += 1
        inst = self.task.get_instance(payload["instance_id"])
        return self.task.global_truth(inst, inst.n_steps)


def test_toy_task_runs_through_the_api() -> None:
    toy = _load_toy()
    task = toy.ArithmeticChainTask(n_steps=3, n_instances={"test": 6})
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(2.0, 2.0))
    human = ReviewQueue(task)
    settings = RunSettings(policy="marginal", knob=0.05, K=5, K_prime=10)
    trajs = run_instances(
        task, llm, settings, [i.instance_id for i in task.instances("test")], human
    )
    assert len(trajs) == 6 and all(len(t.steps) == 3 or t.escalated for t in trajs)
    assert any(t.n_interventions > 0 for t in trajs)
    assert human.checked == sum(t.count(Action.ASK_HUMAN) for t in trajs)
    assert all(d["reason"] for t in trajs for row in t.audit() for d in row["decisions"])


def test_toy_task_runs_through_the_config_path(tmp_path) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "run_id": "toy",
            "plugins": ["examples.toy_task"],
            "task": {"name": "arithmetic", "n_steps": 3, "n_instances": {"test": 5}},
            "llm": {"K": 3, "K_prime": 6},
            "control": {"policy": "per_step_u", "knob": 0.3},
        }
    )
    res = run_experiment(cfg, workers=1, write=True, results_dir=tmp_path)
    assert len(res.trajectories) == 5 and (tmp_path / "toy" / "trajectories.parquet").exists()
