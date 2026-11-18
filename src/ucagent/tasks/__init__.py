"""Research harness: task definitions (synthetic chain, benchmark adapters) and oracles.

Tasks are registered by name; ``ExperimentConfig.task.name`` selects one and
the rest of the ``task`` block is passed to its factory. Adopters register
their own task from a plugin module (``plugins:`` in the config).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ucagent.core.protocols import Task

TaskFactory = Callable[[dict[str, Any]], Task]
TASKS: dict[str, TaskFactory] = {}


def register_task(name: str, factory: TaskFactory) -> None:
    """Register a task factory taking the ``task`` config block as a dict."""
    TASKS[name] = factory


def make_task(block: dict[str, Any]) -> Task:
    """Instantiate the task named in a ``task`` config block."""
    name = str(block.get("name", "synthetic_chain"))
    try:
        factory = TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}") from exc
    return factory(block)


def _synthetic(block: dict[str, Any]) -> Task:
    from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig

    return SyntheticChainTask(SyntheticConfig.model_validate(block))


def _musique(block: dict[str, Any]) -> Task:
    from ucagent.tasks.musique import MusiqueTask

    return MusiqueTask(**{k: v for k, v in block.items() if k != "name"})


def _twowiki(block: dict[str, Any]) -> Task:
    from ucagent.tasks.twowiki import TwoWikiTask

    return TwoWikiTask(**{k: v for k, v in block.items() if k != "name"})


register_task("synthetic_chain", _synthetic)
register_task("musique", _musique)
register_task("twowiki", _twowiki)


def _finqa(block: dict[str, Any]) -> Task:
    from ucagent.tasks.finqa import FinqaTask

    return FinqaTask(**{k: v for k, v in block.items() if k != "name"})


register_task("finqa", _finqa)
