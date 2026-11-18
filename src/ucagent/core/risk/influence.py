"""Downstream influence ``w_i``: priors, and counterfactual replay by checkpoint forking.

``w_i`` is the probability that a local error at step ``i`` is *fatal*: that it
flips the final answer when the remaining steps run under policy ``none``.
It is estimated by running each train instance once under ``none`` with
synchronous checkpoints and, for every step ``i`` whose prefix ``1..i-1`` was
on the true chain, forking the checkpoint taken right after ``control``
committed step ``i`` twice: once with a locally wrong alternative and once
with the gold answer (the base run itself when the committed answer was the
gold one), resuming both under ``none``. With ``p_wrong`` and ``p_correct``
the final-failure rates of the two continuations, the paired estimate is

    w_i = 1 - (1 - p_wrong) / (1 - p_correct)        (clipped to [0, 1])

which under independent local errors is exactly the fatality probability,
so that the product form ``S_i = prod (1 - w_j eps_j)`` is exact. The
*absolute* estimate ``w_i = p_wrong`` (the earlier definition) counts the
later steps' own failures as well, which the future factor then counts a
second time: with a step-2 error rate of 0.2 that always overwrites step 1,
the true final failure is 0.2, the absolute estimate gives
``1 - (1 - 0.5 * 0.2)(1 - 0.2) = 0.28`` and the paired estimate gives 0.2.
Both estimators are available (``estimate_influence``); ``paired`` is the
default. Steps whose prefix was already off track are skipped: on such a
prefix a "locally wrong" answer can be the globally correct entity and
repair the path, which is not what the product form (first error at a step)
models. A forced-answer replay from scratch (``RunSettings.force_answers``)
computes the same quantities without forking and serves as a cross-check.

# OPEN-QUESTION: influence is one number per (n, i); downstream error
# correction under an *acting* policy may differ from the policy-none value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ucagent.core.graph.build import Graph
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.runtime import SETTINGS_KEY, RunSettings, init_state, make_config
from ucagent.core.graph.state import next_attempt, next_draw
from ucagent.core.protocols import HumanInterface, Instance, Task
from ucagent.core.records import StepRecord
from ucagent.core.tables import Tables
from ucagent.core.uq.normalize import normalize_answer
from ucagent.core.utils import stable_hash

Method = Literal["fork", "replay"]


def ones(n: int) -> dict[tuple[int, int], float]:
    """``w_i = 1`` for every step (the default prior)."""
    return {(n, i): 1.0 for i in range(1, n + 1)}


def position_prior(
    n: int, shape: Literal["decreasing", "increasing"] = "decreasing", floor: float = 0.5
) -> dict[tuple[int, int], float]:
    """Linear prior in the step index, in ``[floor, 1]``, in either direction.

    Which direction is right is an empirical question (instructions, Section
    2.3); both are provided for ablations and neither is a default.
    """
    if n <= 1:
        return {(n, 1): 1.0}
    out: dict[tuple[int, int], float] = {}
    for i in range(1, n + 1):
        t = (i - 1) / (n - 1)
        value = 1.0 - (1.0 - floor) * t if shape == "decreasing" else floor + (1.0 - floor) * t
        out[(n, i)] = value
    return out


Kind = Literal["wrong", "correct"]
Estimator = Literal["paired", "absolute"]


@dataclass(frozen=True)
class CounterfactualOutcome:
    """What happened when step ``step`` was replaced by ``alternative``.

    ``kind`` says whether the alternative was a locally wrong answer or the
    gold one (the paired control); the base run stands in for the control
    when its committed answer already was the gold one.
    """

    step: int
    alternative: str
    final_wrong: bool
    kind: Kind = "wrong"


def wrong_alternative(task: Task, instance: Instance, record: StepRecord, seed: int) -> str:
    """A locally wrong answer for the step: a minority cluster, a distractor, or a random entity."""
    i, context = record.step, record.context
    for cluster in record.clusters[1:]:
        if not task.is_locally_correct(instance, i, context, cluster.representative):
            return cluster.representative
    rng = random.Random(stable_hash(seed, "counterfactual", instance.instance_id, i))
    wrong = [
        d
        for d in task.distractor_names(instance, i, context)
        if not task.is_locally_correct(instance, i, context, d)
    ]
    if wrong:
        return rng.choice(wrong)
    for _ in range(100):
        candidate = task.random_entity(instance, rng)
        if not task.is_locally_correct(instance, i, context, candidate):
            return candidate
    raise RuntimeError(f"no wrong alternative found for {instance.instance_id} step {i}")


def _none_settings(settings: RunSettings) -> RunSettings:
    return settings.model_copy(update={"policy": "none", "knob": None, "force_answers": {}})


def prefix_on_track(task: Task, instance: Instance, history: list[StepRecord], i: int) -> bool:
    """Whether steps ``1..i-1`` of ``history`` equal the global truths."""
    return all(
        normalize_answer(h.answer) == normalize_answer(task.global_truth(instance, h.step))
        for h in history[: i - 1]
    )


def _survival(history: list[StepRecord]) -> float:
    s = 1.0
    for h in history:
        s *= 1.0 - h.w * h.eps
    return s


def counterfactuals_by_forking(
    graph: Graph,
    task: Task,
    instance: Instance,
    settings: RunSettings,
    tables: Tables,
    human: HumanInterface,
) -> list[CounterfactualOutcome]:
    """Run once under ``none`` with synchronous checkpoints, then fork at every on-track step.

    Per step two outcomes are recorded, the wrong fork first and the correct
    (gold-answer) continuation second.
    """
    base = _none_settings(settings)
    n = instance.n_steps
    thread = f"cf:{instance.instance_id}:{base.seed}"
    config = make_config(thread, base, tables, n)
    final = run_with_human(graph, init_state(instance, base), config, human, durability="sync")
    history: list[StepRecord] = final["history"]
    base_wrong = not bool(final["final_correct"])
    snapshots = list(graph.get_state_history(config))

    def fork(i: int, answer: str) -> bool:
        snap = next(
            s
            for s in snapshots
            if len(s.values.get("history", [])) == i
            and s.values.get("pending") is None
            and s.next == ("prepare_step",)
        )
        new_history = list(snap.values["history"])
        new_history[i - 1] = history[i - 1].model_copy(
            update={"answer": answer, "source": "counterfactual"}
        )
        update: dict[str, Any] = {
            "history": new_history,
            "survival": _survival(new_history),
            "attempt": next_attempt(snap.values.get("interventions", []), i + 1),
            "draw": next_draw(snap.values["samples"], i + 1),
            "pending": None,
        }
        forked = graph.update_state(
            snap.config, Command(update=update, goto="prepare_step"), as_node="control"
        )
        forked_config: RunnableConfig = {
            "configurable": {
                **config["configurable"],
                **forked["configurable"],
                SETTINGS_KEY: base,
            },
            "recursion_limit": config["recursion_limit"],
        }
        out = run_with_human(graph, None, forked_config, human, durability="sync")
        assert out["history"][i - 1].answer == answer
        return not bool(out["final_correct"])

    outcomes: list[CounterfactualOutcome] = []
    for i in range(1, n + 1):
        if not prefix_on_track(task, instance, history, i):
            continue
        record = history[i - 1]
        alt = wrong_alternative(task, instance, record, base.seed)
        gold = task.global_truth(instance, i)
        committed_is_gold = normalize_answer(record.answer) == normalize_answer(gold)
        if i == n:
            outcomes.append(CounterfactualOutcome(i, alt, not task.is_final_correct(instance, alt)))
            outcomes.append(
                CounterfactualOutcome(i, gold, not task.is_final_correct(instance, gold), "correct")
            )
            continue
        outcomes.append(CounterfactualOutcome(i, alt, fork(i, alt)))
        correct_wrong = base_wrong if committed_is_gold else fork(i, gold)
        outcomes.append(CounterfactualOutcome(i, gold, correct_wrong, "correct"))
    checkpointer = graph.checkpointer
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread)
    return outcomes


def counterfactuals_by_replay(
    graph: Graph,
    task: Task,
    instance: Instance,
    settings: RunSettings,
    tables: Tables,
    human: HumanInterface,
) -> list[CounterfactualOutcome]:
    """Same quantities without forking: rerun from scratch with the step's answer forced."""
    base = _none_settings(settings)
    n = instance.n_steps
    config = make_config(f"cfr:{instance.instance_id}:{base.seed}", base, tables, n)
    final = run_with_human(graph, init_state(instance, base), config, human)
    history: list[StepRecord] = final["history"]
    base_wrong = not bool(final["final_correct"])

    def replay(i: int, answer: str) -> bool:
        forced = base.model_copy(update={"force_answers": {i: answer}})
        cfg = make_config(f"cfr:{instance.instance_id}:{base.seed}:{i}:{answer}", forced, tables, n)
        out = run_with_human(graph, init_state(instance, forced), cfg, human)
        return not bool(out["final_correct"])

    outcomes: list[CounterfactualOutcome] = []
    for i in range(1, n + 1):
        if not prefix_on_track(task, instance, history, i):
            continue
        record = history[i - 1]
        alt = wrong_alternative(task, instance, record, base.seed)
        gold = task.global_truth(instance, i)
        committed_is_gold = normalize_answer(record.answer) == normalize_answer(gold)
        if i == n:
            outcomes.append(CounterfactualOutcome(i, alt, not task.is_final_correct(instance, alt)))
            outcomes.append(
                CounterfactualOutcome(i, gold, not task.is_final_correct(instance, gold), "correct")
            )
            continue
        outcomes.append(CounterfactualOutcome(i, alt, replay(i, alt)))
        correct_wrong = base_wrong if committed_is_gold else replay(i, gold)
        outcomes.append(CounterfactualOutcome(i, gold, correct_wrong, "correct"))
    return outcomes


def estimate_influence(
    outcomes_by_instance: list[tuple[int, list[CounterfactualOutcome]]],
    estimator: Estimator = "paired",
) -> dict[tuple[int, int], float]:
    """Influence per ``(n, i)`` from the forked outcomes.

    ``absolute``: the final-failure rate of the wrong forks (counts later
    steps' own failures). ``paired``: ``1 - (1 - p_wrong) / (1 - p_correct)``
    clipped to ``[0, 1]``, and ``0`` when the correct continuation always
    fails (nothing is left for the step to spoil); falls back to ``absolute``
    for a step without any correct outcome (older callers).
    """
    sums: dict[tuple[Kind, tuple[int, int]], float] = {}
    counts: dict[tuple[Kind, tuple[int, int]], int] = {}
    for n, outcomes in outcomes_by_instance:
        for o in outcomes:
            key = (o.kind, (n, o.step))
            sums[key] = sums.get(key, 0.0) + float(o.final_wrong)
            counts[key] = counts.get(key, 0) + 1
    result: dict[tuple[int, int], float] = {}
    for kind, ni in sorted(counts):
        if kind != "wrong":
            continue
        p_wrong = sums[(kind, ni)] / counts[(kind, ni)]
        if estimator == "absolute" or ("correct", ni) not in counts:
            result[ni] = p_wrong
            continue
        p_correct = sums[("correct", ni)] / counts[("correct", ni)]
        if p_correct >= 1.0:
            result[ni] = 0.0
        else:
            result[ni] = min(1.0, max(0.0, 1.0 - (1.0 - p_wrong) / (1.0 - p_correct)))
    return result
