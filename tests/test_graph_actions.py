"""Resample, backtrack and escalate on the mock: expansion semantics (2026-11-08)."""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError

from tests.helpers.policies import register_forced
from ucagent.core.control.base import register_policy
from ucagent.core.graph.build import build_graph
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.nodes import GraphNodes
from ucagent.core.graph.runtime import (
    CostTable,
    RunSettings,
    init_state,
    make_config,
    recursion_limit_for,
)
from ucagent.core.graph.state import samples_key
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import Action, Decision, Sample
from ucagent.core.tables import Tables
from ucagent.core.uq.calibration import CalibrationArtifact, IsotonicData, ResampleTableData
from ucagent.experiments.runner import trajectory_from_state
from ucagent.tasks.base import make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig

COSTS = CostTable(llm_call=0.01, resample=0.0, backtrack=0.5, ask_human=1.0, escalate=10.0)


def _run(
    policy: str,
    n: int = 4,
    K: int = 2,
    K_prime: int = 4,
    tables: Tables | None = None,
    recursion_limit: int | None = None,
    **kw,
):
    task = SyntheticChainTask(SyntheticConfig(n_steps=n, ambiguity=0.5, n_instances={"test": 3}))
    graph = build_graph(task, MockChainLLM(task=task, seed=0))
    settings = RunSettings(policy=policy, K=K, K_prime=K_prime, costs=COSTS, **kw)
    inst = task.get_instance("test-00000-s0")
    config = make_config("t", settings, tables or Tables(), n)
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    out = run_with_human(graph, init_state(inst, settings), config, make_human(task, settings))
    return task, inst, out, settings


def _by_index(samples: list[Sample]) -> list[Sample]:
    return sorted(samples, key=lambda s: s.sample_index)


def test_resample_expands_the_batch_in_place() -> None:
    name = register_forced("forced_resample", {(2, 0): (Action.RESAMPLE, None)})
    _, _, out, _ = _run(name)
    _, _, plain, _ = _run("none")
    assert "2:0" in out["samples"] and "2:1" not in out["samples"]
    batch = _by_index(out["samples"]["2:0"])
    assert [s.sample_index for s in batch] == [0, 1, 2, 3]
    # the original two samples are kept; the expansion only adds indices 2 and 3
    assert [s.answer for s in batch[:2]] == [s.answer for s in _by_index(plain["samples"]["2:0"])]
    committed = next(h for h in out["history"] if h.step == 2)
    assert (committed.attempt, committed.draw, committed.resample_count) == (1, 0, 1)
    assert committed.n_samples == 4 and committed.samples_key == "2:0"
    resamples = [iv for iv in out["interventions"] if iv.action == Action.RESAMPLE]
    assert len(resamples) == 1 and (resamples[0].step, resamples[0].attempt) == (2, 0)
    assert resamples[0].cost == pytest.approx(2 * 0.01)  # only the two extra draws
    assert len(out["history"]) == 4
    assert out["llm_calls"] == 4 * 2 + 2
    assert out["total_cost"] == pytest.approx(out["llm_calls"] * 0.01)


def test_backtrack_expands_the_target_and_redraws_later_steps() -> None:
    name = register_forced("forced_backtrack", {(3, 0): (Action.BACKTRACK, 1)})
    _, _, out, _ = _run(name)
    hist = out["history"]
    assert [h.step for h in hist] == [1, 2, 3, 4]
    # the target keeps its draw and is expanded to K'
    assert (hist[0].attempt, hist[0].draw, hist[0].n_samples, hist[0].samples_key) == (
        1,
        0,
        4,
        "1:0",
    )
    # the redone later steps get fresh draws with K samples
    assert (hist[1].attempt, hist[1].draw, hist[1].n_samples) == (1, 1, 2)
    assert (hist[2].attempt, hist[2].draw, hist[2].n_samples) == (1, 1, 2)
    assert (hist[3].attempt, hist[3].draw) == (0, 0)
    assert set(out["samples"]) == {"1:0", "2:0", "3:0", "2:1", "3:1", "4:0"}
    assert len(out["samples"]["1:0"]) == 4
    assert out["backtrack_count"] == 1
    backtrack = next(iv for iv in out["interventions"] if iv.action == Action.BACKTRACK)
    assert backtrack.cost == pytest.approx(0.5 + 2 * 0.01 + 2 * 2 * 0.01)
    assert out["llm_calls"] == 3 * 2 + 2 + 2 * 2 + 2
    assert out["fixed_cost"] == pytest.approx(0.5)
    survival = 1.0
    for h in hist:
        survival *= 1 - h.w * h.eps
    assert out["survival"] == pytest.approx(survival)


def test_escalate_ends_the_trajectory_with_the_oracle_answer() -> None:
    name = register_forced("forced_escalate", {(2, 0): (Action.ESCALATE, None)})
    _task, inst, out, _s = _run(name)
    assert out["escalated"] is True and out["done"] is True
    assert out["final_answer"] == inst.entities[-1] and out["final_correct"] is True
    assert len(out["history"]) == 1 and out["llm_calls"] == 4
    assert out["fixed_cost"] == pytest.approx(10.0)
    assert out["total_cost"] == pytest.approx(10.0 + 4 * 0.01)
    assert [iv.action for iv in out["interventions"]][-1] == Action.ESCALATE


def test_the_audit_of_an_escalated_run_shows_the_escalation() -> None:
    name = register_forced("forced_escalate_audit", {(2, 0): (Action.ESCALATE, None)})
    task, inst, out, settings = _run(name)
    trajectory = trajectory_from_state(task, settings, "t", out, inst.instance_id)
    assert trajectory.escalated and [s.step for s in trajectory.steps] == [1]
    audit = trajectory.audit()
    assert [(row["step"], row["committed"]) for row in audit] == [(1, True), (2, False)]
    decisions = audit[1]["decisions"]
    assert [d["action"] for d in decisions] == ["escalate"]
    assert decisions[0]["cost"] == 10.0  # the price of the takeover, which the old view lost


def _two_level_artifact(K: int = 2, K_prime: int = 4) -> CalibrationArtifact:
    return CalibrationArtifact(
        estimator="disagreement",
        equivalence="normalized_exact",
        K=K,
        K_prime=K_prime,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        isotonic_K=IsotonicData(x=[0.0], y=[0.9]),
        isotonic_K_prime=IsotonicData(x=[0.0], y=[0.1]),
        resample_table=ResampleTableData(k_prime=K_prime),
    )


def test_resampled_batch_is_calibrated_by_its_size() -> None:
    """Review finding R1: a K'-sample batch goes through the K' map, not the K map."""
    name = register_forced("forced_resample_r1", {(2, 0): (Action.RESAMPLE, None)})
    _, _, out, _ = _run(name, tables=_two_level_artifact().to_tables())
    by_step = {h.step: h for h in out["history"]}
    assert by_step[2].n_samples == 4 and by_step[2].eps == pytest.approx(0.1)
    assert all(
        by_step[i].n_samples == 2 and by_step[i].eps == pytest.approx(0.9) for i in (1, 3, 4)
    )
    # the reviewer's direct unit case: four unanimous samples under the K = 2 settings
    settings = RunSettings(K=2, K_prime=4)
    state = {
        "i": 1,
        "attempt": 1,
        "draw": 0,
        "n_steps": 1,
        "instance_id": "demo",
        "current": "",
        "samples": {
            samples_key(1, 0): [Sample(sample_index=i, text="a", answer="a") for i in range(4)]
        },
        "resample_count": 1,
    }
    record = GraphNodes(None, None).estimate(  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        make_config("demo", settings, _two_level_artifact().to_tables(), 1),
    )["pending"]
    assert record.n_samples == 4 and record.eps == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no calibrator fitted for batches of 3"):
        _two_level_artifact().to_tables().eps(0.0, 3)


def test_second_expansion_is_infeasible_and_degrades_to_proceed() -> None:
    script = {(2, 0): (Action.RESAMPLE, None), (2, 1): (Action.RESAMPLE, None)}
    name = register_forced("forced_double_resample", script)
    _, _, out, _ = _run(name, max_resamples=2)
    step2 = [iv for iv in out["interventions"] if iv.step == 2]
    assert [iv.action for iv in step2] == [Action.RESAMPLE, Action.PROCEED]
    assert step2[1].numbers["infeasible"] == 1.0 and step2[1].reason.startswith(
        "infeasible resample"
    )
    assert len(out["samples"]["2:0"]) == 4 and out["done"]
    # a forced backtrack to a step that already holds K' samples is degraded the same way
    script = {(1, 0): (Action.RESAMPLE, None), (3, 0): (Action.BACKTRACK, 1)}
    name = register_forced("forced_backtrack_to_full", script)
    _, _, out, _ = _run(name)
    step3 = next(iv for iv in out["interventions"] if iv.step == 3)
    assert step3.action == Action.PROCEED and step3.numbers["infeasible"] == 1.0
    assert out["backtrack_count"] == 0 and len(out["history"]) == 4


def test_k_prime_equal_k_never_expands() -> None:
    name = register_forced("forced_resample_kk", {(2, 0): (Action.RESAMPLE, None)})
    _, _, out, _ = _run(name, K=2, K_prime=2)
    step2 = next(iv for iv in out["interventions"] if iv.step == 2)
    assert step2.action == Action.PROCEED and step2.numbers["infeasible"] == 1.0
    assert "nothing to expand" in step2.reason
    assert set(out["samples"]) == {"1:0", "2:0", "3:0", "4:0"} and out["llm_calls"] == 8


def test_worst_case_resample_after_backtrack_fits_recursion_limit() -> None:
    """Review finding R11: redone steps may resample again; the bound must allow it."""
    n = 6
    script: dict[tuple[int, int], tuple[Action, int | None]] = {}
    for step in range(2, n + 1):
        script[(step, 0)] = (Action.RESAMPLE, None)  # first pass
        script[(step, 2)] = (Action.RESAMPLE, None)  # second pass, after the backtrack
    script[(n, 1)] = (Action.BACKTRACK, 1)
    name = register_forced("forced_worst_case", script)
    _, _, out, s = _run(name, n=n, K=1, K_prime=2, max_resamples=1, max_backtracks=1)
    assert out["done"] and out["backtrack_count"] == 1
    assert sum(iv.action == Action.RESAMPLE for iv in out["interventions"]) == 2 * (n - 1)
    assert out["llm_calls"] == n + (n - 1) + 1 + (n - 1) + (n - 1)
    old_additive_bound = 4 * n + 1 + 5 * n + (1 + 4 * n) + n + 1 + 10  # the pre-fix formula: 97
    with pytest.raises(GraphRecursionError):
        _run(
            name,
            n=n,
            K=1,
            K_prime=2,
            max_resamples=1,
            max_backtracks=1,
            recursion_limit=old_additive_bound,
        )
    assert recursion_limit_for(n, s) > old_additive_bound


class _CappedPolicy:
    """The reviewer's policy: resamples while the counter allows, backtracks once at the end."""

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings = settings

    def decide(self, state, step) -> Decision:
        if state["resample_count"] < self.settings.max_resamples:
            return Decision(action=Action.RESAMPLE, reason="capped: resample")
        if (
            state["i"] == state["n_steps"]
            and state["backtrack_count"] < self.settings.max_backtracks
        ):
            return Decision(action=Action.BACKTRACK, target_step=1, reason="capped: backtrack")
        return Decision(action=Action.PROCEED, reason="capped: proceed")


class _AwareCappedPolicy:
    """As above but respecting the batch-size rule; step 1 is left for the backtrack to expand."""

    def __init__(self, settings: RunSettings, tables: Tables) -> None:
        self.settings = settings

    def decide(self, state, step) -> Decision:
        if step.step > 1 and step.n_samples < self.settings.K_prime:
            return Decision(action=Action.RESAMPLE, reason="aware: expand")
        first = state["history"][0] if state["history"] else None
        if (
            state["i"] == state["n_steps"]
            and state["backtrack_count"] < self.settings.max_backtracks
            and first is not None
            and first.n_samples < self.settings.K_prime
        ):
            return Decision(action=Action.BACKTRACK, target_step=1, reason="aware: backtrack")
        return Decision(action=Action.PROCEED, reason="aware: proceed")


def test_capped_policies_complete_at_the_default_limit() -> None:
    """Review finding R11: resampling again after a backtrack is legal and must fit the bound."""
    register_policy("review_capped", _CappedPolicy)
    register_policy("review_capped_aware", _AwareCappedPolicy)
    # The reviewer's policy asks for a second expansion of every step: degraded, flagged, and the
    # trajectory completes (its backtrack branch is never reached, because the degraded resample
    # commits the step).
    _, _, out, _ = _run("review_capped", n=4, K=1, K_prime=2, max_resamples=2, max_backtracks=1)
    assert out["done"]
    flagged = [iv for iv in out["interventions"] if iv.numbers.get("infeasible") == 1.0]
    assert len(flagged) == 4 and all(iv.action == Action.PROCEED for iv in flagged)
    # A policy that respects the batch-size rule: one expansion per step in both passes and one
    # backtrack that expands step 1, all within the default recursion limit.
    _, _, out, _ = _run(
        "review_capped_aware", n=4, K=1, K_prime=2, max_resamples=2, max_backtracks=1
    )
    assert out["done"] and out["backtrack_count"] == 1
    assert sum(iv.action == Action.RESAMPLE for iv in out["interventions"]) == 6
    assert all(h.n_samples == 2 for h in out["history"])
    assert not any(iv.numbers.get("infeasible") for iv in out["interventions"])
