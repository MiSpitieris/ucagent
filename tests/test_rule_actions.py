"""scripts/rule_actions.py: the summary by target of the actions the rule takes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "rule_actions.py"
    spec = importlib.util.spec_from_file_location("rule_actions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def cell(mod: ModuleType, target: float, **actions: float):
    full = {"ask_human": 0.0, "resample": 0.0, "backtrack": 0.0, "escalate": 0.0, **actions}
    return mod.Cell("b", "e", target, 0.1, 0.05, 0.5, full)


def test_the_summary_counts_selections_that_take_each_action(mod: ModuleType) -> None:
    cells = [
        cell(mod, 0.1, ask_human=0.5, resample=0.2),
        cell(mod, 0.1, ask_human=0.3, backtrack=0.1),
        cell(mod, 0.1, ask_human=0.4),
        cell(mod, 0.05, ask_human=1.2),
    ]
    lines = mod.summary(cells)
    assert lines[0].startswith("- Target 0.1: 3 selections; 3 take checks, 1 take resamples")
    assert "1 take backtracks, 0 take escalations; 1 only check;" in lines[0]
    assert "per question: checks 0.400, resamples 0.067, backtracks 0.033" in lines[0]
    assert lines[1].startswith("- Target 0.05: 1 selections;")
    assert "1 only check" in lines[1]


def test_a_selection_that_takes_no_action_does_not_count_as_only_checking(mod: ModuleType) -> None:
    lines = mod.summary([cell(mod, 0.2), cell(mod, 0.2, resample=0.1), cell(mod, 0.2, ask_human=1)])
    assert "1 take checks, 1 take resamples" in lines[0]
    assert "; 1 only check;" in lines[0]


def test_the_error_expected_after_a_resample_is_recovered_from_the_recorded_numbers(
    mod: ModuleType,
) -> None:
    # delta = S G w (eps - ehat), as core/risk/effects.py computes it
    numbers = {"S_prev": 0.9, "G": 0.8, "w": 1.0, "eps": 0.30}
    numbers["delta_resample"] = 0.9 * 0.8 * 1.0 * (0.30 - 0.10)
    assert mod.expected_after(numbers) == pytest.approx(0.10)
    assert mod.expected_after({"eps": 0.3}) is None  # the decision valued no resample
    assert mod.expected_after({**numbers, "S_prev": 0.0}) is None


def test_a_score_just_below_a_range_edge_is_recognised(mod: ModuleType) -> None:
    assert mod.just_below_edge(1 - 0.8)  # four of five agree: 0.19999999999999996
    assert mod.just_below_edge(1 - 8 / 10)
    assert not mod.just_below_edge(0.2)
    assert not mod.just_below_edge(0.19)
    assert not mod.just_below_edge(0.0)


def test_a_resample_is_paired_with_the_next_decision_at_its_step(mod: ModuleType) -> None:
    from ucagent.core.records import Action, Intervention, Trajectory

    def decision(step: int, action: Action, eps: float, **more: float) -> Intervention:
        numbers = {"S_prev": 1.0, "G": 1.0, "w": 1.0, "eps": eps, **more}
        return Intervention(step=step, attempt=0, action=action, numbers=numbers)

    trajectory = Trajectory(
        run_id="r",
        instance_id="i",
        split="test",
        seed=0,
        n_steps=2,
        policy="marginal",
        knob=0.1,
        estimator="disagreement",
        K=5,
        K_prime=10,
        steps=[],
        interventions=[
            decision(1, Action.RESAMPLE, 0.30, delta_resample=0.20),  # expected 0.10 after
            decision(1, Action.PROCEED, 0.15),  # the expanded batch showed 0.15
            decision(2, Action.RESAMPLE, 0.40, delta_resample=0.10),  # never decided again
        ],
        final_answer="a",
        final_correct=True,
        escalated=False,
        total_cost=0.2,
        fixed_cost=0.1,
        llm_calls=20,
        u_final=0.0,
        risk_naive_final=0.0,
        risk_weighted_final=0.0,
        survival_final=1.0,
    )
    first, second = mod.resamples_of(trajectory)
    assert (first.before, first.realised) == (0.30, 0.15)
    assert first.predicted == pytest.approx(0.10)
    assert first.below_edge is None  # no estimator given: the batch is not rebuilt
    assert second.realised is None
    row = mod.resample_row("x", [first, second])
    # expected 0.100, seen 0.150: the rule expected 0.050 more than the expanded batch showed
    assert row == "| x | 2 | 1 | 0.300 | 0.100 | 0.150 | +0.050 | 0 of 0 |"
    assert mod.resample_row("y", [second]) == "| y | 1 | - | - | - | - | - | - |"


class _Chain:
    """A two-step stub task: the truth of a step is its number, after an on-track context."""

    def get_instance(self, instance_id: str) -> str:
        return instance_id

    def initial_context(self, instance: str) -> str:
        return "start"

    def has_local_truth(self, instance: str, step: int, current: str) -> bool:
        return step == 1 or current == "1"

    def local_truth(self, instance: str, step: int, current: str) -> str:
        return str(step)

    def is_locally_correct(self, instance: str, step: int, current: str, answer: str) -> bool:
        return self.has_local_truth(instance, step, current) and answer == str(step)


def _majority(answers: tuple[str, ...]) -> str:
    return max(sorted(set(answers)), key=answers.count)


def _trajectory(decisions: list, samples: dict[str, list[str]], final: str):  # type: ignore[no-untyped-def]
    from ucagent.core.records import Trajectory

    return Trajectory(
        run_id="r",
        instance_id="q",
        split="test",
        seed=0,
        n_steps=2,
        policy="marginal",
        knob=0.1,
        estimator="disagreement",
        K=2,
        K_prime=4,
        steps=[],
        interventions=decisions,
        samples=samples,
        final_answer=final,
        final_correct=final == "2",
        escalated=False,
        total_cost=0.2,
        fixed_cost=0.1,
        llm_calls=6,
        u_final=0.0,
        risk_naive_final=0.0,
        risk_weighted_final=0.0,
        survival_final=1.0,
    )


def test_an_expansion_is_labelled_before_and_after_by_the_tasks_own_rule(mod: ModuleType) -> None:
    from ucagent.core.records import Action, Intervention

    def decision(step: int, action: Action, eps: float, **more: float) -> Intervention:
        numbers = {"S_prev": 1.0, "G": 1.0, "w": 1.0, "eps": eps, **more}
        return Intervention(step=step, attempt=0, action=action, numbers=numbers)

    # step 1: the first two samples are "x" and "y" (wrong); with four, "1" has the majority
    fixed = _trajectory(
        [
            decision(1, Action.RESAMPLE, 0.40, delta_resample=0.30),
            decision(1, Action.PROCEED, 0.20),
            decision(2, Action.PROCEED, 0.10),
        ],
        {"1:0": ["x", "y", "1", "1"], "2:0": ["2", "2"]},
        final="2",
    )
    n, found = mod.replay_expansions(fixed, _Chain(), _majority)
    assert n == 1 and len(found) == 1
    e = found[0]
    assert (e.wrong_before, e.wrong_after) == (True, False)
    assert e.expected_after == pytest.approx(0.10) and e.seen_after == 0.20
    assert e.key == ("q", 1, (), 0, ("x", "y", "1", "1"))

    # an off-track second step has no local label: it counts as a decision only
    off = _trajectory(
        [
            decision(1, Action.PROCEED, 0.30),
            decision(2, Action.RESAMPLE, 0.50, delta_resample=0.10),
            decision(2, Action.PROCEED, 0.45),
        ],
        {"1:0": ["x", "x"], "2:0": ["2", "2", "2", "2"]},
        final="2",
    )
    assert mod.replay_expansions(off, _Chain(), _majority) == (1, [])

    # a replay that does not end at the recorded final answer is refused
    with pytest.raises(ValueError, match="ends at another answer"):
        mod.replay_expansions(
            _trajectory(
                [decision(1, Action.PROCEED, 0.3), decision(2, Action.PROCEED, 0.1)],
                {"1:0": ["1", "1"], "2:0": ["2", "2"]},
                final="other",
            ),
            _Chain(),
            _majority,
        )

    # one case met at two knobs counts once; the row is over distinct cases
    row = mod.label_row("x", 2, 2, [e])
    assert row == "| x | 2 | 2 | 1 | 1 | 1 | 0 | 1 | 0 | 0.100 | 0.000 |"
    assert mod.label_row("y", 3, 0, []) == "| y | 3 | 0 | 0 | - | - | - | - | - | - | - |"


def test_a_check_and_a_backtrack_are_followed_through_the_history(mod: ModuleType) -> None:
    from ucagent.core.records import Action, Intervention

    def decision(step: int, action: Action, eps: float, target: int | None = None, **more: float):  # type: ignore[no-untyped-def]
        numbers = {"S_prev": 1.0, "G": 1.0, "w": 1.0, "eps": eps, **more}
        return Intervention(
            step=step, attempt=0, action=action, target_step=target, numbers=numbers
        )

    # step 1 is wrong ("x") and committed; at step 2 the rule backtracks to step 1, which is then
    # decided on its batch expanded to four samples, checked by a person, and step 2 is drawn anew
    t = _trajectory(
        [
            decision(1, Action.PROCEED, 0.30),
            decision(2, Action.BACKTRACK, 0.60, target=1),
            decision(1, Action.ASK_HUMAN, 0.50),
            decision(2, Action.RESAMPLE, 0.30, delta_resample=0.10),
            decision(2, Action.PROCEED, 0.25),
        ],
        {"1:0": ["x", "x", "x", "1"], "2:0": ["z", "z"], "2:1": ["2", "z", "2", "2"]},
        final="2",
    )
    n, found = mod.replay_expansions(t, _Chain(), _majority)
    assert n == 1 and len(found) == 1
    # the expansion at step 2 followed the reviewer's answer at step 1, and used the second draw
    assert found[0].key == ("q", 2, ("1",), 1, ("2", "z", "2", "2"))
    assert (found[0].wrong_before, found[0].wrong_after) == (False, False)
