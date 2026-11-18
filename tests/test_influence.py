import pytest

from tests.helpers.configs import small_config
from ucagent.api import fit_calibrators
from ucagent.core.graph.build import build_graph
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.risk.influence import (
    CounterfactualOutcome,
    counterfactuals_by_forking,
    counterfactuals_by_replay,
    estimate_influence,
    ones,
    position_prior,
    wrong_alternative,
)
from ucagent.core.tables import Tables
from ucagent.experiments.runner import build_tables, run_influence
from ucagent.tasks.base import make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig


def _setup(n=4, rho=0.0, n_inst=10, ambiguity=0.5):
    task = SyntheticChainTask(
        SyntheticConfig(
            n_steps=n, ambiguity=ambiguity, reconvergence=rho, n_instances={"train": n_inst}
        )
    )
    llm = MockChainLLM(task=task, seed=0)
    settings = RunSettings(policy="none", K=3)
    return task, build_graph(task, llm), settings, make_human(task, settings)


def test_priors() -> None:
    assert ones(3) == {(3, 1): 1.0, (3, 2): 1.0, (3, 3): 1.0}
    dec = position_prior(5, "decreasing")
    inc = position_prior(5, "increasing")
    assert dec[(5, 1)] == 1.0 and dec[(5, 5)] == 0.5 and inc[(5, 1)] == 0.5 and inc[(5, 5)] == 1.0
    assert position_prior(1) == {(1, 1): 1.0}


def test_forking_and_replay_agree_and_only_change_later_steps() -> None:
    task, graph, settings, human = _setup()
    agree = 0
    for inst in task.instances("train"):
        fork = counterfactuals_by_forking(graph, task, inst, settings, Tables(), human)
        replay = counterfactuals_by_replay(graph, task, inst, settings, Tables(), human)
        wrong_steps = [o.step for o in fork if o.kind == "wrong"]
        assert wrong_steps == sorted(set(wrong_steps))
        assert fork and fork[0].step == 1  # step 1's prefix is always on track
        assert [(o.step, o.alternative, o.kind) for o in fork] == [
            (o.step, o.alternative, o.kind) for o in replay
        ]
        assert [o.final_wrong for o in fork] == [o.final_wrong for o in replay]
        agree += 1
    assert agree == 10


def test_wrong_alternative_is_locally_wrong() -> None:
    task, _graph, _settings, _human = _setup(n=3)
    inst = task.get_instance("train-00000-s0")
    from ucagent.core.records import Cluster, StepRecord

    truth = task.local_truth(inst, 2, inst.entities[1])
    rec = StepRecord(
        step=2,
        attempt=0,
        context=inst.entities[1],
        answer=truth,
        u=0.3,
        entropy=0.3,
        eps=0.3,
        clusters=[
            Cluster(representative=truth, members=[0, 1], proportion=0.67),
            Cluster(representative=inst.distractors[1][0], members=[2], proportion=0.33),
        ],
    )
    alt = wrong_alternative(task, inst, rec, seed=0)
    assert alt == inst.distractors[1][0]
    rec2 = rec.model_copy(update={"clusters": [rec.clusters[0]]})
    alt2 = wrong_alternative(task, inst, rec2, seed=0)
    assert not task.is_locally_correct(inst, 2, inst.entities[1], alt2)


def test_influence_is_one_on_a_pure_chain() -> None:
    cfg = small_config(
        task={"n_steps": 4, "reconvergence": 0.0, "n_instances": {"train": 30}}, llm={"K": 3}
    )
    w = run_influence(cfg, workers=4)
    assert set(w) == {(4, i) for i in range(1, 5)}
    assert all(v >= 0.95 for v in w.values()), w


def test_influence_is_heterogeneous_with_reconvergence() -> None:
    cfg = small_config(
        task={
            "n_steps": 5,
            "reconvergence": [0.5, 0.5, 0.25, 0.0, 0.0],
            "n_instances": {"train": 80},
        },
        llm={"K": 3},
    )
    w = run_influence(cfg, workers=4)
    assert w[(5, 5)] >= 0.95 and w[(5, 1)] <= w[(5, 5)] - 0.15, w
    assert w[(5, 1)] <= 0.85, w


def test_fit_artifact_with_counterfactual_influence_and_table_variants(tmp_path) -> None:
    cfg = small_config(
        task={"n_steps": 3, "reconvergence": 0.5, "n_instances": {"train": 20}},
        llm={"K": 3, "K_prime": 6},
        risk={"influence": "counterfactual"},
    )
    art = fit_calibrators(cfg, workers=2, out_path=tmp_path / "a.json")
    assert art.influence_method == "counterfactual:paired"
    assert set(art.influence) == {"3:1", "3:2", "3:3"}
    cf = build_tables(cfg.with_updates(uq={"artifact": str(tmp_path / "a.json")}))
    assert cf.w(3, 1) == pytest.approx(art.influence["3:1"])
    one = build_tables(
        cfg.with_updates(uq={"artifact": str(tmp_path / "a.json")}, risk={"influence": "ones"})
    )
    assert one.w(3, 1) == 1.0
    dec = build_tables(cfg.with_updates(risk={"influence": "position_prior:decreasing"}))
    assert dec.w(3, 3) == 0.5 and dec.w(3, 1) == 1.0
    with pytest.raises(ValueError):
        build_tables(cfg.with_updates(risk={"influence": "nope"}))
    assert estimate_influence([]) == {}


def test_paired_estimator_reproduces_the_review_counterexample() -> None:
    # Step 2 overwrites step 1 and fails 20 % of the time whatever step 1 did: an injected
    # step-1 error is never fatal, so its influence is 0; the absolute rate says 0.2.
    wrong = [CounterfactualOutcome(1, "x", k < 2) for k in range(10)]
    correct = [CounterfactualOutcome(1, "gold", k < 2, "correct") for k in range(10)]
    outcomes = [(2, wrong + correct)]
    assert estimate_influence(outcomes, estimator="paired") == {(2, 1): 0.0}
    assert estimate_influence(outcomes, estimator="absolute") == {(2, 1): 0.2}
    # fatal with probability 1 - (1 - 0.6) / (1 - 0.2) = 0.5
    wrong = [CounterfactualOutcome(1, "x", k < 6, "wrong") for k in range(10)]
    assert abs(estimate_influence([(2, wrong + correct)])[(2, 1)] - 0.5) < 1e-12
    # the correct continuation always fails: nothing left to spoil
    doomed = [CounterfactualOutcome(1, "gold", True, "correct") for _ in range(5)]
    assert estimate_influence([(2, wrong + doomed)])[(2, 1)] == 0.0
    # clipping when the wrong forks fail less often than the correct ones
    lucky = [CounterfactualOutcome(1, "x", False, "wrong") for _ in range(5)]
    assert estimate_influence([(2, lucky + correct)])[(2, 1)] == 0.0
    # no correct outcome recorded (older callers): the absolute rate is used
    assert estimate_influence([(2, wrong)])[(2, 1)] == 0.6


def test_forking_emits_a_correct_outcome_per_on_track_step() -> None:
    task, graph, settings, human = _setup(n=3, n_inst=6)
    for inst in task.instances("train"):
        fork = counterfactuals_by_forking(graph, task, inst, settings, Tables(), human)
        by_step: dict[int, list[CounterfactualOutcome]] = {}
        for o in fork:
            by_step.setdefault(o.step, []).append(o)
        for step, pair in by_step.items():
            assert [o.kind for o in pair] == ["wrong", "correct"], (step, pair)
        first = by_step[1]
        start = inst.entities[0]  # the context of step 1 on the synthetic chain
        assert task.is_locally_correct(inst, 1, start, first[1].alternative)
        assert not task.is_locally_correct(inst, 1, start, first[0].alternative)
        last = by_step.get(inst.n_steps)
        if last is not None:
            assert last[1].final_wrong is False  # the gold final answer is final-correct
