import pytest
from langgraph.errors import GraphRecursionError

import tests.helpers.policies  # noqa: F401  (registers ask_last / ask_all)
from ucagent.core.graph.build import NODE_NAMES, build_graph, mermaid
from ucagent.core.graph.human import run_with_human
from ucagent.core.graph.runtime import RunSettings, init_state, make_config, recursion_limit_for
from ucagent.core.llm.mock import MockChainLLM
from ucagent.core.records import Action
from ucagent.core.tables import Tables
from ucagent.tasks.base import make_human
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig


def _setup(n: int = 3, **llm_kw):
    task = SyntheticChainTask(SyntheticConfig(n_steps=n, ambiguity=0.5, n_instances={"test": 20}))
    llm = MockChainLLM(task=task, seed=0, **llm_kw)
    return task, llm, build_graph(task, llm)


def _run(task, graph, instance_id: str, settings: RunSettings, **cfg_overrides):
    inst = task.get_instance(instance_id)
    config = make_config(f"t:{instance_id}", settings, Tables(), inst.n_steps)
    config.update(cfg_overrides)
    return run_with_human(graph, init_state(inst, settings), config, make_human(task, settings))


def test_plain_loop_runs_to_done() -> None:
    task, _, graph = _setup(n=3)
    settings = RunSettings(policy="none", K=1)
    out = _run(task, graph, "test-00000-s0", settings)
    assert out["done"] and len(out["history"]) == 3
    assert out["llm_calls"] == 3 and out["final_answer"] == out["history"][-1].answer
    assert [iv.action for iv in out["interventions"]] == [Action.PROCEED] * 3
    assert all(iv.reason for iv in out["interventions"])
    assert out["total_cost"] == pytest.approx(3 * settings.costs.llm_call)
    assert [h.step for h in out["history"]] == [1, 2, 3]
    assert out["history"][1].context == out["history"][0].answer


def test_easy_model_is_always_right() -> None:
    task, _, graph = _setup(n=4, difficulty_beta=(1.0, 1e6))
    settings = RunSettings(policy="none", K=1)
    for inst in task.instances("test"):
        out = _run(task, graph, inst.instance_id, settings)
        assert out["final_correct"] is True


def test_k_samples_fan_out() -> None:
    task, _, graph = _setup(n=2)
    settings = RunSettings(policy="none", K=6)
    out = _run(task, graph, "test-00001-s0", settings)
    assert out["llm_calls"] == 12
    assert sorted(s.sample_index for s in out["samples"]["1:0"]) == list(range(6))
    assert out["history"][0].n_samples == 6 and 0.0 <= out["history"][0].u <= 1.0


def test_recursion_limit_formula_is_sufficient_and_tight() -> None:
    task, _, graph = _setup(n=5)
    settings = RunSettings(policy="none", K=2)
    assert _run(task, graph, "test-00002-s0", settings)["done"]
    with pytest.raises(GraphRecursionError):
        _run(task, graph, "test-00003-s0", settings, recursion_limit=4 * 5)  # supersteps - 1
    assert recursion_limit_for(5, settings) > 4 * 5 + 1


def test_ask_human_commits_oracle_answer() -> None:
    task, _, graph = _setup(n=3)
    settings = RunSettings(policy="ask_all", K=1)
    out = _run(task, graph, "test-00004-s0", settings)
    inst = task.get_instance("test-00004-s0")
    assert all(h.source == "human" and h.eps == 0.0 for h in out["history"])
    assert out["history"][0].answer == task.local_truth(inst, 1, inst.entities[0])
    assert out["final_correct"] is True
    assert out["fixed_cost"] == pytest.approx(3 * settings.costs.ask_human)
    assert out["total_cost"] == pytest.approx(
        3 * settings.costs.ask_human + 3 * settings.costs.llm_call
    )
    assert [iv.action for iv in out["interventions"]] == [Action.ASK_HUMAN] * 3


def test_noisy_oracle_sets_eps_to_p_h() -> None:
    task, _, graph = _setup(n=2)
    settings = RunSettings(policy="ask_all", K=1, p_h=0.3)
    out = _run(task, graph, "test-00005-s0", settings)
    assert all(h.eps == pytest.approx(0.3) for h in out["history"])
    assert out["survival"] == pytest.approx(0.7**2)


def test_mermaid_lists_every_node() -> None:
    _, _, graph = _setup()
    text = mermaid(graph)
    for name in NODE_NAMES:
        assert name in text
    assert "ask_human" in text and "escalate" in text


def test_checkpoint_round_trip_emits_no_unregistered_type_warning() -> None:
    import warnings

    task, _, graph = _setup(n=2)
    settings = RunSettings(policy="ask_all", K=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(task, graph, "test-00006-s0", settings)
    assert out["done"]
    assert not [w for w in caught if "unregistered type" in str(w.message)]
