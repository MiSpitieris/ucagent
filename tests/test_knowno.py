"""KnowNo-style per-step conformal gate: quantile, prediction set, artifact scores."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ucagent.core.control  # noqa: F401
from tests.helpers.configs import small_config
from ucagent.api import fit_calibrators
from ucagent.core.control.base import make_policy
from ucagent.core.graph.runtime import RunSettings, init_state
from ucagent.core.protocols import Instance
from ucagent.core.records import Action, Cluster, StepRecord
from ucagent.core.tables import Tables
from ucagent.core.uq.calibration import StepObservation, fit_artifact, load_artifact

SCORES = [0.0, 0.2, 0.4, 0.6, 0.8]


def _tables(scores: list[float] | None = SCORES, k: int = 5) -> Tables:
    return Tables(conformal_scores={k: list(scores)} if scores is not None else {})


def _step(shares: list[float], k: int = 5) -> StepRecord:
    clusters = [
        Cluster(representative=f"a{j}", members=[j], proportion=p) for j, p in enumerate(shares)
    ]
    return StepRecord(
        step=1,
        attempt=0,
        context="c",
        answer="a0",
        u=1 - shares[0],
        entropy=0.0,
        eps=0.0,
        clusters=clusters,
        n_samples=k,
    )


def _state():
    inst = Instance(instance_id="i", split="test", index=0, question="q", n_steps=3)
    return init_state(inst, RunSettings())


def _decide(eps: float, shares: list[float], tables: Tables):
    policy = make_policy(RunSettings(policy="knowno", knob=eps), tables)
    return policy.decide(_state(), _step(shares))


def test_qhat_rank_rule() -> None:
    t = _tables()
    assert t.qhat(5, 0.1) == 1.0  # rank 6 of 5 scores: everything qualifies
    assert t.qhat(5, 0.5) == 0.4  # rank ceil(3.0) = 3
    assert t.qhat(5, 0.9) == 0.0  # rank ceil(0.6) = 1
    assert t.qhat(5, 0.0) == 1.0 and t.qhat(5, 1.0) == 0.0
    assert t.qhat(10, 0.5) is None and Tables().qhat(5, 0.5) is None


def test_singleton_set_proceeds_and_larger_set_asks() -> None:
    d = _decide(0.5, [0.6, 0.4], _tables())  # qhat 0.4: only p >= 0.6 qualifies
    assert d.action == Action.PROCEED and d.numbers["set_size"] == 1 and d.numbers["qhat"] == 0.4
    d = _decide(0.1, [0.6, 0.4], _tables())  # qhat 1.0: both clusters in the set
    assert d.action == Action.ASK_HUMAN and d.numbers["set_size"] == 2
    assert "knowno: ask" in d.reason and d.numbers["m"] == 5
    assert _decide(0.1, [1.0], _tables()).action == Action.PROCEED  # unanimous: singleton


def test_uncalibrated_fallback_without_scores() -> None:
    d = _decide(0.3, [0.6, 0.4], _tables(None))
    assert d.action == Action.ASK_HUMAN and d.numbers["uncalibrated"] == 1.0
    assert _decide(0.5, [0.8, 0.2], _tables(None)).action == Action.PROCEED


@settings(max_examples=80, deadline=None)
@given(
    st.floats(0.001, 0.999),
    st.floats(0.001, 0.999),
    st.lists(st.floats(0.01, 1.0), min_size=1, max_size=5),
    st.lists(st.floats(0.0, 1.0), min_size=1, max_size=12),
)
def test_asking_is_monotone_in_the_knob(e1, e2, weights, scores) -> None:
    lo, hi = sorted([e1, e2])
    total = sum(weights)
    shares = sorted((w / total for w in weights), reverse=True)
    tables = _tables(sorted(scores))
    ask_hi = _decide(hi, shares, tables).action == Action.ASK_HUMAN
    ask_lo = _decide(lo, shares, tables).action == Action.ASK_HUMAN
    assert ask_lo or not ask_hi


def test_artifact_stores_sorted_scores_and_old_artifacts_still_load(tmp_path) -> None:
    obs = [
        StepObservation(
            n_steps=2,
            step=1,
            u_K=0.1,
            wrong_K=False,
            u_K_prime=0.1,
            wrong_K_prime=False,
            p_true_K=0.8,
            p_true_K_prime=0.9,
        ),
        StepObservation(
            n_steps=2,
            step=2,
            u_K=0.7,
            wrong_K=True,
            u_K_prime=0.5,
            wrong_K_prime=False,
            p_true_K=0.0,
            p_true_K_prime=0.6,
        ),
    ]
    art = fit_artifact(
        obs,
        estimator="disagreement",
        equivalence="normalized_exact",
        K=5,
        K_prime=10,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=1,
        trajectories=[obs],
    )
    assert art.conformal_scores["K"] == pytest.approx([0.2, 1.0])
    assert art.conformal_scores["K_prime"] == pytest.approx([0.1, 0.4])
    tables = art.to_tables()
    assert set(tables.conformal_scores) == {5, 10} and tables.qhat(10, 0.5) == pytest.approx(0.4)
    path = art.save(tmp_path / "a.json")
    assert load_artifact(path).conformal_scores == art.conformal_scores
    old = json.loads(path.read_text())
    del old["conformal_scores"]
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert load_artifact(tmp_path / "old.json").to_tables().conformal_scores == {}


def test_fit_calibrators_records_the_correct_cluster_share() -> None:
    cfg = small_config(llm={"K": 2, "K_prime": 4}, task={"n_instances": {"train": 12}})
    art = fit_calibrators(cfg, workers=1)
    scores = art.conformal_scores
    assert len(scores["K"]) == 12 * 3 and len(scores["K_prime"]) == 12 * 3
    assert all(0.0 <= s <= 1.0 for s in scores["K"]) and scores["K"] == sorted(scores["K"])
    assert any(s < 1.0 for s in scores["K"])  # the mock gets some steps right
