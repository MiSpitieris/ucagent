"""M1 acceptance: compounding with n, slower decay with reconvergence, oracle degeneracy."""

import itertools

import pytest

import tests.helpers.policies  # noqa: F401
from tests.helpers.configs import small_config
from ucagent.experiments.runner import run_experiment

pytestmark = pytest.mark.slow


def _accuracy(n: int, rho: float, n_instances: int = 100, **control) -> float:
    cfg = small_config(
        run_id=f"acc-n{n}-rho{rho}",
        task={
            "n_steps": n,
            "ambiguity": 0.3,
            "distractors": 3,
            "reconvergence": rho,
            "n_instances": {"test": n_instances},
        },
        llm={"K": 1, "difficulty_beta": [2, 5], "clean_factor": 0.2},
        control={"policy": "none", **control},
    )
    res = run_experiment(cfg, workers=4, write=False)
    return sum(t.final_correct for t in res.trajectories) / len(res.trajectories)


def test_h1_compounding_on_mock() -> None:
    acc = {n: _accuracy(n, 0.0) for n in (1, 2, 3, 5, 8)}
    ns = sorted(acc)
    for a, b in itertools.pairwise(ns):
        assert acc[b] <= acc[a] + 0.02, acc
    assert acc[1] - acc[8] >= 0.2, acc


def test_reconvergence_slows_the_decay() -> None:
    assert _accuracy(8, 0.5) >= _accuracy(8, 0.0) + 0.05


def test_global_oracle_degeneracy_vs_local_oracle() -> None:
    base = dict(
        task={
            "n_steps": 4,
            "ambiguity": 0.5,
            "distractors": 3,
            "reconvergence": 0.0,
            "n_instances": {"test": 60},
        },
        llm={"K": 1},
        control={"policy": "ask_last"},
    )
    g = run_experiment(
        small_config(run_id="deg-global", oracle={"scope": "global"}, **base),
        workers=4,
        write=False,
        allow_global_oracle=True,
    )
    assert all(t.final_correct for t in g.trajectories)
    assert all(t.n_interventions == 1 for t in g.trajectories)
    lo = run_experiment(
        small_config(run_id="deg-local", oracle={"scope": "local"}, **base), workers=4, write=False
    )
    local_error = sum(not t.final_correct for t in lo.trajectories) / len(lo.trajectories)
    assert local_error >= 0.1, local_error
