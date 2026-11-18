"""scripts/randomized_gate.py: a check-or-proceed rule read from the two recorded branches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "randomized_gate.py"
    spec = importlib.util.spec_from_file_location("randomized_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree(mod: ModuleType, unanimous: list[list[bool]], wrong: list[list[list[bool]]]):
    ids = tuple(f"q{n}" for n in range(len(unanimous)))
    return mod.Tree(ids, np.array(unanimous, dtype=bool), np.array(wrong, dtype=bool))


def test_the_coin_decides_only_the_unanimous_steps(mod: ModuleType) -> None:
    # question 0: both hops unanimous; question 1: hop 1 disagrees; question 2: hop 2 disagrees
    # only after an unchecked hop 1
    t = tree(
        mod,
        [[True, True, True], [False, True, True], [True, False, True]],
        [
            [[True, False], [True, False]],  # wrong unless hop 2 is checked
            [[True, True], [False, False]],  # wrong unless hop 1 is checked
            [[True, False], [False, False]],
        ],
    )
    never = np.ones((3, 2)) * 0.5  # draws below q = 1: the coin never asks
    always = np.ones((3, 2)) * 0.5  # at q = 0 every draw is at least q
    errors, checks = mod.decide(t, 1.0, never, ties=False)
    assert errors.tolist() == [True, True, True] and checks.tolist() == [0, 0, 0]
    errors, checks = mod.decide(t, 0.0, always, ties=False)
    assert errors.tolist() == [False, False, False] and checks.tolist() == [2, 2, 2]
    # with ties the disagreements are checked whatever the coin says
    errors, checks = mod.decide(t, 1.0, never, ties=True)
    assert checks.tolist() == [0, 1, 1]  # hop 1 of question 1, hop 2 of question 2
    assert errors.tolist() == [True, False, False]
    # the second hop is the one that follows what was done at the first: after a checked hop 1
    # question 2's second batch is unanimous, so it is not checked again
    draws = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    errors, checks = mod.decide(t, 0.5, draws, ties=True)
    assert checks.tolist() == [1, 1, 1] and errors.tolist() == [True, False, False]


def test_coins_are_those_of_random_asking_and_change_with_the_seed(mod: ModuleType) -> None:
    import random

    from ucagent.core.utils import stable_hash

    ids = ("musique-test-00001-s0", "musique-test-00002-s0")
    draws = mod.coins(ids, 0)
    assert draws.shape == (2, 2) and ((draws >= 0) & (draws < 1)).all()
    expected = random.Random(stable_hash(0, "random_ask", ids[1], 2, 0)).random()
    assert draws[1, 1] == expected
    assert not np.array_equal(draws, mod.coins(ids, 1))
    assert np.array_equal(draws, mod.coins(ids, 0))


def test_the_knob_is_selected_along_the_grid_of_random_asking(mod: ModuleType) -> None:
    n = 200
    grid = [0.0, 0.5, 1.0]
    # every question is unanimous at both hops and right whatever is done: nothing to check
    easy = tree(mod, [[True, True, True]] * n, [[[False, False], [False, False]]] * n)
    assert mod.select(easy, grid, seed=0, alpha=0.1, delta=0.1) == 1.0
    # every question is unanimous and wrong unless its second hop is checked: only q = 0 passes
    hard = tree(mod, [[True, True, True]] * n, [[[True, False], [True, False]]] * n)
    assert mod.select(hard, grid, seed=0, alpha=0.1, delta=0.1) == 0.0
    # wrong whatever is done: not even checking every step passes
    lost = tree(mod, [[True, True, True]] * n, [[[True, True], [True, True]]] * n)
    assert mod.select(lost, grid, seed=0, alpha=0.1, delta=0.1) is None
