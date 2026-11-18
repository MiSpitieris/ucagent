import pytest

from ucagent.experiments.metrics import auroc, ece


def test_auroc_hand_values() -> None:
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5], [1, 0, 1]) == 0.5
    assert auroc([0.1, 0.9], [0, 0]) == 0.5


def test_ece_hand_values() -> None:
    assert ece([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == 0.0
    assert ece([0.25] * 4, [1, 0, 0, 0], n_bins=4) == pytest.approx(0.0)
    assert ece([0.25] * 4, [1, 1, 1, 1], n_bins=4) == pytest.approx(0.75)
    assert ece([], []) == 0.0
