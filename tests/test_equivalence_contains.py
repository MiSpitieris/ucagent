"""The containment judge used to cluster free-text answers."""

from ucagent.core.uq.equivalence import NormalizedContains, make_judge


def test_normalized_contains() -> None:
    judge = make_judge("normalized_contains")
    assert isinstance(judge, NormalizedContains)
    assert judge.same("Bombardier", "Bombardier Inc.", None)
    assert judge.same("the Bombardier Inc", "bombardier inc.", None)
    assert judge.same("Palembang", "Palembang, Indonesia", None)
    assert not judge.same("Bombardier", "Bombardment", None)  # whole words only
    assert judge.same("Green", "Grant Green", None)
    assert not judge.same("", "x", None) and judge.same("", "", None)
