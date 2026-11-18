import random

import pytest

from ucagent.core.llm.mock import parse_header
from ucagent.core.uq.normalize import normalize_answer
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig


def make(**kw):
    base = dict(
        n_steps=4,
        ambiguity=0.5,
        distractors=3,
        reconvergence=0.0,
        n_instances={"train": 30, "calib": 30, "test": 30},
        seed=0,
    )
    base.update(kw)
    return SyntheticChainTask(SyntheticConfig(**base))


def test_deterministic_and_split_disjoint() -> None:
    a = make().get_instance("test-00003-s0")
    b = make().get_instance("test-00003-s0")
    assert a == b
    c = make().get_instance("train-00003-s0")
    assert a.entities != c.entities
    assert len(list(make().instances("calib"))) == 30


def test_names_are_confusable_but_distinct() -> None:
    inst = make().get_instance("test-00000-s0")
    for i in range(1, inst.n_steps + 1):
        truth = normalize_answer(inst.entities[i])
        for d in inst.distractors[i - 1]:
            assert normalize_answer(d) != truth
    assert len({normalize_answer(x) for x in inst.pool}) == len(inst.pool)


def test_kb_is_total() -> None:
    task = make()
    inst = task.get_instance("test-00001-s0")
    for i in range(1, inst.n_steps + 1):
        starts = [inst.entities[i - 1]] + (inst.distractors[i - 2] if i >= 2 else [])
        for x in starts:
            assert task.local_truth(inst, i, x)
        assert task.local_truth(inst, i, "Nobody Known")  # fallback for unknown entities
        assert task.local_truth(inst, i, "Nobody Known") == task.local_truth(
            inst, i, "Nobody Known"
        )


@pytest.mark.parametrize("rho,expect", [(0.0, False), (1.0, True)])
def test_reconvergence_extremes(rho: float, expect: bool) -> None:
    task = make(reconvergence=rho)
    for inst in task.instances("test"):
        for i in range(2, inst.n_steps + 1):
            for x in inst.distractors[i - 2]:
                assert (task.local_truth(inst, i, x) == inst.entities[i]) is expect


def test_reconvergence_rate_is_controlled() -> None:
    task = make(reconvergence=0.5, n_instances={"test": 200})
    hits = total = 0
    for inst in task.instances("test"):
        for i in range(2, inst.n_steps + 1):
            for x in inst.distractors[i - 2]:
                total += 1
                hits += task.local_truth(inst, i, x) == inst.entities[i]
    assert abs(hits / total - 0.5) < 0.05


def test_distractors_iff_ambiguous_and_exclude_truth() -> None:
    task = make()
    seen_amb = seen_clean = False
    for inst in task.instances("test"):
        for i in range(1, inst.n_steps + 1):
            current = inst.entities[i - 1]
            names = task.distractor_names(inst, i, current)
            if inst.ambiguity[i - 1]:
                seen_amb = True
                assert names and inst.entities[i] not in names
            else:
                seen_clean = True
                assert names == []
    assert seen_amb and seen_clean


def test_local_and_global_truth_diverge_off_path() -> None:
    task = make()
    inst = task.get_instance("test-00002-s0")
    wrong = inst.distractors[0][0]
    assert task.local_truth(inst, 2, wrong) != task.global_truth(inst, 2)
    assert task.local_truth(inst, 2, inst.entities[1]) == task.global_truth(inst, 2)
    assert task.is_locally_correct(inst, 2, wrong, task.local_truth(inst, 2, wrong))
    assert not task.is_final_correct(inst, wrong)
    assert task.is_final_correct(inst, inst.entities[-1].upper())


def test_prompt_header_and_parsing() -> None:
    task = make()
    inst = task.get_instance("test-00000-s0")
    msgs = task.step_prompt(inst, 1, [], attempt=2)
    text = str(msgs[-1].content)
    header = parse_header(text)
    assert header == {
        "instance": inst.instance_id,
        "step": "1",
        "current": inst.entities[0],
        "attempt": "2",
    }
    assert "Passage:" in text and inst.entities[0] in text
    assert task.parse_answer("Answer: Kalo Mire.\nConfidence: 0.9") == "Kalo Mire"
    assert task.parse_answer("  Kalo Mire \n") == "Kalo Mire"
    assert task.random_entity(inst, random.Random(1)) in inst.pool


def test_per_hop_and_dict_forms() -> None:
    cfg = SyntheticConfig(
        n_steps=3, ambiguity={"per_hop": [0.1, 0.2, 0.3]}, reconvergence={"global": 0.4}
    )
    assert cfg.q == [0.1, 0.2, 0.3] and cfg.rho == [0.4, 0.4, 0.4]
    with pytest.raises(ValueError):
        _ = SyntheticConfig(n_steps=3, ambiguity=[0.1, 0.2]).q
