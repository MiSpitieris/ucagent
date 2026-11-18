from concurrent.futures import ThreadPoolExecutor

from ucagent.core.graph.nodes import parse_confidence
from ucagent.core.llm.mock import MockChainLLM
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig


def _setup(**kw):
    task = SyntheticChainTask(SyntheticConfig(n_steps=3, ambiguity=1.0, n_instances={"test": 50}))
    llm = MockChainLLM(task=task, seed=0, **kw)
    return task, llm


def _ambiguous_step(task, llm, lo=0.25, hi=0.75):
    for inst in task.instances("test"):
        for i in range(1, 4):
            current = inst.entities[i - 1]
            p = llm.p_wrong(inst.instance_id, i, current)
            if lo < p < hi:
                return inst, i, current, p
    raise AssertionError("no step with moderate difficulty found")


def test_deterministic_and_thread_safe() -> None:
    task, llm = _setup()
    inst = task.get_instance("test-00000-s0")
    msgs = task.step_prompt(inst, 1, [], 0)
    seq = [llm.invoke(msgs, sample_index=k).content for k in range(32)]
    assert seq == [llm.invoke(msgs, sample_index=k).content for k in range(32)]
    with ThreadPoolExecutor(16) as pool:
        par = list(pool.map(lambda k: llm.invoke(msgs, sample_index=k).content, range(32)))
    assert par == seq


def test_answers_are_truth_or_distractors() -> None:
    task, llm = _setup()
    inst, i, current, _ = _ambiguous_step(task, llm)
    msgs = task.step_prompt(inst, i, [], 0) if i == 1 else None
    # build a prompt with the wanted current entity by faking history
    from ucagent.core.records import StepRecord

    history = [
        StepRecord(step=j, attempt=0, context="", answer=inst.entities[j], u=0, entropy=0, eps=0)
        for j in range(1, i)
    ]
    msgs = task.step_prompt(inst, i, history, 0)
    truth = task.local_truth(inst, i, current)
    allowed = {truth, *task.distractor_names(inst, i, current)}
    answers = {task.parse_answer(str(llm.invoke(msgs, sample_index=k).content)) for k in range(40)}
    assert answers <= allowed and truth in answers and len(answers) > 1


def test_wrong_rate_tracks_difficulty() -> None:
    task, llm = _setup()
    inst, i, current, p = _ambiguous_step(task, llm)
    from ucagent.core.records import StepRecord

    history = [
        StepRecord(step=j, attempt=0, context="", answer=inst.entities[j], u=0, entropy=0, eps=0)
        for j in range(1, i)
    ]
    msgs = task.step_prompt(inst, i, history, 0)
    truth = task.local_truth(inst, i, current)
    wrong = sum(
        task.parse_answer(str(llm.invoke(msgs, sample_index=k).content)) != truth
        for k in range(400)
    )
    assert abs(wrong / 400 - p) < 0.08


def test_attempt_changes_draws_and_clean_steps_are_easier() -> None:
    task, llm = _setup()
    inst, i, _current, _ = _ambiguous_step(task, llm)
    from ucagent.core.records import StepRecord

    history = [
        StepRecord(step=j, attempt=0, context="", answer=inst.entities[j], u=0, entropy=0, eps=0)
        for j in range(1, i)
    ]
    a0 = [
        llm.invoke(task.step_prompt(inst, i, history, 0), sample_index=k).content for k in range(30)
    ]
    a1 = [
        llm.invoke(task.step_prompt(inst, i, history, 1), sample_index=k).content for k in range(30)
    ]
    assert a0 != a1
    clean_task = SyntheticChainTask(
        SyntheticConfig(n_steps=3, ambiguity=0.0, n_instances={"test": 5})
    )
    clean_llm = MockChainLLM(task=clean_task, seed=0, clean_factor=0.2)
    cinst = clean_task.get_instance("test-00000-s0")
    assert (
        clean_llm.p_wrong(cinst.instance_id, 1, cinst.entities[0])
        == clean_llm.difficulty(cinst.instance_id, 1) * 0.2
    )


def test_verbalized_confidence_line() -> None:
    task, llm = _setup()
    inst = task.get_instance("test-00000-s0")
    text = str(llm.invoke(task.step_prompt(inst, 1, [], 0), sample_index=0).content)
    conf = parse_confidence(text)
    assert conf is not None and 0.0 <= conf <= 1.0
    quiet = MockChainLLM(task=task, seed=0, verbalized=False)
    assert parse_confidence(str(quiet.invoke(task.step_prompt(inst, 1, [], 0)).content)) is None
