"""MuSiQue adapter on a three-question fixture (no large files needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ucagent.api import fit_calibrators, observations_from, run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import HEADER_RE, MockChainLLM
from ucagent.core.records import Action, StepRecord, Trajectory
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import run_experiment
from ucagent.tasks import make_task
from ucagent.tasks.base import LocalOracle, make_human
from ucagent.tasks.musique import (
    RELATION_QUESTIONS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NL,
    MusiqueInstance,
    MusiqueTask,
    is_linear,
    render_subquestion,
    sanitize_context,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL = Path("data/musique")


def _task(**kw) -> MusiqueTask:
    base = {
        "data_dir": FIXTURES,
        "train_file": "musique_mini.jsonl",
        "dev_file": "musique_mini.jsonl",
        "n_instances": {"train": 3, "calib": 1, "test": 1},
        "n_distractors": 2,
    }
    return MusiqueTask(**{**base, **kw})


def _task_block(**kw) -> dict:
    return {
        "name": "musique",
        "data_dir": str(FIXTURES),
        "train_file": "musique_mini.jsonl",
        "dev_file": "musique_mini.jsonl",
        "n_steps": 2,
        "n_instances": {"train": 3, "calib": 1, "test": 1},
        "n_distractors": 2,
        **kw,
    }


def _rec(step: int, context: str, answer: str) -> StepRecord:
    return StepRecord(
        step=step, attempt=0, context=context, answer=answer, u=0.0, entropy=0.0, eps=0.0
    )


def test_instances_ids_splits_and_fields() -> None:
    task = _task()
    train = list(task.instances("train"))
    assert [i.instance_id for i in train] == [f"musique-train-0000{j}-s0" for j in range(3)]
    assert {i.qid for i in train} == {
        "2hop__460946_294723",
        "2hop__481349_302087",
        "2hop__765180_123956",
    }
    calib, test = list(task.instances("calib")), list(task.instances("test"))
    assert calib[0].qid != test[0].qid  # disjoint halves of the dev file
    inst = task.get_instance("musique-train-00000-s0")
    assert isinstance(inst, MusiqueInstance) and inst.n_steps == 2 and task.n_steps(inst) == 2
    assert task.get_instance(inst.instance_id) is inst  # cached
    for hop in inst.hops:
        assert hop.support_idx not in hop.distractor_idx and len(hop.distractor_idx) == 2
        assert all(not inst.paragraphs[j].is_supporting for j in hop.distractor_idx)
    with pytest.raises(KeyError):
        task.get_instance("musique-train-00000-s9")
    with pytest.raises(ValueError, match="requested"):
        list(_task(n_instances={"train": 99}).instances("train"))


def test_registry_builds_the_task_from_a_config_block() -> None:
    task = make_task(_task_block())
    assert isinstance(task, MusiqueTask) and task.hops == (2,)
    assert make_task(_task_block(hops=[2, 3])).hops == (2, 3)


def test_prompt_substitutes_references_and_shows_passages() -> None:
    task = _task()
    inst = next(i for i in task.instances("train") if i.qid == "2hop__460946_294723")
    first = task.step_prompt(inst, 1, [], 0)[1].content
    assert isinstance(first, str) and HEADER_RE.match(first.splitlines()[0])
    assert (
        "Overall question (context only, do not answer it): "
        "Who is the spouse of the Green performer?" in first
    )
    assert "Sub-question for step 1 of 2 (answer this one): Who performed Green?" in first
    assert ">>" not in first and "\nQuestion:" not in first
    assert first.count("\n[") == 3  # supporting paragraph + 2 distractors
    support = next(p for p in inst.paragraphs if p.idx == inst.hops[0].support_idx)
    assert support.text[:40] in first
    second = task.step_prompt(inst, 2, [_rec(1, "", "Steve Hillage")], 1)[1].content
    assert isinstance(second, str)
    assert 'current="Steve Hillage" attempt="1"' in second.splitlines()[0]
    assert (
        "Sub-question for step 2 of 2 (answer this one): Who is the spouse of Steve Hillage?"
        in second
    )
    assert ">>" not in second
    assert "Answers so far: #1: Steve Hillage" in second
    assert task.sub_question(inst, 2, {}) == "Who is the spouse of #1?"  # unresolved ref kept


def test_header_sanitising_keeps_truth_round_trip() -> None:
    task = _task()
    inst = task.get_instance("musique-train-00000-s0")
    prompt = task.step_prompt(inst, 2, [_rec(1, "", 'Steve "the man" Hillage')], 0)[1].content
    assert isinstance(prompt, str)
    header = HEADER_RE.match(prompt.splitlines()[0])
    assert header is not None and '"' not in header.group("current")
    assert sanitize_context('a "b"\nc') == "a 'b' c"
    assert task.has_local_truth(inst, 2, header.group("current")) is False  # it is wrong anyway
    assert task.has_local_truth(inst, 2, sanitize_context("Steve Hillage")) is True


def test_local_and_global_truth_on_and_off_track() -> None:
    task = _task()
    inst = next(i for i in task.instances("train") if i.qid == "2hop__460946_294723")
    assert (
        task.global_truth(inst, 1) == "Steve Hillage"
        and task.global_truth(inst, 2) == "Miquette Giraudy"
    )
    assert task.has_local_truth(inst, 1, "") and task.local_truth(inst, 1, "") == "Steve Hillage"
    assert task.has_local_truth(inst, 2, "steve hillage")  # normalised comparison
    assert not task.has_local_truth(inst, 2, "Grant Green")
    assert task.is_locally_correct(inst, 2, "Steve Hillage", "Miquette Giraudy")
    assert not task.is_locally_correct(inst, 2, "Grant Green", "Miquette Giraudy")  # off track
    assert task.is_final_correct(inst, "miquette giraudy") and not task.is_final_correct(inst, None)
    bombardier = next(i for i in task.instances("train") if i.qid == "2hop__481349_302087")
    assert task.is_final_correct(bombardier, "Bombardier")  # alias
    assert task.is_locally_correct(bombardier, 2, "Bombardier Aerospace", "Bombardier")
    assert not task.is_locally_correct(
        bombardier, 1, "", "Bombardier"
    )  # alias only on the final hop
    assert task.parse_answer("Answer: Palembang.\nConfidence: 0.8") == "Palembang"
    assert (
        task.parse_answer("Answer: Western and Majorcan >> Confidence: 0.8")
        == "Western and Majorcan"
    )
    assert task.parse_answer("Answer: NSA >> ") == "NSA"
    assert task.parse_answer("**Answer:** Paris\n**Confidence:** 0.9") == "Paris"
    assert task.parse_answer("Answer: **Paris**") == "Paris"
    assert (
        task.parse_answer("Answer: New\u202fYork\u202fYankees\nConfidence: 1") == "New York Yankees"
    )
    assert task.parse_answer("Answer: 'Bombardier Inc.' Confidence=0.9") == "Bombardier Inc"
    assert task.parse_answer("\n Palembang \n") == "Palembang" and task.parse_answer("") == ""
    assert set(task.distractor_names(inst, 1, "")) and "Steve Hillage" not in task.distractor_names(
        inst, 1, ""
    )
    import random

    assert task.random_entity(inst, random.Random(0)) in {p.title for p in inst.paragraphs}


def test_local_oracle_confirms_when_off_track() -> None:
    task = _task()
    inst = task.get_instance("musique-train-00000-s0")
    oracle = LocalOracle(task)
    on = {
        "instance_id": inst.instance_id,
        "step": 2,
        "attempt": 0,
        "current": "Steve Hillage",
        "committed": "x",
    }
    off = {**on, "current": "Grant Green", "committed": "whatever the model said"}
    assert oracle(on) == "Miquette Giraudy"
    assert oracle(off) == "whatever the model said"


def test_mock_runs_end_to_end_and_asks() -> None:
    task = _task()
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(3.0, 2.0))
    settings = RunSettings(policy="per_step_u", knob=0.0, K=3, K_prime=6)
    ids = [i.instance_id for i in task.instances("train")]
    trajectories = run_instances(task, llm, settings, ids, make_human(task, settings))
    assert len(trajectories) == 3 and all(len(t.steps) == 2 for t in trajectories)
    assert all(isinstance(t.steps[0].locally_correct, bool) for t in trajectories)
    assert any(iv.action == Action.ASK_HUMAN for t in trajectories for iv in t.interventions)


def test_config_path_observations_skip_off_track_and_fit_scores(tmp_path: Path) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "run_id": "musique_test",
            "split": "train",
            "task": _task_block(),
            "llm": {"backend": "mock", "K": 2, "K_prime": 4, "difficulty_beta": [4.0, 1.0]},
            "uq": {"estimator": "semantic_entropy", "equivalence": "normalized_contains"},
            "control": {"policy": "none"},
        }
    )
    result = run_experiment(cfg, workers=1, write=True, results_dir=tmp_path)
    assert len(result.trajectories) == 3
    fit_run = run_experiment(cfg.with_updates(llm={"K": cfg.llm.K_prime}), workers=1, write=False)
    per_traj = observations_from(cfg, fit_run.trajectories)
    for t, rows in zip(fit_run.trajectories, per_traj, strict=True):
        expected = 2 if t.steps[0].globally_correct else 1  # off-track hop 2 has no label
        assert len(rows) == expected
    art = fit_calibrators(cfg, workers=1)
    assert art.conformal_scores["K"] and all(0.0 <= s <= 1.0 for s in art.conformal_scores["K"])
    row = Trajectory.from_row(result.trajectories[0].to_row())
    assert row.n_steps == 2


@pytest.mark.slow
def test_real_files_load_two_hop_questions() -> None:
    if not (REAL / "musique_ans_v1.0_dev.jsonl").exists():
        pytest.skip("data/musique is not downloaded")
    task = MusiqueTask(data_dir=REAL, n_steps=2, n_instances={"train": 5, "calib": 5, "test": 5})
    for split in ("train", "calib", "test"):
        got = list(task.instances(split))
        assert len(got) == 5 and all(i.n_steps == 2 for i in got)
        assert all(i.qid.startswith("2hop__") for i in got)
    assert {i.qid for i in task.instances("calib")}.isdisjoint(
        {i.qid for i in task.instances("test")}
    )
    # every arrow-form relation of the real 2-hop data has a hand-written question
    import json
    import re

    arrow = re.compile(r"^(.*?)\s*>>\s*(.*)$")
    relations: set[str] = set()
    for name in ("musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl"):
        with open(REAL / name, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if not row["id"].startswith("2hop"):
                    continue
                for hop in row["question_decomposition"]:
                    m = arrow.match(hop["question"])
                    if m:
                        relations.add(m.group(2).strip())
    assert relations and relations <= set(RELATION_QUESTIONS)  # no fallback on real data
    three = MusiqueTask(
        data_dir=REAL,
        n_steps=3,
        hops=[3],
        linear_only=True,
        n_instances={"train": 5, "calib": 5, "test": 5},
    )
    assert len(three._candidates("calib")) + len(three._candidates("test")) == 568  # dev 3hop1 rows
    assert all(i.n_steps == 3 and i.qid.startswith("3hop1__") for i in three.instances("test"))


def test_render_subquestion_table_fallback_and_idempotence() -> None:
    assert len(RELATION_QUESTIONS) == 92
    for relation, question in RELATION_QUESTIONS.items():
        assert question.count("{x}") == 1 and question.endswith("?"), relation
        assert ">>" not in question and "{relation}" not in question, relation
    assert render_subquestion("Green >> performer") == "Who performed Green?"
    assert (
        render_subquestion("#1 , season 4 >> original broadcaster")
        == "What is the original broadcaster of #1 , season 4?"
    )
    assert render_subquestion("The Road to #1 >> author") == "Who is the author of The Road to #1?"
    assert (
        render_subquestion("Foo >> brand new relation") == "What is the brand new relation of Foo?"
    )
    natural = "On what date did Battle of #1 end?"
    assert render_subquestion(natural) == natural
    for template in ("Green >> performer", natural, "Foo >> brand new relation"):
        once = render_subquestion(template)
        assert render_subquestion(once) == once
    assert render_subquestion("{x} >> country") == "Which country is {x} in?"  # braces never raise


def test_render_relations_flag_keeps_legacy_prompt() -> None:
    legacy = _task(render_relations=False)
    inst = next(i for i in legacy.instances("train") if i.qid == "2hop__460946_294723")
    messages = legacy.step_prompt(inst, 1, [], 0)
    assert messages[0].content == SYSTEM_PROMPT and ">> relation" in SYSTEM_PROMPT
    body = str(messages[1].content)
    assert "Question: Who is the spouse of the Green performer?" in body  # legacy layout
    assert "Step 1 of 2: Green >> performer" in body and "Overall question" not in body
    assert legacy.sub_question(inst, 2, {}) == "#1 >> spouse"
    rendered = _task()
    assert rendered.step_prompt(inst, 1, [], 0)[0].content == SYSTEM_PROMPT_NL
    assert ">>" not in SYSTEM_PROMPT_NL
    assert make_task(_task_block(render_relations=False)).render_relations is False
    assert make_task(_task_block()).render_relations is True


def test_local_match_contains_labels_hops_but_not_the_final_answer() -> None:
    exact, lenient = _task(), _task(local_match="contains")
    inst = next(i for i in exact.instances("train") if i.qid == "2hop__460946_294723")
    assert not exact.is_locally_correct(inst, 1, "", "Steve Hillage, guitarist")
    assert lenient.is_locally_correct(inst, 1, "", "Steve Hillage, guitarist")
    assert not exact.has_local_truth(inst, 2, "Steve Hillage, guitarist")
    assert lenient.has_local_truth(inst, 2, "Steve Hillage, guitarist")
    assert lenient.is_locally_correct(inst, 2, "Steve Hillage, guitarist", "Giraudy")  # partial
    assert not lenient.is_locally_correct(inst, 1, "", "") and not lenient.has_local_truth(
        inst, 2, ""
    )
    assert not lenient.is_locally_correct(inst, 1, "", "Hill")  # whole words only
    assert not lenient.is_final_correct(inst, "Miquette Giraudy, musician")  # final stays strict
    assert make_task(_task_block(local_match="contains")).local_match == "contains"
    with pytest.raises(ValueError):
        _task(local_match="fuzzy")


def test_final_match_contains_applies_the_hop_rule_to_the_final_answer() -> None:
    both = _task(local_match="contains", final_match="contains")
    inst = next(i for i in both.instances("train") if i.qid == "2hop__460946_294723")
    assert both.final_match == "contains" and _task().final_match == "exact"
    assert both.is_final_correct(inst, "Miquette Giraudy, musician")  # verbose but right
    assert both.is_final_correct(inst, "Giraudy")  # partial, either way, as the hop rule
    assert not both.is_final_correct(inst, "Gir") and not both.is_final_correct(inst, None)
    assert not both.is_final_correct(inst, "Steve Hillage")  # a wrong entity still fails
    # the final rule is independent of the per-hop rule
    final_only = _task(final_match="contains")
    assert final_only.is_final_correct(inst, "Miquette Giraudy, musician")
    assert not final_only.is_locally_correct(inst, 1, "", "Steve Hillage, guitarist")
    assert make_task(_task_block(final_match="contains")).final_match == "contains"
    with pytest.raises(ValueError):
        _task(final_match="fuzzy")


def test_off_track_hops_are_unlabelled_in_trajectories() -> None:
    task = _task()
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(3.0, 2.0))
    settings = RunSettings(policy="none", K=3, K_prime=6)
    ids = [i.instance_id for i in task.instances("train")]
    trajectories = run_instances(task, llm, settings, ids, make_human(task, settings))
    assert all(isinstance(t.steps[0].locally_correct, bool) for t in trajectories)
    off = [t for t in trajectories if t.steps[0].locally_correct is False]
    assert off, "the mock at this difficulty should get at least one first hop wrong"
    for t in off:
        assert t.steps[1].locally_correct is None  # no local truth off track: missing, not wrong
    for t in trajectories:
        if t.steps[0].locally_correct:
            assert isinstance(t.steps[1].locally_correct, bool)


def test_offset_moves_the_window_and_keeps_absolute_ids() -> None:
    task = _task(n_instances={"train": 1, "calib": 1, "test": 1}, offset={"train": 1})
    (inst,) = list(task.instances("train"))
    assert inst.instance_id == "musique-train-00001-s0"
    assert inst.qid == _task().get_instance("musique-train-00001-s0").qid
    with pytest.raises(ValueError):
        list(
            _task(n_instances={"train": 3, "calib": 1, "test": 1}, offset={"train": 1}).instances(
                "train"
            )
        )
    with pytest.raises(ValueError):
        _task(offset={"train": -1})
    assert make_task(_task_block(offset={"calib": 0})).offsets == {"calib": 0}


def test_linear_only_filter_selects_linear_three_hop_chains() -> None:
    both = _task(hops=[3], n_instances={"train": 2, "calib": 1, "test": 1})
    assert {i.qid for i in both.instances("train")} == {
        "3hop1__900001_900002_900003",
        "3hop2__900011_900012_900013",
    }
    linear = _task(hops=[3], linear_only=True, n_instances={"train": 1, "calib": 1, "test": 1})
    (inst,) = list(linear.instances("train"))
    assert inst.qid == "3hop1__900001_900002_900003" and inst.n_steps == 3
    with pytest.raises(ValueError):  # only one linear 3-hop row in the fixture
        list(
            _task(
                hops=[3], linear_only=True, n_instances={"train": 2, "calib": 1, "test": 1}
            ).instances("train")
        )
    assert linear.sub_question(inst, 3, {1: "Fennwick", 2: "Marrow"}) == (
        "What is the administrative centre of Marrow?"
    )
    assert linear.sub_question(inst, 2, {1: "Fennwick"}) == (
        "In which administrative territorial entity is Fennwick located?"
    )
    assert linear.has_local_truth(inst, 3, "Marrow") and not linear.has_local_truth(
        inst, 3, "Fennwick"
    )
    assert linear.is_locally_correct(inst, 3, "Marrow", "Harrowgate")
    assert linear.is_final_correct(inst, "harrowgate")
    lines = (FIXTURES / "musique_mini.jsonl").read_text(encoding="utf-8").splitlines()
    rows = {json.loads(line)["id"]: json.loads(line) for line in lines if line.strip()}
    assert is_linear(rows["3hop1__900001_900002_900003"]) and not is_linear(
        rows["3hop2__900011_900012_900013"]
    )
    assert all(is_linear(r) for i, r in rows.items() if i.startswith("2hop"))
    # the 2-hop fixture subsets are untouched by the flag (default off)
    assert [i.qid for i in _task().instances("train")] == [
        i.qid for i in _task(linear_only=True).instances("train")
    ]
