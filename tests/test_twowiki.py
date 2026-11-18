"""2WikiMultiHopQA adapter on a five-record fixture (no large files needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ucagent.api import fit_calibrators, run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import HEADER_RE, MockChainLLM
from ucagent.core.records import Action
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.runner import run_experiment
from ucagent.tasks import make_task
from ucagent.tasks.base import make_human
from ucagent.tasks.musique import Hop, MusiqueInstance, MusiqueTask, Paragraph
from ucagent.tasks.twowiki import (
    TwoWikiTask,
    chain_rows,
    question_flags,
    render_relation,
    same_date,
    unchecked,
    uses_fallback,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "twowiki_mini.jsonl"
REAL = Path("data/twowiki")


def _raw() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    with open(tmp_path / "mini.jsonl", "w", encoding="utf-8") as fh:
        for row in chain_rows(_raw(), exclude_relations=()):  # keep the date question
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tmp_path


def _task(data_dir: Path, **kw) -> TwoWikiTask:
    base = {
        "data_dir": data_dir,
        "train_file": "mini.jsonl",
        "dev_file": "mini.jsonl",
        "n_instances": {"train": 3, "calib": 1, "test": 1},
        "n_distractors": 2,
        "local_match": "contains",
        "final_match": "contains",
    }
    return TwoWikiTask(**{**base, **kw})


def _by_question(task: TwoWikiTask, fragment: str):
    for inst in task.instances("train"):
        if fragment in inst.question:
            return inst
    raise AssertionError(fragment)


def test_chain_rows_keep_chains_only_in_musique_format() -> None:
    assert [r["type"] for r in chain_rows(_raw())] == ["compositional", "inference"]  # "has part"
    assert chain_rows(
        _raw(), exclude_relations=(), exclude_ids=["8813f87c0bdd11eba7f7acde48001122"]
    )
    rows = chain_rows(_raw(), exclude_relations=())
    assert [r["type"] for r in rows] == ["compositional", "compositional", "inference"]
    date_row = rows[0]
    hop1, hop2 = date_row["question_decomposition"]
    assert hop1["question"] == "List of The Dillinger Escape Plan band members >> has part"
    assert hop1["answer"] == "Ben Weinman"
    assert hop2["question"] == "#1 >> date of birth" and hop2["answer"] == "August 8, 1975"
    by_idx = {p["idx"]: p for p in date_row["paragraphs"]}
    assert by_idx[hop1["paragraph_support_idx"]]["is_supporting"]
    assert by_idx[hop2["paragraph_support_idx"]]["title"] == "Ben Weinman"
    assert sum(p["is_supporting"] for p in date_row["paragraphs"]) == 2
    assert date_row["answer"] == hop2["answer"] and date_row["answer_aliases"] == []


def test_ids_splits_registry_and_musique_ids_unchanged(data_dir: Path) -> None:
    task = _task(data_dir)
    ids = [i.instance_id for i in task.instances("train")]
    assert ids == [f"twowiki-train-{k:05d}-s0" for k in range(3)]
    assert task.get_instance(ids[1]).instance_id == ids[1]
    assert all(task.get_instance(i).n_steps == 2 for i in ids)
    with pytest.raises(KeyError):
        task.get_instance("musique-train-00000-s0")
    built = make_task(
        {
            "name": "twowiki",
            "data_dir": str(data_dir),
            "train_file": "mini.jsonl",
            "dev_file": "mini.jsonl",
            "n_instances": {"train": 3, "calib": 1, "test": 1},
        }
    )
    assert isinstance(built, TwoWikiTask)
    assert MusiqueTask.id_prefix == "musique" and TwoWikiTask.id_prefix == "twowiki"


def test_rendering_substitution_and_prompt(data_dir: Path) -> None:
    task = _task(data_dir)
    inst = _by_question(task, "Dillinger")
    assert task.sub_question(inst, 1, {}) == (
        "What is a part of List of The Dillinger Escape Plan band members?"
    )
    assert task.sub_question(inst, 2, {1: "Ben Weinman"}) == "When was Ben Weinman born?"
    assert render_relation("X >> place of burial") == "Where was X buried?"
    assert render_relation("X >> director") == "Who directed X?"  # MuSiQue's template
    assert uses_fallback("unheard-of relation") and not uses_fallback("date of death")
    from ucagent.core.records import StepRecord

    hist = [StepRecord(step=1, attempt=0, context="", answer="Ben Weinman", u=0, entropy=0, eps=0)]
    messages = task.step_prompt(inst, 2, hist, attempt=0)
    text = messages[1].content
    assert HEADER_RE.match(text.splitlines()[0])
    assert "Sub-question for step 2 of 2 (answer this one): When was Ben Weinman born?" in text
    assert "[Ben Weinman]" in text  # the gold paragraph of hop 2 is shown
    assert task.step_prompt(inst, 2, hist, attempt=0) == messages  # byte-identical on reuse


def test_date_aware_labels_and_on_track_logic(data_dir: Path) -> None:
    task = _task(data_dir)
    inst = _by_question(task, "Dillinger")
    assert task.is_locally_correct(inst, 1, "", "Ben Weinman")
    assert task.is_locally_correct(inst, 2, "Ben Weinman", "8 August 1975")
    assert task.is_locally_correct(inst, 2, "Ben Weinman", "1975-08-08")
    assert not task.is_locally_correct(inst, 2, "Ben Weinman", "9 August 1975")
    assert not task.has_local_truth(inst, 2, "Greg Puciato")
    assert task.is_final_correct(inst, "August 8, 1975") and task.is_final_correct(
        inst, "8 Aug 1975"
    )
    assert not task.is_final_correct(inst, "1976")
    polish = _by_question(task, "Polish-Russian War")
    assert task.has_local_truth(polish, 2, "Xawery Zuławski")  # combining accents are dropped
    assert not task.has_local_truth(polish, 2, "Xawery Zulawski")  # "ł" is a letter, not "l"
    assert task.is_locally_correct(polish, 2, "Xawery Żuławski", "Małgorzata Braunek")
    assert same_date("June 12, 1516", "12 june 1516") and not same_date("Paris", "1983")


def test_mock_runs_end_to_end_and_asks(data_dir: Path) -> None:
    task = _task(data_dir)
    llm = MockChainLLM(task=task, seed=0, difficulty_beta=(3.0, 2.0))
    settings = RunSettings(policy="per_step_u", knob=0.0, K=3, K_prime=6)
    ids = [i.instance_id for i in task.instances("train")]
    trajectories = run_instances(task, llm, settings, ids, make_human(task, settings))
    assert len(trajectories) == 3 and all(len(t.steps) == 2 for t in trajectories)
    assert any(iv.action == Action.ASK_HUMAN for t in trajectories for iv in t.interventions)


def test_config_path_runs_and_fits(data_dir: Path, tmp_path: Path) -> None:
    block = {
        "name": "twowiki",
        "data_dir": str(data_dir),
        "train_file": "mini.jsonl",
        "dev_file": "mini.jsonl",
        "n_steps": 2,
        "n_instances": {"train": 3, "calib": 1, "test": 1},
        "n_distractors": 2,
        "local_match": "contains",
        "final_match": "contains",
    }
    cfg = ExperimentConfig.model_validate(
        {
            "run_id": "twowiki_test",
            "split": "train",
            "task": block,
            "llm": {"backend": "mock", "K": 2, "K_prime": 4, "difficulty_beta": [4.0, 1.0]},
            "uq": {"estimator": "semantic_entropy", "equivalence": "normalized_contains"},
            "control": {"policy": "none"},
        }
    )
    result = run_experiment(cfg, workers=1, write=True, results_dir=tmp_path / "runs")
    assert len(result.trajectories) == 3
    art = fit_calibrators(cfg, workers=1)
    assert art.conformal_scores["K"]


@pytest.mark.slow
def test_real_files_when_present() -> None:
    if not (REAL / "twowiki_dev.jsonl").exists():
        pytest.skip("run scripts/prepare_twowiki.py first")
    task = TwoWikiTask(data_dir=REAL, n_instances={"train": 600, "calib": 400, "test": 400})
    assert len(list(task.instances("train"))) == 600
    inst = next(iter(task.instances("test")))
    assert inst.n_steps == 2 and inst.instance_id.startswith("twowiki-test-")


def test_dash_artefacts_of_the_smoke_run_are_fixed_and_musique_is_unchanged(
    data_dir: Path,
) -> None:
    task = _task(data_dir)
    assert task.canonical_dashes and not MusiqueTask().canonical_dashes
    nb = f"Princess Marie Melita of Hohenlohe{chr(0x2011)}Langenburg"  # as gpt-oss wrote it
    assert task.parse_answer(f"Answer: {nb}\nConfidence: 1.0") == nb.replace(chr(0x2011), "-")
    assert MusiqueTask().parse_answer(f"Answer: {nb}") == nb  # recorded MuSiQue runs replay as run
    gold = {"princess marie melita of hohenlohelangenburg"}
    assert task._matches(nb, gold) and not MusiqueTask(local_match="contains")._matches(nb, gold)
    merged = "Frederick Michael, Count Palatine of Zweibrücken-Birkenfeld"
    spaced_gold = {"frederick michael count palatine of zweibrucken"}
    assert task._matches(merged, spaced_gold)
    assert not MusiqueTask(local_match="contains")._matches(merged, spaced_gold)


def test_word_subset_rule_and_tokens_judge(data_dir: Path) -> None:
    from ucagent.core.uq.equivalence import make_judge

    task = _task(data_dir)
    gold = {"albert i of germany"}
    assert task._matches("Albert I, King of Germany", gold)
    assert not task._matches("Albert II of Germany", gold)
    assert not _task(data_dir, local_match="exact")._matches("Albert I, King of Germany", gold)
    judge = make_judge("normalized_tokens")
    assert judge.same("Albert I, King of Germany", "Albert I of Germany", None)
    assert judge.same("Hohenlohe-Langenburg", "Hohenlohe Langenburg", None)
    assert not judge.same("William I", "William II", None)
    assert not make_judge("normalized_contains").same(
        "Albert I, King of Germany", "Albert I of Germany", None
    )


def test_windows_file_lists_absolute_indices(data_dir: Path, tmp_path: Path) -> None:
    windows = tmp_path / "windows.yaml"
    windows.write_text("test: [1, 0]\n", encoding="utf-8")
    sizes = {"train": 3, "calib": 1, "test": 2}
    plain = _task(data_dir, n_instances=sizes)
    task = _task(data_dir, n_instances=sizes, windows_file=str(windows))
    ids = [i.instance_id for i in task.instances("test")]
    assert ids == ["twowiki-test-00001-s0", "twowiki-test-00000-s0"]  # listed order, absolute
    assert task.sizes["test"] == 2
    assert task.get_instance(ids[0]).qid == plain.get_instance(ids[0]).qid
    assert [i.instance_id for i in task.instances("calib")] == [
        i.instance_id for i in plain.instances("calib")
    ]  # an unlisted split keeps its offset window
    built = make_task(
        {
            "name": "twowiki",
            "data_dir": str(data_dir),
            "train_file": "mini.jsonl",
            "dev_file": "mini.jsonl",
            "n_instances": sizes,
            "windows_file": str(windows),
        }
    )
    assert [i.instance_id for i in built.instances("test")] == ids
    with pytest.raises(ValueError, match="offset"):
        _task(data_dir, n_instances=sizes, windows_file=str(windows), offset={"test": 1})
    with pytest.raises(ValueError, match="n_instances"):
        _task(data_dir, windows_file=str(windows))  # one test question configured, two listed
    windows.write_text("test: [0, 5]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index 5"):
        list(_task(data_dir, n_instances=sizes, windows_file=str(windows)).instances("test"))
    for bad in ("test: [0, 0]\n", "test: [-1]\n", "valid: [0]\n"):
        windows.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            _task(data_dir, n_instances={"train": 3, "calib": 1}, windows_file=str(windows))


def test_question_flags_and_unchecked_golds() -> None:
    assert unchecked("place of birth", "St. Louis", "He was born in St. Louis, Missouri.")
    assert not unchecked("place of birth", "Paris", "He was born in Paris.")
    assert not unchecked("director", "John C. Smith", "It was directed by John C. Smith.")
    inst = MusiqueInstance(
        instance_id="twowiki-test-00000-s0",
        split="test",
        index=0,
        question="Where was the father of A born?",
        n_steps=2,
        qid="x",
        hops=[
            Hop(
                template="A >> father",
                answer="John C. Breckinridge",
                support_idx=0,
                distractor_idx=[],
            ),
            Hop(template="#1 >> place of birth", answer="Dutch", support_idx=1, distractor_idx=[]),
        ],
        paragraphs=[
            Paragraph(
                idx=0, title="A", text="A was the son of John C. Breckinridge.", is_supporting=True
            ),
            Paragraph(
                idx=1, title="B", text="Breckinridge was a Dutch politician.", is_supporting=True
            ),
        ],
        answer="Dutch",
        answer_aliases=[],
    )
    assert question_flags(inst) == ["hop1:unchecked", "hop2:demonym", "hop2:keyword_missing"]


def test_split3_windows_reproduce_and_pass_the_filter() -> None:
    windows = Path("configs/twowiki_split3_windows.yaml")
    if not (REAL / "twowiki_dev.jsonl").exists() or not windows.exists():
        pytest.skip("run scripts/prepare_twowiki.py first")
    import yaml

    listed = yaml.safe_load(windows.read_text(encoding="utf-8"))
    task = TwoWikiTask(
        data_dir=REAL, n_instances={"train": 600, "calib": 200, "test": 200}, windows_file=windows
    )
    used = {f"twowiki-{s}-{k:05d}-s0" for s in ("calib", "test") for k in range(400)}
    used |= {f"twowiki-test-{k:05d}-s0" for k in range(900, 932)}  # the third smoke run
    for split in ("calib", "test"):
        insts = list(task.instances(split))
        assert [i.index for i in insts] == listed[split] and len(insts) == 200
        assert not any(i.instance_id in used or question_flags(i) for i in insts)
        # the first 200 usable rows: every unused, unlisted row below the last one is flagged
        skipped = set(range(max(listed[split]))) - set(listed[split])
        for k in skipped:
            iid = task.instance_id(split, k)
            assert iid in used or question_flags(task.get_instance(iid))
