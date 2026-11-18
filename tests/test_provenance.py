"""A resumed suite checks what its results were computed from (code review of 2026-11-11, C04).

The review's reproduction: run a suite whose windows file says ``test: [0]``, change the file
to ``test: [1]`` and resume. The configuration is the same, so the suite used to return the
trajectory of question 0 for a request of question 1. Each test below is one way the inputs can
change under an unchanged configuration; the suite must refuse, and say what changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ucagent.core.uq.calibration import load_artifact
from ucagent.experiments import provenance
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.runner import model_digest
from ucagent.experiments.suite import ResumeMismatch, SuiteConfig, load_manifest, run_suite
from ucagent.tasks.finqa import FinqaTask
from ucagent.tasks.twowiki import chain_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    raw = [
        json.loads(line)
        for line in (FIXTURES / "twowiki_mini.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with open(tmp_path / "mini.jsonl", "w", encoding="utf-8") as fh:
        for row in chain_rows(raw, exclude_relations=()):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (tmp_path / "windows.yaml").write_text("test: [0]\n", encoding="utf-8")
    return tmp_path


def suite(data_dir: Path) -> SuiteConfig:
    data: dict[str, Any] = {
        "kind": "suite",
        "suite_id": "guarded",
        "base": {
            "run_id": "guarded",
            "task": {
                "name": "twowiki",
                "data_dir": str(data_dir),
                "train_file": "mini.jsonl",
                "dev_file": "mini.jsonl",
                "n_steps": 2,
                "n_distractors": 2,
                "n_instances": {"train": 3, "calib": 1, "test": 1},
                "windows_file": str(data_dir / "windows.yaml"),
                "local_match": "contains",
                "final_match": "contains",
            },
            "llm": {"backend": "mock", "K": 2, "K_prime": 4},
            "uq": {"estimator": "semantic_entropy", "equivalence": "normalized_tokens"},
            "limits": {"max_concurrency": 1},
        },
        "policies": {"none": None},
        "splits": ["test"],
    }
    path = data_dir / "suite.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SuiteConfig.from_yaml(path)


def run(data_dir: Path, **kwargs: Any) -> Path:
    return run_suite(suite(data_dir), workers=1, results_dir=data_dir / "results", **kwargs)


def returned(root: Path) -> list[str]:
    path = load_manifest(root)["path"].iloc[0]
    return [t.instance_id for t in read_trajectories(path)]


def test_unchanged_inputs_resume_without_a_new_run(data_dir: Path) -> None:
    root = run(data_dir)
    assert returned(root) == ["twowiki-test-00000-s0"]
    run_dir = Path(load_manifest(root)["path"].iloc[0]).parent
    record = provenance.read(run_dir)
    assert record is not None and set(record["files"]) == {"train", "dev", "windows"}
    assert record["n"] == 1 and len(record["source"]) == 16
    parquet = run_dir / "trajectories.parquet"
    before = parquet.stat().st_mtime_ns
    assert run(data_dir) == root  # the same request resumes quietly
    assert parquet.stat().st_mtime_ns == before  # and nothing was run again


def test_resume_refuses_a_changed_windows_file(data_dir: Path) -> None:
    root = run(data_dir)
    (data_dir / "windows.yaml").write_text("test: [1]\n", encoding="utf-8")
    with pytest.raises(ResumeMismatch, match="other inputs") as caught:
        run(data_dir)
    assert "the file 'windows'" in str(caught.value)  # the fit read it too: refused first
    assert returned(run(data_dir, resume=False)) == ["twowiki-test-00001-s0"]
    assert root.exists()


def test_resume_refuses_a_changed_data_file(data_dir: Path) -> None:
    run(data_dir)
    text = (data_dir / "mini.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    for row in rows:  # the same questions, another passage text
        row["paragraphs"][0]["paragraph_text"] += " (revised)"
    (data_dir / "mini.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ResumeMismatch, match="other inputs") as caught:
        run(data_dir)
    assert "the file 'dev'" in str(caught.value) or "fitted on other inputs" in str(caught.value)


def test_resume_refuses_changed_code(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run(data_dir)
    monkeypatch.setattr(provenance, "source_digest", lambda: "another version")
    with pytest.raises(ResumeMismatch, match="other inputs"):
        run(data_dir)


def test_resume_refuses_results_without_a_record(data_dir: Path) -> None:
    root = run(data_dir)
    run_dir = Path(load_manifest(root)["path"].iloc[0]).parent
    (run_dir / provenance.FILE_NAME).unlink()
    with pytest.raises(ResumeMismatch, match="no record of the inputs"):
        run(data_dir)
    (run_dir / "config.yaml").unlink()
    with pytest.raises(ResumeMismatch, match=r"no config\.yaml"):
        run(data_dir)


def test_resume_refuses_an_artifact_fitted_on_other_inputs(data_dir: Path) -> None:
    root = run(data_dir)
    path = root / "artifacts" / "seed0.json"
    artifact = load_artifact(path)
    assert artifact.inputs["n"] == 3 and set(artifact.inputs["files"]) == {
        "train",
        "dev",
        "windows",
    }
    artifact.check_compatible(
        artifact.estimator, artifact.K, artifact.K_prime, inputs=artifact.inputs
    )
    with pytest.raises(ValueError, match="fitted on other inputs"):
        artifact.check_compatible(
            artifact.estimator, artifact.K, artifact.K_prime, inputs={**artifact.inputs, "n": 4}
        )
    artifact.model_copy(update={"inputs": {}}).save(path)  # as written before the record
    with pytest.raises(ResumeMismatch, match="no record of the inputs"):
        run(data_dir)


def test_a_changed_question_window_is_named(data_dir: Path) -> None:
    root = run(data_dir)
    run_dir = Path(load_manifest(root)["path"].iloc[0]).parent
    stored = provenance.read(run_dir)
    assert stored is not None
    (data_dir / "windows.yaml").write_text("test: [1]\n", encoding="utf-8")
    from ucagent.experiments.runner import build_task

    task = build_task(suite(data_dir).base)
    current = provenance.record(task, [i.instance_id for i in task.instances("test")])
    changed = provenance.differences(stored, current)
    assert "the questions (run ids and source ids)" in changed and "the file 'windows'" in changed


def test_records_compare_in_words() -> None:
    rec = {
        "schema": 1,
        "source": "a",
        "files": {"dev": "1"},
        "n": 2,
        "questions": "q",
        "content": "c",
    }
    assert provenance.differences(rec, rec) == []
    assert provenance.differences(None, rec) == ["no record of the inputs (the results predate it)"]
    other = {**rec, "source": "b", "files": {"dev": "2", "windows": "3"}}
    assert provenance.differences(rec, other) == [
        "the package's source",
        "the file 'dev'",
        "the file 'windows'",
    ]
    assert provenance.file_digest("no/such/file") == "missing"
    assert len(provenance.source_digest()) == 16


def test_finqa_refuses_options_it_would_ignore(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no option 'windows_file'"):
        FinqaTask(data_dir=tmp_path, windows_file="windows.yaml")
    with pytest.raises(ValueError, match="no option 'protocal'"):
        FinqaTask(data_dir=tmp_path, protocal=2)
    task = FinqaTask(data_dir=tmp_path, n_steps=2)  # nominal in every config: accepted
    assert set(task.input_files()) == {"train", "dev", "test"}


def test_the_model_digest_is_asked_of_ollama_only() -> None:
    assert model_digest("mock") is None and model_digest("openai:gpt-4o") is None
