"""Freshness is judged by source question, not by run id (code review of 2026-11-11, C06).

A run id encodes split, index and seed. Under another seed the halves are drawn again, so a
calibration question of one config can be a test question of another under a different run
id. The checks that keep evaluation questions fresh compare the dataset's own ids.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from ucagent.tasks import make_task
from ucagent.tasks.twowiki import chain_rows

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "twowiki_mini.jsonl"


def load(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configs(tmp_path: Path) -> tuple[Path, Path, dict]:
    raw = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    with open(tmp_path / "mini.jsonl", "w", encoding="utf-8") as fh:
        for row in chain_rows(raw, exclude_relations=()):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    block = {
        "name": "twowiki",
        "data_dir": str(tmp_path),
        "train_file": "mini.jsonl",
        "dev_file": "mini.jsonl",
        "n_distractors": 2,
        "n_instances": {"train": 0, "calib": 1, "test": 1},
        "local_match": "contains",
        "final_match": "contains",
    }
    paths = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}.yaml"
        suite = {"kind": "suite", "base": {"task": {**block, "seed": seed}}}
        path.write_text(yaml.safe_dump(suite), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1], block


def test_the_same_record_under_two_seeds_is_an_overlap(configs: tuple[Path, Path, dict]) -> None:
    first, second, block = configs
    mod = load("validate_benchmark")
    ids0, _, _ = mod.check_config(first)
    ids1, _, _ = mod.check_config(second)
    run_ids = {
        seed: {
            i.instance_id
            for s in ("calib", "test")
            for i in make_task({**block, "seed": seed}).instances(s)
        }
        for seed in (0, 1)
    }
    assert not run_ids[0] & run_ids[1]  # by run id the two configs share nothing
    clash, same_text = mod.shared(ids1, ids0, mod.TEXTS[second], mod.TEXTS[first])
    assert len(clash) == 2 and len(same_text) == 2  # by source question they share both
    assert all(not q.startswith("twowiki-") for q in clash)


def test_split3_excludes_by_source_question(configs: tuple[Path, Path, dict]) -> None:
    _, second, block = configs
    mod = load("make_twowiki_split3")
    used = mod.used_ids(second)  # the questions the seed-1 config evaluates on
    assert len(used) == 2 and all(not q.startswith("twowiki-") for q in used)
    task = make_task({**block, "seed": 0})
    # every seed-0 evaluation question is one of them, though no run id agrees
    assert {i.qid for s in ("calib", "test") for i in task.instances(s)} == used
    with pytest.raises(SystemExit, match="only 0 usable questions"):
        mod.walk(task, "calib", 1, used)  # so nothing is left to keep for a fresh split
