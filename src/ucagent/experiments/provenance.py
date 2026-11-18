"""What a run was computed from: recorded beside its results, compared when a suite resumes.

A resumed suite used to compare configurations only. A windows file or a data file that
changed under the same path, or a changed label rule, was then reused without a word (code
review of 2026-11-11, finding C04). The record holds what a configuration does not say: the
content of the files the task reads, the questions as the task presents them, and the code.

A task names its files through an optional method ``input_files() -> dict[str, Path]``; a task
without it records no file, and its questions and the code still count.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import ucagent
from ucagent.core.utils import stable_digest

SCHEMA = 1
FILE_NAME = "inputs.json"
# what each field of the record stands for, for the message of a refused resume
MEANING = {
    "schema": "the format of the record",
    "source": "the package's source",
    "n": "the number of questions",
    "questions": "the questions (run ids and source ids)",
    "content": "what the task shows and expects for the questions",
}


@lru_cache(maxsize=256)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file's content, or ``missing`` when there is no such file."""
    p = Path(path)
    if not p.is_file():
        return "missing"
    stat = p.stat()
    return _sha256(str(p.resolve()), stat.st_size, stat.st_mtime_ns)


def source_digest() -> str:
    """One digest over every Python file of the ``ucagent`` package that is imported."""
    root = Path(ucagent.__file__).resolve().parent
    return stable_digest(
        [[str(p.relative_to(root)), file_digest(p)] for p in sorted(root.rglob("*.py"))]
    )


def record(task: Any, instance_ids: Sequence[str]) -> dict[str, Any]:
    """The inputs of a run of ``task`` over ``instance_ids``."""
    instances = [task.get_instance(i) for i in instance_ids]
    files = getattr(task, "input_files", None)
    return {
        "schema": SCHEMA,
        "source": source_digest(),
        "files": {
            name: file_digest(path) for name, path in sorted((files() if files else {}).items())
        },
        "n": len(instances),
        "questions": stable_digest(
            [[i.instance_id, str(getattr(i, "qid", ""))] for i in instances]
        ),
        "content": stable_digest([i.model_dump(mode="json") for i in instances]),
    }


def differences(stored: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """What differs between a stored record and the current one, in words (empty: nothing)."""
    if stored is None:
        return ["no record of the inputs (the results predate it)"]
    out = [MEANING[key] for key in MEANING if stored.get(key) != current.get(key)]
    before, now = stored.get("files", {}), current.get("files", {})
    out += [
        f"the file {name!r}"
        for name in sorted(set(before) | set(now))
        if before.get(name) != now.get(name)
    ]
    return out


def write(run_dir: str | Path, rec: dict[str, Any]) -> Path:
    """Write the record into a run directory."""
    path = Path(run_dir) / FILE_NAME
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read(run_dir: str | Path) -> dict[str, Any] | None:
    """The record of a run directory, or None when it has none."""
    path = Path(run_dir) / FILE_NAME
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
