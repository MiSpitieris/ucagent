"""Split 3 of 2WikiMultiHopQA: fresh calibration and test questions that pass the data filter.

Usage::

    uv run python scripts/make_twowiki_split3.py --out configs/twowiki_split3_windows.yaml \\
        --exclude configs/twowiki_confagree.yaml configs/twowiki_split2_confagree.yaml \\
        configs/twowiki_smoke3.yaml

Walks the seeded calibration and test halves in order (the task block of ``--like``, so the
data, seed and order are those of splits 1 and 2) and keeps a question when

- no calibration or test window of an ``--exclude`` config uses it (compared by the
  dataset's own question id, so a config with another seed or split role would still
  exclude its questions), and
- ``ucagent.tasks.twowiki.question_flags`` finds nothing on either hop: no support flag
  and no keyword relation whose gold the triage could not locate in a sentence.

It stops at ``--n`` questions per split and writes their absolute indices as a windows file
for the task's ``windows_file`` option, with the counts in its header. The filter reads the
data only (user decision of 2026-11-09, made before any split 3 call). No model calls.

With ``--coverage WINDOWS`` nothing is written: for the questions a windows file keeps, it
prints on how many hops the filter could check the relation by keyword. The filter has
keywords for the relations of ``ucagent.tasks.twowiki.RELATION_KEYWORDS`` only; a hop with
another relation passes it unchecked (but for the test of place names).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ucagent.tasks import make_task
from ucagent.tasks.twowiki import RELATION_KEYWORDS, question_flags


def used_ids(path: Path) -> set[str]:
    """Source question ids of the calibration and test windows a config runs."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("kind") == "suite":
        block = dict(data["base"]["task"])
        splits = [s for s in data.get("splits", ["calib", "test"]) if s != "train"]
    else:
        block, splits = dict(data["task"]), [str(data.get("split", "test"))]
    task: Any = make_task(block)
    return {inst.qid for split in splits for inst in task.instances(split)}


def walk(
    task: Any, split: str, n: int, used: set[str]
) -> tuple[list[int], int, int, Counter[str], int]:
    """The first ``n`` usable indices, questions skipped as used and as filtered, the filter's
    flag kinds over the filtered questions, and the last row read."""
    keep: list[int] = []
    n_used = n_filtered = 0
    kinds: Counter[str] = Counter()
    last = -1
    for index in range(len(task._candidates(split))):
        last = index
        inst = task.get_instance(task.instance_id(split, index))
        if inst.qid in used:
            n_used += 1
            continue
        flags = question_flags(inst)
        if flags:
            n_filtered += 1
            kinds.update({f.split(":", 1)[1] for f in flags})
            continue
        keep.append(index)
        if len(keep) == n:
            break
    if len(keep) < n:
        raise SystemExit(f"split {split!r}: only {len(keep)} usable questions, {n} needed")
    return keep, n_used, n_filtered, kinds, last


def checked_hops(inst: Any) -> int:
    """On how many hops of a question the filter has keywords for the relation."""
    relations = [
        h.template.split(">>")[-1].strip() if ">>" in h.template else "" for h in inst.hops
    ]
    return sum(1 for relation in relations if RELATION_KEYWORDS.get(relation))


def coverage(task: Any, windows: dict[str, list[int]]) -> list[str]:
    """Per split: the kept questions by the number of hops the filter checked by keyword."""
    lines = []
    for split, indices in windows.items():
        counts = Counter(
            checked_hops(task.get_instance(task.instance_id(split, i))) for i in indices
        )
        lines.append(
            f"{split}: {len(indices)} kept questions; relation checked by keyword on both hops "
            f"{counts[2]}, on one hop {counts[1]}, on neither {counts[0]}"
        )
    return lines


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--like", type=Path, default=Path("configs/twowiki_confagree.yaml"))
    parser.add_argument("--exclude", nargs="+", type=Path, default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=None,
        help="a windows file: print on how many hops its questions were checked; write nothing",
    )
    args = parser.parse_args()
    base = yaml.safe_load(args.like.read_text(encoding="utf-8"))["base"]["task"]
    block = {k: v for k, v in base.items() if k not in ("offset", "windows_file")}
    task: Any = make_task(block)
    if args.coverage is not None:
        kept = yaml.safe_load(args.coverage.read_text(encoding="utf-8"))
        print("\n".join(coverage(task, kept)))
        return
    if args.exclude is None or args.out is None:
        parser.error("--exclude and --out are required unless --coverage is given")
    used: set[str] = set()
    for path in args.exclude:
        used |= used_ids(path)
    windows: dict[str, list[int]] = {}
    header = [
        f"# Split 3 of 2WikiMultiHopQA, made by scripts/make_twowiki_split3.py on "
        f"{dt.date.today().isoformat()} (data only, no model calls).",
        f"# The first {args.n} questions of each seeded half (task block of {args.like}) that no "
        "calibration or test window of",
        "# " + ", ".join(str(p) for p in args.exclude) + " uses, and on which",
        "# ucagent.tasks.twowiki.question_flags finds nothing on either hop (user decision of "
        "2026-11-09).",
    ]
    for split in ("calib", "test"):
        keep, n_used, n_filtered, kinds, last = walk(task, split, args.n, used)
        windows[split] = keep
        counts = ", ".join(f"{k} {v}" for k, v in kinds.most_common())
        line = (
            f"# {split}: rows 0..{last} read, {len(keep)} kept, {n_used} used by an earlier "
            f"window, {n_filtered} filtered (questions with: {counts})"
        )
        header.append(line)
        print(line[2:])
    body = yaml.safe_dump(windows, default_flow_style=None, width=96, sort_keys=False)
    args.out.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
    print(f"wrote {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
