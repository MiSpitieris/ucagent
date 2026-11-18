"""Convert the 2WikiMultiHopQA dev split into the two-hop chain files the ``twowiki`` task reads.

Usage::

    uv run python scripts/prepare_twowiki.py data/twowiki/dev.parquet --out data/twowiki

Keeps the compositional and inference questions whose two evidence triples form a chain
(``ucagent.tasks.twowiki.chain_rows``), shuffles them with a fixed seed, writes the first
``--train-pool`` rows to ``twowiki_train.jsonl`` (the pool the calibration fit draws from)
and the rest to ``twowiki_dev.jsonl`` (calibration and test are its seeded halves), and
prints counts and SHA-256 digests. The input is read as data only.

With ``--count-only`` nothing is written: it prints how many chain questions there are before
the exclusion of relations (``AMBIGUOUS_RELATIONS``: relations with several valid values or
several forms of one value, for which a single gold answer misjudges right answers), how many
the exclusion leaves out, and how many of those have each excluded relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from ucagent.tasks.twowiki import (
    AMBIGUOUS_RELATIONS,
    DEV_FILE,
    TRAIN_FILE,
    chain_rows,
    uses_fallback,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relations_of(row: dict) -> set[str]:
    """The relations of the two hops of a chain row."""
    return {h["question"].split(">>")[-1].strip() for h in row["question_decomposition"]}


def exclusion_counts(records: list[dict], skip: list[str]) -> list[str]:
    """What the exclusion of relations leaves out of the chain questions."""
    kept = chain_rows(records, exclude_ids=skip)
    every = chain_rows(records, exclude_relations=(), exclude_ids=skip)
    kept_ids = {r["id"] for r in kept}
    left_out = [r for r in every if r["id"] not in kept_ids]
    by_relation = Counter(
        rel for r in left_out for rel in sorted(relations_of(r) & set(AMBIGUOUS_RELATIONS))
    )
    return [
        f"chain questions before the exclusion of relations: {len(every)}",
        f"left out by it: {len(left_out)} ({len(AMBIGUOUS_RELATIONS)} relations excluded)",
        f"kept: {len(kept)}",
        "left out, by excluded relation (a question with two of them counts under both): "
        + ", ".join(f"{rel} {n}" for rel, n in by_relation.most_common()),
    ]


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/twowiki"))
    parser.add_argument("--train-pool", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-ids", type=Path, default=None, help="file of question ids to leave out"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="print what the exclusion of relations leaves out, and write nothing",
    )
    args = parser.parse_args()
    records = pd.read_parquet(args.parquet).to_dict("records")
    skip = (
        [ln.strip() for ln in args.exclude_ids.read_text().splitlines() if ln.strip()]
        if args.exclude_ids
        else []
    )
    if args.count_only:
        print(f"excluded question ids: {len(skip)}")
        print("\n".join(exclusion_counts(records, skip)))
        return
    rows = chain_rows(records, exclude_ids=skip)
    print(f"excluded question ids: {len(skip)}")
    random.Random(f"twowiki-pool-{args.seed}").shuffle(rows)
    pool, rest = rows[: args.train_pool], rows[args.train_pool :]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in ((TRAIN_FILE, pool), (DEV_FILE, rest)):
        with open(args.out / name, "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    relations = Counter(
        h["question"].split(">>")[-1].strip() for r in rows for h in r["question_decomposition"]
    )
    print(f"input {args.parquet} ({_sha256(args.parquet)}): {len(records)} records")
    print(f"chains kept: {len(rows)} ({Counter(r['type'] for r in rows)})")
    for name, part in ((TRAIN_FILE, pool), (DEV_FILE, rest)):
        print(f"wrote {args.out / name}: {len(part)} rows, sha256 {_sha256(args.out / name)}")
    fallback = {r: n for r, n in relations.items() if uses_fallback(r)}
    print(
        f"relations: {len(relations)}; without a template (generic question): {fallback or 'none'}"
    )


if __name__ == "__main__":
    main()
