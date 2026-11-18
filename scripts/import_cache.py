"""Import JSONL rows into the sqlite response cache.

Usage: python scripts/import_cache.py in.jsonl cache.sqlite
"""

from __future__ import annotations

import json
import sys

from ucagent.core.llm.cache import SqliteCache


def main(src: str, dst: str) -> int:
    cache = SqliteCache(dst)
    n = 0
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                cache.import_row(json.loads(line))
                n += 1
    print(f"imported {n} rows into {dst} ({len(cache)} total)")
    return n


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
