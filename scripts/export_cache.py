"""Export the sqlite response cache to JSONL.

Usage: python scripts/export_cache.py cache.sqlite out.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ucagent.core.llm.cache import SqliteCache


def main(src: str, dst: str) -> int:
    cache = SqliteCache(src)
    n = 0
    with open(dst, "w", encoding="utf-8") as fh:
        for row in cache.rows():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n} rows to {Path(dst)}")
    return n


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
