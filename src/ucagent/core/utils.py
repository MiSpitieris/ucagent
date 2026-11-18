"""Small shared helpers: stable hashing for seeds and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(*parts: object) -> int:
    """Deterministic 64-bit hash of a tuple of simple values.

    Uses blake2b over ``repr`` of each part, so the value is identical across
    processes, machines and Python versions (unlike builtin ``hash``). Used to
    seed every random draw in the mock, the task generator and the oracles.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def stable_digest(obj: Any) -> str:
    """Hex fingerprint of a JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
