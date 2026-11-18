import subprocess
import sys

from ucagent.core.utils import stable_digest, stable_hash


def test_stable_across_processes() -> None:
    here = stable_hash("a", 1, 2.5, ("x", 3))
    code = "from ucagent.core.utils import stable_hash; print(stable_hash('a', 1, 2.5, ('x', 3)))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "12345"},
    )
    assert int(out.stdout.strip()) == here


def test_distinct_inputs_distinct_hashes() -> None:
    values = {stable_hash("s", i, j) for i in range(50) for j in range(50)}
    assert len(values) == 2500
    assert stable_hash(1) != stable_hash("1")


def test_digest_is_order_independent_for_dicts() -> None:
    assert stable_digest({"a": 1, "b": [1, 2]}) == stable_digest({"b": [1, 2], "a": 1})
