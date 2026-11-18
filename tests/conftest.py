import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip real-LLM tests unless explicitly enabled (no key, no network in CI)."""
    if os.environ.get("UCAGENT_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set UCAGENT_LLM_TESTS=1 to run tests marked llm")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
