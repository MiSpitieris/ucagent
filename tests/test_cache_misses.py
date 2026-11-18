"""The dry replay answers from the cache, counts what it lacks, and writes nothing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ucagent.core.llm.backend import get_chat_model


class Fixed(BaseChatModel):
    """A stand-in for the real model: one fixed reply."""

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Answer: 42"))])


@pytest.fixture
def mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "cache_misses.py"
    spec = importlib.util.spec_from_file_location("cache_misses", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hits_come_from_the_cache_and_misses_are_counted_once(
    tmp_path: Path, mod: ModuleType
) -> None:
    db = tmp_path / "c.sqlite"
    kwargs = {"reasoning": "low"}
    real = get_chat_model("test:model", 0.7, cache_path=db, inner=Fixed(), model_kwargs=kwargs)
    asked = [HumanMessage(content="what is 6 times 7?")]
    real.invoke(asked, sample_index=0)  # one real call: now in the cache
    before = db.read_bytes()

    dry = mod.DryModel(db=str(db), model_name="test:model", temperature=0.7, model_kwargs=kwargs)
    assert dry.invoke(asked, sample_index=0).content == "Answer: 42"  # the same key: a hit
    assert not mod.MISSES and mod.COUNTS == {"requests": 1, "hits": 1}
    for _ in range(2):  # another sample index is not cached: a miss, counted once
        assert dry.invoke(asked, sample_index=1).content == mod.PLACEHOLDER
    other = mod.DryModel(db=str(db), model_name="test:model", temperature=0.7)  # other options
    assert other.invoke(asked, sample_index=0).content == mod.PLACEHOLDER
    assert len(mod.MISSES) == 2 and mod.COUNTS == {"requests": 4, "hits": 1}
    assert mod.counter(str(db)) == 1  # the one real call; the replay charged nothing
    assert db.read_bytes() == before  # and wrote nothing
