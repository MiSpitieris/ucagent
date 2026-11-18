"""Real chat-model backends: ``init_chat_model`` wrapped in the cache and cost guard.

Any provider works (``anthropic:...``, ``openai:...``, ``ollama:...``); no
provider is hardcoded. Real models are always wrapped, because the graph
passes ``sample_index`` as a model kwarg that provider SDKs do not accept.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ucagent.core.llm.cache import CachedChatModel, SqliteCache
from ucagent.core.llm.guard import CostGuard


def init_model(name: str, temperature: float, **model_kwargs: Any) -> BaseChatModel:
    """Provider-agnostic instantiation via ``init_chat_model`` (needs the ``llm`` extra).

    ``model_kwargs`` reach the provider class unchanged (for ``ChatOllama``:
    ``reasoning``, ``num_ctx``, ``num_predict``, ...).
    """
    from langchain.chat_models import init_chat_model

    model = init_chat_model(name, temperature=temperature, **model_kwargs)
    if not isinstance(model, BaseChatModel):
        raise TypeError(f"init_chat_model({name!r}) did not return a BaseChatModel")
    return model


def get_chat_model(
    name: str,
    temperature: float = 0.7,
    *,
    cache_path: str | Path = "cache/llm_cache.sqlite",
    max_calls: int = 2000,
    max_usd: float = 2.0,
    inner: BaseChatModel | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> CachedChatModel:
    """A cached, cost-guarded chat model for ``provider:model``.

    Non-empty ``model_kwargs`` are passed to the provider class and become
    part of the cache key (:func:`ucagent.core.llm.cache.model_tag`).
    """
    extra = dict(model_kwargs or {})
    cache = SqliteCache(cache_path)
    guard = CostGuard(max_calls=max_calls, max_usd=max_usd, model_name=name, shared_counter=cache)
    return CachedChatModel(
        inner=inner if inner is not None else init_model(name, temperature, **extra),
        store=cache,
        model_name=name,
        temperature=temperature,
        guard=guard,
        model_kwargs=extra,
    )
