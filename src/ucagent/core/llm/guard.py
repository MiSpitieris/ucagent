"""Cost guard: abort a run when real-LLM calls or estimated spend exceed a limit.

The call counter is shared across worker processes through the response
cache's sqlite file (``SqliteCache.increment_calls``), so the limit holds for
the whole run, not per worker. Mock calls never pass through the guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("ucagent")

# USD per million input / output tokens; unknown models fall back to a
# conservative default so the USD guard errs on the side of stopping early.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "ollama": (0.0, 0.0),
    "anthropic:claude-haiku-5-5": (1.0, 5.0),
    "anthropic:claude-sonnet-5-5": (3.0, 15.0),
    "anthropic:claude-opus-5-5": (15.0, 75.0),
    "openai:gpt-4o-mini": (0.15, 0.6),
    "openai:gpt-4o": (2.5, 10.0),
}
DEFAULT_PRICE = (3.0, 15.0)


class CostGuardExceeded(RuntimeError):
    """Raised when a limit is hit; the message names the limit and the count."""


def price_for(model_name: str) -> tuple[float, float]:
    """Price table lookup by exact name, then by provider prefix."""
    if model_name in PRICES_USD_PER_MTOK:
        return PRICES_USD_PER_MTOK[model_name]
    provider = model_name.split(":", 1)[0]
    return PRICES_USD_PER_MTOK.get(provider, DEFAULT_PRICE)


@dataclass
class CostGuard:
    """Counts real calls and estimated USD; raises past ``max_calls`` or ``max_usd``."""

    max_calls: int
    max_usd: float
    model_name: str = ""
    calls: int = 0
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    shared_counter: object = field(default=None, repr=False)

    def check(self) -> None:
        """Raise if another call would exceed a limit (call before the request)."""
        total = self._total_calls()
        if total >= self.max_calls:
            raise CostGuardExceeded(
                f"cost guard: {total} real LLM calls reached the limit UCAGENT_MAX_LLM_CALLS="
                f"{self.max_calls}; raise the limit or use the cache"
            )
        if self.usd >= self.max_usd:
            raise CostGuardExceeded(
                f"cost guard: estimated spend {self.usd:.2f} USD reached UCAGENT_MAX_USD="
                f"{self.max_usd}"
            )

    def charge(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record one completed call (call after the request)."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        pin, pout = price_for(self.model_name)
        self.usd += (input_tokens * pin + output_tokens * pout) / 1e6
        if self.shared_counter is not None:
            self.shared_counter.increment_calls()  # type: ignore[attr-defined]

    def _total_calls(self) -> int:
        if self.shared_counter is not None:
            return int(self.shared_counter.total_calls())  # type: ignore[attr-defined]
        return self.calls
