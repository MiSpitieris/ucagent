"""Process-level settings read from the environment (prefix ``UCAGENT_``).

Experiment configuration (the YAML of Appendix B) lives in
:mod:`ucagent.experiments.config`; this module only holds the few settings
that must come from the environment: the cost guard and file locations.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings.

    Attributes:
        max_llm_calls: Abort a run after this many real-LLM calls (mock calls
            do not count). Small by default on purpose.
        max_usd: Abort a run once the estimated spend exceeds this amount.
        cache_path: Location of the sqlite response cache for real models.
        results_dir: Root directory for run outputs.
        llm_tests: Whether tests marked ``llm`` run (``UCAGENT_LLM_TESTS=1``).
    """

    model_config = SettingsConfigDict(env_prefix="UCAGENT_", env_file=".env", extra="ignore")

    max_llm_calls: int = 2000
    max_usd: float = 2.0
    cache_path: Path = Path("cache/llm_cache.sqlite")
    results_dir: Path = Path("results")
    llm_tests: bool = False


def get_settings() -> Settings:
    """Return settings read from the environment (and ``.env`` if present)."""
    return Settings()
