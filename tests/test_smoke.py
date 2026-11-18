import ucagent
from ucagent.config import Settings


def test_version() -> None:
    assert ucagent.__version__


def test_settings_defaults_are_small(monkeypatch) -> None:
    monkeypatch.delenv("UCAGENT_MAX_LLM_CALLS", raising=False)
    s = Settings(_env_file=None)
    assert s.max_llm_calls <= 5000
    assert s.max_usd <= 10


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("UCAGENT_MAX_LLM_CALLS", "7")
    assert Settings(_env_file=None).max_llm_calls == 7
