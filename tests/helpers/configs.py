"""Small experiment configs for tests."""

from __future__ import annotations

from typing import Any

from ucagent.experiments.config import ExperimentConfig


def small_config(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "run_id": "test",
        "split": "test",
        "seed": 0,
        "plugins": ["tests.helpers.policies"],
        "task": {
            "n_steps": 3,
            "ambiguity": 0.5,
            "distractors": 3,
            "reconvergence": 0.0,
            "n_instances": {"train": 20, "calib": 20, "test": 20},
            "seed": 0,
        },
        "llm": {"backend": "mock", "K": 1, "K_prime": 2},
        "control": {"policy": "none"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    llm = base["llm"]
    if "K_prime" not in overrides.get("llm", {}) and llm["K_prime"] < llm["K"]:
        llm["K_prime"] = llm["K"]  # K' >= K is required; a test that sets only K gets K' = K
    return ExperimentConfig.model_validate(base)
