"""History-conditional, hash-seeded mock chat model for offline experiments.

The mock reads the header the task puts in every step prompt
(``<ucagent instance=".." step=".." current=".." attempt="..">``), asks the
task for the *local* truth and the distractor names of that step given the
current context, and answers correctly with probability ``1 - p_wrong``. The
latent difficulty ``d[instance, step]`` is Beta-distributed; ``p_wrong = d``
when distractors are present and ``d * clean_factor`` otherwise. Every draw
is seeded by a stable hash of ``(seed, instance, step, current, attempt,
sample_index)``, so the model is stateless, deterministic, and safe to call
from threads and processes. The header's ``attempt`` is the draw id of the
batch: a new draw yields fresh samples, while an expansion of the batch (a
resample) keeps the draw and adds indices ``K..K'-1``, which are exactly the
samples a ``K'``-sample run of the same prompt would have drawn.
"""

from __future__ import annotations

import random
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ucagent.core.protocols import Task
from ucagent.core.utils import stable_hash

HEADER_RE = re.compile(
    r'<ucagent instance="(?P<instance>[^"]+)" step="(?P<step>\d+)" '
    r'current="(?P<current>[^"]*)" attempt="(?P<attempt>\d+)">'
)


def render_header(instance_id: str, step: int, current: str, attempt: int) -> str:
    """The structured header every step prompt must start with."""
    return (
        f'<ucagent instance="{instance_id}" step="{step}" current="{current}" attempt="{attempt}">'
    )


def parse_header(text: str) -> dict[str, str]:
    """Parse the header fields out of a prompt; raises if absent."""
    m = HEADER_RE.search(text)
    if m is None:
        raise ValueError("prompt has no <ucagent ...> header")
    return m.groupdict()


class MockChainLLM(BaseChatModel):
    """Mock chat model with controllable per-(instance, step) difficulty."""

    task: Any = None
    seed: int = 0
    difficulty_beta: tuple[float, float] = (2.0, 5.0)
    clean_factor: float = 0.2
    verbalized: bool = True
    verbalized_sigma: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "mock-chain"

    def difficulty(self, instance_id: str, step: int) -> float:
        """Latent difficulty ``d`` of a step, drawn once per (instance, step)."""
        rng = random.Random(stable_hash(self.seed, instance_id, step, "difficulty"))
        a, b = self.difficulty_beta
        return rng.betavariate(a, b)

    def p_wrong(self, instance_id: str, step: int, current: str) -> float:
        """Probability that one sample is locally wrong."""
        task: Task = self.task
        instance = task.get_instance(instance_id)
        d = self.difficulty(instance_id, step)
        distractors = task.distractor_names(instance, step, current)
        return d if distractors else d * self.clean_factor

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        sample_index = int(kwargs.get("sample_index", 0))
        text = "\n".join(str(m.content) for m in messages)
        header = parse_header(text)
        instance_id = header["instance"]
        step = int(header["step"])
        current = header["current"]
        attempt = int(header["attempt"])
        task: Task = self.task
        instance = task.get_instance(instance_id)
        truth = task.local_truth(instance, step, current)
        distractors = [d for d in task.distractor_names(instance, step, current) if d != truth]
        d = self.difficulty(instance_id, step)
        p_wrong = d if distractors else d * self.clean_factor
        rng = random.Random(
            stable_hash(self.seed, instance_id, step, current, attempt, sample_index)
        )
        if rng.random() < p_wrong:
            answer = rng.choice(distractors) if distractors else task.random_entity(instance, rng)
        else:
            answer = truth
        content = f"Answer: {answer}"
        if self.verbalized:
            conf = min(1.0, max(0.0, 1.0 - p_wrong + rng.gauss(0.0, self.verbalized_sigma)))
            content += f"\nConfidence: {conf:.2f}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
