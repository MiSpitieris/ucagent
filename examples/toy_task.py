"""A minimal custom task: an arithmetic chain. Start value, then n operations.

This is the worked example of docs/ADOPTION.md. It implements the
``ucagent.core.protocols.Task`` protocol and registers itself under the name
``arithmetic``, so a config with ``plugins: [examples.toy_task]`` and
``task: {name: arithmetic, ...}`` runs through every ucagent command.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ucagent.core.llm.mock import render_header
from ucagent.core.protocols import Instance
from ucagent.core.records import StepRecord
from ucagent.core.utils import stable_hash
from ucagent.tasks import register_task

OPS = [("add", 3), ("add", 7), ("double", 0), ("subtract", 4), ("triple", 0)]


class ArithmeticInstance(Instance):
    start: int
    ops: list[tuple[str, int]]


def apply(op: tuple[str, int], value: int) -> int:
    kind, k = op
    return {"add": value + k, "subtract": value - k, "double": 2 * value, "triple": 3 * value}[kind]


class ArithmeticChainTask:
    """Each step applies one operation to the previous (possibly wrong) value."""

    name = "arithmetic"

    def __init__(
        self, n_steps: int = 3, n_instances: dict[str, int] | None = None, seed: int = 0, **_: Any
    ) -> None:
        self.n = n_steps
        self.sizes = n_instances or {"train": 50, "calib": 50, "test": 50}
        self.seed = seed

    # --- instances ---------------------------------------------------------------
    def instances(self, split: str) -> Iterable[Instance]:
        for index in range(self.sizes[split]):
            yield self.get_instance(f"{split}-{index:05d}-s{self.seed}")

    @lru_cache(maxsize=4096)  # noqa: B019
    def get_instance(self, instance_id: str) -> ArithmeticInstance:
        split, index, _ = instance_id.split("-")
        rng = random.Random(stable_hash("arith", self.seed, split, int(index)))
        start = rng.randint(1, 20)
        ops = [rng.choice(OPS) for _ in range(self.n)]
        words = ", then ".join(f"{k} {v}" if v else k for k, v in ops)
        return ArithmeticInstance(
            instance_id=instance_id,
            split=split,
            index=int(index),
            n_steps=self.n,
            question=f"Start at {start}; {words}. What is the result?",
            start=start,
            ops=ops,
        )

    def n_steps(self, instance: Instance) -> int:
        return instance.n_steps

    def initial_context(self, instance: Instance) -> str:
        return str(self._inst(instance).start)

    # --- truths (local = relative to the current value, global = the true chain) ----
    def local_truth(self, instance: Instance, i: int, current: str) -> str:
        inst = self._inst(instance)
        try:
            value = int(float(current))
        except ValueError:
            value = inst.start
        return str(apply(inst.ops[i - 1], value))

    def global_truth(self, instance: Instance, i: int) -> str:
        inst = self._inst(instance)
        value = inst.start
        for op in inst.ops[:i]:
            value = apply(op, value)
        return str(value)

    def distractor_names(self, instance: Instance, i: int, current: str) -> list[str]:
        truth = int(self.local_truth(instance, i, current))
        return [str(truth + d) for d in (-2, -1, 1, 2)]  # always ambiguous: near misses

    def random_entity(self, instance: Instance, rng: random.Random) -> str:
        return str(rng.randint(-50, 200))

    def is_final_correct(self, instance: Instance, answer: str | None) -> bool:
        truth = self.global_truth(instance, instance.n_steps)
        return answer is not None and self._same(answer, truth)

    def is_locally_correct(self, instance: Instance, i: int, current: str, answer: str) -> bool:
        return self._same(answer, self.local_truth(instance, i, current))

    # --- prompts ------------------------------------------------------------------
    def step_prompt(
        self, instance: Instance, i: int, history: Sequence[StepRecord], attempt: int
    ) -> list[BaseMessage]:
        inst = self._inst(instance)
        current = history[-1].answer if history else str(inst.start)
        kind, k = inst.ops[i - 1]
        op_text = f"{kind} {k}" if k else kind
        body = "\n".join(
            [
                render_header(inst.instance_id, i, current, attempt),
                f"Question: {inst.question}",
                f"Step {i} of {inst.n_steps}: apply '{op_text}' to {current}.",
            ]
        )
        return [
            SystemMessage(content="Reply as 'Answer: <integer>' then 'Confidence: <0-1>'."),
            HumanMessage(content=body),
        ]

    def parse_answer(self, text: str) -> str:
        m = re.search(r"Answer:\s*(-?\d+)", text)
        return m.group(1) if m else text.strip().split("\n")[0]

    @staticmethod
    def _same(a: str, b: str) -> bool:
        try:
            return int(float(a)) == int(float(b))
        except ValueError:
            return a.strip() == b.strip()

    @staticmethod
    def _inst(instance: Instance) -> ArithmeticInstance:
        assert isinstance(instance, ArithmeticInstance)
        return instance


register_task(
    "arithmetic",
    lambda block: ArithmeticChainTask(**{k: v for k, v in block.items() if k != "name"}),
)
