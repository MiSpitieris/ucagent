"""Synthetic knowledge-base chain task with controllable ambiguity and reconvergence.

Each instance is a chain ``e_0 -r_1-> e_1 -> ... -> e_n`` over its own small
knowledge base, generated deterministically from the seed. Every hop ``i``
has a set of distractor entities ``D_i`` with names confusable with ``e_i``.
The passage shown at step ``i`` is built for the *current* entity (whatever
the agent committed), so a wrong path stays coherent and errors compound.

Controls (per hop or global): ambiguity ``q_i`` (distractor facts are injected
into the passage), reconvergence ``rho_i`` (following ``r_i`` from a wrong
entity lands on the true ``e_i``), number of distractors ``m_d``.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ucagent.core.llm.mock import render_header
from ucagent.core.protocols import Instance
from ucagent.core.records import StepRecord
from ucagent.core.uq.normalize import normalize_answer
from ucagent.core.utils import stable_hash

RELATIONS = [
    "was founded by",
    "is led by",
    "is headquartered in",
    "reports to",
    "is owned by",
    "was succeeded by",
    "is advised by",
    "partners with",
    "is audited by",
    "is supplied by",
]
SYLLABLES = [
    "ka", "lo", "mi", "ren", "tas", "vor", "el", "ub", "sha", "dri",
    "nok", "pel", "zu", "ith", "gar", "wen", "os", "tiv", "mur", "ja",
]  # fmt: skip
VOWELS = "aeiou"

SYSTEM_PROMPT = (
    "You solve one hop of a multi-hop lookup. Use only the passage. "
    "Reply with exactly one entity name on the first line as 'Answer: <name>', "
    "then your confidence that it is right, as 'Confidence: <number between 0 and 1>'."
)

_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)
_ID_RE = re.compile(r"^(?P<split>[a-z]+)-(?P<index>\d+)-s(?P<seed>-?\d+)$")


def _syllable_word(rng: random.Random) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


def fresh_name(rng: random.Random, taken: set[str]) -> str:
    """A new two-word name whose normalised form is not in ``taken``."""
    while True:
        name = f"{_syllable_word(rng)} {_syllable_word(rng)}"
        key = normalize_answer(name)
        if key not in taken:
            taken.add(key)
            return name


def confusable_variant(name: str, rng: random.Random, taken: set[str]) -> str:
    """A name that looks like ``name`` but normalises differently."""
    for _ in range(200):
        words = name.split(" ")
        w = rng.randrange(len(words))
        word = words[w]
        strategy = rng.randrange(3)
        if strategy == 0 and len(word) > 3:
            p = rng.randrange(1, len(word) - 1)
            word = word[:p] + word[p + 1] + word[p] + word[p + 2 :]
        elif strategy == 1:
            vowels = [k for k, ch in enumerate(word.lower()) if ch in VOWELS]
            if vowels:
                p = rng.choice(vowels)
                new = rng.choice([v for v in VOWELS if v != word[p].lower()])
                word = word[:p] + new + word[p + 1 :]
        else:
            word = word + rng.choice("aeiosn")
        words[w] = word[0].upper() + word[1:]
        candidate = " ".join(words)
        key = normalize_answer(candidate)
        if key not in taken:
            taken.add(key)
            return candidate
    raise RuntimeError("could not generate a confusable variant")


def _per_hop(value: float | list[float] | dict[str, Any], n: int, what: str) -> list[float]:
    if isinstance(value, dict):
        if "per_hop" in value:
            value = list(value["per_hop"])
        elif "global" in value:
            value = float(value["global"])
        else:
            raise ValueError(f"{what}: dict needs 'global' or 'per_hop'")
    if isinstance(value, int | float):
        return [float(value)] * n
    if len(value) != n:
        raise ValueError(f"{what}: per-hop list must have length n_steps={n}")
    return [float(v) for v in value]


class SyntheticConfig(BaseModel):
    """Generator parameters (Appendix B ``task`` block)."""

    name: Literal["synthetic_chain"] = "synthetic_chain"
    n_steps: int = 5
    ambiguity: float | list[float] | dict[str, Any] = 0.3
    distractors: int = 3
    reconvergence: float | list[float] | dict[str, Any] = 0.0
    n_instances: dict[str, int] = Field(
        default_factory=lambda: {"train": 300, "calib": 300, "test": 300}
    )
    seed: int = 0

    @property
    def q(self) -> list[float]:
        """Per-hop ambiguity probabilities."""
        return _per_hop(self.ambiguity, self.n_steps, "ambiguity")

    @property
    def rho(self) -> list[float]:
        """Per-hop reconvergence probabilities."""
        return _per_hop(self.reconvergence, self.n_steps, "reconvergence")


class SyntheticInstance(Instance):
    """One generated chain with its local knowledge base."""

    entities: list[str]
    relations: list[str]
    distractors: list[list[str]]
    ambiguity: list[bool]
    kb: list[dict[str, str]]
    pool: list[str]


class SyntheticChainTask:
    """The synthetic chain task; implements :class:`ucagent.core.protocols.Task`."""

    name = "synthetic_chain"

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self._q = config.q
        self._rho = config.rho

    # --- instances ------------------------------------------------------------

    def instance_id(self, split: str, index: int) -> str:
        """Instance id encoding split, index and task seed."""
        return f"{split}-{index:05d}-s{self.config.seed}"

    def instances(self, split: str) -> Iterable[SyntheticInstance]:
        """All instances of a split, generated lazily."""
        for index in range(self.config.n_instances[split]):
            yield self.get_instance(self.instance_id(split, index))

    def get_instance(self, instance_id: str) -> SyntheticInstance:
        """Regenerate an instance from its id (cached)."""
        return self._cached(instance_id)

    @lru_cache(maxsize=4096)  # noqa: B019 - instances are immutable, task is long-lived
    def _cached(self, instance_id: str) -> SyntheticInstance:
        m = _ID_RE.match(instance_id)
        if m is None:
            raise ValueError(f"bad instance id {instance_id!r}")
        if int(m["seed"]) != self.config.seed:
            raise ValueError(f"instance {instance_id!r} belongs to another task seed")
        return self._generate(m["split"], int(m["index"]))

    def _generate(self, split: str, index: int) -> SyntheticInstance:
        cfg = self.config
        n = cfg.n_steps
        rng = random.Random(stable_hash("synthetic_chain", cfg.seed, split, index))
        taken: set[str] = set()
        entities = [fresh_name(rng, taken) for _ in range(n + 1)]
        relations = [rng.choice(RELATIONS) for _ in range(n)]
        distractors = [
            [confusable_variant(entities[i], rng, taken) for _ in range(cfg.distractors)]
            for i in range(1, n + 1)
        ]
        ambiguity = [rng.random() < self._q[i - 1] for i in range(1, n + 1)]
        kb: list[dict[str, str]] = []
        for i in range(1, n + 1):
            mapping = {entities[i - 1]: entities[i]}
            if i >= 2:
                for x in distractors[i - 2]:
                    if rng.random() < self._rho[i - 1]:
                        mapping[x] = entities[i]
                    elif distractors[i - 1]:
                        mapping[x] = rng.choice(distractors[i - 1])
                    else:
                        mapping[x] = confusable_variant(entities[i], rng, taken)
            kb.append(mapping)
        pool = list(entities) + [d for ds in distractors for d in ds]
        hops = ", ".join(f"({k}) {r}" for k, r in enumerate(relations, 1))
        question = (
            f"Start at {entities[0]}. Follow these relations in order: {hops}. "
            "Which entity do you reach at the end?"
        )
        return SyntheticInstance(
            instance_id=self.instance_id(split, index),
            split=split,
            index=index,
            question=question,
            n_steps=n,
            entities=entities,
            relations=relations,
            distractors=distractors,
            ambiguity=ambiguity,
            kb=kb,
            pool=pool,
        )

    # --- truths -----------------------------------------------------------------

    def n_steps(self, instance: Instance) -> int:
        """Number of hops."""
        return instance.n_steps

    def initial_context(self, instance: Instance) -> str:
        """The start entity ``e_0``."""
        return self._inst(instance).entities[0]

    def local_truth(self, instance: Instance, i: int, current: str) -> str:
        """``follow(current, r_i)``; total: unknown entities get a deterministic target."""
        inst = self._inst(instance)
        mapping = inst.kb[i - 1]
        if current in mapping:
            return mapping[current]
        key = normalize_answer(current)
        for name, target in mapping.items():
            if normalize_answer(name) == key:
                return target
        rng = random.Random(stable_hash("fallback", inst.instance_id, i, key))
        if rng.random() < self._rho[i - 1]:
            return inst.entities[i]
        if inst.distractors[i - 1]:
            return rng.choice(inst.distractors[i - 1])
        return confusable_variant(inst.entities[i], rng, set())

    def global_truth(self, instance: Instance, i: int) -> str:
        """The true chain entity ``e_i``."""
        return self._inst(instance).entities[i]

    def distractor_names(self, instance: Instance, i: int, current: str) -> list[str]:
        """Names in the passage of hop ``i`` that are not the local truth (if ambiguous).

        Off the true chain the globally correct entity is never among them, so
        a wrong path rejoins the chain only through reconvergence ``rho`` or a
        hallucination; this keeps ``rho`` the single control of error correction.
        """
        inst = self._inst(instance)
        if not inst.ambiguity[i - 1]:
            return []
        truth = normalize_answer(self.local_truth(instance, i, current))
        on_track = normalize_answer(current) == normalize_answer(inst.entities[i - 1])
        names = [*inst.distractors[i - 1]]
        if on_track:
            names.append(inst.entities[i])  # never offered off-track: rejoining needs rho
        return [d for d in names if normalize_answer(d) != truth]

    def random_entity(self, instance: Instance, rng: random.Random) -> str:
        """A random name from the instance's entity pool."""
        return rng.choice(self._inst(instance).pool)

    def is_final_correct(self, instance: Instance, answer: str | None) -> bool:
        """Normalised exact match against ``e_n``."""
        if answer is None:
            return False
        inst = self._inst(instance)
        return normalize_answer(answer) == normalize_answer(inst.entities[inst.n_steps])

    def is_locally_correct(self, instance: Instance, i: int, current: str, answer: str) -> bool:
        """Normalised exact match against the local truth."""
        return normalize_answer(answer) == normalize_answer(self.local_truth(instance, i, current))

    # --- prompts ------------------------------------------------------------------

    def passage(self, instance: Instance, i: int, current: str) -> list[str]:
        """Fact lines shown at hop ``i`` for the current entity."""
        inst = self._inst(instance)
        relation = inst.relations[i - 1]
        truth = self.local_truth(instance, i, current)
        lines = [f"{current} {relation} {truth}."]
        distractors = self.distractor_names(instance, i, current)
        if distractors:
            rng = random.Random(stable_hash("passage", inst.instance_id, i, current))
            taken = {normalize_answer(current)}
            for d in distractors:
                lines.append(f"{confusable_variant(current, rng, taken)} {relation} {d}.")
            lines.append(f"Some records claim that {current} {relation} {distractors[0]}.")
            rng.shuffle(lines)
        return lines

    def step_prompt(
        self, instance: Instance, i: int, history: Sequence[StepRecord], attempt: int
    ) -> list[BaseMessage]:
        """System + human messages for hop ``i`` (header first, passage last)."""
        inst = self._inst(instance)
        current = history[-1].answer if history else inst.entities[0]
        relation = inst.relations[i - 1]
        chain = inst.entities[0]
        for rec in history:
            chain += f" --{inst.relations[rec.step - 1]}--> {rec.answer}"
        lines = [
            render_header(inst.instance_id, i, current, attempt),
            f"Question: {inst.question}",
            f"Chain so far: {chain}",
            f'Step {i} of {inst.n_steps}: which entity does "{current} {relation}" point to?',
            "Passage:",
            *(f"- {line}" for line in self.passage(instance, i, current)),
        ]
        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(lines))]

    def parse_answer(self, text: str) -> str:
        """``Answer: <name>`` on any line, else the first non-empty line."""
        m = _ANSWER_RE.search(text)
        if m:
            return m.group(1).strip().rstrip(".")
        for line in text.splitlines():
            if line.strip():
                return line.strip().rstrip(".")
        return ""

    @staticmethod
    def _inst(instance: Instance) -> SyntheticInstance:
        if not isinstance(instance, SyntheticInstance):
            raise TypeError("SyntheticChainTask needs SyntheticInstance objects")
        return instance
