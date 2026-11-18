"""MuSiQue (Trivedi et al., 2022) as a multi-hop ``Task`` with real per-hop labels.

Each MuSiQue question ships its decomposition: one sub-question per hop with
its gold answer and supporting paragraph, where later sub-questions refer to
earlier answers as ``#k``. That is exactly the per-hop ground truth the local
labels need. Hop ``i`` of the agent asks sub-question ``i`` with ``#k``
replaced by the answers committed so far, over the supporting paragraph plus
``n_distractors`` non-supporting paragraphs of the same question.

Sub-questions of the form ``X >> relation`` (Wikidata-style, about 40 % of
the 2-hop hops) are rendered as natural-language questions before prompting
(:func:`render_subquestion`: a table of every relation in the 2-hop data plus
a generic fallback) unless the task is built with ``render_relations=False``,
which keeps the raw templates, the legacy system prompt that explains the
arrow syntax and the legacy prompt layout, as the archived ``llama3.1:8b`` runs
used. The rendered layout labels the overall question as context and the
sub-question as the thing to answer.

Local versus global truth on a real benchmark: the gold hop answer is the
local truth only while the chain is *on track* (the previous committed answer
is the gold answer of the previous hop). Off track, the sub-question asks
about a wrong entity and the data holds no answer for it, so
``has_local_truth`` is False: such steps are excluded from calibration, count
as locally wrong, and a local reviewer asked to check one can only confirm
what is there (``tasks.base.LocalOracle``). ``local_truth`` still returns the
gold hop answer so the mock has something to say; this lets the mock "repair"
off-track chains, which is why mock runs on this task validate the pipeline
only. Per-hop labels use exact match by default; ``local_match="contains"``
accepts whole-word containment either way (a capable model answers "Foshan,
Guangdong" for gold "Foshan"), which also decides when a chain is on track.
Final correctness is MuSiQue's exact match against the answer and its aliases
by default (``final_match="exact"``); ``final_match="contains"`` applies the
containment rule to the final answer too, so that the final label agrees with
the step checks a human reviewer makes (a verbose-but-right final answer is
then not counted as a failure that no step-level control could have caught).

Files: ``musique_ans_v1.0_train.jsonl`` (the ``train`` split) and
``musique_ans_v1.0_dev.jsonl`` (``calib`` and ``test`` are disjoint seeded
halves of it); download them to ``data_dir`` (for example from the Hugging
Face mirror ``dgslibisey/MuSiQue``).

Windows: a split draws ``n_instances[split]`` questions from index ``offset[split]`` of
its seeded order, or, when ``windows_file`` (YAML ``{split: [indices]}``) lists the split,
exactly the listed indices, for windows left after a data-only filter. Ids always carry
the absolute index, so windows of different configs can be compared by id.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from ucagent.core.llm.mock import render_header
from ucagent.core.protocols import Instance
from ucagent.core.records import StepRecord
from ucagent.core.uq.equivalence import NormalizedContains
from ucagent.core.uq.normalize import normalize_answer
from ucagent.core.utils import stable_hash

SYSTEM_PROMPT = (
    "You answer one hop of a multi-hop question using only the passages given. "
    "A sub-question written as 'X >> relation' asks for the relation of X "
    "(for example 'Paris >> country' asks which country Paris is in). "
    "Reply with the answer only, as a short phrase, on the first line as "
    "'Answer: <answer>', then your confidence that it is right as "
    "'Confidence: <number between 0 and 1>'."
)
SYSTEM_PROMPT_NL = (
    "You answer one hop of a multi-hop question using only the passages given. "
    "Each step asks one sub-question: answer only the sub-question of the current "
    "step, never the overall question, which is shown for context. "
    "Reply with the answer only, as a short phrase, on the first line as "
    "'Answer: <answer>', then your confidence that it is right as "
    "'Confidence: <number between 0 and 1>'."
)
TRAIN_FILE = "musique_ans_v1.0_train.jsonl"
_contains = NormalizedContains()
DEV_FILE = "musique_ans_v1.0_dev.jsonl"

_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)
_ID_PATTERN = r"^{prefix}-(?P<split>[a-z]+)-(?P<index>\d+)-s(?P<seed>-?\d+)$"
_ID_RE = re.compile(_ID_PATTERN.format(prefix="musique"))
_REF_RE = re.compile(r"#(\d+)")


_CONFIDENCE_RE = re.compile(r"\s*(>>)?\s*Confidence\s*[:=].*$", re.IGNORECASE)


_DASHES = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-"
)


def canonical_dashes(text: str) -> str:
    """Unicode hyphens and dashes as the ASCII hyphen (gpt-oss writes U+2011 inside names)."""
    return text.translate(_DASHES)


def _clean_answer(raw: str) -> str:
    """Strip an inline confidence tag, sub-question arrows, markdown emphasis, punctuation."""
    text = _CONFIDENCE_RE.sub("", raw.strip()).replace("\u202f", " ").replace("\u00a0", " ")
    for _ in range(2):  # quotes and trailing punctuation may wrap each other
        text = text.strip().strip("\"'*_`").rstrip(" >.").strip()
    return text


_ARROW_RE = re.compile(r"^(?P<x>.+?)\s*>>\s*(?P<relation>[^>]+?)\s*$")
FALLBACK_QUESTION = "What is the {relation} of {x}?"
#: Every relation of the arrow-form sub-questions in the 2-hop train and dev
#: files (92, in data-frequency order), as a question with the subject ``{x}``.
RELATION_QUESTIONS: dict[str, str] = {
    "located in the administrative territorial entity": (
        "In which administrative territorial entity is {x} located?"
    ),
    "country": "Which country is {x} in?",
    "place of birth": "Where was {x} born?",
    "performer": "Who performed {x}?",
    "place of death": "Where did {x} die?",
    "member of sports team": "Which sports team is {x} a member of?",
    "part of": "What is {x} part of?",
    "shares border with": "What does {x} share a border with?",
    "spouse": "Who is the spouse of {x}?",
    "employer": "Who is the employer of {x}?",
    "headquarters location": "Where is the headquarters of {x}?",
    "record label": "What is the record label of {x}?",
    "educated at": "Where was {x} educated?",
    "author": "Who is the author of {x}?",
    "location": "What is the location of {x}?",
    "composer": "Who composed {x}?",
    "cast member": "Who is a cast member of {x}?",
    "sibling": "Who is a sibling of {x}?",
    "child": "Who is a child of {x}?",
    "father": "Who is the father of {x}?",
    "country of citizenship": "Which country is {x} a citizen of?",
    "founded by": "Who founded {x}?",
    "location of formation": "Where was {x} formed?",
    "instance of": "What is {x} an instance of?",
    "owned by": "Who owns {x}?",
    "lyrics by": "Who wrote the lyrics of {x}?",
    "original broadcaster": "What is the original broadcaster of {x}?",
    "director": "Who directed {x}?",
    "operator": "Who operates {x}?",
    "producer": "Who produced {x}?",
    "military branch": "Which military branch did {x} serve in?",
    "creator": "Who created {x}?",
    "screenwriter": "Who is the screenwriter of {x}?",
    "publisher": "Who is the publisher of {x}?",
    "developer": "Who developed {x}?",
    "capital": "What is the capital of {x}?",
    "member of political party": "Which political party is {x} a member of?",
    "located on terrain feature": "On which terrain feature is {x} located?",
    "capital of": "What is {x} the capital of?",
    "genre": "What is the genre of {x}?",
    "applies to jurisdiction": "Which jurisdiction does {x} apply to?",
    "member of": "What is {x} a member of?",
    "languages spoken, written or signed": "Which language does {x} speak, write or sign?",
    "present in work": "In which work does {x} appear?",
    "original language of film or TV show": "What is the original language of {x}?",
    "located in or next to body of water": "Which body of water is {x} located in or next to?",
    "continent": "Which continent is {x} on?",
    "residence": "What is the residence of {x}?",
    "contains administrative territorial entity": (
        "Which administrative territorial entity does {x} contain?"
    ),
    "licensed to broadcast to": "Which place is {x} licensed to broadcast to?",
    "manufacturer": "Who is the manufacturer of {x}?",
    "main subject": "What is the main subject of {x}?",
    "conflict": "Which conflict was {x} involved in?",
    "part of the series": "Which series is {x} part of?",
    "instrument": "Which instrument does {x} play?",
    "named after": "What is {x} named after?",
    "sport": "Which sport is {x} associated with?",
    "country of origin": "What is the country of origin of {x}?",
    "has part": "What is a part of {x}?",
    "family": "Which family does {x} belong to?",
    "narrative location": "Where is {x} set?",
    "distributed by": "Who distributed {x}?",
    "mouth of the watercourse": "What does {x} flow into?",
    "crosses": "What does {x} cross?",
    "participant": "Who participated in {x}?",
    "award received": "Which award did {x} receive?",
    "winner": "Who won {x}?",
    "language of work or name": "What is the language of {x}?",
    "based on": "What is {x} based on?",
    "field of work": "What is the field of work of {x}?",
    "followed by": "What is {x} followed by?",
    "mother": "Who is the mother of {x}?",
    "movement": "Which movement is {x} associated with?",
    "notable work": "What is a notable work of {x}?",
    "follows": "What does {x} follow?",
    "native language": "What is the native language of {x}?",
    "participant in": "What did {x} participate in?",
    "work location": "Where did {x} work?",
    "position held": "What position did {x} hold?",
    "occupant": "Who is the occupant of {x}?",
    "place served by transport hub": "Which place does {x} serve?",
    "facet of": "What is {x} a facet of?",
    "twinned administrative body": "What is {x} twinned with?",
    "league": "Which league does {x} play in?",
    "subclass of": "What is {x} a subclass of?",
    "parent taxon": "What is the parent taxon of {x}?",
    "legislated by": "Who legislated {x}?",
    "religion": "What is the religion of {x}?",
    "production company": "Which production company made {x}?",
    "occupation": "What is the occupation of {x}?",
    "basin country": "Which country is the basin of {x} in?",
    "drafted by": "Which team drafted {x}?",
}


def render_subquestion(template: str) -> str:
    """``X >> relation`` as a natural question; any other template is returned unchanged.

    ``X`` is kept verbatim (it may be a phrase and may still hold an unresolved
    ``#k``); a relation outside :data:`RELATION_QUESTIONS` gets the generic
    fallback. ``str.replace`` rather than ``str.format`` so braces in ``X``
    can never raise.
    """
    m = _ARROW_RE.match(template.strip())
    if m is None:
        return template
    x, relation = m.group("x").strip(), m.group("relation").strip()
    question = RELATION_QUESTIONS.get(relation, FALLBACK_QUESTION.replace("{relation}", relation))
    return question.replace("{x}", x)


class Paragraph(BaseModel, frozen=True):
    """One passage of a MuSiQue question."""

    idx: int
    title: str
    text: str
    is_supporting: bool


class Hop(BaseModel, frozen=True):
    """One sub-question of the decomposition with its gold answer and passages."""

    template: str  # may contain "#k" references to earlier hops
    answer: str
    support_idx: int
    distractor_idx: list[int]


class MusiqueInstance(Instance):
    """A MuSiQue question as a chain of hops."""

    qid: str
    hops: list[Hop]
    paragraphs: list[Paragraph]
    answer: str
    answer_aliases: list[str]


def sanitize_context(text: str) -> str:
    """Make an answer safe for the prompt header (no quotes or newlines)."""
    return text.replace('"', "'").replace("\n", " ").strip()


def is_linear(row: dict[str, Any]) -> bool:
    """Hop 1 references nothing and every later hop ``k`` references exactly ``#(k-1)``.

    MuSiQue's ``2hop``, ``3hop1`` and ``4hop1`` questions are linear chains; the
    ``3hop2``, ``4hop2`` and ``4hop3`` types fan in or out. The on-track test
    (``has_local_truth``) compares the previous committed answer with gold, which
    is the right test only for linear chains.
    """
    for k, hop in enumerate(row["question_decomposition"], start=1):
        refs = sorted({int(m) for m in _REF_RE.findall(hop["question"])})
        if refs != ([k - 1] if k > 1 else []):
            return False
    return True


def _read_windows(path: str | Path) -> dict[str, list[int]]:
    """Listed windows, ``{split: [absolute indices of the seeded order]}``, from a YAML file.

    Used for windows that are not contiguous, such as fresh questions left after a data-only
    filter. Indices must be non-negative and unique within a split.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    windows: dict[str, list[int]] = {}
    for split, indices in data.items():
        if split not in ("train", "calib", "test"):
            raise ValueError(f"windows file {path}: unknown split {split!r}")
        listed = [int(i) for i in indices]
        if any(i < 0 for i in listed) or len(set(listed)) != len(listed):
            raise ValueError(f"windows file {path}: split {split!r} needs unique indices >= 0")
        windows[str(split)] = listed
    return windows


@lru_cache(maxsize=8)
def _load_rows(
    path: str, hops: tuple[int, ...], linear_only: bool = False
) -> tuple[dict[str, Any], ...]:
    """Answerable rows with the requested hop counts, in file order (cached per process).

    ``linear_only`` keeps linear chains only (see :func:`is_linear`); it is off by
    default so that the 2-hop subsets drawn so far keep their seeded order.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("answerable", True):
                continue
            if len(row["question_decomposition"]) not in hops:
                continue
            if linear_only and not is_linear(row):
                continue
            rows.append(row)
    return tuple(rows)


class MusiqueTask:
    """The MuSiQue multi-hop QA benchmark as a :class:`ucagent.core.protocols.Task`."""

    name = "musique"
    #: Instance ids are ``<id_prefix>-<split>-<index>-s<seed>``; a subclass for another
    #: benchmark in the same row format changes the prefix (the prompt header carries the id).
    id_prefix = "musique"

    def __init__(
        self,
        data_dir: str | Path = "data/musique",
        n_steps: int = 2,
        hops: Sequence[int] | None = None,
        n_instances: dict[str, int] | None = None,
        seed: int = 0,
        n_distractors: int = 3,
        train_file: str = TRAIN_FILE,
        dev_file: str = DEV_FILE,
        render_relations: bool = True,
        local_match: str = "exact",
        offset: dict[str, int] | None = None,
        linear_only: bool = False,
        final_match: str = "exact",
        canonical_dashes: bool = False,
        windows_file: str | Path | None = None,
        **_: Any,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.hops = tuple(int(h) for h in hops) if hops else (int(n_steps),)
        self.sizes = dict(n_instances or {"train": 60, "calib": 60, "test": 60})
        self.windows = _read_windows(windows_file) if windows_file else {}
        self.windows_file = Path(windows_file) if windows_file else None
        for split, indices in self.windows.items():
            if split in (offset or {}):
                raise ValueError(f"split {split!r} has both an offset and a listed window")
            if split in (n_instances or {}) and int((n_instances or {})[split]) != len(indices):
                raise ValueError(
                    f"split {split!r}: n_instances says {(n_instances or {})[split]}, "
                    f"the windows file lists {len(indices)}"
                )
            self.sizes[split] = len(indices)
        self.seed = int(seed)
        self.n_distractors = int(n_distractors)
        self.train_file = train_file
        self.dev_file = dev_file
        self.render_relations = bool(render_relations)
        self.system_prompt = SYSTEM_PROMPT_NL if self.render_relations else SYSTEM_PROMPT
        if local_match not in ("exact", "contains"):
            raise ValueError(f"local_match must be 'exact' or 'contains', not {local_match!r}")
        self.local_match = local_match
        if final_match not in ("exact", "contains"):
            raise ValueError(f"final_match must be 'exact' or 'contains', not {final_match!r}")
        self.final_match = final_match
        self.offsets = {k: int(v) for k, v in (offset or {}).items()}
        self.linear_only = bool(linear_only)
        # off by default so the recorded MuSiQue runs replay unchanged; see parse_answer
        self.canonical_dashes = bool(canonical_dashes)
        if any(v < 0 for v in self.offsets.values()):
            raise ValueError("offset values must be non-negative")

    def input_files(self) -> dict[str, Path]:
        """The files this task reads (``ucagent.experiments.provenance`` records their content)."""
        files = {"train": self.data_dir / self.train_file, "dev": self.data_dir / self.dev_file}
        if self.windows_file is not None:
            files["windows"] = self.windows_file
        return files

    # --- data ---------------------------------------------------------------------

    def _candidates(self, split: str) -> list[dict[str, Any]]:
        """Rows available to a split, in the seeded order instances are drawn from."""
        if split == "train":
            rows = list(
                _load_rows(str(self.data_dir / self.train_file), self.hops, self.linear_only)
            )
            source = "train"
        elif split in ("calib", "test"):
            rows = list(_load_rows(str(self.data_dir / self.dev_file), self.hops, self.linear_only))
            source = "dev"
        else:
            raise KeyError(f"unknown split {split!r}; use train, calib or test")
        order = list(range(len(rows)))
        random.Random(stable_hash("musique", self.seed, source)).shuffle(order)
        half = len(order) // 2
        if split == "calib":
            order = order[:half]
        elif split == "test":
            order = order[half:]
        return [rows[j] for j in order]

    def instance_id(self, split: str, index: int) -> str:
        """Instance id encoding split, index and task seed."""
        return f"{self.id_prefix}-{split}-{index:05d}-s{self.seed}"

    def instances(self, split: str) -> Iterable[MusiqueInstance]:
        """``n_instances[split]`` instances of a split, from index ``offset[split]`` (lazy).

        Ids carry the absolute index of the seeded order, so a window that
        starts at 60 yields ``musique-<split>-00060-s<seed>`` onwards: fresh
        questions for a confirmatory calibration that never overlap the first 60.
        A split listed in the windows file yields exactly the listed indices, in order.
        """
        available = len(self._candidates(split))
        if split in self.windows:
            listed = self.windows[split]
            if listed and max(listed) >= available:
                raise ValueError(
                    f"split {split!r} has {available} questions, the windows file lists "
                    f"index {max(listed)}"
                )
            for index in listed:
                yield self.get_instance(self.instance_id(split, index))
            return
        n = self.sizes[split]
        start = self.offsets.get(split, 0)
        if start + n > available:
            raise ValueError(
                f"split {split!r} has {available} questions, {n} requested from index {start}"
            )
        for index in range(start, start + n):
            yield self.get_instance(self.instance_id(split, index))

    def get_instance(self, instance_id: str) -> MusiqueInstance:
        """Build an instance from its id (cached)."""
        return self._cached(instance_id)

    @lru_cache(maxsize=4096)  # noqa: B019 - instances are immutable, task is long-lived
    def _cached(self, instance_id: str) -> MusiqueInstance:
        m = re.match(_ID_PATTERN.format(prefix=re.escape(self.id_prefix)), instance_id)
        if m is None or int(m.group("seed")) != self.seed:
            raise KeyError(f"not an instance of this task: {instance_id!r}")
        split, index = m.group("split"), int(m.group("index"))
        row = self._candidates(split)[index]
        rng = random.Random(stable_hash("musique", self.seed, row["id"], "distractors"))
        hops: list[Hop] = []
        for h in row["question_decomposition"]:
            support = int(h["paragraph_support_idx"])
            others = [
                int(p["idx"])
                for p in row["paragraphs"]
                if int(p["idx"]) != support and not p["is_supporting"]
            ]
            picked = rng.sample(others, min(self.n_distractors, len(others)))
            hops.append(
                Hop(
                    template=str(h["question"]),
                    answer=str(h["answer"]),
                    support_idx=support,
                    distractor_idx=picked,
                )
            )
        paragraphs = [
            Paragraph(
                idx=int(p["idx"]),
                title=str(p["title"]),
                text=str(p["paragraph_text"]),
                is_supporting=bool(p["is_supporting"]),
            )
            for p in row["paragraphs"]
        ]
        return MusiqueInstance(
            instance_id=instance_id,
            split=split,
            index=index,
            question=str(row["question"]),
            n_steps=len(hops),
            qid=str(row["id"]),
            hops=hops,
            paragraphs=paragraphs,
            answer=str(row["answer"]),
            answer_aliases=[str(a) for a in row.get("answer_aliases") or []],
        )

    # --- protocol -----------------------------------------------------------------

    def n_steps(self, instance: Instance) -> int:
        """Number of hops."""
        return instance.n_steps

    def initial_context(self, instance: Instance) -> str:
        """Nothing is committed before hop 1."""
        return ""

    def sub_question(self, instance: Instance, i: int, answers: dict[int, str]) -> str:
        """Sub-question ``i`` with ``#k`` replaced by committed answers, then rendered.

        An unresolved ``#k`` stays visible inside the rendered question.
        """
        template = self._inst(instance).hops[i - 1].template
        text = _REF_RE.sub(lambda m: answers.get(int(m.group(1)), m.group(0)), template)
        return render_subquestion(text) if self.render_relations else text

    def passages(self, instance: Instance, i: int) -> list[Paragraph]:
        """Supporting plus distractor paragraphs of hop ``i`` in a seeded order."""
        inst = self._inst(instance)
        hop = inst.hops[i - 1]
        by_idx = {p.idx: p for p in inst.paragraphs}
        chosen = [by_idx[hop.support_idx], *(by_idx[j] for j in hop.distractor_idx)]
        random.Random(stable_hash("musique", self.seed, inst.qid, i, "order")).shuffle(chosen)
        return chosen

    def step_prompt(
        self, instance: Instance, i: int, history: Sequence[StepRecord], attempt: int
    ) -> list[BaseMessage]:
        """System + human messages for hop ``i`` (header first, passages last).

        With ``render_relations`` the overall question is labelled as context and
        the sub-question as the thing to answer: a capable model otherwise answers
        the overall question at hop 1 (seen with gpt-oss:20b).
        """
        inst = self._inst(instance)
        answers = {rec.step: rec.answer for rec in history}
        current = history[-1].answer if history else ""
        chain = "; ".join(f"#{k}: {answers[k]}" for k in sorted(answers)) or "(none yet)"
        sub = self.sub_question(instance, i, answers)
        if self.render_relations:  # the overall question is context; the sub-question is the task
            ask = [
                f"Overall question (context only, do not answer it): {inst.question}",
                f"Answers so far: {chain}",
                f"Sub-question for step {i} of {inst.n_steps} (answer this one): {sub}",
            ]
        else:  # legacy layout of the archived llama3.1:8b runs (byte-identical prompts)
            ask = [
                f"Question: {inst.question}",
                f"Answers so far: {chain}",
                f"Step {i} of {inst.n_steps}: {sub}",
            ]
        lines = [
            render_header(inst.instance_id, i, sanitize_context(current), attempt),
            *ask,
            "Passages:",
            *(f"[{p.title}] {p.text}" for p in self.passages(instance, i)),
        ]
        return [SystemMessage(content=self.system_prompt), HumanMessage(content="\n".join(lines))]

    def parse_answer(self, text: str) -> str:
        """``Answer: <answer>`` on any line, else the first non-empty line.

        A confidence the model put on the same line (``... >> Confidence: 0.8``)
        is cut off, as is a trailing ``>>`` left over from the sub-question syntax.
        """
        m = _ANSWER_RE.search(text)
        raw = m.group(1) if m else next((ln for ln in text.splitlines() if ln.strip()), "")
        answer = _clean_answer(raw)
        # canonical_dashes: "Hohenlohe\u2011Langenburg" and "Hohenlohe-Langenburg" are one answer
        return canonical_dashes(answer) if self.canonical_dashes else answer

    def _truths(self, inst: MusiqueInstance, i: int) -> set[str]:
        golds = [inst.hops[i - 1].answer]
        if i == inst.n_steps:
            golds += list(inst.answer_aliases)
        truths = {normalize_answer(g) for g in golds}
        if self.canonical_dashes:  # a hyphen also separates words ("Zweibrucken Birkenfeld")
            truths |= {normalize_answer(canonical_dashes(g).replace("-", " ")) for g in golds}
        return truths

    def _matches(self, answer: str, truths: set[str], rule: str | None = None) -> bool:
        """Label rule: exact match, or whole-word containment either way.

        ``rule`` is ``local_match`` (the default, for the per-hop labels and the on-track
        test) or ``final_match`` (for the final answer). With ``"contains"``,
        "Greenfield, Indiana" counts for gold "Greenfield".
        """
        forms = [normalize_answer(answer)]
        if self.canonical_dashes:
            dashed = canonical_dashes(answer)
            forms = [normalize_answer(dashed), normalize_answer(dashed.replace("-", " "))]
        if any(f in truths for f in forms):
            return True
        if (rule or self.local_match) != "contains" or not forms[0]:
            return False
        return any(_contains.same(f, t, None) for f in forms for t in truths if t)

    def has_local_truth(self, instance: Instance, i: int, current: str) -> bool:
        """True while the chain is on track (hop 1, or the previous answer is gold)."""
        if i <= 1:
            return True
        inst = self._inst(instance)
        return self._matches(current, self._truths(inst, i - 1))

    def local_truth(self, instance: Instance, i: int, current: str) -> str:
        """The gold hop answer (meaningful only when ``has_local_truth``)."""
        return self._inst(instance).hops[i - 1].answer

    def global_truth(self, instance: Instance, i: int) -> str:
        """The gold answer of hop ``i``."""
        return self._inst(instance).hops[i - 1].answer

    def distractor_names(self, instance: Instance, i: int, current: str) -> list[str]:
        """Titles of the distractor paragraphs shown for hop ``i`` (for the mock)."""
        inst = self._inst(instance)
        truth = normalize_answer(inst.hops[i - 1].answer)
        by_idx = {p.idx: p for p in inst.paragraphs}
        names = [by_idx[j].title for j in inst.hops[i - 1].distractor_idx]
        return [n for n in dict.fromkeys(names) if normalize_answer(n) != truth]

    def random_entity(self, instance: Instance, rng: random.Random) -> str:
        """A random paragraph title (hallucination stand-in for the mock)."""
        return rng.choice(self._inst(instance).paragraphs).title

    def is_final_correct(self, instance: Instance, answer: str | None) -> bool:
        """The final label under ``final_match``.

        ``"exact"`` is MuSiQue's exact match against the answer and its aliases;
        ``"contains"`` is the whole-word containment rule of the per-hop labels.
        """
        if answer is None:
            return False
        inst = self._inst(instance)
        return self._matches(answer, self._truths(inst, inst.n_steps), self.final_match)

    def is_locally_correct(self, instance: Instance, i: int, current: str, answer: str) -> bool:
        """On track and matching the gold hop answer (aliases count on the final hop).

        The per-hop rule is ``local_match``; the final label :meth:`is_final_correct`
        follows ``final_match``, which is exact match unless the task says otherwise.
        """
        if not self.has_local_truth(instance, i, current):
            return False
        return self._matches(answer, self._truths(self._inst(instance), i))

    @staticmethod
    def _inst(instance: Instance) -> MusiqueInstance:
        if not isinstance(instance, MusiqueInstance):
            raise TypeError("MusiqueTask needs MusiqueInstance objects")
        return instance
