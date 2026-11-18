"""2WikiMultiHopQA (Ho et al., 2020) as a two-hop chain ``Task``, in MuSiQue's row format.

2WikiMultiHopQA questions carry their reasoning path as Wikidata evidence triples
(subject, relation, object). Compositional and inference questions whose two triples
form a chain (the second triple's subject is the first triple's object) are linear
two-hop questions: hop 1 asks for the first relation of the first subject, hop 2 for
the second relation of the hop-1 answer. :func:`chain_rows` writes them in MuSiQue's
row format (sub-questions ``X >> relation`` and ``#1 >> relation``, the supporting
paragraph of each hop, the record's other paragraphs as distractors), so that the
MuSiQue adapter's passages, prompt layout, labels, on-track test and seeded splits
apply unchanged. :class:`TwoWikiTask` changes three things:

- the instance-id prefix (``twowiki-<split>-<index>-s<seed>``);
- question templates for the relations MuSiQue's table lacks (dates of birth and
  death, burial, cause of death, inception and a few others);
- a date-aware label rule: a date-valued gold answer also matches an answer that
  names the same date in another format ("June 12, 1516" for "12 June 1516");
- ``canonical_dashes`` on by default: Unicode hyphens in answers become ASCII hyphens,
  and a hyphen also separates words when labels are matched (the smoke run's audit
  found both as label artefacts: "Hohenlohe\u2011Langenburg", and
  "Zweibrücken-Birkenfeld" for gold "Zweibrücken").

Comparison and bridge-comparison questions combine two lookups and are not chains,
so they are left out: the framework labels a step against the previous step's
answer only. Hop 1 names its subject by the supporting paragraph's title, which
disambiguates films ("Polish-Russian War (film)").

Data: the dev split (``dev.parquet`` of the Hugging Face mirror
``xanhho/2WikiMultihopQA``, Apache 2.0), converted by ``scripts/prepare_twowiki.py``
into ``twowiki_train.jsonl`` (the pool for the calibration fit) and
``twowiki_dev.jsonl`` (calibration and test are its seeded halves).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ucagent.core.protocols import Instance
from ucagent.core.uq.equivalence import answer_tokens
from ucagent.tasks.musique import (
    _ARROW_RE,
    _REF_RE,
    FALLBACK_QUESTION,
    RELATION_QUESTIONS,
    MusiqueInstance,
    MusiqueTask,
)

CHAIN_TYPES = ("compositional", "inference")
#: Relations left out because a single gold answer misjudges right answers (smoke audits):
#: several valid values (spouse, child, sibling, education, employer, awards, parts),
#: mixed forms of one value (citizenship: "American", "America", "United States"), or
#: synonyms (cause of death), plus rare relations of the same kinds.
AMBIGUOUS_RELATIONS = frozenset(
    {
        "spouse",
        "child",
        "sibling",
        "educated at",
        "employer",
        "award received",
        "has part",
        "country of citizenship",
        "country",
        "cause of death",
        "student of",
        "place of detention",
    }
)
TRAIN_FILE = "twowiki_train.jsonl"
DEV_FILE = "twowiki_dev.jsonl"

#: Templates for relations of the 2WikiMultiHopQA chains that MuSiQue's table lacks;
#: the rest come from :data:`ucagent.tasks.musique.RELATION_QUESTIONS`.
TWOWIKI_RELATIONS: dict[str, str] = {
    "date of birth": "When was {x} born?",
    "date of death": "When did {x} die?",
    "place of burial": "Where was {x} buried?",
    "cause of death": "What was the cause of death of {x}?",
    "inception": "When was {x} founded?",
    "student of": "Whose student was {x}?",
    "place of detention": "Where was {x} detained?",
    "editor": "Who is the editor of {x}?",
    "presenter": "Who presented {x}?",
    "doctoral advisor": "Who was the doctoral advisor of {x}?",
    "nominated for": "What was {x} nominated for?",
    "religion": "What is the religion of {x}?",
}

_YEAR_RE = re.compile(r"\b\d{3,4}\b")


#: Words that a sentence stating the relation usually contains (triage for the label audit).
RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "place of birth": ("born", "birth", "native of"),
    "date of birth": ("born", "birth"),
    "place of death": ("died", "death", "killed", "murdered", "executed"),
    "date of death": ("died", "death", "killed", "murdered", "executed"),
    "place of burial": ("buried", "burial", "interred", "grave", "tomb"),
    "father": ("son", "daughter", "father", "child", "born to"),
    "mother": ("son", "daughter", "mother", "child", "born to"),
}
_PLACE_RELATIONS = frozenset({"place of birth", "place of death", "place of burial"})
_KEYWORD_FIRST = frozenset({"place of birth", "place of death", "place of burial"})
_DEMONYM_RE = re.compile(r"^[A-Z][a-z]+(?:ish|ese|ian|i|ean|er)$")
_DEMONYMS = frozenset(
    {"American", "German", "Mexican", "Italian", "Dutch", "Swiss", "Greek", "Thai", "French"}
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def support_flags(relation: str, gold: str, paragraph: str) -> list[str]:
    """Reasons to doubt that ``paragraph`` states ``relation`` with value ``gold`` (triage only).

    - ``keyword_missing``: no sentence containing the gold has a word of the relation;
    - ``keyword_after``: for place relations, the gold never follows such a word in its
      sentence ("his parents were from France but he was born in Mexico" for gold France);
    - ``demonym``: a place relation whose gold is a nationality adjective ("British").

    An empty list is no evidence of support: a sentence can contain both and not state it.
    Relations without a keyword list get only the ``demonym`` check.
    """
    flags: list[str] = []
    word = gold.strip()
    if relation in _PLACE_RELATIONS and (word in _DEMONYMS or _DEMONYM_RE.match(word)):
        flags.append("demonym")
    keys = RELATION_KEYWORDS.get(relation)
    if not keys or not word:
        return flags
    low = word.lower()
    sentences = [x.lower() for x in _SENTENCE_RE.split(paragraph) if low in x.lower()]
    if not sentences:
        return flags
    if not any(any(k in x for k in keys) for x in sentences):
        flags.append("keyword_missing")
    elif relation in _KEYWORD_FIRST and not any(
        any(k in x and x.index(k) < x.index(low) for k in keys) for x in sentences
    ):
        flags.append("keyword_after")
    return flags


def unchecked(relation: str, gold: str, paragraph: str) -> bool:
    """True when :func:`support_flags` cannot test a keyword relation: no sentence has the gold.

    The sentence split also breaks at abbreviations, so a gold such as "St. Louis" or
    "John C. Breckinridge" lies across two pieces and its keyword is never looked for.
    """
    low = gold.strip().lower()
    return (
        bool(RELATION_KEYWORDS.get(relation))
        and bool(low)
        and not any(low in x.lower() for x in _SENTENCE_RE.split(paragraph))
    )


def question_flags(instance: MusiqueInstance) -> list[str]:
    """The data-only filter of a question: each hop's support flags, plus ``unchecked``.

    Flags are ``hop<k>:<flag>``. A question without flags has passages that state each gold
    in a sentence with a word of its relation, as far as the triage can tell; split 3 keeps
    only such questions (user decision of 2026-11-09).
    """
    out: list[str] = []
    for k, hop in enumerate(instance.hops, start=1):
        relation = hop.template.split(">>")[-1].strip() if ">>" in hop.template else ""
        text = next(p.text for p in instance.paragraphs if p.idx == hop.support_idx)
        flags = support_flags(relation, hop.answer, text)
        if unchecked(relation, hop.answer, text):
            flags.append("unchecked")
        out += [f"hop{k}:{f}" for f in flags]
    return out


def render_relation(template: str) -> str:
    """``X >> relation`` as a question (2WikiMultiHopQA templates first, then MuSiQue's)."""
    m = _ARROW_RE.match(template.strip())
    if m is None:
        return template
    x, relation = m.group("x").strip(), m.group("relation").strip()
    question = TWOWIKI_RELATIONS.get(relation) or RELATION_QUESTIONS.get(relation)
    if question is None:
        question = FALLBACK_QUESTION.replace("{relation}", relation)
    return question.replace("{x}", x)


def uses_fallback(relation: str) -> bool:
    """True when ``relation`` has no template and gets the generic question."""
    return relation not in TWOWIKI_RELATIONS and relation not in RELATION_QUESTIONS


def _date_parts(text: str) -> dict[str, int] | None:
    """Year, month and day stated in ``text`` (parts absent from the text are left out)."""
    if not _YEAR_RE.search(text):
        return None
    from dateutil import parser  # type: ignore[import-untyped]

    try:
        low = parser.parse(text, default=datetime(1, 1, 1))
        high = parser.parse(text, default=datetime(2, 2, 2))
    except (ValueError, OverflowError, TypeError):
        return None
    parts = {
        name: getattr(low, name)
        for name in ("year", "month", "day")
        if getattr(low, name) == getattr(high, name)
    }
    return parts if "year" in parts else None


def same_date(answer: str, gold: str) -> bool:
    """``answer`` states every date part that ``gold`` states, with the same values."""
    want = _date_parts(gold)
    if want is None:
        return False
    got = _date_parts(answer)
    return got is not None and all(got.get(k) == v for k, v in want.items())


def _decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def chain_rows(
    records: Iterable[Mapping[str, Any]],
    exclude_relations: Iterable[str] = AMBIGUOUS_RELATIONS,
    exclude_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """MuSiQue-format rows for the two-hop chain questions among raw records.

    Kept: type compositional or inference; exactly two evidence triples with the second
    subject equal to the first object and neither relation in ``exclude_relations``; two
    distinct supporting titles present among the paragraphs; the final answer equal to the
    second object; an id not in ``exclude_ids`` (questions already looked at). Everything
    else is skipped.
    """
    skip_rel, skip_ids = set(exclude_relations), set(exclude_ids)
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("type") not in CHAIN_TYPES or str(rec.get("_id")) in skip_ids:
            continue
        evidences = _decode(rec["evidences"])
        facts = _decode(rec["supporting_facts"])
        context = _decode(rec["context"])
        if len(evidences) != 2 or evidences[1][0] != evidences[0][2]:
            continue
        if {str(evidences[0][1]), str(evidences[1][1])} & skip_rel:
            continue
        if str(rec["answer"]) != str(evidences[1][2]):
            continue
        titles = list(dict.fromkeys(str(t) for t, _ in facts))
        paragraphs: list[dict[str, Any]] = []
        index_of: dict[str, int] = {}
        for idx, (title, sentences) in enumerate(context):
            text = " ".join(str(s).strip() for s in sentences).strip()
            paragraphs.append(
                {
                    "idx": idx,
                    "title": str(title),
                    "paragraph_text": text,
                    "is_supporting": str(title) in titles,
                }
            )
            index_of.setdefault(str(title), idx)
        if len(titles) != 2 or any(t not in index_of for t in titles):
            continue
        (_, rel1, obj1), (_, rel2, obj2) = evidences
        out.append(
            {
                "id": str(rec["_id"]),
                "type": str(rec["type"]),
                "question": str(rec["question"]),
                "question_decomposition": [
                    {
                        "id": 1,
                        "question": f"{titles[0]} >> {rel1}",
                        "answer": str(obj1),
                        "paragraph_support_idx": index_of[titles[0]],
                    },
                    {
                        "id": 2,
                        "question": f"#1 >> {rel2}",
                        "answer": str(obj2),
                        "paragraph_support_idx": index_of[titles[1]],
                    },
                ],
                "paragraphs": paragraphs,
                "answer": str(rec["answer"]),
                "answer_aliases": [],
                "answerable": True,
            }
        )
    return out


class TwoWikiTask(MusiqueTask):
    """2WikiMultiHopQA two-hop chains as a :class:`ucagent.core.protocols.Task`."""

    name = "twowiki"
    id_prefix = "twowiki"

    def __init__(
        self,
        data_dir: str | Path = "data/twowiki",
        train_file: str = TRAIN_FILE,
        dev_file: str = DEV_FILE,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("canonical_dashes", True)  # gpt-oss mixes U+2011 and "-" in names
        super().__init__(data_dir=data_dir, train_file=train_file, dev_file=dev_file, **kwargs)

    def sub_question(self, instance: Instance, i: int, answers: dict[int, str]) -> str:
        """Sub-question ``i`` with ``#1`` replaced by the committed answer, then rendered."""
        template = self._inst(instance).hops[i - 1].template
        text = _REF_RE.sub(lambda m: answers.get(int(m.group(1)), m.group(0)), template)
        return render_relation(text) if self.render_relations else text

    def _matches(self, answer: str, truths: set[str], rule: str | None = None) -> bool:
        """MuSiQue's rule, every gold word in the answer, or the same date written otherwise.

        The word rule takes name variants ("Albert I, King of Germany" for "Albert I of
        Germany"); like containment, it applies only under the ``contains`` rules.
        """
        if super()._matches(answer, truths, rule):
            return True
        if (rule or self.local_match) == "contains":
            words = answer_tokens(answer)
            if words and any(t and set(t.split()) <= words for t in truths):
                return True
        return any(same_date(answer, t) for t in truths if t)
