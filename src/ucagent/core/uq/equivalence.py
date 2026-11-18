"""Semantic-equivalence judges and greedy union-find clustering of samples.

``NormalizedExactMatch`` is the default and is sufficient for entity and
number answers. ``LLMJudge`` asks a (cached) chat model; ``NLIJudge`` uses a
small cross-encoder behind the optional ``nli`` extra and is never imported
unless installed. The judge is the extension point for structured outputs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ucagent.core.records import Cluster, Sample
from ucagent.core.uq.normalize import normalize_answer


class EquivalenceJudge(Protocol):
    """Decides whether two answers mean the same thing."""

    name: str

    def same(self, a: str, b: str, context: Any) -> bool:
        """Whether ``a`` and ``b`` are equivalent answers in this context."""
        ...


class NormalizedExactMatch:
    """Equivalent iff the normalised strings are equal."""

    name = "normalized_exact"

    def same(self, a: str, b: str, context: Any) -> bool:
        """Compare after :func:`normalize_answer`."""
        return normalize_answer(a) == normalize_answer(b)


class NormalizedContains:
    """Equivalent iff normalised strings are equal or one is a whole-word part of the other.

    For clustering free-text answers ("Bombardier" and "Bombardier Inc."); labels
    keep using exact match.
    """

    name = "normalized_contains"

    def same(self, a: str, b: str, context: Any) -> bool:
        """Normalised equality or whole-word containment either way."""
        na, nb = normalize_answer(a), normalize_answer(b)
        if na == nb:
            return True
        if not na or not nb:
            return False
        short, long_ = sorted((na, nb), key=len)
        return re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long_) is not None


_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d-"


def answer_tokens(text: str) -> set[str]:
    """Normalised words of an answer, a hyphen (any dash) counting as a space."""
    for ch in _DASH_CHARS:
        text = text.replace(ch, " ")
    return set(normalize_answer(text).split())


class NormalizedTokens(NormalizedContains):
    """Containment, or every word of one answer among the words of the other.

    Clusters name variants that containment splits ("Albert I, King of Germany" and
    "Albert I of Germany"); dashes separate words. The label rule of the ``twowiki``
    task is the same, so that clustering and labels agree.
    """

    name = "normalized_tokens"

    def same(self, a: str, b: str, context: Any) -> bool:
        """Containment either way, or one answer's words a subset of the other's."""
        if super().same(a, b, context):
            return True
        ta, tb = answer_tokens(a), answer_tokens(b)
        return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
NUMERIC_REL_TOL = 0.01


def parse_number(text: str) -> tuple[float, bool] | None:
    """The first number in ``text`` and whether it is written as a percentage.

    Thousands separators and currency signs are ignored, any dash before the digits is a
    minus sign, and an accounting form "(5.2)" is negative.
    """
    t = text
    for ch in _DASH_CHARS:
        t = t.replace(ch, "-")
    t = t.replace(",", "").replace("$", "").strip()
    t = re.sub(r"^\((.*)\)$", r"-\1", t)
    m = _NUMBER_RE.search(t.replace(" ", ""))
    if m is None:
        return None
    return float(m.group(0)), "%" in t


class NumericJudge:
    """Two answers are equivalent iff their numbers agree within a relative tolerance."""

    name = "numeric"

    def same(self, a: str, b: str, context: Any) -> bool:
        """Numbers within ``NUMERIC_REL_TOL`` of each other; otherwise equal stripped strings."""
        pa, pb = parse_number(a), parse_number(b)
        if pa is None or pb is None:
            return a.strip().lower() == b.strip().lower()
        x, y = pa[0], pb[0]
        return abs(x - y) <= max(NUMERIC_REL_TOL * max(abs(x), abs(y)), 5e-4)


class LLMJudge:
    """Asks a chat model whether two answers are the same; exact matches short-circuit.

    Every call goes through the model given, so with the cached backend of
    milestone M4 the calls are cached and counted by the cost guard.
    """

    name = "llm_judge"
    prompt = (
        "Do the following two answers refer to the same entity or value? "
        "Reply with exactly one word: yes or no.\nAnswer A: {a}\nAnswer B: {b}"
    )

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._exact = NormalizedExactMatch()

    def same(self, a: str, b: str, context: Any) -> bool:
        """Exact match first, then one model call."""
        if self._exact.same(a, b, context):
            return True
        reply = self.llm.invoke([HumanMessage(content=self.prompt.format(a=a, b=b))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        return text.strip().lower().startswith("yes")


class NLIJudge:
    """Bidirectional entailment with a cross-encoder (optional extra ``nli``)."""

    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("NLIJudge needs the 'nli' extra: uv sync --extra nli") from exc
        self.model = CrossEncoder(model_name)
        self._exact = NormalizedExactMatch()

    def _entails(self, premise: str, hypothesis: str) -> bool:
        scores = self.model.predict([(premise, hypothesis)])[0]
        labels = ["contradiction", "entailment", "neutral"]
        return labels[int(scores.argmax())] == "entailment"

    def same(self, a: str, b: str, context: Any) -> bool:
        """Exact match, else entailment in both directions."""
        if self._exact.same(a, b, context):
            return True
        return self._entails(a, b) and self._entails(b, a)


JudgeFactory = Callable[..., EquivalenceJudge]
JUDGES: dict[str, JudgeFactory] = {
    "normalized_exact": lambda **_: NormalizedExactMatch(),
    "normalized_contains": lambda **_: NormalizedContains(),
    "normalized_tokens": lambda **_: NormalizedTokens(),
    "numeric": lambda **_: NumericJudge(),
    "llm_judge": lambda llm=None, **_: LLMJudge(llm),
    "nli": lambda **_: NLIJudge(),
}


def make_judge(name: str, **kwargs: Any) -> EquivalenceJudge:
    """Instantiate a registered judge by name (``llm=`` is passed to judges that need it)."""
    try:
        factory = JUDGES[name]
    except KeyError as exc:
        raise KeyError(f"unknown equivalence judge {name!r}; known: {sorted(JUDGES)}") from exc
    return factory(**kwargs)


def cluster_samples(
    samples: Sequence[Sample], judge: EquivalenceJudge, context: Any
) -> list[Cluster]:
    """Greedy union-find clustering over pairwise judgments.

    Samples are processed in ``sample_index`` order so the result does not
    depend on the order in which parallel samples arrived. Clusters are sorted
    by size (descending) then by their first member, and the representative
    is the answer of the first member.
    """
    ordered = sorted(samples, key=lambda s: s.sample_index)
    parent = list(range(len(ordered)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(ordered)):
        for b in range(a + 1, len(ordered)):
            ra, rb = find(a), find(b)
            if ra != rb and judge.same(ordered[a].answer, ordered[b].answer, context):
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for idx in range(len(ordered)):
        groups.setdefault(find(idx), []).append(idx)
    k = max(1, len(ordered))
    clusters = [
        Cluster(
            representative=ordered[members[0]].answer,
            members=[ordered[m].sample_index for m in members],
            proportion=len(members) / k,
        )
        for members in groups.values()
    ]
    clusters.sort(key=lambda c: (-c.proportion, c.members[0]))
    return clusters
