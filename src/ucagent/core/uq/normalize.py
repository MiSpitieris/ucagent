"""Answer normalisation shared by equivalence judges and correctness checks."""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents, punctuation, articles and extra whitespace.

    Numbers lose thousands separators (``1,000`` -> ``1000``) so that numeric
    answers compare by value as written.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()
