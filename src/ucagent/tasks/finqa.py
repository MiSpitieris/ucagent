"""FinQA (Chen et al., 2021) as a multi-step numeric ``Task``: one step per program operation.

Each FinQA question over a financial report comes with a gold reasoning program such as
``subtract(5829, 5735), divide(#0, 5735)``. Linear programs (every operation after the
first uses the previous result ``#(k-1)`` and otherwise only numbers from the report or
constants) become chains: step k asks for the result of operation k. A step names its
arguments without giving their values: a table number by its row and column ("the net
revenue of 2019"), a number from the text by its sentence with the number blanked out, the
previous result by the committed answer of the previous step (as MuSiQue substitutes
``#k``). So each step is a retrieval and a computation. The evidence shown is the
question's supporting facts plus seeded distractor facts from the same report, in a seeded
order (the analogue of MuSiQue's gold paragraph plus distractors).

Labels are numeric: an answer matches a gold value within a relative tolerance, also as a
percentage of it or as a fraction of it ("1.64%" for 0.0164), or when it equals the gold
rounded to the answer's own precision with at least two significant digits. Programs with
table aggregations or comparisons are left out. Every program is executed here, and a
question whose execution does not reproduce FinQA's ``exe_ans`` is skipped.

Files: ``train.json`` (the fit pool), ``dev.json`` and ``test.json`` (calibration and test
are seeded halves of their union) from https://github.com/czyssrs/FinQA (MIT License),
downloaded to ``data_dir``.

Two protocols. ``protocol: 1`` (the default) is the adapter as first run on 2026-11-09, kept
so that every recorded number reproduces. Reviews of 2026-11-11 found three defects in it: the
label rule accepts answers off by a factor of 100 and zero for a small value; the simulated
reviewer writes values of a million or more in exponent notation, which the rule misreads; and
some step descriptions do not single out their number. ``protocol: 2`` repairs the three:

- A number without a percent sign is right when it is the gold value (relative tolerance, or
  the gold rounded to the answer's own precision). No factor-100 forms, no absolute tolerance.
- A number written with a percent sign is right as the percent number or as the fraction,
  because FinQA's programs use both conventions and the sign says that it is a percentage.
- The final answer of a question that FinQA itself answers with a percentage may also be
  given without the sign, as the fraction or as the percent number; FinQA's answer string
  says which of the two the program's result is, and a signed answer must then be the
  percent number.
- The reviewer writes plain decimals.
- A description that fails ``description_issue`` is replaced by the quoted fact with the
  number blanked (or by the number itself where two facts differ in nothing else).

Under both protocols the same questions are loaded in the same order and the same facts are
shown, and a question that no defect touches gets the same prompts, so most samples of the
first run replay from the cache. The admission test of ``convert`` therefore keeps the first
protocol's rule and parser.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from ucagent.core.llm.mock import render_header
from ucagent.core.protocols import Instance
from ucagent.core.records import StepRecord
from ucagent.core.uq.equivalence import NUMERIC_REL_TOL as REL_TOL
from ucagent.core.uq.equivalence import parse_number
from ucagent.core.utils import stable_hash
from ucagent.tasks.musique import _clean_answer, canonical_dashes, sanitize_context

SYSTEM_PROMPT = (
    "You compute one step of a calculation over facts from a financial report. "
    "Use only the facts given and the results of earlier steps. "
    "Reply with the number only (no units, no currency, no thousands separators; a "
    "percentage may end with %), on the first line as 'Answer: <number>', then your "
    "confidence that it is right as 'Confidence: <number between 0 and 1>'."
)
OPS = {"add": "plus", "subtract": "minus", "multiply": "times", "divide": "divided by"}
_OP_RE = re.compile(r"(\w+)\(([^)]*)\)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ID_RE = re.compile(r"^finqa-(?P<split>[a-z]+)-(?P<index>\d+)-s(?P<seed>-?\d+)$")
_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def _decimals(text: str) -> int:
    m = re.search(r"\d+\.(\d+)", text.replace(",", ""))
    return len(m.group(1)) if m else 0


def _sig_digits(text: str) -> int:
    digits = re.sub(r"\D", "", (_NUM_RE.search(text.replace(",", "")) or [""])[0])
    return len(digits.lstrip("0"))


def numbers_match(answer: str, gold: float) -> bool:
    """``answer`` states ``gold``: within tolerance, as a percentage or fraction, or rounded."""
    parsed = parse_number(answer)
    if parsed is None or not math.isfinite(gold):
        return False
    value, _ = parsed
    for g in (gold, gold * 100.0, gold / 100.0):
        if abs(value - g) <= max(REL_TOL * abs(g), 5e-4):
            return True
        if _sig_digits(answer) >= 2 and round(g, _decimals(answer)) == value:
            return True
    return False


def _value(arg: str, results: list[float]) -> float:
    if arg.startswith("#"):
        return results[int(arg[1:])]
    if arg.startswith("const_"):
        name = arg[6:]
        return -1.0 if name == "m1" else float(name)
    parsed = parse_number(arg)
    if parsed is None:
        raise ValueError(f"not a number: {arg!r}")
    return parsed[0] / 100.0 if parsed[1] else parsed[0]


def execute(ops: Sequence[tuple[str, list[str]]]) -> list[float]:
    """Results of each operation of a program."""
    results: list[float] = []
    for op, args in ops:
        a, b = (_value(x, results) for x in args)
        if op == "add":
            results.append(a + b)
        elif op == "subtract":
            results.append(a - b)
        elif op == "multiply":
            results.append(a * b)
        elif op == "divide":
            results.append(a / b if b else math.nan)
        else:
            raise ValueError(f"operation {op!r} not supported")
    return results


def _facts(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Gold facts (FinQA's supporting facts) and the report's other facts, linearised."""
    gold = [str(v).strip() for v in row["qa"].get("gold_inds", {}).values()]
    others: list[str] = [s.strip() for s in [*row["pre_text"], *row["post_text"]] if len(s) > 20]
    table = row.get("table") or []
    if table:
        header = table[0]
        for r in table[1:]:
            cells = [
                f"the {header[j]} of {r[0]} is {r[j]}"
                for j in range(1, min(len(r), len(header)))
                if r[j]
            ]
            if cells:
                others.append(" ; ".join(cells) + " ;")
    gold_set = set(gold)
    return gold, [f for f in others if f not in gold_set and not any(f in g for g in gold)]


def describe(arg: str, gold: Sequence[str]) -> str:
    """A description of a literal argument that does not state its value."""
    if arg.startswith("const_"):
        return _value(arg, []).__format__("g")
    token = arg.strip()
    pattern = re.compile(rf"(?<![\d.]){re.escape(token)}(?![\d.]|\d)")
    for fact in gold:
        if not pattern.search(fact):
            continue
        for segment in fact.split(";"):
            if pattern.search(segment) and " is " in segment:
                return pattern.sub("___", segment.split(" is ")[0].strip())  # labels may repeat it
        blanked = pattern.sub("___", fact)  # every occurrence ("17% ( 17 % )")
        return f'the number given in "{blanked}"'
    return token  # not found among the facts: the program's own constant


_PLAIN_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def program_ops(program: str) -> list[tuple[str, list[str]]]:
    """A FinQA program as ``[(operation, [argument, ...]), ...]``."""
    return [(op, [a.strip() for a in args.split(",")]) for op, args in _OP_RE.findall(program)]


def table_head(row: dict[str, Any]) -> str:
    """The first header cell of a record's table; FinQA writes it before a row's first cell."""
    table = row.get("table") or []
    return str(table[0][0]).strip() if table and table[0] else ""


def is_table_row(fact: str) -> bool:
    """A fact that is a table row: each of its cells ends with ';' (a sentence does not)."""
    return fact.rstrip().endswith(";")


def _stand_alone(token: str) -> re.Pattern[str]:
    """``token`` where it stands as a number of its own: what ``describe`` blanks."""
    return re.compile(rf"(?<![\d.]){re.escape(token)}(?![\d.]|\d)")


def _without_head(label: str, head: str) -> str:
    """``label`` without the table's head cell (FinQA respaces it: "31," becomes "31 ,")."""
    label = label.strip()
    target = "".join(head.split())
    if not target:
        return label
    seen = 0
    for pos, ch in enumerate(label):
        if ch.isspace():
            continue
        if seen == len(target):
            return label[pos:].strip()
        if ch != target[seen]:
            return label
        seen += 1
    return label


def _magnitudes(text: str) -> set[float]:
    return {float(n) for n in _PLAIN_NUM_RE.findall(text)}


def _other_number_fits(blanked: str, texts: Sequence[str], mine: set[float]) -> bool:
    """Blanking a number other than the operand's in one of ``texts`` gives ``blanked`` too."""
    for text in texts:
        for n in set(_PLAIN_NUM_RE.findall(text)):
            if float(n) not in mine and _stand_alone(n).sub("___", text) == blanked:
                return True
    return False


def description_issue(arg: str, gold: Sequence[str], head: str = "") -> str | None:
    """Why ``describe(arg, gold)`` does not single out its number among the gold facts.

    None when it does. The cases, in the order they are tested:

    - ``sentence cut at " is "``: the fact is a sentence and the description is its text
      before the first " is ". What follows is lost, and with it what tells two sentences apart
      ("the maturity date for $ 30 million of the facility is ..." and "... $ 164 million
      ...") or which number of the sentence is meant.
    - ``blanked label fits another number``: the number stands in a table label and another
      label gives the same text once its number is blanked (a header row that is data).
    - ``cell holds several numbers``: the label names a cell such as "a347/$ 78".
    - ``label occurs with another value``: the table repeats a column header, so that two
      cells of the row have one label.
    - ``quoted fact fits another number``: two facts are the same but for the number.

    ``head`` is the table's first header cell (``table_head``). The check reads the gold facts
    only, so it does not depend on the distractors drawn.
    """
    if arg.startswith("const_"):
        return None
    token = arg.strip()
    pattern = _stand_alone(token)
    mine = _magnitudes(token)
    for fact in gold:
        if not pattern.search(fact):
            continue
        for segment in fact.split(";"):
            if not (pattern.search(segment) and " is " in segment):
                continue
            if not is_table_row(fact):
                return 'sentence cut at " is "'
            label, value = segment.split(" is ", 1)
            cells = [
                (_without_head(lab, head), val.strip())
                for row in gold
                if is_table_row(row)
                for seg in row.split(";")
                if " is " in seg
                for lab, val in [seg.split(" is ", 1)]
            ]
            if pattern.search(label):
                blanked = _without_head(pattern.sub("___", label), head)
                if _other_number_fits(blanked, [lab for lab, _ in cells], mine):
                    return "blanked label fits another number"
                return None
            if _magnitudes(value) != mine:
                return "cell holds several numbers"
            mine_label = _without_head(label, head)
            if any(lab == mine_label and val != value.strip() for lab, val in cells):
                return "label occurs with another value"
            return None
        if _other_number_fits(pattern.sub("___", fact), gold, mine):
            return "quoted fact fits another number"
        return None
    return None


def printed_negative(arg: str, gold: Sequence[str]) -> bool:
    """The program uses a magnitude where the report prints a negative.

    FinQA renders an accounting negative "(2547)" as "-2547 ( 2547 )". True when ``arg`` has
    no minus sign and every place it stands alone in the gold facts is inside such a cell, so
    that a step described by the cell's label asks for a number of the other sign.
    """
    if arg.startswith(("-", "const_", "#")):
        return False
    number = re.escape(arg.strip().rstrip("%"))
    alone = re.compile(rf"(?<![\d.]){number}(?![\d.]|\d)")
    negative = re.compile(rf"-\s?\$?\s?{number}\s?%?\s*\(\s*\$?\s?{number}\s?%?\s*\)")
    hits = sum(len(alone.findall(fact)) for fact in gold)
    inside = sum(2 * len(negative.findall(fact)) for fact in gold)
    return hits > 0 and inside >= hits


# --- protocol 2 -------------------------------------------------------------------------

_WRITTEN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_WRITTEN_RE = re.compile(rf"(?P<number>{_WRITTEN})(?P<percent>\s*%)?")
_ACCOUNTING_RE = re.compile(rf"^\(\s*(?P<body>{_WRITTEN}\s*%?)\s*\)(?P<rest>.*)$", re.DOTALL)
_SPACED_SIGN_RE = re.compile(r"(?<![\w.)])([-+])\s+(?=[\d.])")


class Written(NamedTuple):
    """A number as an answer writes it."""

    value: float  # the number itself: 12.5 for "12.5%"
    percent: bool  # a percent sign follows it
    digits: int  # significant digits written
    decimals: int  # digits written after the decimal point (negative for 7e+07)


def read_number(text: str) -> Written | None:
    """The first number of ``text`` as written, or None.

    Reads exponent notation and a leading decimal point, any dash as a minus sign, an
    accounting form "(5.2)" as negative, and ignores thousands separators and currency
    signs. A percent sign counts only directly after the number, so "0.125 (12.5%)" is
    0.125 without a sign.
    """
    t = canonical_dashes(text).replace(",", "").replace("$", "").strip()
    accounting = _ACCOUNTING_RE.match(t)
    if accounting and not accounting.group("body").startswith(("-", "+")):
        t = "-" + accounting.group("body") + accounting.group("rest")
    t = _SPACED_SIGN_RE.sub(r"\1", t)
    found = _WRITTEN_RE.search(t)
    if found is None:
        return None
    try:
        number = Decimal(found.group("number"))
    except InvalidOperation:
        return None
    _, digits, exponent = number.as_tuple()
    significant = len("".join(map(str, digits)).lstrip("0"))
    return Written(float(number), found.group("percent") is not None, significant, -int(exponent))


def _near(written: Written, target: float) -> bool:
    """``written`` states ``target``.

    Within the relative tolerance, or the target rounded to the precision written (two
    significant digits or more). Zero is stated by zero only.
    """
    if not math.isfinite(target):
        return False
    if abs(target) < 1e-9:
        return abs(written.value) < 1e-9
    distance = abs(written.value - target)
    if distance <= REL_TOL * abs(target):
        return True
    return written.digits >= 2 and distance <= 0.5 * 10.0 ** (-written.decimals) * (1 + 1e-9)


def step_match(answer: str, gold: float) -> bool:
    """Protocol 2: ``answer`` states the gold value of a step.

    Without a percent sign the number must be the gold. With one it may be the gold as a
    percent number or as a fraction ("0.8%" for 0.8 percentage points, "56.8%" for 0.568):
    the program fixes neither convention, and the sign marks the number as a percentage.
    """
    written = read_number(answer)
    if written is None:
        return False
    if written.percent:
        return _near(written, gold) or _near(written, 100.0 * gold)
    return _near(written, gold)


def final_match(answer: str, gold: float, fraction: float | None) -> bool:
    """Protocol 2: ``answer`` states the final value.

    ``fraction`` is the gold as a fraction when FinQA's own answer is a percentage
    (``percentage_fraction``), else None, and then the step rule applies. A percentage may
    be given as the percent number, with or without a sign, or as the fraction without one.
    A signed answer that is the fraction ("0.125%" for 12.5 %) is wrong.
    """
    if fraction is None:
        return step_match(answer, gold)
    written = read_number(answer)
    if written is None:
        return False
    if written.percent:
        return _near(written, 100.0 * fraction)
    return _near(written, fraction) or _near(written, 100.0 * fraction)


def percentage_fraction(
    answer_text: str, gold: float, ops: Sequence[tuple[str, Sequence[str]]]
) -> float | None:
    """The gold as a fraction when FinQA's own answer to the question is a percentage.

    FinQA writes most answers to percentage questions with a percent sign ("12.5%") while the
    program's result is the fraction (0.125) for most of them and the percent number (12.5)
    for some. The answer string says which: its number has the size of the result (percent
    number) or a hundred times it (fraction). A string without a sign counts as a percentage
    only when its number is a hundred times the result. None: not a percentage question.
    """
    written = read_number(answer_text)
    if written is None or not math.isfinite(gold):
        return None
    stated, result = abs(written.value), abs(gold)
    hundredfold = result > 0 and abs(stated - 100.0 * result) <= 0.05 * 100.0 * result
    if not written.percent and not hundredfold:
        return None
    if stated == 0 or result == 0:  # the string cannot say: by the program's last operation
        op, args = ops[-1]
        percent_number = op == "multiply" and any(a in ("const_100", "100") for a in args)
    else:
        percent_number = stated < 10.0 * result
    return gold / 100.0 if percent_number else gold


def plain_number(value: float) -> str:
    """Six significant digits without exponent notation.

    The same text as ``f"{value:.6g}"`` wherever that has no exponent, so that a reviewer's
    answer under protocol 2 differs from protocol 1 only for values of a million or more and
    below a ten-thousandth.
    """
    text = f"{value:.6g}"
    return format(Decimal(text), "f") if "e" in text else text


def describe2(arg: str, gold: Sequence[str], head: str = "") -> str:
    """Protocol 2: a description of a literal argument that singles out its number.

    The description of protocol 1 where ``description_issue`` finds nothing, so that the
    prompt stays as it was. Otherwise the whole fact, quoted, with every place the number
    stands alone blanked ("___% ( ___ % )" for "1.61% ( 1.61 % )"). If blanking another
    number of the gold facts gives the same text (two facts that differ in nothing else), no
    description can tell them apart, and the number itself is given.
    """
    if description_issue(arg, gold, head) is None:
        return describe(arg, gold)
    token = arg.strip()
    whole = _stand_alone(token)
    bare = _stand_alone(token.rstrip("%").lstrip("-"))
    for fact in gold:
        if whole.search(fact):
            blanked = bare.sub("___", fact)
            if _other_number_fits(blanked, gold, _magnitudes(token)):
                return token
            return f'the number given in "{blanked}"'
    return token


class FinqaStep(BaseModel, frozen=True):
    """One operation: its phrase template and gold value."""

    op: str
    left: str  # description, or "{prev}" for the previous result
    right: str
    gold: float


class FinqaInstance(Instance):
    """A FinQA question as a chain of operations."""

    qid: str
    steps: list[FinqaStep]
    facts: list[str]
    answer: float
    # protocol 2: the answer as a fraction when FinQA's own answer is a percentage
    fraction: float | None = None


def convert(row: dict[str, Any], max_steps: int = 4) -> dict[str, Any] | None:
    """A chain for one FinQA question, or None when its program is not a supported chain."""
    ops = [
        (op, [a.strip() for a in args.split(",")])
        for op, args in _OP_RE.findall(row["qa"]["program"])
    ]
    if not 2 <= len(ops) <= max_steps or any(op not in OPS or len(a) != 2 for op, a in ops):
        return None
    for k, (_, args) in enumerate(ops):
        refs = [a for a in args if a.startswith("#")]
        if refs != ([f"#{k - 1}"] if k else []):
            return None
    try:
        results = execute(ops)
        expected = float(row["qa"]["exe_ans"])
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    if not all(math.isfinite(r) for r in results) or not numbers_match(f"{results[-1]}", expected):
        return None
    gold, others = _facts(row)
    steps = []
    for k, (op, args) in enumerate(ops):
        left, right = ("{prev}" if a.startswith("#") else describe(a, gold) for a in args)
        steps.append({"op": op, "left": left, "right": right, "gold": results[k]})
    return {
        "id": str(row["id"]),
        "question": str(row["qa"]["question"]),
        "steps": steps,
        "gold_facts": gold,
        "other_facts": others,
        "answer": results[-1],
        # for protocol 2; the keys above, and which rows are kept, are those of protocol 1
        "ops": ops,
        "answer_text": str(row["qa"].get("answer", "")),
        "head": table_head(row),
    }


@lru_cache(maxsize=4)
def _load(paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            chain = convert(row)
            if chain is not None:
                rows.append(chain)
    return tuple(rows)


class FinqaTask:
    """FinQA linear programs as a :class:`ucagent.core.protocols.Task`."""

    name = "finqa"

    def __init__(
        self,
        data_dir: str | Path = "data/finqa",
        n_instances: dict[str, int] | None = None,
        seed: int = 0,
        n_distractors: int = 6,
        offset: dict[str, int] | None = None,
        protocol: int = 1,
        **other: Any,
    ) -> None:
        unknown = sorted(set(other) - {"n_steps"})  # n_steps is nominal: programs set the length
        if unknown:
            raise ValueError(
                f"the finqa task has no option {', '.join(map(repr, unknown))}; it takes "
                "data_dir, n_instances, seed, n_distractors, offset and protocol (a windows "
                "file is not supported and would be ignored)"
            )
        if protocol not in (1, 2):
            raise ValueError(f"protocol must be 1 (as first run) or 2 (repaired), not {protocol!r}")
        self.protocol = int(protocol)
        self.data_dir = Path(data_dir)
        self.sizes = dict(n_instances or {"train": 60, "calib": 60, "test": 60})
        self.seed = int(seed)
        self.n_distractors = int(n_distractors)
        self.offsets = {k: int(v) for k, v in (offset or {}).items()}

    def input_files(self) -> dict[str, Path]:
        """The files this task reads (``ucagent.experiments.provenance`` records their content)."""
        return {name: self.data_dir / f"{name}.json" for name in ("train", "dev", "test")}

    def _candidates(self, split: str) -> list[dict[str, Any]]:
        if split == "train":
            rows, source = list(_load((str(self.data_dir / "train.json"),))), "train"
        elif split in ("calib", "test"):
            files = (str(self.data_dir / "dev.json"), str(self.data_dir / "test.json"))
            rows, source = list(_load(files)), "devtest"
        else:
            raise KeyError(f"unknown split {split!r}; use train, calib or test")
        order = list(range(len(rows)))
        random.Random(stable_hash("finqa", self.seed, source)).shuffle(order)
        half = len(order) // 2
        if split == "calib":
            order = order[:half]
        elif split == "test":
            order = order[half:]
        return [rows[j] for j in order]

    def instance_id(self, split: str, index: int) -> str:
        """Instance id encoding split, index and task seed."""
        return f"finqa-{split}-{index:05d}-s{self.seed}"

    def instances(self, split: str) -> Iterable[FinqaInstance]:
        """``n_instances[split]`` instances from index ``offset[split]``."""
        n, start = self.sizes[split], self.offsets.get(split, 0)
        available = len(self._candidates(split))
        if start + n > available:
            raise ValueError(f"split {split!r} has {available} questions, {n} from index {start}")
        for index in range(start, start + n):
            yield self.get_instance(self.instance_id(split, index))

    def get_instance(self, instance_id: str) -> FinqaInstance:
        """Build an instance from its id (cached)."""
        return self._cached(instance_id)

    @lru_cache(maxsize=4096)  # noqa: B019 - instances are immutable, task is long-lived
    def _cached(self, instance_id: str) -> FinqaInstance:
        m = _ID_RE.match(instance_id)
        if m is None or int(m.group("seed")) != self.seed:
            raise KeyError(f"not an instance of this task: {instance_id!r}")
        split, index = m.group("split"), int(m.group("index"))
        row = self._candidates(split)[index]
        rng = random.Random(stable_hash("finqa", self.seed, row["id"], "facts"))
        others = list(row["other_facts"])
        picked = rng.sample(others, min(self.n_distractors, len(others)))
        facts = [*row["gold_facts"], *picked]
        rng.shuffle(facts)
        steps = [FinqaStep(**s) for s in row["steps"]]
        fraction = None
        if self.protocol == 2:
            steps = [
                step.model_copy(
                    update={
                        side: describe2(arg, row["gold_facts"], row["head"])
                        for side, arg in zip(("left", "right"), args, strict=True)
                        if not arg.startswith("#")
                    }
                )
                for step, (_, args) in zip(steps, row["ops"], strict=True)
            ]
            fraction = percentage_fraction(row["answer_text"], float(row["answer"]), row["ops"])
        return FinqaInstance(
            instance_id=instance_id,
            split=split,
            index=index,
            question=row["question"],
            n_steps=len(row["steps"]),
            qid=row["id"],
            steps=steps,
            facts=facts,
            answer=float(row["answer"]),
            fraction=fraction,
        )

    # --- protocol -----------------------------------------------------------------

    def n_steps(self, instance: Instance) -> int:
        """Number of operations."""
        return instance.n_steps

    def initial_context(self, instance: Instance) -> str:
        """Nothing is computed before step 1."""
        return ""

    def sub_question(self, instance: Instance, i: int, previous: str) -> str:
        """Step ``i`` as a question, the previous result substituted for ``{prev}``."""
        step = self._inst(instance).steps[i - 1]
        left, right = (previous if x == "{prev}" else x for x in (step.left, step.right))
        return f"What is {left} {OPS[step.op]} {right}?"

    def step_prompt(
        self, instance: Instance, i: int, history: Sequence[StepRecord], attempt: int
    ) -> list[BaseMessage]:
        """System + human messages for step ``i`` (header first, facts last)."""
        inst = self._inst(instance)
        current = history[-1].answer if history else ""
        chain = "; ".join(f"step {r.step}: {r.answer}" for r in history) or "(none yet)"
        lines = [
            render_header(inst.instance_id, i, sanitize_context(current), attempt),
            f"Overall question (context only, do not answer it): {inst.question}",
            f"Results so far: {chain}",
            f"Step {i} of {inst.n_steps} (answer this one): {self.sub_question(inst, i, current)}",
            "Facts:",
            *(f"- {f}" for f in inst.facts),
        ]
        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(lines))]

    def parse_answer(self, text: str) -> str:
        """``Answer: <number>`` on any line, else the first non-empty line."""
        m = _ANSWER_RE.search(text)
        raw = m.group(1) if m else next((ln for ln in text.splitlines() if ln.strip()), "")
        return canonical_dashes(_clean_answer(raw))

    def _states(self, answer: str, gold: float) -> bool:
        """``answer`` states the gold value of a step, by the protocol's rule."""
        return step_match(answer, gold) if self.protocol == 2 else numbers_match(answer, gold)

    def has_local_truth(self, instance: Instance, i: int, current: str) -> bool:
        """True while the chain is on track (step 1, or the previous result is right)."""
        if i <= 1:
            return True
        return self._states(current, self._inst(instance).steps[i - 2].gold)

    def local_truth(self, instance: Instance, i: int, current: str) -> str:
        """The gold value of step ``i`` (protocol 2: never in exponent notation)."""
        gold = self._inst(instance).steps[i - 1].gold
        return plain_number(gold) if self.protocol == 2 else f"{gold:.6g}"

    def global_truth(self, instance: Instance, i: int) -> str:
        """The gold value of step ``i``."""
        return self.local_truth(instance, i, "")

    def distractor_names(self, instance: Instance, i: int, current: str) -> list[str]:
        """Other numbers from the facts (for the mock)."""
        inst = self._inst(instance)
        gold = inst.steps[i - 1].gold
        nums = [n for f in inst.facts for n in _NUM_RE.findall(f.replace(",", ""))]
        return [n for n in dict.fromkeys(nums) if not self._states(n, gold)][:5]

    def random_entity(self, instance: Instance, rng: random.Random) -> str:
        """A random number from the facts (for the mock)."""
        nums: list[str] = [n for f in self._inst(instance).facts for n in _NUM_RE.findall(f)]
        return str(rng.choice(nums or ["0"]))

    def is_locally_correct(self, instance: Instance, i: int, current: str, answer: str) -> bool:
        """On track and matching the gold value of step ``i`` (the last step: as a final answer)."""
        if not self.has_local_truth(instance, i, current):
            return False
        inst = self._inst(instance)
        if self.protocol == 2 and i == inst.n_steps:
            return final_match(answer, inst.answer, inst.fraction)
        return self._states(answer, inst.steps[i - 1].gold)

    def is_final_correct(self, instance: Instance, answer: str | None) -> bool:
        """The final answer matches the program's result."""
        if answer is None:
            return False
        inst = self._inst(instance)
        if self.protocol == 2:
            return final_match(answer, inst.answer, inst.fraction)
        return numbers_match(answer, inst.answer)

    @staticmethod
    def _inst(instance: Instance) -> FinqaInstance:
        if not isinstance(instance, FinqaInstance):
            raise TypeError("FinqaTask needs FinqaInstance objects")
        return instance
