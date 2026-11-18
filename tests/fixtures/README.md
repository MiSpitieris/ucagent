# Test fixtures

## `musique_mini.jsonl`

Five rows in the format of the MuSiQue answerable files, used by `tests/test_musique.py`:

- Three 2-hop questions taken from MuSiQue (`2hop__460946_294723`, `2hop__481349_302087`, `2hop__765180_123956`). They are **modified**: each keeps 6 of its 20 paragraphs, and the question decomposition is re-indexed to the kept paragraphs. The questions, answers and aliases are unchanged.
- Two rows written by hand for the 3-hop tests (`3hop1__900001_900002_900003`, linear, and `3hop2__900011_900012_900013`, non-linear). They are not MuSiQue data.

MuSiQue is distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), at https://github.com/StonyBrookNLP/musique. The material is provided as is, without warranties, as stated in the license. Source:

> Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot and Ashish Sabharwal. MuSiQue: Multihop Questions via Single-hop Question Composition. *Transactions of the Association for Computational Linguistics*, 2022.

The full MuSiQue files are not part of this repository; download them to `data/musique/` (see the README).

## `twowiki_mini.jsonl`

Five records of the 2WikiMultiHopQA dev split (Ho, Duong Nguyen, Sugawara and Aizawa, *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*, COLING 2020), distributed under the Apache License 2.0 at https://github.com/Alab-NII/2wikimultihop; taken from the Hugging Face mirror `xanhho/2WikiMultihopQA` (`dev.parquet`, revision `612bc50`). Changes: each record keeps its two supporting paragraphs (shortened to the supporting sentence or the first two sentences) and its first two other paragraphs (first sentence only); all other fields are unchanged. The records are two compositional chains (one with a date answer), one inference chain, one comparison question and one compositional question whose evidence names the bridge entity in two spellings; the last two are not chains and are skipped by the `twowiki` task.

## `finqa_mini.json`

Three records of the FinQA dev split (Chen, Chen, Smiley, Shah, Borova, Langdon, Moussa, Beane, Huang, Routledge and Wang, *FinQA: A Dataset of Numerical Reasoning over Financial Data*, EMNLP 2021), distributed under the MIT License at https://github.com/czyssrs/FinQA. Changes: each record keeps the sentences that hold its supporting facts plus one other sentence, the first four table rows, and from its `qa` block only the question, program, execution result and supporting facts. Two records are linear two-step programs (one through a table cell, one through a sentence); the third is a one-step program, which the `finqa` task skips.

## `finqa_protocol.json`

Seven records written by hand in FinQA's format, for the tests of FinQA's second protocol (`tests/test_finqa_protocol.py`, `tests/test_reviewer_accepted.py`). They are not FinQA data: the report ids, numbers and sentences are made up after cases met in the data. They cover a plain table chain whose answer FinQA gives as a percentage, two sentences that the first protocol's descriptions do not tell apart, values of a million or more, a table that repeats a column header (once with amounts and once with percentage points), two facts that differ in nothing but the number, and a result that Python writes in exponent notation, which neither protocol loads.
