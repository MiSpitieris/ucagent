# ucagent

Uncertainty-driven, cost-aware oversight of multi-step language-model agents.

At each step the agent samples the model several times and turns the samples into an estimated probability that the step is wrong. A controller with one knob then decides whether to proceed, sample again, backtrack, ask a person to check the step, or hand the case over. Every action has a price. Learn-then-Test selects the knob for a target on the end-to-end error, with a finite-sample guarantee under the conditions stated in `docs/MATH.md`. The loop is an explicit [LangGraph](https://github.com/langchain-ai/langgraph) graph with human interrupts, a supervisor mode and an audit log of every decision.

![The control loop: prepare a step, sample K times, estimate, control. The controller proceeds, resamples, backtracks, asks a person or escalates. Below the loop: the two calibration steps before deployment.](docs/figures/fig_loop.png)

*The control loop of one task. The two actions in dashed orange involve a person. The two boxes below the loop are the calibration steps, done once before deployment.*

This repository holds the framework, its tests, and the configurations and scripts of its evaluation on three benchmarks.

## Install and try

```bash
pipx install uv      # once
make setup           # Python 3.11 or later; no API key needed
make test            # runs on a mock model, a few minutes
uv run ucagent demo --policy marginal --knob 0.05   # one task; you answer the step checks
make ui              # the same in a web page, with the supervisor mode
```

## What is where

| Path | Contents |
|---|---|
| `src/ucagent/core/` | The framework: uncertainty estimators and their calibration (`uq/`), risk propagation (`risk/`), controllers and Learn-then-Test (`control/`), the graph and the human interface (`graph/`), model backends with a response cache (`llm/`) |
| `src/ucagent/tasks/` | The tasks: MuSiQue, 2WikiMultiHopQA, FinQA and a synthetic chain |
| `src/ucagent/experiments/`, `src/ucagent/ui/` | Running and calibrating suites of runs; the web page |
| `configs/` | The run configurations: model, prices, label rules, questions and knob grids |
| `scripts/` | The scripts that write the evaluation's tables, figures and counts from the run records |
| `docs/MATH.md`, `docs/DESIGN.md`, `docs/ADOPTION.md` | The math, the control loop, and how to plug in your own task (`examples/toy_task.py`) |
| `tests/` | The tests; they need no model server |

## The evaluation

- **Runs.** `uv run ucagent suite -c configs/<name>.yaml` runs every controller of a configuration over its knob grid, and `uv run ucagent calibrate -s results/<name>` selects the knobs with Learn-then-Test. The benchmark runs used `gpt-oss:20b` on an Ollama server.
- **Run records.** The run records and the cache of model replies are not in this repository; they are available from the author on request. With them, the scripts write their tables and figures again without a model call.
- **Data.** The datasets are not redistributed here. Shortened records serve as test fixtures (`tests/fixtures/README.md`).
  - MuSiQue (CC BY 4.0): `musique_ans_v1.0_train.jsonl` and `musique_ans_v1.0_dev.jsonl` go to `data/musique/`.
  - 2WikiMultiHopQA (Apache 2.0): `dev.parquet` of the Hugging Face mirror `xanhho/2WikiMultihopQA` goes to `data/twowiki/`; then run `uv run python scripts/prepare_twowiki.py data/twowiki/dev.parquet --out data/twowiki --exclude-ids configs/twowiki_smoked_ids.txt`.
  - FinQA (MIT): `train.json`, `dev.json` and `test.json` go to `data/finqa/`.
- **Development files.** Comments in some files name an experiment log, a decision log and review notes (`docs/EXPERIMENTS.md`, `docs/DECISIONS.md`, `docs/reviews/`). These are kept in the development repository and are available from the author on request. Some scripts and Makefile targets are named after a preprint that is not public yet.

## Citation and license

See `CITATION.cff`. MIT license.
