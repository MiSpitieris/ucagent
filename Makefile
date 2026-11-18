.RECIPEPREFIX := >
.PHONY: setup lint format typecheck test test-fast test-llm experiments figures results docs ui report preprint preprint-framework readme-figure clean

UV ?= uv
PY := $(UV) run

setup:
> $(UV) sync --extra dev --extra llm --extra bench --extra ui
> $(PY) pre-commit install || true

lint:
> $(PY) ruff check src tests scripts examples
> $(PY) ruff format --check src tests scripts examples

format:
> $(PY) ruff format src tests scripts examples
> $(PY) ruff check --fix src tests scripts examples

typecheck:
> $(PY) mypy

test:
> $(PY) pytest -q -m "not llm"

test-fast:
> $(PY) pytest -q -m "not llm and not slow"

test-llm:
> UCAGENT_LLM_TESTS=1 $(PY) pytest -q -m llm

experiments:
> $(PY) ucagent suite --config configs/suite_small.yaml

figures:
> $(PY) ucagent pipeline --config configs/figures_small.yaml

docs:
> $(PY) python scripts/export_graph.py
ui:
> $(PY) ucagent ui
report:
> cd docs/report && latexmk -pdf -interaction=nonstopmode -halt-on-error ucagent_report.tex
preprint:
> cd docs/preprint && latexmk -pdf -interaction=nonstopmode -halt-on-error ucagent_preprint.tex
preprint-framework:
> cd docs/preprint && latexmk -pdf -interaction=nonstopmode -halt-on-error ucagent_framework.tex

# The control-loop diagram as an image for the README, from the same TikZ source as the paper's Figure 1.
# Needs pdflatex and pdftocairo (poppler); the build files go to docs/preprint/build/, which git ignores.
# The source is under docs/preprint/, which is in the development repository only while the paper is not public.
readme-figure:
> mkdir -p docs/preprint/build
> cd docs/preprint && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build framework_loop_standalone.tex > /dev/null
> pdftocairo -png -r 250 -singlefile docs/preprint/build/framework_loop_standalone.pdf docs/figures/fig_loop

clean:
> rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov .coverage

results:
> $(PY) python scripts/results_table.py results figures_small
