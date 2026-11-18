"""Do the archived runs still match what their configurations ask for?

Usage::

    uv run python scripts/audit_archive.py --also results/musique_confirm2_abstain
    uv run python scripts/audit_archive.py results/finqa_confagree results/twowiki_entropy

Without suite directories it takes the suites of the cross-benchmark tables, as recorded and
as replayed with the repaired effect model (four estimators per benchmark, and the abstention
baseline of the recorded ones); ``--also`` adds others.

A resumed suite compares configurations only, so a windows file or a data file that changed
under the same path, or a changed label rule, would go unnoticed (review of 2026-11-11,
``docs/EXPERIMENTS.md``). This script compares the archive with today's repository, without
model calls. For each suite directory:

1. the stored ``suite.yaml`` against the tracked config of the same name;
2. the questions of every run against the window its config asks for (today's windows file
   and data);
3. every stored final label against the label the task computes now (today's data and rule);
4. the commits the runs recorded (one per suite is expected) and the hashes of the fit
   artifact.

Then, over the configs under ``--configs``:

5. the windows as source questions. Run ids encode split, index and seed, so the same record
   can sit under two ids; the comparison is by the dataset's own id and by question text.
   Reported: questions shared by fit and evaluation inside a config; evaluation questions
   (calibration or test) that another config with other evaluation questions also runs;
   question texts under more than one source id; ids of ``--excluded-ids`` found in a window.
6. every Learn-then-Test selection against the test knobs: a selection needs a test run at
   exactly its knob.
7. the configured grid against the manifest: every knob that a suite's configuration gives a
   controller needs a run on every split. A selection is walked along the configured grid and
   ends at the first knob without a calibration run, so a missing run is reported here.

Exits with status 1 when parts 1 to 4, 6 or 7 find a difference. Part 5 is a report: mock
configs and development runs share questions by design.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_benchmarks as pb
from ucagent.core.control.ltt import calibrate_knob
from ucagent.core.uq.calibration import load_artifact
from ucagent.core.utils import stable_digest
from ucagent.experiments.suite import SuiteConfig, configured_knobs
from ucagent.tasks import make_task

SPLITS = ("train", "calib", "test")
TARGETS = (0.2, 0.1, 0.05, 0.02)
BENCHMARK_TASKS = ("musique", "twowiki", "finqa")


@dataclass
class SuiteAudit:
    """What parts 1 to 4 find for one suite."""

    name: str
    config: str = "no tracked config"  # same, differs
    runs: int = 0
    missing_runs: int = 0
    trajectories: int = 0
    other_questions: int = 0  # runs whose questions are not the configured window
    labels_differ: int = 0
    commits: Counter[str] = field(default_factory=Counter)
    artifact: str = "no artifact"

    @property
    def problems(self) -> int:
        """Differences that make the exit status 1."""
        return (
            (self.config == "differs")
            + self.missing_runs
            + self.other_questions
            + self.labels_differ
            + ("differ" in self.artifact)
        )


def default_suites(results: Path) -> list[Path]:
    """The suites of the cross-benchmark tables in both generations (four estimators each, and
    the abstention baseline, which was not replayed), and those of the runs that a corrected
    run has replaced."""
    out: list[Path] = []
    for bench in (*pb.benchmarks("recorded"), *pb.benchmarks("repaired"), *pb.FIRST_RUNS):
        names = [bench.pattern.format(suffix) for suffix, _ in pb.ESTIMATORS] + [bench.abstain]
        for name in names:
            suite_dir = results / name
            if (suite_dir / "manifest.csv").exists() and suite_dir not in out:
                out.append(suite_dir)
    return out


def grid_rows(root: Path) -> tuple[int, list[str]]:
    """Configured grid runs of ``root`` and those missing from its manifest (part 7)."""
    stored = yaml.safe_load((root / "suite.yaml").read_text(encoding="utf-8"))
    manifest = pd.read_csv(root / "manifest.csv")
    grid = manifest[manifest["label"] == "grid"].dropna(subset=["knob"])
    if "variant" in grid.columns:
        grid = grid[grid["variant"] == "main"]
    n, missing = 0, []
    for policy in stored.get("policies") or {}:
        knobs = configured_knobs(root, str(policy))
        if knobs is None:
            continue
        rows = grid[grid["policy"] == policy]
        for split in stored.get("splits") or ["calib", "test"]:
            for seed in stored.get("seeds") or [0]:
                have = rows[(rows["split"] == split) & (rows["seed"] == seed)]["knob"]
                for knob in knobs:
                    n += 1
                    if have.empty or float((have - knob).abs().min()) > 1e-9 * max(1.0, abs(knob)):
                        missing.append(f"{root.name} {policy} {split} seed {seed}: knob {knob:g}")
    return n, missing


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root.parent.parent / p


def _window(task: Any, split: str) -> list[Any]:
    """The instances a task yields for ``split`` (none when it has no such split)."""
    try:
        return list(task.instances(split))
    except (KeyError, ValueError):
        return []


def audit_suite(root: Path, configs: Path | None) -> SuiteAudit:
    """Parts 1 to 4 for the suite directory ``root``."""
    audit = SuiteAudit(root.name)
    stored = yaml.safe_load((root / "suite.yaml").read_text(encoding="utf-8"))
    tracked = None if configs is None else configs / f"{root.name}.yaml"
    if tracked is not None and tracked.exists():
        same = SuiteConfig.from_yaml(tracked).model_dump(mode="json") == stored
        audit.config = "same" if same else "differs"
    suite = SuiteConfig.model_validate(stored)
    manifest = pd.read_csv(root / "manifest.csv")
    notes = []
    for seed in suite.seeds:
        base = suite.base.model_copy(update={"seed": seed}).with_updates(task={"seed": seed})
        task: Any = make_task(base.task.model_dump())
        window = {split: [i.instance_id for i in _window(task, split)] for split in SPLITS}
        labels: dict[tuple[str, str | None], bool] = {}
        for row in manifest[manifest["seed"] == seed].itertuples():
            path = _resolve(root, str(row.path))
            if not path.exists():
                audit.missing_runs += 1
                continue
            audit.runs += 1
            sha = path.parent / "git_sha.txt"
            if sha.exists():
                audit.commits[sha.read_text(encoding="utf-8").strip()[:7]] += 1
            frame = pq.read_table(
                path, columns=["instance_id", "final_answer", "final_correct"]
            ).to_pandas()
            audit.trajectories += len(frame)
            if sorted(frame["instance_id"]) != sorted(window.get(str(row.split), [])):
                audit.other_questions += 1
                continue
            for iid, answer, correct in zip(
                frame["instance_id"], frame["final_answer"], frame["final_correct"], strict=True
            ):
                text = None if answer is None or answer != answer else str(answer)
                key = (str(iid), text)
                if key not in labels:
                    labels[key] = bool(task.is_final_correct(task.get_instance(str(iid)), text))
                audit.labels_differ += labels[key] != bool(correct)
        artifact = root / "artifacts" / f"seed{seed}.json"
        if artifact.exists():
            art = load_artifact(str(artifact))
            task_ok = art.task_config_hash == stable_digest(base.task.model_dump())
            data_ok = art.data_hash == stable_digest(window.get("train", []))
            notes.append(
                f"task hash {'same' if task_ok else 'differs'}, "
                f"fit questions {'same' if data_ok else 'differ'}"
            )
    if notes:
        audit.artifact = "; ".join(notes)
    return audit


def selection_gaps(root: Path, targets: tuple[float, ...] = TARGETS) -> tuple[int, list[str]]:
    """Selections of ``root`` and those without a test run at exactly their knob (part 6)."""
    manifest = pd.read_csv(root / "manifest.csv")
    grid = manifest[manifest["label"] == "grid"].dropna(subset=["knob"])
    if "variant" in grid.columns:
        grid = grid[grid["variant"] == "main"]
    cal = yaml.safe_load((root / "suite.yaml").read_text(encoding="utf-8"))["base"].get(
        "calibration", {}
    )
    n, gaps = 0, []
    for policy, rows in grid.groupby("policy"):
        calib, test = rows[rows["split"] == "calib"], rows[rows["split"] == "test"]
        if calib.empty:
            continue
        points = [
            (float(k), int(m), round(float(e) * int(m)))
            for k, m, e in zip(calib["knob"], calib["n"], calib["final_error"], strict=True)
        ]
        for alpha in targets:
            star = calibrate_knob(
                str(policy),
                points,
                alpha=alpha,
                delta=float(cal.get("delta", 0.1)),
                pvalue=cal.get("pvalue", "binomial"),
                grid=configured_knobs(root, str(policy)),
            ).knob_star
            if star is None:
                continue
            n += 1
            if test.empty or float((test["knob"] - star).abs().min()) > 1e-9:
                gaps.append(f"{root.name} {policy} at {alpha:g}: no test run at knob {star:g}")
    return n, gaps


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def config_questions(path: Path) -> dict[str, Any] | None:
    """Source ids and question texts per split of a benchmark config (None: not one)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("kind") == "suite":
        block = dict(data.get("base", {}).get("task", {}))
        splits = ["train", *data.get("splits", ["calib", "test"])]
        backend = str(data.get("base", {}).get("llm", {}).get("backend", "mock"))
    elif "task" in data:
        block, splits = dict(data["task"]), [str(data.get("split", "test"))]
        backend = str(data.get("llm", {}).get("backend", "mock"))
    else:
        return None
    if block.get("name") not in BENCHMARK_TASKS:
        return None
    out: dict[str, Any] = {"task": block["name"], "mock": backend == "mock", "splits": {}}
    try:
        task: Any = make_task(block)
        for split in dict.fromkeys(splits):
            if task.sizes.get(split, 0) > 0:  # a window beyond the data raises: reported below
                out["splits"][split] = [(i.qid, _norm(i.question)) for i in task.instances(split)]
    except (ValueError, KeyError, FileNotFoundError) as exc:
        out["skipped"] = f"{type(exc).__name__}: {str(exc)[:110]}"
    return out


def _ids(info: dict[str, Any], splits: tuple[str, ...]) -> set[str]:
    return {q for s in splits for q, _ in info["splits"].get(s, [])}


def _texts(info: dict[str, Any], splits: tuple[str, ...]) -> set[str]:
    return {t for s in splits for _, t in info["splits"].get(s, [])}


def freshness(configs: Path, audited: list[str], excluded: dict[str, set[str]]) -> list[str]:
    """Part 5 as markdown lines."""
    infos = {}
    lines: list[str] = []
    for path in sorted(configs.glob("*.yaml")):
        info = config_questions(path)
        if info is None:
            continue
        if "skipped" in info:
            lines.append(f"- `{path.stem}` could not be built and is left out ({info['skipped']}).")
            continue
        infos[path.stem] = info
    for name in BENCHMARK_TASKS:
        group = {stem: info for stem, info in infos.items() if info["task"] == name}
        if not group:
            continue
        seen: dict[str, set[str]] = defaultdict(set)
        for info in group.values():
            for split in SPLITS:
                for qid, text in info["splits"].get(split, []):
                    seen[text].add(qid)
        twins = sum(len(v) > 1 for v in seen.values())
        hit = {
            stem: len(_ids(info, SPLITS) & excluded.get(name, set()))
            for stem, info in group.items()
        }
        lines.append(
            f"- **{name}** ({len(group)} configs): {twins} question texts stand under more than "
            f"one source id; excluded ids found in a window: {sum(hit.values())}."
        )
        for stem in audited:
            if stem not in group:
                continue
            info = group[stem]
            evaluation, fit = _ids(info, ("calib", "test")), _ids(info, ("train",))
            by_text = len(_texts(info, ("calib", "test")) & _texts(info, ("train",)))
            both_halves = len(_ids(info, ("calib",)) & _ids(info, ("test",)))
            others, mocks = [], []
            for other, o in group.items():
                if _ids(o, ("calib", "test")) == evaluation:
                    continue  # the same evaluation questions by design (the other estimators)
                shared = len(evaluation & _ids(o, SPLITS))
                if shared:
                    (mocks if o["mock"] else others).append(f"`{other}` {shared}")
            lines.append(
                f"  - `{stem}`: {len(evaluation)} evaluation questions; in its own fit "
                f"{len(evaluation & fit)} by id and {by_text} by text; in both halves "
                f"{both_halves}; also run by a config with other evaluation questions: "
                f"{', '.join(others) if others else 'none'}"
                + (f"; by mock configs: {', '.join(mocks)}" if mocks else "")
                + "."
            )
    return lines


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="*", type=Path, help="suite directories")
    parser.add_argument("--also", nargs="*", type=Path, default=[], help="suites to add")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument(
        "--excluded-ids",
        type=Path,
        default=Path("configs/twowiki_smoked_ids.txt"),
        help="source ids of 2WikiMultiHopQA that no window may hold",
    )
    args = parser.parse_args()
    suites = [*(args.suites or default_suites(args.results)), *args.also]
    out = [
        "### Archive audit",
        "",
        "| suite | runs | trajectories | config | runs with other questions | final labels that "
        "differ | commits | fit artifact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    problems = 0
    totals: Counter[str] = Counter()
    for root in suites:
        a = audit_suite(root, args.configs)
        problems += a.problems
        totals.update(suites=1, runs=a.runs, trajectories=a.trajectories)
        commits = ", ".join(f"{sha} ({n})" for sha, n in sorted(a.commits.items())) or "none"
        runs = f"{a.runs}" + (f" ({a.missing_runs} missing)" if a.missing_runs else "")
        out.append(
            f"| {a.name} | {runs} | {a.trajectories} | {a.config} | {a.other_questions} "
            f"| {a.labels_differ} | {commits} | {a.artifact} |"
        )
    out += [
        "",
        f"{totals['suites']} suites, {totals['runs']} runs, {totals['trajectories']} "
        f"trajectories; differences found: {problems}.",
        "",
        "Windows as source questions:",
        "",
    ]
    excluded: dict[str, set[str]] = {}
    if args.excluded_ids.exists():
        excluded["twowiki"] = set(args.excluded_ids.read_text(encoding="utf-8").split())
    out += freshness(args.configs, [root.name for root in suites], excluded)
    n_sel, gaps = 0, []
    n_rows, absent = 0, []
    for root in suites:
        n, g = selection_gaps(root)
        n_sel += n
        gaps += g
        n, g = grid_rows(root)
        n_rows += n
        absent += g
    out += [
        "",
        f"Learn-then-Test selections at targets {', '.join(f'{t:g}' for t in TARGETS)} over every "
        f"controller: {n_sel}; without a test run at exactly the selected knob: {len(gaps)}.",
        *(f"- {g}" for g in gaps),
        "",
        f"Configured grid runs (controller, knob, split and seed): {n_rows}; missing from a "
        f"manifest: {len(absent)}.",
        *(f"- {a}" for a in absent),
    ]
    print("\n".join(out))
    sys.exit(1 if problems or gaps or absent else 0)


if __name__ == "__main__":
    main()
