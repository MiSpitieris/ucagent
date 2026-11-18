"""Offline estimator and calibrator arms on a finished suite, at zero model calls.

Usage::

    uv run python scripts/estimator_arms.py results/<suite_id> [--figure docs/figures/<name>]

What it does:

1. Replays the suite's training fit from the response cache (policy ``none``, ``K'`` samples
   per step) with real model calls forbidden, so every training step has ``K'`` samples with
   their stated confidences.
2. Computes the named step features (``ucagent.core.uq.features``) at ``K``, at smaller
   batch sizes, and for an adaptive rule (start with 3 samples, draw up to ``K'`` only when
   they disagree), with the local-error label of each batch's committed answer.
3. Fits calibrator arms on the training steps: isotonic maps of each single estimator score
   (entropy, disagreement, verbalized ``1 - mean confidence``, confidence-weighted
   agreement) and the logistic, boosted-tree and Gaussian-process calibrators on a
   multi-feature set; evaluates them on the suite's held-out ``none`` runs (calibration and
   test splits, ``K`` samples): AUROC, Brier score, expected calibration error, distinct
   calibrated values, AUROC within unanimous steps and the catch rate of the riskiest fifth.
4. Compares batch sizes and the adaptive rule by two-fold cross-fitting over the training
   questions at matched calls.

Everything is read from the suite's files and the cache; nothing is typed in.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["UCAGENT_MAX_LLM_CALLS"] = "1"  # any cache miss raises: this script makes no calls

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_results as rr
from ucagent.core.records import Sample, Trajectory
from ucagent.core.uq.base import StepContext, make_estimator
from ucagent.core.uq.calibration import IsotonicCalibrator
from ucagent.core.uq.calibrators import fit_calibrator
from ucagent.core.uq.equivalence import make_judge
from ucagent.core.uq.features import step_features
from ucagent.experiments.config import ExperimentConfig
from ucagent.experiments.io import read_trajectories
from ucagent.experiments.runner import build_task, run_experiment

MULTI = ["entropy", "disagreement", "conf_mean", "conf_min", "conf_agreement", "position"]
SINGLE = {  # arm label -> (feature name, sign): isotonic maps are increasing in their input
    "entropy": ("entropy", 1.0),
    "disagreement": ("disagreement", 1.0),
    "verbalized": ("conf_mean", -1.0),
    "confidence-weighted agreement": ("conf_agreement", 1.0),
}


@dataclass
class StepRow:
    """One labelled on-track step re-estimated from its first ``k`` samples."""

    qid: str
    key: str  # question and sample batch: identifies the step across batch sizes
    k: int
    features: dict[str, float]
    wrong: bool
    unanimous: bool


def _base_config(suite_dir: Path) -> ExperimentConfig:
    data = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data["base"])


def _samples(t: Trajectory, key: str) -> list[Sample]:
    answers = t.samples.get(key, [])
    confs = t.sample_confidences.get(key) or [None] * len(answers)
    return [
        Sample(sample_index=j, text=a, answer=a, confidence=c)
        for j, (a, c) in enumerate(zip(answers, confs, strict=True))
    ]


def step_rows(
    cfg: ExperimentConfig, trajectories: Sequence[Trajectory], ks: Sequence[int]
) -> list[StepRow]:
    """Features and labels of every labelled on-track LLM step, for each batch size in ``ks``."""
    task = build_task(cfg)
    estimator = make_estimator("semantic_entropy", make_judge(cfg.uq.equivalence))
    rows: list[StepRow] = []
    for t in trajectories:
        inst = task.get_instance(t.instance_id)
        for s in t.steps:
            if s.source != "llm" or not task.has_local_truth(inst, s.step, s.context):
                continue
            samples = _samples(t, s.samples_key)
            for k in ks:
                if len(samples) < k:
                    continue
                ctx = StepContext(
                    instance_id=t.instance_id,
                    step=s.step,
                    attempt=0,
                    current=s.context,
                    n_steps=t.n_steps,
                )
                est = estimator.estimate(samples[:k], ctx)
                feats = step_features(est.u, samples[:k], est.clusters, s.step, t.n_steps)
                wrong = not task.is_locally_correct(inst, s.step, s.context, est.committed)
                rows.append(
                    StepRow(
                        t.instance_id,
                        f"{t.instance_id}/{s.samples_key}",
                        k,
                        feats,
                        wrong,
                        feats["disagreement"] == 0.0,
                    )
                )
    return rows


def train_trajectories(cfg: ExperimentConfig, workers: int) -> list[Trajectory]:
    """The training fit replayed from the cache (policy none, K' samples per step)."""
    fit_cfg = cfg.with_updates(
        control={"policy": "none", "knob": None}, llm={"K": cfg.llm.K_prime}, uq={"artifact": None}
    ).model_copy(update={"split": "train", "run_id": f"{cfg.run_id}__arms"})
    return run_experiment(fit_cfg, workers=workers, write=False).trajectories


def none_trajectories(suite_dir: Path, split: str) -> list[Trajectory]:
    """The suite's policy-none run on a split."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    row = manifest[(manifest["policy"] == "none") & (manifest["split"] == split)]
    return list(read_trajectories(rr._resolve(suite_dir, str(row["path"].iloc[0]))))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def metrics(rows: Sequence[StepRow], probs: Sequence[float]) -> dict[str, float]:
    """Step-level diagnostics of one arm on held-out steps."""
    y = np.asarray([r.wrong for r in rows], dtype=int)
    p = np.asarray(probs, dtype=float)
    unan = np.asarray([r.unanimous for r in rows])
    out = {
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _ece(y, p),
        "levels": float(len(np.unique(np.round(p, 6)))),
    }
    yu, pu = y[unan], p[unan]
    out["auroc_unanimous"] = (
        float(roc_auc_score(yu, pu))
        if len(set(yu)) > 1 and len(np.unique(pu)) > 1
        else float("nan")
    )
    w = riskiest_share_weights(p, 0.2)  # ask at the riskiest fifth of the steps
    out["caught_wrong"] = float(w[y == 1].mean()) if (y == 1).any() else float("nan")
    uw = (y == 1) & unan
    out["caught_unanimous_wrong"] = float(w[uw].mean()) if uw.any() else float("nan")
    return out


def riskiest_share_weights(p: np.ndarray, share: float) -> np.ndarray:
    """Probability of being asked when asking at exactly ``share`` of the steps, riskiest first.

    Steps above the threshold are asked; steps tied at it are asked with the probability that
    fills the budget exactly (random tie-breaking, in expectation), so arms with few distinct
    values are compared at the same budget as continuous ones.
    """
    q = np.round(np.asarray(p, dtype=float), 9)
    budget = share * len(q)
    if budget <= 0:
        return np.zeros(len(q))
    t = np.sort(q)[::-1][int(np.ceil(budget)) - 1]
    above, tied = q > t, q == t
    w = above.astype(float)
    w[tied] = (budget - above.sum()) / tied.sum()
    return w


def _single_predictor(
    train: Sequence[StepRow], feature: str, sign: float
) -> Callable[[StepRow], float]:
    cal = IsotonicCalibrator.fit(
        [sign * r.features[feature] for r in train], [int(r.wrong) for r in train]
    )
    return lambda r: cal(sign * r.features[feature])


def _multi_predictor(train: Sequence[StepRow], name: str) -> Callable[[StepRow], float]:
    cal = fit_calibrator(name, MULTI, [r.features for r in train], [r.wrong for r in train])
    return lambda r: cal.predict(r.features)


def arms(train: Sequence[StepRow]) -> dict[str, Callable[[StepRow], float]]:
    """Every calibrator arm fitted on the training steps."""
    out = {
        f"isotonic: {label}": _single_predictor(train, f, sign)
        for label, (f, sign) in SINGLE.items()
    }
    for name in ("logistic", "boosted", "gp"):
        out[f"{name}: six features"] = _multi_predictor(train, name)
    return out


def arm_table(train: Sequence[StepRow], held: Mapping[str, Sequence[StepRow]]) -> list[str]:
    """Markdown table of every arm on each held-out split."""
    fitted = arms(train)
    lines = [
        f"Calibrator arms fitted on {len(train)} training steps (K = {train[0].k}); evaluated on "
        "held-out steps of the suite's policy-none runs:",
        "",
        "| split | arm | AUROC | Brier | ECE | levels | AUROC within unanimous "
        "| riskiest fifth: wrong caught | unanimous-wrong caught |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for split, rows in held.items():
        n_wrong = sum(r.wrong for r in rows)
        notes.append(
            f"- {split}: {sum(r.unanimous for r in rows)} of {len(rows)} steps unanimous; "
            f"{n_wrong} wrong, {sum(r.wrong and r.unanimous for r in rows)} of them unanimous."
        )
        for label, predict in fitted.items():
            m = metrics(rows, [predict(r) for r in rows])
            lines.append(
                f"| {split} ({len(rows)} steps) | {label} | {m['auroc']:.3f} | {m['brier']:.4f} | "
                f"{m['ece']:.4f} | {int(m['levels'])} | {m['auroc_unanimous']:.3f} | "
                f"{m['caught_wrong']:.2f} | {m['caught_unanimous_wrong']:.2f} |"
            )
    return [*lines, "", *notes]


def sample_budget_table(rows_by_k: Mapping[int, Sequence[StepRow]], k_max: int) -> list[str]:
    """Batch sizes and the adaptive rule, by two-fold cross-fitting over training questions."""
    qids = sorted({r.qid for rows in rows_by_k.values() for r in rows})
    fold = {q: i % 2 for i, q in enumerate(qids)}
    arms_k: dict[str, tuple[list[StepRow], Callable[[StepRow], int]]] = {
        f"fixed K = {k}": (list(rows), lambda r, k=k: k) for k, rows in sorted(rows_by_k.items())
    }
    small = {r.key: r for r in rows_by_k[3]}
    large = {r.key: r for r in rows_by_k[k_max]}
    if len(small) != len(rows_by_k[3]) or set(small) != set(large):
        raise ValueError("training steps differ between batch sizes; cannot pair them")
    adaptive = [small[key] if small[key].unanimous else large[key] for key in small]
    arms_k[f"adaptive: 3, then {k_max} on disagreement"] = (adaptive, lambda r: r.k)
    lines = [
        "Sample budget, two-fold cross-fitting over training questions (isotonic map per batch "
        "size, fitted on one fold, evaluated on the other):",
        "",
        "| rule | mean calls per step | local error of the committed answer | AUROC (entropy) "
        "| Brier (entropy) | AUROC (confidence-weighted agreement) "
        "| Brier (confidence-weighted agreement) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, (rows, calls) in arms_k.items():
        res: dict[str, list[float]] = {"ae": [], "be": [], "ac": [], "bc": []}
        for held_fold in (0, 1):
            fit_rows = [r for r in rows if fold[r.qid] != held_fold]
            test_rows = [r for r in rows if fold[r.qid] == held_fold]
            for feat, (ka, kb) in (("entropy", ("ae", "be")), ("conf_agreement", ("ac", "bc"))):
                preds: list[float] = []
                for r in test_rows:
                    same = [x for x in fit_rows if x.k == r.k]
                    preds.append(_cached_iso(same, feat)(r.features[feat]))
                m = metrics(test_rows, preds)
                res[ka].append(m["auroc"])
                res[kb].append(m["brier"])
        lines.append(
            f"| {label} | {np.mean([calls(r) for r in rows]):.2f} | "
            f"{np.mean([r.wrong for r in rows]):.3f} | "
            f"{np.mean(res['ae']):.3f} | {np.mean(res['be']):.4f} | {np.mean(res['ac']):.3f} | "
            f"{np.mean(res['bc']):.4f} |"
        )
    extended = [key for key, r in small.items() if not r.unanimous]
    lines += [
        "",
        f"Adaptive rule: {len(extended)} of {len(small)} training steps disagree within the first "
        f"three samples and are extended to {k_max}; local error of the committed answer on those "
        f"steps {np.mean([small[q].wrong for q in extended]):.3f} with three samples and "
        f"{np.mean([large[q].wrong for q in extended]):.3f} after the extension.",
    ]
    return lines


_ISO_CACHE: dict[tuple[int, int, str], Any] = {}


def _cached_iso(rows: Sequence[StepRow], feat: str) -> Any:
    key = (id(rows[0]) if rows else 0, len(rows), feat)
    if key not in _ISO_CACHE:
        _ISO_CACHE[key] = IsotonicCalibrator.fit(
            [r.features[feat] for r in rows], [int(r.wrong) for r in rows]
        )
    return _ISO_CACHE[key]


def reliability_figure(
    train: Sequence[StepRow], held: Sequence[StepRow], out: Path, caption: str
) -> None:
    """Reliability diagram of four arms on one held-out split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ucagent.experiments.figures import STYLE, _save

    fitted = arms(train)
    chosen = {
        "isotonic: entropy": ("#2a78d6", "o"),
        "isotonic: confidence-weighted agreement": ("#eb6834", "s"),
        "logistic: six features": ("#1baf7a", "^"),
        "gp: six features": ("#6250d6", "D"),
    }
    y = np.asarray([r.wrong for r in held], dtype=int)
    with plt.style.context(str(STYLE)):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot([0, 1], [0, 1], color="#b9b8b1", linestyle=":", linewidth=1.0, zorder=1)
        for label, (color, marker) in chosen.items():
            p = np.asarray([fitted[label](r) for r in held])
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 7)))
            xs, ys = [], []
            for lo, hi in itertools.pairwise(edges):
                mask = (p >= lo) & (p <= hi)
                if mask.sum() >= 5:
                    xs.append(p[mask].mean())
                    ys.append(y[mask].mean())
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.4,
                label=label,
                zorder=3,
            )
        ax.set_xlabel("predicted probability that the step is wrong")
        ax.set_ylabel("observed share of wrong steps")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()
        for path in _save(fig, out.parent, out.name, caption):
            print("wrote", path)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--figure", type=Path, default=None, help="path stem of the reliability figure"
    )
    args = parser.parse_args()
    cfg = _base_config(args.suite_dir)
    k, k_max = cfg.llm.K, cfg.llm.K_prime
    train_trajs = train_trajectories(cfg, args.workers)
    ks = sorted({3, k, k_max})
    train_rows = step_rows(cfg, train_trajs, ks)
    rows_by_k = {kk: [r for r in train_rows if r.k == kk] for kk in ks}
    held = {
        split: step_rows(cfg, none_trajectories(args.suite_dir, split), [k])
        for split in ("calib", "test")
    }
    out = [f"### Offline arms on {args.suite_dir.name} ({cfg.llm.backend}; zero model calls)", ""]
    out += [*arm_table(rows_by_k[k], held), ""]
    out += sample_budget_table(rows_by_k, k_max)
    print("\n".join(out))
    if args.figure is not None:
        caption = (
            f"Reliability of four calibrator arms on the test split of {args.suite_dir.name} "
            f"({len(held['test'])} labelled on-track steps, K = {k}; {cfg.llm.backend}); each arm "
            f"fitted on {len(rows_by_k[k])} training steps. Points are six equal-count bins of the "
            "predicted probability; the dotted diagonal is perfect calibration."
        )
        reliability_figure(rows_by_k[k], held["test"], args.figure, caption)


if __name__ == "__main__":
    main()
