"""Preprint figure: held-out error against cost, one panel per estimator.

Usage::

    uv run python scripts/preprint_figures.py \\
        --suite "disagreement=results/musique_confirm2_disagreement" \\
        --suite "cluster entropy=results/musique_confirm2_entropy" \\
        --suite "verbalized confidence=results/musique_confirm2_verbalized" \\
        --suite "confidence-weighted agreement=results/musique_confirm2_confagree" \\
        --out docs/figures/preprint

Each panel shows, on the test split, the random-asking baseline (each step checked
independently with a fixed probability; the curve is the measured one, not a mixture of never
and always asking), the calibrated gate (``per_step_eps``) and the value-of-intervention
rule (``marginal``) along their knob grids, and marks the Learn-then-Test selections at the
two targets. Everything is read from the suites' manifests (test curves) and calibration rows
(selections); nothing is typed in.

The panels may come from different benchmarks, with other targets (``--targets 0.1,0.05``)
and their own wording in the caption file (``--setting``, ``--panels``)::

    uv run python scripts/preprint_figures.py --name fig_benchmarks --targets 0.1,0.05 \\
        --suite "2WikiMultiHopQA: confidence-weighted agreement=results/twowiki_confagree" \\
        --suite "2WikiMultiHopQA: cluster entropy=results/twowiki_entropy" \\
        --suite "FinQA: confidence-weighted agreement=results/finqa_confagree" \\
        --suite "FinQA: cluster entropy=results/finqa_entropy" \\
        --setting "2WikiMultiHopQA split 1 (a, b) and FinQA (c, d); K = 5, K' = 10" \\
        --panels "benchmark and uncertainty estimator"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ucagent.core.control.ltt import calibrate_knob
from ucagent.experiments.figures import STYLE, _save, aggregate_curves
from ucagent.experiments.suite import configured_knobs

# Categorical slots 1 and 2 of the validated reference palette (white surface: every check
# passes, both above 3:1 contrast); the baseline is a reference mark in a neutral ink, told
# apart by its dashes and its label, not by hue.
SERIES = {
    "random_ask": {"color": "#52514e", "linestyle": "--", "marker": "", "label": "random asking"},
    "per_step_eps": {
        "color": "#2a78d6",
        "linestyle": "-",
        "marker": "s",
        "label": "calibrated gate",
    },
    "marginal": {
        "color": "#eb6834",
        "linestyle": "-",
        "marker": "o",
        "label": "value-of-intervention rule",
    },
}
TEXT_PRIMARY, TEXT_MUTED, RULE = "#0b0b0b", "#52514e", "#b9b8b1"
TARGETS: tuple[float, ...] = (0.2, 0.1)  # --targets overrides
SETTING = (
    "MuSiQue 2-hop; K = 5, K' = 10; final answer judged by the same containment rule as the hops"
)


def selections(suite_dir: Path, policy: str) -> dict[float, float]:
    """Learn-then-Test knob at each target from the suite's calibration rows (feasible only)."""
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    cal = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))["base"].get(
        "calibration", {}
    )
    rows = manifest[
        (manifest["policy"] == policy)
        & (manifest["split"] == "calib")
        & (manifest["label"] == "grid")
    ].dropna(subset=["knob"])
    if "variant" in rows.columns:
        rows = rows[rows["variant"] == "main"]
    points = [
        (float(k), int(n), round(float(e) * int(n)))
        for k, n, e in zip(rows["knob"], rows["n"], rows["final_error"], strict=True)
    ]
    out: dict[float, float] = {}
    for alpha in TARGETS:
        res = calibrate_knob(
            policy,
            points,
            alpha=alpha,
            delta=float(cal.get("delta", 0.1)),
            pvalue=cal.get("pvalue", "binomial"),
            grid=configured_knobs(suite_dir, policy),
        )
        if res.feasible and res.knob_star is not None:
            out[alpha] = res.knob_star
    return out


def selected_point(sub: pd.DataFrame, knob: float, what: str) -> tuple[float, float]:
    """Cost and held-out error of the test run at exactly ``knob``.

    A selection is marked only where it was evaluated. With no test run at the selected knob
    (an interrupted suite) the figure stops; it does not mark the nearest other knob.
    """
    match = sub[(sub["knob"] - knob).abs() < 1e-9]
    if match.empty:
        raise SystemExit(
            f"{what}: no test run at the selected knob {knob:g}; the suite is incomplete"
        )
    return float(match["total_cost"].iloc[0]), float(match["final_error"].iloc[0])


def panel(ax: plt.Axes, suite_dir: Path, title: str) -> None:
    """One estimator: the three curves, the target lines and the labelled selections."""
    curves = aggregate_curves(pd.read_csv(suite_dir / "manifest.csv"), split="test")
    labelled: dict[tuple[float, float], list[str]] = {}  # one label per point, targets merged
    for alpha in TARGETS:
        ax.axhline(alpha, color=RULE, linestyle=":", linewidth=0.9, zorder=1)
        ax.annotate(
            f"target {alpha:g}",
            xy=(1.0, alpha),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=TEXT_MUTED,
        )
    for policy, style in SERIES.items():
        sub = curves[(curves["policy"] == policy) & (curves["label"] == "grid")]
        sub = sub.sort_values("total_cost")
        ax.plot(
            sub["total_cost"],
            sub["final_error"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"] or None,
            markersize=4.5,
            linewidth=1.4,
            zorder=3,
        )
        for alpha, knob in selections(suite_dir, policy).items():
            x, y = selected_point(sub, knob, f"{suite_dir.name} {policy} at target {alpha:g}")
            ax.plot(
                x,
                y,
                marker="*",
                markersize=12,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.4,
                linestyle="none",
                zorder=4,
            )
            # selections that nearly coincide share one label, so their texts do not collide
            near = [k for k in labelled if abs(k[0] - x) <= 0.06 and abs(k[1] - y) <= 0.01]
            texts = (
                labelled[near[0]] if near else labelled.setdefault((round(x, 3), round(y, 3)), [])
            )
            if f"{alpha:g}" not in texts:
                texts.append(f"{alpha:g}")
    for (x, y), texts in labelled.items():
        ax.annotate(
            ", ".join(texts),
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=TEXT_PRIMARY,
            zorder=5,
        )
    ax.set_title(title, loc="left", color=TEXT_PRIMARY)
    ax.set_xlabel("cost per question (one human check of one step = 1)")
    ax.set_xlim(0.0, 2.25)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="LABEL=PATH of a suite directory, one panel each, in the order given",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/figures/preprint"))
    parser.add_argument("--name", default="fig_estimators")
    parser.add_argument("--targets", default="0.2,0.1", help="the two targets to mark")
    parser.add_argument("--setting", default=SETTING, help="benchmark and settings (caption)")
    parser.add_argument("--panels", default="uncertainty estimator", help="what a panel is")
    args = parser.parse_args()
    global TARGETS
    TARGETS = tuple(float(x) for x in args.targets.split(","))
    if len(TARGETS) != 2:
        parser.error("give exactly two targets")
    suites = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.suite]
    manifest = pd.read_csv(suites[0][1] / "manifest.csv")
    n_test = int(manifest[manifest["split"] == "test"]["n"].iloc[0])
    n_cal = int(manifest[manifest["split"] == "calib"]["n"].iloc[0])
    backend = ", ".join(sorted(set(manifest["backend"].astype(str))))
    cols = 1 if len(suites) == 1 else 2
    rows = (len(suites) + cols - 1) // cols
    with plt.style.context(str(STYLE)):
        fig, grid = plt.subplots(
            rows, cols, figsize=(10.5, 3.6 * rows), sharex=True, sharey=True, squeeze=False
        )
        axes = list(grid.flat)
        for k, (ax, (label, path)) in enumerate(zip(axes, suites, strict=False)):
            panel(ax, path, f"({chr(ord('a') + k)}) {label}")
        for ax in axes[len(suites) :]:
            ax.set_visible(False)
        for ax in grid[:, 0]:
            ax.set_ylabel("held-out error")
        top = max(ax.get_ylim()[1] for ax in axes[: len(suites)])
        for ax in axes[: len(suites)]:
            ax.set_ylim(0.0, top)
            if rows > 1 and ax not in list(grid[-1, :]):
                ax.set_xlabel("")
        handles = [
            Line2D(
                [],
                [],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"] or None,
                markersize=4.5,
                linewidth=1.4,
                label=st["label"],
            )
            for st in SERIES.values()
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=TEXT_MUTED,
                linestyle="none",
                label="Learn-then-Test selection (labelled with its target)",
            )
        )
        fig.tight_layout()
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
        caption = (
            f"Held-out error against cost per question on the test split ({n_test} questions; "
            f"selections from {n_cal} calibration questions; {backend}; {args.setting}; one seed), "
            f"one panel per {args.panels}, each a pipeline with its own selections. Each "
            "curve runs along the controller's knob grid; random asking checks each step "
            "independently with a fixed probability. Stars mark the Learn-then-Test selections "
            f"at targets {TARGETS[0]:g} and {TARGETS[1]:g} "
            "(delta 0.1); a controller without a star at a target has no selection there."
        )
        # saved inside the style context so that its tight bounding box keeps the legend
        for path in _save(fig, args.out, args.name, caption):
            print("wrote", path)


if __name__ == "__main__":
    main()
