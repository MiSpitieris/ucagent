"""Figures from suite manifests and parquet files only; never from typed-in numbers.

Every figure is written as PDF and PNG with a caption file that states the
backend (mock or model name) and the sample sizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt

STYLE = Path(__file__).with_name("ucagent.mplstyle")
POLICY_ORDER = [
    "none",
    "random_ask",
    "final_u",
    "final_gate",
    "per_step_u",
    "per_step_eps",
    "budgeted",
    "marginal",
    "knowno",
    "abstain_ask",
]
POLICY_LABEL = {
    "none": "none",
    "random_ask": "random ask (mixing baseline)",
    "final_u": "final gate on u_n",
    "final_gate": "final gate on R_n (score & route)",
    "per_step_u": "per-step gate on u",
    "per_step_eps": "per-step gate on w·eps",
    "budgeted": "budgeted (spending)",
    "marginal": "marginal (value of intervention)",
    "knowno": "knowno (per-step conformal set)",
    "abstain_ask": "ask on abstention (baseline)",
}


def _save(fig: Figure, out_dir: Path, name: str, caption: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p)
    (out_dir / f"{name}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    plt.close(fig)
    return paths


def _backend_text(manifest: pd.DataFrame) -> str:
    backends = sorted(set(manifest["backend"].astype(str)))
    return "mock LLM" if backends == ["mock"] else ", ".join(backends)


def aggregate_curves(
    manifest: pd.DataFrame, split: str = "test", variant: str = "main"
) -> pd.DataFrame:
    """Mean error, cost and interventions per (policy, knob, label) over seeds."""
    rows = manifest[(manifest["split"] == split)].copy()
    if "variant" in rows.columns:
        rows = rows[rows["variant"] == variant]
    rows["knob"] = rows["knob"].astype(float)
    cols = ["final_error", "total_cost", "interventions_total", "escalation_rate", "n"]
    return (
        rows.groupby(["policy", "label", "knob"], dropna=False)[cols]
        .mean()
        .reset_index()
        .sort_values(["policy", "total_cost"])
    )


def fig2(
    suite_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    split: str = "test",
    selected: dict[str, float] | None = None,
) -> list[Path]:
    """Headline figure: error vs total cost (and vs interventions), one curve per policy."""
    suite_dir = Path(suite_dir)
    out_dir = Path(out_dir) if out_dir is not None else suite_dir / "figures"
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    curves = aggregate_curves(manifest, split)
    alpha = float(manifest["alpha"].iloc[0])
    if selected is None:
        from ucagent.experiments.suite import selected_knobs

        selected = selected_knobs(suite_dir) or None
    n_seeds = manifest["seed"].nunique()
    n_traj = int(manifest[manifest["split"] == split]["n"].iloc[0])
    with plt.style.context(str(STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, xcol, xlabel in [
            (axes[0], "total_cost", "total cost per trajectory"),
            (axes[1], "interventions_total", "interventions per trajectory"),
        ]:
            for policy in POLICY_ORDER:
                sub = curves[(curves["policy"] == policy) & (curves["label"] == "grid")]
                if sub.empty:
                    continue
                sub = sub.sort_values(xcol)
                marker = "s" if policy == "none" else "o"
                ax.plot(sub[xcol], sub["final_error"], marker=marker, label=POLICY_LABEL[policy])
                if selected and policy in selected:
                    pick = sub.iloc[(sub["knob"] - selected[policy]).abs().argsort()[:1]]
                    ax.plot(
                        pick[xcol],
                        pick["final_error"],
                        marker="*",
                        markersize=14,
                        color="black",
                        linestyle="none",
                    )
            extras = curves[curves["label"] != "grid"]
            for _, row in extras.iterrows():
                ax.plot(row[xcol], row["final_error"], marker="D", color="black", linestyle="none")
                ax.annotate(
                    row["label"],
                    (row[xcol], row["final_error"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
            ax.axhline(alpha, color="gray", linestyle=":", label=f"alpha = {alpha:g}")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(f"end-to-end error ({split} split)")
        axes[1].legend(loc="upper right")
        star = " Black stars mark the Learn-then-Test operating points." if selected else ""
        fig.suptitle("Error versus cost per policy")
        caption = (
            f"fig2: end-to-end error versus total cost (left) and interventions per trajectory "
            f"(right), one curve per policy over its knob grid; {_backend_text(manifest)}; "
            f"{n_traj} {split} trajectories per point, averaged over {n_seeds} seed(s).{star}"
        )
    return _save(fig, out_dir, "fig2_error_vs_cost", caption)


def plot_risk_vs_knob(suite_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Calibration-split risk versus knob per policy, with the selected knobs marked."""
    from ucagent.experiments.suite import selected_knobs

    suite_dir = Path(suite_dir)
    out_dir = Path(out_dir) if out_dir is not None else suite_dir / "figures"
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    rows = manifest[(manifest["split"] == "calib") & (manifest["label"] == "grid")].dropna(
        subset=["knob"]
    )
    if rows.empty:
        raise ValueError("no calibration-split grid rows in the manifest")
    alpha = float(manifest["alpha"].iloc[0])
    picks = selected_knobs(suite_dir)
    curves = rows.groupby(["policy", "knob"])["final_error"].mean().reset_index()
    with plt.style.context(str(STYLE)):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for policy in POLICY_ORDER:
            sub = curves[curves["policy"] == policy].sort_values("knob")
            if sub.empty:
                continue
            ax.plot(sub["knob"], sub["final_error"], marker="o", label=POLICY_LABEL[policy])
            if policy in picks:
                ax.axvline(picks[policy], color="gray", alpha=0.3)
                ax.plot(
                    [picks[policy]],
                    [sub["final_error"].iloc[(sub["knob"] - picks[policy]).abs().argmin()]],
                    marker="*",
                    markersize=13,
                    color="black",
                    linestyle="none",
                )
        ax.axhline(alpha, color="gray", linestyle=":", label=f"alpha = {alpha:g}")
        if (curves["knob"] > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel("knob (tau for gates and marginal, lambda for budgeted)")
        ax.set_ylabel("calibration-split error")
        ax.legend(loc="best")
        caption = (
            "risk versus knob on the calibration split, one curve per policy; stars mark the "
            f"Learn-then-Test selections (median over seeds); {_backend_text(manifest)}; "
            f"{int(rows['n'].iloc[0])} trajectories per point."
        )
    return _save(fig, out_dir, "risk_vs_knob", caption)


def fig1(summary_csv: str | Path, out_dir: str | Path) -> list[Path]:
    """Error versus chain length, and AUROC of three failure predictors per ``n``."""
    summary = pd.read_csv(summary_csv)
    out_dir = Path(out_dir)
    with plt.style.context(str(STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        g = summary.groupby("n_steps")
        ns = sorted(summary["n_steps"].unique())
        err = g["final_error"].mean().reindex(ns)
        axes[0].plot(ns, err.values, marker="o")
        axes[0].set_xlabel("chain length n")
        axes[0].set_ylabel("end-to-end error (policy none)")
        axes[0].set_ylim(bottom=0)
        for col, label in [
            ("auroc_u_final", "last-step u_n"),
            ("auroc_risk_naive", "naive R_n (w = 1)"),
            ("auroc_risk_weighted", "weighted R_n (counterfactual w)"),
        ]:
            if col in summary.columns:
                axes[1].plot(ns, g[col].mean().reindex(ns).values, marker="o", label=label)
        axes[1].axhline(0.5, color="gray", linestyle=":")
        axes[1].set_xlabel("chain length n")
        axes[1].set_ylabel("AUROC for final failure")
        axes[1].legend(loc="best")
        fig.suptitle("Compounding with chain length (H1)")
        n_test = int(summary["n"].iloc[0]) if "n" in summary.columns else 0
        backend = "mock LLM" if set(summary.get("backend", ["mock"])) == {"mock"} else "real model"
        caption = (
            f"fig1: left, end-to-end error of the uncontrolled agent versus chain length; right, "
            f"AUROC of the last step's uncertainty, the naive propagated risk and the "
            f"influence-weighted propagated risk as predictors of final failure; {backend}; "
            f"{n_test} test trajectories per n, averaged over {summary['seed'].nunique()} seed(s)."
        )
    return _save(fig, out_dir, "fig1_error_vs_n", caption)


def _steps_of_interventions(parquet: Path) -> pd.Series:
    """Non-proceed decisions per step index (mean per trajectory) from one run."""
    import json

    frame = pd.read_parquet(parquet, columns=["interventions_json", "n_steps"])
    counts: dict[int, float] = {}
    for text in frame["interventions_json"]:
        for iv in json.loads(text):
            if iv["action"] != "proceed":
                counts[int(iv["step"])] = counts.get(int(iv["step"]), 0.0) + 1.0
    n = int(frame["n_steps"].iloc[0]) if len(frame) else 0
    total = max(len(frame), 1)
    return pd.Series({i: counts.get(i, 0.0) / total for i in range(1, n + 1)})


def fig3(
    suite_dir: str | Path, out_dir: str | Path | None = None, *, split: str = "test"
) -> list[Path]:
    """Where interventions land: step index x policy heatmap, with ``w_i`` as a side panel."""
    from ucagent.experiments.suite import load_artifact_for, selected_knobs

    suite_dir = Path(suite_dir)
    out_dir = Path(out_dir) if out_dir is not None else suite_dir / "figures"
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    rows = manifest[(manifest["split"] == split) & (manifest["label"] == "grid")]
    if "variant" in rows.columns:
        rows = rows[rows["variant"] == "main"]
    picks = selected_knobs(suite_dir)
    seed0 = int(rows["seed"].min())
    table: dict[str, pd.Series] = {}
    knob_used: dict[str, float | None] = {}
    for policy in POLICY_ORDER:
        sub = rows[(rows["policy"] == policy) & (rows["seed"] == seed0)]
        if sub.empty or policy == "none":
            continue
        knobs = sub["knob"].dropna().astype(float)
        if knobs.empty:
            continue
        target = picks.get(policy, float(knobs.sort_values().iloc[len(knobs) // 2]))
        row = sub.iloc[(knobs - target).abs().argsort().iloc[0]]
        table[policy] = _steps_of_interventions(Path(row["path"]))
        knob_used[policy] = float(row["knob"])
    if not table:
        raise ValueError("no policy runs to plot")
    heat = pd.DataFrame(table).T.fillna(0.0)
    n = heat.shape[1]
    try:
        art = load_artifact_for(suite_dir, seed0)
        w = [art.influence.get(f"{n}:{i}", 1.0) for i in range(1, n + 1)]
        w_label = art.influence_method
    except FileNotFoundError:
        w, w_label = [1.0] * n, "ones"
    with plt.style.context(str(STYLE)):
        fig, (ax, side) = plt.subplots(
            1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [4, 1.2]}, sharey=False
        )
        im = ax.imshow(heat.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(n), [str(i) for i in range(1, n + 1)])
        ax.set_yticks(range(len(heat.index)), [POLICY_LABEL[p] for p in heat.index])
        ax.set_xlabel("step index")
        ax.grid(False)
        for r in range(heat.shape[0]):
            for c in range(n):
                ax.text(c, r, f"{heat.values[r, c]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="interventions per trajectory")
        side.bar(range(1, n + 1), w)
        side.set_ylim(0, 1.05)
        side.set_xlabel("step index")
        side.set_title(f"influence w_i ({w_label})", fontsize=9)
        fig.suptitle("Where interventions land (at the calibrated knob)")
        knob_txt = ", ".join(f"{p}={k:.3g}" for p, k in knob_used.items() if k is not None)
        caption = (
            f"fig3: mean non-proceed decisions per trajectory by step index and policy on the "
            f"{split} split at each policy's Learn-then-Test knob (or the middle grid knob when "
            f"none was selected): {knob_txt}; side panel: downstream influence w_i "
            f"({w_label}); {_backend_text(manifest)}; seed {seed0}."
        )
    return _save(fig, out_dir, "fig3_intervention_locations", caption)


def fig4(
    suite_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    split: str = "test",
    n_bins: int = 8,
) -> list[Path]:
    """Reliability of projected risk versus observed error; certificate coverage and validity."""
    suite_dir = Path(suite_dir)
    out_dir = Path(out_dir) if out_dir is not None else suite_dir / "figures"
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    rows = manifest[manifest["split"] == split]
    if "variant" in rows.columns:
        rows = rows[rows["variant"] == "main"]
    none_rows = rows[rows["policy"] == "none"]
    if none_rows.empty:
        raise ValueError("fig4 needs policy-none runs")
    frames = [
        pd.read_parquet(p, columns=["risk_weighted_final", "final_correct"])
        for p in none_rows["path"]
    ]
    data = pd.concat(frames, ignore_index=True)
    data["wrong"] = ~data["final_correct"].astype(bool)
    alpha = float(manifest["alpha"].iloc[0])
    edges = [k / n_bins for k in range(n_bins + 1)]
    centers, observed, counts = [], [], []
    for lo, hi in itertools.pairwise(edges):
        mask = (data["risk_weighted_final"] >= lo) & (
            (data["risk_weighted_final"] < hi) if hi < 1 else (data["risk_weighted_final"] <= hi)
        )
        if mask.sum() >= 3:
            centers.append(float(data.loc[mask, "risk_weighted_final"].mean()))
            observed.append(float(data.loc[mask, "wrong"].mean()))
            counts.append(int(mask.sum()))
    budget = rows[(rows["policy"] == "budgeted") & (rows["label"] == "grid")].copy()
    with plt.style.context(str(STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":")
        axes[0].plot(centers, observed, marker="o")
        for x, y, c in zip(centers, observed, counts, strict=True):
            axes[0].annotate(str(c), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        axes[0].set_xlabel("projected final risk R_n (weighted)")
        axes[0].set_ylabel("observed error rate")
        axes[0].set_title("reliability (policy none)", fontsize=10)
        if not budget.empty:
            b = (
                budget.groupby("knob")[
                    ["certificate_coverage", "certificate_validity", "final_error"]
                ]
                .mean()
                .sort_index()
            )
            axes[1].plot(
                b.index, b["certificate_coverage"], marker="o", label="certificate coverage"
            )
            axes[1].plot(
                b.index, b["certificate_validity"], marker="s", label="error among certified"
            )
            axes[1].plot(b.index, b["final_error"], marker="^", label="error, all outputs")
            axes[1].axhline(alpha, color="gray", linestyle=":", label=f"alpha = {alpha:g}")
            axes[1].set_xlabel("lambda (budgeted controller)")
            axes[1].legend(loc="best")
            axes[1].set_title("per-output certificate (budgeted)", fontsize=10)
        fig.suptitle("Calibration of projected risk and the certificate")
        caption = (
            f"fig4: left, observed error rate against the projected weighted risk R_n of the "
            f"uncontrolled agent in {n_bins} equal-width bins (counts annotated), {len(data)} "
            f"{split} trajectories pooled over seeds; right, fraction of outputs delivered with "
            f"R_n <= alpha, the error among them, and the overall error of the budgeted "
            f"controller along its lambda grid; {_backend_text(manifest)}."
        )
    return _save(fig, out_dir, "fig4_reliability_certificate", caption)
