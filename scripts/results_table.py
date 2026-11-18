"""Render the results tables of docs/EXPERIMENTS.md from run outputs (never by hand).

Usage: uv run python scripts/results_table.py [results_dir] [pipeline_id]
Prints Markdown; the tables are pasted into docs/EXPERIMENTS.md under "Results".

``--suite <dir>`` renders one suite directory. With ``--tex <path>`` after it, the
Learn-then-Test table of that suite is also written as a LaTeX table body (the validation
table of the preprint's appendix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from ucagent.core.control.ltt import calibrate_knob
from ucagent.experiments.figures import POLICY_ORDER, aggregate_curves
from ucagent.experiments.suite import configured_knobs


def ltt_table(suite_dir: Path) -> str:
    ltt = pd.read_csv(suite_dir / "ltt.csv")
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    none_rows = manifest[(manifest["policy"] == "none") & (manifest["split"] == "test")]
    alpha = float(ltt["alpha"].iloc[0])
    lines = [
        f"Learn-then-Test at alpha = {alpha:g}, delta = 0.1, per policy, mean over "
        f"{ltt['seed'].nunique()} seeds (held-out test split at the selected knob); "
        f"uncontrolled (policy none) test error {none_rows['final_error'].mean():.3f}, "
        f"cost {none_rows['total_cost'].mean():.2f}.",
        "",
        "| policy | feasible seeds | test error at knob* | test cost at knob* "
        "| interventions at knob* | coverage |",
        "|---|---|---|---|---|---|",
    ]
    cov = json.loads((suite_dir / "coverage.json").read_text())
    for policy in POLICY_ORDER:
        sub = ltt[ltt["policy"] == policy]
        if sub.empty:
            continue
        feasible = sub[sub["feasible"]]
        coverage = cov.get(policy, {}).get("guarantee_coverage")
        cov_txt = "-" if coverage is None else f"{coverage:.2f}"
        if feasible.empty:
            lines.append(f"| {policy} | 0/{len(sub)} | - | - | - | {cov_txt} |")
            continue
        lines.append(
            f"| {policy} | {len(feasible)}/{len(sub)} | "
            f"{feasible['test_error_at_star'].mean():.3f} | "
            f"{feasible['test_cost_at_star'].mean():.2f} | "
            f"{feasible['test_interventions_at_star'].mean():.2f} | {cov_txt} |"
        )
    return "\n".join(lines)


TEX_NAME = {
    "random_ask": "random asking",
    "final_u": "final gate on the raw score",
    "final_gate": "final gate (score and route)",
    "per_step_u": "gate on the raw score",
    "per_step_eps": "calibrated gate",
    "budgeted": "budgeted",
    "marginal": "value-of-intervention rule",
    "knowno": "KnowNo-style gate",
    "abstain_ask": "abstention rule",
}


def ltt_tex(suite_dir: Path) -> str:
    """The Learn-then-Test table of a suite as a LaTeX table body (means over feasible seeds)."""
    ltt = pd.read_csv(suite_dir / "ltt.csv")
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    none_rows = manifest[(manifest["policy"] == "none") & (manifest["split"] == "test")]
    lines = [
        "% written by scripts/results_table.py; do not edit by hand",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"controller & feasible seeds & held-out error & cost & interventions \\",
        r"\midrule",
        f"none (uncontrolled) & -- & {none_rows['final_error'].mean():.3f} & "
        f"{none_rows['total_cost'].mean():.2f} & 0.00 \\\\",
    ]
    for policy in POLICY_ORDER:
        sub = ltt[ltt["policy"] == policy]
        if sub.empty or policy == "none":
            continue
        feasible = sub[sub["feasible"]]
        name = TEX_NAME.get(policy, policy)
        if feasible.empty:
            lines.append(f"{name} & 0/{len(sub)} & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{name} & {len(feasible)}/{len(sub)} & "
            f"{feasible['test_error_at_star'].mean():.3f} & "
            f"{feasible['test_cost_at_star'].mean():.2f} & "
            f"{feasible['test_interventions_at_star'].mean():.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def curve_table(suite_dir: Path) -> str:
    curves = aggregate_curves(pd.read_csv(suite_dir / "manifest.csv"))
    lines = [
        "Error and cost along each policy's knob grid (test split, mean over seeds):",
        "",
        "| policy | knob | error | cost | interventions |",
        "|---|---|---|---|---|",
    ]
    for policy in POLICY_ORDER:
        sub = curves[(curves["policy"] == policy) & (curves["label"] == "grid")].sort_values("knob")
        for _, r in sub.iterrows():
            knob = "-" if pd.isna(r["knob"]) else f"{r['knob']:.3g}"
            lines.append(
                f"| {policy} | {knob} | {r['final_error']:.3f} | {r['total_cost']:.2f} | "
                f"{r['interventions_total']:.2f} |"
            )
    return "\n".join(lines)


def fig1_table(csv: Path) -> str:
    s = pd.read_csv(csv)
    lines = [
        "Chain-length stage (policy none, 100 test trajectories per n):",
        "",
        "| n | error | AUROC last-step u | AUROC naive R_n | AUROC weighted R_n |",
        "|---|---|---|---|---|",
    ]
    for _, r in s.sort_values("n_steps").iterrows():
        lines.append(
            f"| {int(r['n_steps'])} | {r['final_error']:.3f} | {r['auroc_u_final']:.3f} | "
            f"{r['auroc_risk_naive']:.3f} | {r['auroc_risk_weighted']:.3f} |"
        )
    return "\n".join(lines)


def variant_ltt_table(suite_dir: Path) -> str:
    """Learn-then-Test per (variant, policy) from the manifest, for ablation suites.

    ``ucagent calibrate`` writes ``ltt.csv`` for the ``main`` variant only; this
    table re-runs the same fixed-sequence test for every variant.
    """
    manifest = pd.read_csv(suite_dir / "manifest.csv")
    if "variant" not in manifest.columns or manifest["variant"].nunique() < 2:
        return ""
    suite = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8"))
    cal = suite["base"].get("calibration", {})
    delta, pvalue = float(cal.get("delta", 0.1)), cal.get("pvalue", "binomial")
    grid = manifest[manifest["label"] == "grid"]
    lines = [
        f"Learn-then-Test per variant (delta = {delta:g}; alpha from each run's config; "
        "held-out test split at the selected knob):",
        "",
        "| variant | policy | feasible | knob* | calib risk | test error | test cost "
        "| interventions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in list(dict.fromkeys(grid["variant"])):
        rows = grid[grid["variant"] == variant]
        for policy in POLICY_ORDER:
            sub = rows[rows["policy"] == policy]
            if sub.empty or policy == "none":
                continue
            c = sub[sub["split"] == "calib"].dropna(subset=["knob"])
            points = [
                (float(k), int(n), round(float(e) * int(n)))
                for k, n, e in zip(c["knob"], c["n"], c["final_error"], strict=True)
            ]
            res = calibrate_knob(
                policy,
                points,
                alpha=float(sub["alpha"].iloc[0]),
                delta=delta,
                pvalue=pvalue,
                grid=configured_knobs(suite_dir, policy),
            )
            t = sub[(sub["split"] == "test") & (sub["knob"] == res.knob_star)]
            if not res.feasible or t.empty or res.selected is None:
                lines.append(
                    f"| {variant} | {policy} | no | - | {res.min_risk_hat:.3f} | - | - | - |"
                )
                continue
            lines.append(
                f"| {variant} | {policy} | yes | {res.knob_star:.4g} | "
                f"{res.selected.risk_hat:.3f} | {float(t['final_error'].iloc[0]):.3f} | "
                f"{float(t['total_cost'].iloc[0]):.2f} | "
                f"{float(t['interventions_total'].iloc[0]):.2f} |"
            )
    return "\n".join(lines)


def influence_line(suite_dir: Path) -> str:
    art = json.loads((suite_dir / "artifacts" / "seed0.json").read_text())
    w = art.get("influence", {})
    if not w:
        return ""
    items = ", ".join(f"w_{k.split(':')[1]} = {v:.2f}" for k, v in sorted(w.items()))
    return f"Counterfactual influence fitted for seed 0 ({art.get('influence_method')}): {items}."


def main(results_dir: str = "results", pipeline_id: str = "figures_small", *rest: str) -> None:
    if results_dir == "--suite":  # one suite directory, e.g. an ablation run on its own
        suite_dir = Path(pipeline_id)
        print(f"### Suite `{suite_dir}`\n")
        if (suite_dir / "ltt.csv").exists():
            print(ltt_table(suite_dir) + "\n")
            if len(rest) == 2 and rest[0] == "--tex":
                out = Path(rest[1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(ltt_tex(suite_dir), encoding="utf-8")
                print(f"wrote {out}\n")
        variants = variant_ltt_table(suite_dir)
        if variants:
            print(variants + "\n")
        print(curve_table(suite_dir) + "\n")
        return
    root = Path(results_dir)
    pipe = json.loads((root / pipeline_id / "pipeline.json").read_text())
    suite_dir = Path(pipe["suite_dir"])
    print(f"### Main suite (`{suite_dir}`)\n")
    print(influence_line(suite_dir) + "\n")
    print(ltt_table(suite_dir) + "\n")
    print(curve_table(suite_dir) + "\n")
    if "reconverge_dir" in pipe:
        rec = Path(pipe["reconverge_dir"])
        print(f"### Reconvergence suite (`{rec}`)\n")
        print(influence_line(rec) + "\n")
        print(ltt_table(rec) + "\n")
    print("### Chain length\n")
    print(fig1_table(root / pipeline_id / "fig1" / "fig1_summary.csv") + "\n")
    t = pipe.get("timings_s", {})
    print(
        f"Wall time of `make figures`: {t.get('total', float('nan')):.0f} s on "
        f"{pipe.get('workers')} workers (suite {t.get('suite', 0):.0f} s, reconvergence suite "
        f"{t.get('reconverge_suite', 0):.0f} s, chain-length stage {t.get('fig1_stage', 0):.0f} s)."
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
