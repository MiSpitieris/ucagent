"""Re-price the recorded runs of the controllers whose decisions do not depend on prices.

Usage::

    uv run python scripts/price_sensitivity.py \\
        --suite "confidence-weighted agreement=results/musique_confirm2_confagree" \\
        --suite "verbalized confidence=results/musique_confirm2_verbalized" \\
        --targets 0.1,0.2 --llm-prices 0.01,0.05,0.2

The calibrated gate (``per_step_eps``) and random asking (``random_ask``) only choose between
proceeding and asking a person, so their decisions, their calibration errors and hence their
Learn-then-Test selections are the same under any price table. Their test runs can therefore
be re-priced exactly: a trajectory costs its model calls times the price per call plus a fixed
price per action taken. The script first checks that re-pricing at the configured prices
reproduces every stored cost. It then prints, for each target, the selection's held-out error,
checks and calls per question, its cost at each price per call (a check costs 1), and the price
per call at which the selection costs as much as a person checking every step without the model
(the mean number of steps per question, which on a benchmark of mixed lengths is less than the
longest question). The value-of-intervention rule and the budgeted controller weigh prices
when they decide, so their runs cannot be re-priced and are left out. No model calls.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from ucagent.core.records import Action
from ucagent.experiments.io import read_trajectories

PRICE_FREE = ("per_step_eps", "random_ask")


def _test_run(suite_dir: Path, policy: str, knob: float) -> Path:
    grid = pr._grid(suite_dir)
    rows = grid[(grid["policy"] == policy) & (grid["split"] == "test")]
    rows = rows[(rows["knob"] - knob).abs() < 1e-9]
    path = Path(str(rows["path"].iloc[0]))
    return path if path.is_absolute() else suite_dir.parent.parent / path


def direct_review_checks(n_steps: Sequence[int]) -> float:
    """Checks per question of a person who checks every step without the model.

    The mean length of the questions. The longest question would overstate what direct review
    costs whenever lengths differ (FinQA: 4 against a mean of 2.12).
    """
    return sum(n_steps) / len(n_steps)


def break_even(direct: float, checks: float, calls: float) -> float:
    """Price per call at which ``checks`` checks plus ``calls`` calls cost as much as ``direct``."""
    return (direct - checks) / calls if calls else float("nan")


def counts(suite_dir: Path, policy: str, knob: float) -> dict[str, float]:
    """Mean calls, checks and other actions per question at a knob, after the cost check."""
    costs = pr._base(suite_dir)["control"]["costs"]
    trajs = read_trajectories(str(_test_run(suite_dir, policy, knob)))
    n = len(trajs)
    out = {
        "calls": sum(t.llm_calls for t in trajs) / n,
        "checks": sum(t.count(Action.ASK_HUMAN) for t in trajs) / n,
        "other": sum(
            t.count(Action.RESAMPLE) + t.count(Action.BACKTRACK) + t.count(Action.ESCALATE)
            for t in trajs
        )
        / n,
        "n_steps": direct_review_checks([t.n_steps for t in trajs]),
    }
    for t in trajs:
        repriced = (
            t.llm_calls * costs["llm_call"]
            + t.count(Action.ASK_HUMAN) * costs["ask_human"]
            + t.count(Action.RESAMPLE) * costs["resample"]
            + t.count(Action.BACKTRACK) * costs["backtrack"]
            + t.count(Action.ESCALATE) * costs["escalate"]
        )
        if abs(repriced - t.total_cost) > 1e-9:
            raise SystemExit(f"{suite_dir.name} {t.instance_id}: {repriced} != {t.total_cost}")
    if out["other"] > 0:
        raise SystemExit(f"{suite_dir.name} {policy} took actions other than asking")
    return out


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--targets", default="0.1,0.2")
    parser.add_argument("--llm-prices", default="0.01,0.05,0.2")
    args = parser.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    prices = [float(x) for x in args.llm_prices.split(",")]
    suites = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.suite]
    head = " | ".join(f"cost at {p:g}" for p in prices)
    lines = [
        "Re-priced test runs of the controllers that decide without prices (a check costs 1; "
        "every stored cost reproduced at the configured prices):",
        "",
        f"| estimator | controller | target | knob | held-out error | checks | calls | {head} "
        "| price per call at which it costs as much as checking every step |",
        "|---|---|---|---|---|---|---|" + "---|" * len(prices) + "---|",
    ]
    rows = []
    for label, suite_dir in suites:
        for policy in PRICE_FREE:
            for alpha in targets:
                sel = pr.select(suite_dir, policy, alpha)
                if not sel.feasible or sel.knob is None:
                    lines.append(f"| {label} | {policy} | {alpha:g} | not certified |")
                    continue
                c = counts(suite_dir, policy, sel.knob)
                cells = " | ".join(f"{c['calls'] * p + c['checks']:.2f}" for p in prices)
                even = break_even(c["n_steps"], c["checks"], c["calls"])
                lines.append(
                    f"| {label} | {policy} | {alpha:g} | {sel.knob:.4g} | {sel.test_error:.3f} "
                    f"| {c['checks']:.3f} | {c['calls']:.2f} | {cells} | {even:.3f} |"
                )
                rows.append({"label": label, "policy": policy, "alpha": alpha, **c})
    print("\n".join(lines))
    if rows:
        same = pd.DataFrame(rows).groupby("alpha")["calls"].nunique().max() == 1
        print()
        print(
            "Calls per question are the same at every selection, so a price per call shifts "
            "every cost above by the same amount."
            if same
            else "Calls per question differ between selections."
        )


if __name__ == "__main__":
    main()
