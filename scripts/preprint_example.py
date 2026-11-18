"""The recorded decision of the value-of-intervention rule quoted in the preprint (Section 3).

Usage::

    uv run python scripts/preprint_example.py results/musique_wide_confagree \\
        --instance musique-test-00060-s0 --target 0.1

Prints the rule's Learn-then-Test selection at the target in the development suite, then,
for one test question at that knob, every decision on its first step with the numbers the
controller recorded (the step error, the expected survival of the steps to come, the projected
final risk, and the risk reduction and net value of each action), the stated confidences of
the samples, and whether the committed answer was locally correct. Read from the run records;
no model calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprint_results as pr
from price_sensitivity import _test_run
from ucagent.experiments.io import read_trajectories


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    sel = pr.select(args.suite_dir, "marginal", args.target)
    if not sel.feasible or sel.knob is None:
        raise SystemExit("the rule certifies nothing at this target")
    print(f"rule (marginal) selected at {args.target:g}: knob tau = {sel.knob:.4g}")
    run = _test_run(args.suite_dir, "marginal", sel.knob)
    traj = next(t for t in read_trajectories(str(run)) if t.instance_id == args.instance)
    for iv in traj.interventions:
        if iv.step != args.step:
            continue
        nums = ", ".join(f"{k} {v:.4f}" for k, v in iv.numbers.items())
        print(
            f"step {iv.step}, attempt {iv.attempt}: {iv.action.value} (price {iv.cost:g}); {nums}"
        )
    for s in traj.steps:
        if s.step != args.step:
            continue
        conf = traj.sample_confidences.get(s.samples_key) or []
        samples = traj.samples.get(s.samples_key, [])
        print(
            f"step {s.step}, attempt {s.attempt}: {len(samples)} samples, "
            f"{len(set(samples))} distinct answers, stated confidences "
            f"{', '.join('-' if c is None else f'{c:g}' for c in conf)}; u {s.u:.4f}, "
            f"eps {s.eps:.4f}; committed answer locally correct: {s.locally_correct}"
        )


if __name__ == "__main__":
    main()
