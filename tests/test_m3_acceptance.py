"""M3 acceptance on the mock: more interventions as the knob tightens; fig2 ordering report."""

import itertools
import logging

import pytest
import yaml

from ucagent.experiments.figures import aggregate_curves
from ucagent.experiments.suite import SuiteConfig, load_manifest, run_suite

pytestmark = pytest.mark.slow
log = logging.getLogger("ucagent.tests")


def _suite(tmp_path, policies: dict, n_test: int, n_train: int = 100) -> SuiteConfig:
    data = {
        "suite_id": "m3",
        "base": {
            "run_id": "m3",
            "task": {
                "n_steps": 5,
                "ambiguity": 0.3,
                "distractors": 3,
                "n_instances": {"train": n_train, "calib": n_test, "test": n_test},
            },
            "llm": {"K": 10, "K_prime": 20},
            "uq": {"estimator": "semantic_entropy"},
            "control": {"alpha": 0.1, "max_resamples": 1, "max_backtracks": 1},
            "limits": {"max_concurrency": 8},
        },
        "policies": policies,
        "seeds": [0],
        "splits": ["test"],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data))
    return SuiteConfig.from_yaml(path)


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        for rank, idx in enumerate(order):
            r[idx] = float(rank)
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def test_tighter_knob_means_more_cost_and_less_error(tmp_path) -> None:
    """Tightening the knob never lowers cost nor raises error (within tolerance).

    The *count* of interventions need not be monotone for ``marginal``: one
    escalation can replace several step checks. Cost and error are the
    monotone quantities; the count is checked for the fixed gate only.
    """
    policies = {
        "per_step_eps": {"kind": "log", "min": 0.0005, "max": 0.5, "points": 6},
        "marginal": {"kind": "log", "min": 0.002, "max": 1.0, "points": 6},
    }
    root = run_suite(_suite(tmp_path, policies, n_test=100), workers=8, results_dir=tmp_path / "r")
    manifest = load_manifest(root)
    for policy in policies:
        sub = manifest[manifest["policy"] == policy].sort_values("knob")
        cost = sub["total_cost"].tolist()
        errors = sub["final_error"].tolist()
        inter = sub["interventions_total"].tolist()
        for a, b in itertools.pairwise(cost):
            assert b <= a + 0.05, (policy, cost)
        for a, b in itertools.pairwise(errors):
            assert b >= a - 0.02, (policy, errors)
        assert errors[0] <= errors[-1] - 0.05, (policy, errors)
        assert cost[0] >= cost[-1] + 0.3, (policy, cost)
        if policy == "per_step_eps":
            for a, b in itertools.pairwise(inter):
                assert b <= a + 0.05, (policy, inter)
            assert inter[0] >= inter[-1] + 0.3, (policy, inter)


def test_fig2_ordering_is_reported_not_asserted(tmp_path, capsys) -> None:
    policies = {
        "none": None,
        "final_u": {"kind": "linear", "min": 0.1, "max": 0.9, "points": 3},
        "final_gate": {"kind": "log", "min": 0.02, "max": 0.5, "points": 3},
        "per_step_u": {"kind": "linear", "min": 0.1, "max": 0.9, "points": 3},
        "per_step_eps": {"kind": "log", "min": 0.01, "max": 0.5, "points": 3},
        "budgeted": {"kind": "linear", "min": 0.5, "max": 2.0, "points": 3},
        "marginal": {"kind": "log", "min": 0.01, "max": 1.0, "points": 3},
    }
    root = run_suite(
        _suite(tmp_path, policies, n_test=40, n_train=60), workers=8, results_dir=tmp_path / "r2"
    )
    curves = aggregate_curves(load_manifest(root))
    lines = ["policy knob error cost interventions"]
    for _, row in curves.iterrows():
        lines.append(
            f"{row['policy']:>13} {row['knob']!s:>8} {row['final_error']:.3f} "
            f"{row['total_cost']:.2f} {row['interventions_total']:.2f}"
        )
    report = "\n".join(lines)
    (root / "ordering.md").write_text("```\n" + report + "\n```\n")
    print(report)
    assert set(curves["policy"]) == set(policies)
