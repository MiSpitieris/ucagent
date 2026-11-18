"""M6 acceptance on the mock: weighted risk beats naive risk where influence is heterogeneous."""

import pytest
import yaml

from ucagent.experiments.io import read_trajectories
from ucagent.experiments.metrics import auroc
from ucagent.experiments.suite import SuiteConfig, load_manifest, run_suite

pytestmark = pytest.mark.slow


def test_h2_weighted_risk_predicts_failure_better_on_reconverge(tmp_path) -> None:
    data = {
        "suite_id": "m6h2",
        "base": {
            "run_id": "m6h2",
            "task": {
                "n_steps": 5,
                "ambiguity": 0.5,
                "distractors": 3,
                "reconvergence": [0.6, 0.6, 0.3, 0.0, 0.0],
                "n_instances": {"train": 150, "calib": 1, "test": 300},
            },
            "llm": {"K": 10, "K_prime": 20},
            "uq": {"estimator": "semantic_entropy"},
            "risk": {"influence": "counterfactual"},
            "control": {"alpha": 0.1},
            "limits": {"max_concurrency": 8},
        },
        "policies": {"none": None},
        "seeds": [0, 1],
        "splits": ["test"],
    }
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(data))
    root = run_suite(SuiteConfig.from_yaml(path), workers=8, results_dir=tmp_path / "r")
    manifest = load_manifest(root)
    gains = []
    for _, row in manifest.iterrows():
        trajs = read_trajectories(row["path"])
        wrong = [not t.final_correct for t in trajs]
        naive = auroc([t.risk_naive_final for t in trajs], wrong)
        weighted = auroc([t.risk_weighted_final for t in trajs], wrong)
        last = auroc([t.u_final or 0.0 for t in trajs], wrong)
        gains.append((row["seed"], naive, weighted, last))
    for _seed, naive, weighted, last in gains:
        assert weighted >= naive + 0.01, gains
        assert weighted > last, gains
