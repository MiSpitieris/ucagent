"""M2 acceptance on the mock: informative u, calibration reduces ECE, resampling helps."""

import pytest

from tests.helpers.configs import small_config
from ucagent.api import fit_calibrators, observations_from
from ucagent.core.uq.calibration import IsotonicCalibrator, bucket_of
from ucagent.experiments.metrics import auroc, ece, step_pairs
from ucagent.experiments.runner import run_experiment

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("estimator", ["disagreement", "semantic_entropy"])
def test_u_is_informative_and_calibration_helps(tmp_path, estimator: str) -> None:
    cfg = small_config(
        run_id=f"m2-{estimator}",
        task={
            "n_steps": 3,
            "ambiguity": 0.5,
            "distractors": 3,
            "n_instances": {"train": 200, "calib": 200},
        },
        llm={"K": 10, "K_prime": 20},
        uq={"estimator": estimator},
    )
    art = fit_calibrators(cfg, workers=4, out_path=tmp_path / "art.json")
    assert art.n_trajectories == 200
    calib_cfg = cfg.with_updates(uq={"artifact": str(tmp_path / "art.json")}).model_copy(
        update={"split": "calib"}
    )
    res = run_experiment(calib_cfg, workers=4, write=False)
    u, eps, wrong = step_pairs(res.trajectories)
    assert len(u) == 600
    assert auroc(u, wrong) >= 0.65, auroc(u, wrong)
    e_u, e_eps = ece(u, wrong), ece(eps, wrong)
    assert e_eps <= e_u - 0.02 and e_eps <= 0.08, (e_u, e_eps)
    cal = IsotonicCalibrator(art.isotonic_K)
    assert all(abs(e - cal(x)) < 1e-9 for x, e in zip(u, eps, strict=True))


def test_resampling_lowers_expected_error_in_uncertain_buckets(tmp_path) -> None:
    cfg = small_config(
        run_id="m2-resample",
        task={"n_steps": 3, "ambiguity": 0.5, "n_instances": {"train": 200}},
        llm={"K": 10, "K_prime": 20},
    )
    art = fit_calibrators(cfg, workers=4)
    fit_cfg = cfg.with_updates(control={"policy": "none"}, llm={"K": 20}).model_copy(
        update={"split": "train"}
    )
    obs = [
        o
        for rows in observations_from(
            cfg, run_experiment(fit_cfg, workers=4, write=False).trajectories
        )
        for o in rows
    ]
    cal = IsotonicCalibrator(art.isotonic_K)
    edges = art.resample_table.bucket_edges
    by_bucket: dict[int, list[float]] = {}
    for o in obs:
        by_bucket.setdefault(bucket_of(o.u_K, edges), []).append(cal(o.u_K))
    checked = 0
    for b in sorted(by_bucket, reverse=True)[:2]:
        before = sum(by_bucket[b]) / len(by_bucket[b])
        after = art.resample_table.ehat[str(b)]
        assert after <= before - 0.02, (b, before, after)
        checked += 1
    assert checked == 2
