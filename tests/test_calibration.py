import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ucagent.core.control  # noqa: F401
from ucagent.core.control.base import make_policy
from ucagent.core.graph.runtime import CostTable, RunSettings, init_state
from ucagent.core.protocols import Instance
from ucagent.core.records import Action, StepRecord
from ucagent.core.tables import Tables
from ucagent.core.uq.calibration import (
    CalibrationArtifact,
    IsotonicCalibrator,
    StepObservation,
    bucket_of,
    fit_artifact,
    load_artifact,
)


def test_isotonic_hand_example() -> None:
    cal = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 1, 1, 1])
    assert [round(cal(u), 6) for u in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] == [0, 0.5, 0.5, 1, 1, 1]
    assert cal(0.0) == 0.0 and cal(1.0) == 1.0  # clipped outside the fitted range
    assert cal(0.25) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        IsotonicCalibrator.fit([], [])


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.tuples(st.floats(0, 1), st.booleans()), min_size=2, max_size=40),
    st.lists(st.floats(0, 1), min_size=2, max_size=10),
)
def test_isotonic_is_monotone_and_bounded(data, probes) -> None:
    cal = IsotonicCalibrator.fit([u for u, _ in data], [w for _, w in data])
    values = [cal(p) for p in sorted(probes)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(values))


def test_bucket_of() -> None:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8]
    assert [bucket_of(u, edges) for u in [0.0, 0.19, 0.2, 0.59, 0.95, 1.0]] == [0, 0, 1, 2, 4, 4]


def test_a_score_that_equals_an_edge_up_to_rounding_is_in_the_range_above() -> None:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8]
    assert 1 - 0.8 < 0.2  # four of five samples agree: 0.19999999999999996
    for top_share in (4 / 5, 8 / 10, 16 / 20):
        assert bucket_of(1 - top_share, edges) == 1
    assert bucket_of(0.2 - 1e-6, edges) == 0  # a score really below the edge stays below
    assert bucket_of(1 - 0.8, edges, tol=0.0) == 0  # how tables before schema 2 were fitted


def _obs(n: int, step: int, uk: float, wk: bool, ukp: float, wkp: bool) -> StepObservation:
    return StepObservation(
        n_steps=n, step=step, u_K=uk, wrong_K=wk, u_K_prime=ukp, wrong_K_prime=wkp
    )


def _artifact() -> CalibrationArtifact:
    trajs = [
        [_obs(2, 1, 0.1, False, 0.05, False), _obs(2, 2, 0.7, True, 0.3, False)],
        [_obs(2, 1, 0.3, False, 0.1, False), _obs(2, 2, 0.9, True, 0.6, True)],
        [_obs(2, 1, 0.6, True, 0.2, False), _obs(2, 2, 0.2, False, 0.1, False)],
    ]
    flat = [o for t in trajs for o in t]
    return fit_artifact(
        flat,
        estimator="disagreement",
        equivalence="normalized_exact",
        K=5,
        K_prime=10,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=3,
        trajectories=trajs,
    )


def test_fit_artifact_tables_and_round_trip(tmp_path) -> None:
    art = _artifact()
    tables = art.to_tables()
    assert set(tables.calibrators) == {5, 10}  # one map per fitted batch size
    assert tables.eps(0.9, 5) >= tables.eps(0.1, 5)
    assert tables.eps(0.5, 10) == pytest.approx(art.isotonic_K_prime(0.5))
    with pytest.raises(ValueError, match="no calibrator fitted for batches of 7"):
        tables.eps(0.5, 7)
    assert set(art.future_factor) == {"2:1", "2:2"}
    assert art.future_factor["2:2"] == pytest.approx(1.0)  # nothing after the last step
    assert 0.0 <= art.future_factor["2:1"] <= 1.0
    assert set(art.eps_mean_by_position) == {"2:1", "2:2"}
    assert art.resample_table.k_prime == 10 and art.resample_table.ehat
    high = tables.ehat_r(0.85, 10)
    assert high is not None and 0.0 <= high <= 1.0
    assert tables.ehat_r(0.5, 99) is None
    path = art.save(tmp_path / "a.json")
    back = load_artifact(path)
    assert back == art


def test_compatibility_refusal() -> None:
    art = _artifact()
    art.check_compatible("disagreement", 5, 10)
    art.check_compatible("disagreement", 5, 10, "normalized_exact", "t")
    with pytest.raises(ValueError, match="artifact fitted"):
        art.check_compatible("semantic_entropy", 5, 10)
    with pytest.raises(ValueError):
        art.check_compatible("disagreement", 6, 10)
    with pytest.raises(ValueError, match="equivalence"):
        art.check_compatible("disagreement", 5, 10, "normalized_contains")
    with pytest.raises(ValueError, match="task config"):
        art.check_compatible("disagreement", 5, 10, "normalized_exact", "other-task")


def _fit(observations: list[StepObservation]) -> CalibrationArtifact:
    return fit_artifact(
        observations,
        estimator="disagreement",
        equivalence="normalized_exact",
        K=5,
        K_prime=10,
        task_config_hash="t",
        config_hash="c",
        data_hash="d",
        n_trajectories=len(observations),
    )


def test_a_fit_puts_four_to_one_splits_in_the_second_range() -> None:
    split = 1 - 0.8  # the disagreement score of a four-to-one split
    observations = [_obs(1, 1, 0.0, False, 0.0, False) for _ in range(6)]
    observations += [_obs(1, 1, split, w, 0.3, w) for w in (True, False, False, True)]
    art = _fit(observations)
    assert art.schema_version == 2
    assert art.resample_table.counts == {"0": 6, "1": 4}


def test_the_resample_map_is_the_isotonic_fit_of_the_expanded_label_on_the_score() -> None:
    scores = [0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.4]
    wrong_expanded = [False, False, True, False, False, True, False, True]
    art = _fit([_obs(1, 1, u, False, u, w) for u, w in zip(scores, wrong_expanded, strict=True)])
    assert art.resample_map is not None and art.resample_map.n == 8
    # level means 1/3, 0, 2/3: the first two violate the order and pool to (1 + 0) / 5
    tables = art.to_tables()
    assert tables.ehat_r(0.0, 10) == pytest.approx(0.2)
    assert tables.ehat_r(0.2, 10) == pytest.approx(0.2)
    assert tables.ehat_r(0.4, 10) == pytest.approx(2 / 3)
    assert tables.ehat_r(0.4, 20) is None  # fitted for the expansion to 10 samples only


def test_the_lookup_prefers_the_map_and_falls_back_to_the_table() -> None:
    tables = Tables(
        resample_bucket_edges=[0.0, 0.5],
        resample_effect={(10, 0): 0.4, (20, 0): 0.3},
        resample_map={10: lambda u: 0.1},
    )
    assert tables.ehat_r(0.2, 10) == 0.1  # the map
    assert tables.ehat_r(0.2, 20) == 0.3  # no map for 20 samples: the table
    assert tables.ehat_r(0.2, 30) is None


def test_an_artifact_from_before_the_map_loads_and_reads_its_table_as_fitted() -> None:
    data = _artifact().model_dump()
    del data["resample_map"]
    data["schema_version"] = 1
    data["resample_table"] = {
        "bucket_edges": [0.0, 0.2, 0.4, 0.6, 0.8],
        "k_prime": 10,
        "ehat": {"0": 0.05, "2": 0.3},  # as stored: four-to-one splits were counted in range 0
        "counts": {"0": 9, "2": 1},
    }
    old = CalibrationArtifact.model_validate(data)
    assert old.resample_map is None
    assert old.to_tables().ehat_r(1 - 0.8, 10) == 0.05  # read as it was fitted
    # the same table in a current artifact meets the edge: range 1, which this table lacks
    current = CalibrationArtifact.model_validate({**data, "schema_version": 2})
    assert current.to_tables().ehat_r(1 - 0.8, 10) is None


def _decisions(art: CalibrationArtifact, scores: list[float], policy: str, knob: float) -> list:
    """The controller's action at each score, first step of a two-step task."""
    tables = art.to_tables()
    settings = RunSettings(
        policy=policy,
        knob=knob,
        K=5,
        K_prime=10,
        alpha=0.2,
        costs=CostTable(ask_human=1.0, escalate=10.0, llm_call=0.01),
    )
    instance = Instance(instance_id="i", split="test", index=0, question="q", n_steps=2)
    out = []
    for u in scores:
        step = StepRecord(
            step=1,
            attempt=0,
            context="c",
            answer="a",
            u=u,
            entropy=u,
            eps=tables.eps(u, 5),
            w=1.0,
            G=tables.G(2, 1),
            n_samples=5,
        )
        out.append(make_policy(settings, tables).decide(init_state(instance, settings), step))
    return [d.action for d in out]


def _graded() -> list[StepObservation]:
    """Steps on four score levels whose error falls a little when the batch is expanded."""
    rows = []
    for u, n, wrong_k, wrong_kp in (
        (0.0, 40, 2, 2),
        (0.2, 20, 6, 4),
        (0.4, 20, 9, 6),
        (0.6, 10, 8, 7),
    ):
        for j in range(n):
            rows.append(_obs(2, 1, u, j < wrong_k, u, j < wrong_kp))
    return rows


def test_a_monotone_rescaling_of_the_score_leaves_the_decisions_unchanged() -> None:
    plain = _graded()
    rescaled = [
        o.model_copy(update={"u_K": math.sqrt(o.u_K), "u_K_prime": math.sqrt(o.u_K_prime)})
        for o in plain
    ]
    levels = [0.0, 0.2, 0.4, 0.6]
    for policy, knobs in (("marginal", (0.02, 0.1, 0.3, 0.6)), ("budgeted", (0.5, 1.0, 2.0))):
        for knob in knobs:
            a = _decisions(_fit(plain), levels, policy, knob)
            b = _decisions(_fit(rescaled), [math.sqrt(u) for u in levels], policy, knob)
            assert a == b, (policy, knob)
    # the rule does use the map: where a check is no longer worth its price (0.30 < 0.3 * 1),
    # it expands the level whose error the expansion cuts from 0.30 to 0.20
    assert _decisions(_fit(plain), levels, "marginal", 0.3)[1] == Action.RESAMPLE


def test_no_resample_where_the_expansion_cannot_change_the_answer() -> None:
    # the expanded label equals the label at K on every step: the map is the K map
    same = [_obs(2, 1, u, w, u, w) for o in _graded() for u, w in [(o.u_K, o.wrong_K)]]
    art = _fit(same)
    tables = art.to_tables()
    for u in (0.0, 0.2, 0.4, 0.6):
        assert tables.ehat_r(u, 10) == pytest.approx(tables.eps(u, 5))
    for knob in (0.02, 0.1, 0.3):
        assert Action.RESAMPLE not in _decisions(art, [0.0, 0.2, 0.4, 0.6], "marginal", knob)
