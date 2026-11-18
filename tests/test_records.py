import math

from ucagent.core.records import Action, Intervention, StepRecord, Trajectory


def _traj(knob: float | None = 0.1) -> Trajectory:
    steps = [
        StepRecord(step=1, attempt=0, context="A", answer="B", u=0.2, entropy=0.1, eps=0.2),
        StepRecord(
            step=2,
            attempt=1,
            context="B",
            answer="C",
            u=0.0,
            entropy=0.0,
            eps=0.0,
            source="human",
            locally_correct=True,
        ),
    ]
    ivs = [
        Intervention(step=1, attempt=0, action=Action.PROCEED, reason="r1"),
        Intervention(step=2, attempt=0, action=Action.RESAMPLE, cost=0.1, reason="r2"),
        Intervention(step=2, attempt=1, action=Action.ASK_HUMAN, cost=1.0, reason="r3"),
    ]
    return Trajectory(
        run_id="r",
        instance_id="test-00000-s0",
        split="test",
        seed=0,
        n_steps=2,
        policy="marginal",
        knob=knob,
        estimator="disagreement",
        K=5,
        K_prime=10,
        steps=steps,
        interventions=ivs,
        final_answer="C",
        final_correct=True,
        escalated=False,
        total_cost=1.2,
        fixed_cost=1.1,
        llm_calls=10,
        u_final=0.0,
        risk_naive_final=0.2,
        risk_weighted_final=0.2,
        survival_final=0.8,
        samples={"1:0": ["B", "B"], "2:0": ["C", "D"]},
        sample_confidences={"1:0": [0.9, None], "2:0": [0.8, 0.7]},
    )


def test_counts_and_audit() -> None:
    t = _traj()
    assert t.n_interventions == 2
    assert t.count(Action.ASK_HUMAN) == 1
    audit = t.audit()
    assert [a["step"] for a in audit] == [1, 2]
    assert [d["action"] for d in audit[1]["decisions"]] == ["resample", "ask_human"]
    assert all(a["committed"] for a in audit)
    assert audit[1]["decisions"][0] == {
        "action": "resample",
        "attempt": 0,
        "target_step": None,
        "cost": 0.1,
        "reason": "r2",
        "numbers": {},
    }


def test_audit_lists_the_decision_at_a_step_that_was_never_committed() -> None:
    # the case is escalated at step 2: step 2 has no record, and its decision must still show
    base = _traj()
    escalation = Intervention(
        step=2,
        attempt=0,
        action=Action.ESCALATE,
        cost=10.0,
        reason="final gate: escalate",
        numbers={"risk": 0.41, "tau": 0.3},
    )
    t = base.model_copy(
        update={
            "steps": base.steps[:1],
            "interventions": [base.interventions[0], escalation],
            "escalated": True,
        }
    )
    audit = t.audit()
    assert [a["step"] for a in audit] == [1, 2]
    assert audit[0]["committed"] and audit[0]["answer"] == "B"
    last = audit[1]
    assert not last["committed"] and last["answer"] is None and last["eps"] is None
    assert [(d["action"], d["cost"]) for d in last["decisions"]] == [("escalate", 10.0)]
    assert last["decisions"][0]["numbers"] == {"risk": 0.41, "tau": 0.3}


def test_row_round_trip() -> None:
    t = _traj()
    row = t.to_row()
    assert row["n_ask_human"] == 1 and row["n_resample"] == 1 and row["n_committed"] == 2
    assert "sample_confidences_json" in row
    assert Trajectory.from_row(row) == t  # confidences survive persistence (review finding R3)


def test_rows_without_confidences_still_load() -> None:
    row = _traj().to_row()
    row.pop("sample_confidences_json")  # a parquet written before schema version 2
    back = Trajectory.from_row(row)
    assert back.sample_confidences == {} and back.samples == {"1:0": ["B", "B"], "2:0": ["C", "D"]}


def test_row_round_trip_with_nan_knob() -> None:
    t = _traj(knob=None)
    row = t.to_row()
    row["knob"] = math.nan  # what parquet gives back for a null float column
    assert Trajectory.from_row(row).knob is None
