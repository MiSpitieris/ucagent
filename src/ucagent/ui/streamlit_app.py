"""The Streamlit page: one trajectory, supervised by you.

Launch with ``ucagent ui`` (or ``streamlit run src/ucagent/ui/streamlit_app.py``
with ``UCAGENT_UI_CONFIG`` / ``UCAGENT_UI_RESULTS_DIR`` in the environment).

Widget callbacks only *queue* an action; the body executes it under a
spinner before rendering, so every run shows the pause the action produced
and the headless test harness drives the page click by click.
"""

from __future__ import annotations

import os
import time
from typing import Any, cast

import streamlit as st

import ucagent.core.control
import ucagent.core.uq  # noqa: F401  (registers the estimators)
from ucagent.core.control.base import POLICIES
from ucagent.core.control.policies import DEFAULT_KNOB
from ucagent.core.graph.session import InteractiveSession, Pause
from ucagent.core.llm.guard import CostGuardExceeded
from ucagent.core.records import Action, Decision
from ucagent.core.uq.base import ESTIMATORS
from ucagent.ui import bootstrap, render

ss = st.session_state
NO_CONFIG = "(defaults)"
ollama_status = st.cache_data(ttl=30, show_spinner=False)(bootstrap.ollama_status)


# --- callbacks: queue only ---------------------------------------------------------------


def _queue(*action: Any) -> None:
    """Widget callback: remember what to do; the body does it before rendering."""
    ss["pending_action"] = action


def _queue_override() -> None:
    action = str(ss.get("sel_action") or Action.PROCEED.value)
    target = ss.get("sel_target") if action == Action.BACKTRACK.value else None
    ss["pending_action"] = ("override", action, target, str(ss.get("txt_reason") or ""))


def _queue_answer(key: str) -> None:
    ss["pending_action"] = ("answer", str(ss.get(key) or ""))


def _on_config_change() -> None:
    path = ss["cfg_file"]
    if path != NO_CONFIG:
        try:
            ss.update(bootstrap.defaults_from_config(bootstrap.base_config(path)))
        except Exception as exc:
            ss["error"] = f"could not read {path}: {exc}"


def _on_policy_change() -> None:
    ss["knob"] = float(DEFAULT_KNOB.get(ss["policy"], 0.0))


# --- actions (run in the body) -----------------------------------------------------------


def _choices() -> bootstrap.UiChoices:
    backend = "mock" if ss["backend"] == "mock" else f"ollama:{ss['ollama_model']}"
    return bootstrap.UiChoices(
        config_path=None if ss["cfg_file"] == NO_CONFIG else ss["cfg_file"],
        backend=backend,
        policy=ss["policy"],
        knob=float(ss["knob"]),
        use_ltt_knob=bool(ss["use_ltt_knob"]),
        K=int(ss["k"]),
        K_prime=int(ss["k_prime"]),
        alpha=float(ss["alpha"]),
        estimator=ss["estimator"],
        artifact_mode=ss["artifact_mode"],
        n_steps=int(ss["n_steps"]),
        ambiguity=float(ss["ambiguity"]),
        split=ss["split"],
        instance_index=int(ss["instance_index"]),
        seed=int(ss["seed"]),
        supervise=bool(ss["supervise"]),
        human_mode=ss["human_mode"],
        p_h=float(ss["p_h"]),
        results_dir=ss["results_dir"],
    )


def _log(text: str) -> None:
    ss["events"].append(text)


def _session() -> InteractiveSession:
    session = ss.get("session")
    if session is None or not session.started:
        raise RuntimeError("no run in progress: press Start run")
    return cast(InteractiveSession, session)


def _after(pause: Pause) -> None:
    """Log the advance, let the oracle answer if asked to, prime the answer boxes."""
    session = _session()
    worker = ss["worker"]
    _log(f"{render.trace_summary(session.last_trace)} => {pause.kind} (step {pause.step})")
    while pause.needs_answer and ss["human_mode"] == "oracle":
        human = worker.human
        answer = (
            human.check_step(pause.payload)
            if pause.kind == "check_step"
            else human.take_over(pause.payload)
        )
        _log(f"oracle answered {answer!r} at step {pause.step}")
        pause = session.answer(answer)
        _log(f"{render.trace_summary(session.last_trace)} => {pause.kind} (step {pause.step})")
    if pause.kind == "check_step":
        ss["txt_answer"] = str(pause.payload.get("committed", ""))
    elif pause.kind == "take_over":
        ss["txt_final"] = str(pause.payload.get("pending") or "")
    elif pause.kind == "done":
        ss["seconds"] = time.perf_counter() - ss["t0"]
        _log("finished")


def _do_start() -> None:
    choices = _choices()
    cfg, notes = bootstrap.prepare_run(choices)
    key = bootstrap.worker_key(cfg)
    if ss.get("worker_key") != key:
        with st.spinner("building the task, the model and the graph"):
            ss["worker"] = bootstrap.build_worker(cfg)
        ss["worker_key"] = key
    worker = ss["worker"]
    old = ss.get("session")
    if old is not None:
        old.close()
    session = InteractiveSession(
        worker.graph, worker.settings, worker.tables, supervise=choices.supervise, run_id="ui"
    )
    instance = bootstrap.instance_at(worker.task, choices.split, choices.instance_index)
    ss.update(
        {
            "session": session,
            "instance": instance,
            "config": cfg,
            "notes": notes,
            "events": [],
            "t0": time.perf_counter(),
            "seconds": 0.0,
        }
    )
    with st.spinner(f"sampling {cfg.llm.K} answers per step from {cfg.llm.backend}"):
        _after(session.start(instance))


def _do_new_run() -> None:
    session = ss.get("session")
    if session is not None:
        session.close()
    ss.update({"session": None, "events": [], "notes": []})


def _process_pending() -> None:
    action = ss.pop("pending_action", None)
    if not action:
        return
    ss["error"] = None
    name = str(action[0])
    try:
        if name == "start":
            _do_start()
        elif name == "new_run":
            _do_new_run()
        elif name == "reset_calls":
            ss["worker"].llm.store.reset_calls()
            _log("reset the shared real-call counter")
        else:
            session = _session()
            with st.spinner("running the agent"):
                if name == "accept":
                    _log("accepted the policy's proposal")
                    _after(session.accept())
                elif name == "override":
                    act = Action(str(action[1]))
                    target = int(action[2]) if action[2] is not None else None
                    reason = str(action[3]).strip()
                    _log(f"override: {act.value}" + (f" to step {target}" if target else ""))
                    _after(
                        session.override(Decision(action=act, target_step=target, reason=reason))
                    )
                elif name == "answer":
                    text = str(action[1])
                    _log(f"answered {text!r}" if text else "kept the committed answer")
                    _after(session.answer(text))
    except CostGuardExceeded as exc:
        ss["error"] = f"stopped by the cost guard: {exc}"
    except Exception as exc:
        ss["error"] = f"{type(exc).__name__}: {exc}"


# --- rendering ----------------------------------------------------------------------------


def _init_defaults(config_path: str | None, results_dir: str) -> None:
    defaults: dict[str, Any] = {
        "cfg_file": config_path or NO_CONFIG,
        "backend": "mock",
        "ollama_model": bootstrap.OLLAMA_DEFAULT_MODEL,
        "policy": "marginal",
        "knob": float(DEFAULT_KNOB["marginal"]),
        "use_ltt_knob": False,
        "k": 5,
        "k_prime": 10,
        "alpha": 0.1,
        "estimator": "semantic_entropy",
        "artifact_mode": "auto",
        "n_steps": 5,
        "ambiguity": 0.5,
        "split": "test",
        "instance_index": 0,
        "seed": 0,
        "supervise": True,
        "human_mode": "me",
        "p_h": 0.0,
        "results_dir": results_dir,
        "events": [],
        "error": None,
        "notes": [],
        "session": None,
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value
    if config_path and not ss.get("_config_applied"):
        ss["_config_applied"] = True
        _on_config_change()


def _sidebar() -> None:
    with st.sidebar:
        st.header("Run setup")
        configs = [NO_CONFIG, *bootstrap.list_configs()]
        if ss["cfg_file"] not in configs:
            configs.append(ss["cfg_file"])
        st.selectbox("Config file", configs, key="cfg_file", on_change=_on_config_change)
        st.radio("Backend", ["mock", "ollama"], key="backend", horizontal=True)
        if ss["backend"] == "ollama":
            st.text_input("Ollama model", key="ollama_model")
            reachable, models = ollama_status()
            if reachable:
                st.caption("Ollama is up; models: " + (", ".join(models) or "none"))
            else:
                st.warning("no Ollama server answers at OLLAMA_BASE_URL or localhost:11434")
        st.selectbox("Policy", sorted(POLICIES), key="policy", on_change=_on_policy_change)
        st.number_input("Knob (tau or lambda)", min_value=0.0, step=0.01, format="%.4f", key="knob")
        ltt_value, ltt_note = bootstrap.ltt_knob(ss["policy"], ss["results_dir"])
        st.checkbox("Use the Learn-then-Test knob", key="use_ltt_knob", disabled=ltt_value is None)
        st.caption(ltt_note)
        c1, c2 = st.columns(2)
        c1.number_input("K", min_value=1, max_value=64, key="k")
        c2.number_input("K' (resample)", min_value=1, max_value=128, key="k_prime")
        st.number_input("alpha (risk target)", min_value=0.0, max_value=1.0, step=0.01, key="alpha")
        st.selectbox("Estimator", sorted(ESTIMATORS), key="estimator")
        st.selectbox(
            "Calibration artifact",
            ["auto", "none"],
            key="artifact_mode",
            help="auto: the first fitted artifact under results/ that matches the estimator and K",
        )
        st.number_input("Steps (synthetic chain)", min_value=1, max_value=12, key="n_steps")
        st.slider("Ambiguity (synthetic chain)", 0.0, 1.0, step=0.05, key="ambiguity")
        st.selectbox("Split", ["test", "calib", "train"], key="split")
        st.number_input("Instance index", min_value=0, max_value=100000, key="instance_index")
        st.number_input("Seed", min_value=0, max_value=1_000_000, key="seed")
        st.toggle("Supervisor mode: pause before every decision", key="supervise")
        st.radio(
            "Who answers the agent",
            ["me", "oracle"],
            key="human_mode",
            horizontal=True,
            help="oracle: the simulated human of the experiments (local truth, error rate p_h)",
        )
        if ss["human_mode"] == "oracle":
            st.slider("Oracle error rate p_h", 0.0, 1.0, step=0.05, key="p_h")
        st.button("Start run", key="btn_start", type="primary", on_click=_queue, args=("start",))


def _intro() -> None:
    st.markdown(
        "Pick a backend, a policy and an instance in the sidebar, then **Start run**. "
        "The agent samples the model K times per step, turns the disagreement into a "
        "calibrated local error and a projected end-to-end risk, and the controller picks "
        "the cheapest action worth its cost. In **supervisor mode** the run pauses before every "
        "decision so you can accept the proposal or override it; whenever the agent asks a "
        "human (`ask_human`) or hands the case over (`escalate`), you answer here."
    )


def _header(values: Any) -> None:
    worker = ss["worker"]
    cfg = ss["config"]
    instance = ss["instance"]
    st.markdown(
        f"**Instance** `{instance.instance_id}` | **policy** `{cfg.control.policy}` "
        f"knob `{worker.settings.knob}` | **backend** `{cfg.llm.backend}` | "
        f"**estimator** `{cfg.uq.estimator}` K={cfg.llm.K} K'={cfg.llm.K_prime} | "
        f"alpha={cfg.control.alpha}"
    )
    for note in ss["notes"]:
        st.caption(note)
    st.markdown(f"**Question.** {instance.question}")
    nums = render.status_numbers(values, worker.settings)
    cols = st.columns(6)
    cols[0].metric("Step", f"{nums['step']} / {nums['n_steps']}")
    cols[1].metric("Survival", f"{nums['survival']:.3f}")
    cols[2].metric("Risk so far", f"{nums['risk_so_far']:.3f}")
    cols[3].metric("Cost", f"{nums['cost']:.2f}")
    cols[4].metric("LLM calls", nums["llm_calls"])
    cols[5].metric("Interventions", nums["interventions"])
    counters = bootstrap.llm_counters(worker.llm)
    if counters is not None:
        st.caption(
            f"real model calls (shared counter): {counters['real_calls']} of "
            f"{counters.get('max_calls', '?')} allowed | cache rows {counters['cache_rows']} "
            f"| {counters['cache_path']}"
        )
        st.button(
            "Reset the shared call counter",
            key="btn_reset_calls",
            on_click=_queue,
            args=("reset_calls",),
            help="The counter is a lifetime cap per cache file (UCAGENT_MAX_LLM_CALLS).",
        )
    series = render.risk_series(values, cfg.control.alpha)
    if not series.empty:
        st.line_chart(series, height=220)


def _control_view(session: InteractiveSession, values: Any) -> None:
    worker = ss["worker"]
    cfg = ss["config"]
    pending = values["pending"]
    assert pending is not None
    st.subheader(
        f"Step {pending.step}, decision round {pending.attempt}: the controller waits for you"
    )
    with st.expander("Prompt sent to the model"):
        st.code(render.prompt_text(values["prompt"]))
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"**Provisional answer:** `{pending.answer}` from {pending.n_samples} samples "
            f"(batch {pending.samples_key})"
        )
        st.dataframe(render.clusters_frame(pending), hide_index=True, width="stretch")
        with st.expander("Samples"):
            st.dataframe(
                render.samples_frame(values, pending.samples_key),
                hide_index=True,
                width="stretch",
            )
    with right:
        st.markdown("**Uncertainty and risk of this step**")
        m = st.columns(2)
        m[0].metric("u (uncertainty)", f"{pending.u:.3f}")
        m[1].metric("eps (local error)", f"{pending.eps:.3f}")
        m = st.columns(2)
        m[0].metric("w (influence)", f"{pending.w:.2f}")
        m[1].metric("G (future factor)", f"{pending.G:.2f}")
        projected = render.pending_projection(values)
        st.metric(
            "Projected final risk if committed",
            f"{projected:.3f}" if projected is not None else "n/a",
            delta=f"target alpha {cfg.control.alpha}",
            delta_color="off",
        )
    proposal = session.proposal()
    target = f" to step {proposal.target_step}" if proposal.target_step else ""
    st.markdown(f"**Policy proposal ({cfg.control.policy}):** `{proposal.action.value}`{target}")
    st.caption(proposal.reason)
    if proposal.numbers:
        with st.expander("Numbers behind the proposal"):
            st.dataframe(render.proposal_frame(proposal), hide_index=True)
    st.markdown("**Expected effect of each feasible action**")
    st.dataframe(
        render.effects_frame(session.effects(), worker.settings.knob),
        hide_index=True,
        width="stretch",
    )
    st.button(
        f"Accept the proposal: {proposal.action.value}",
        key="btn_accept",
        type="primary",
        on_click=_queue,
        args=("accept",),
    )
    allowed = session.allowed_actions()
    options = [a.value for a, why in allowed.items() if why is None]
    blocked = [f"{a.value} ({why})" for a, why in allowed.items() if why is not None]
    if ss.get("sel_action") not in options:
        ss["sel_action"] = options[0]
    with st.container(border=True):
        st.markdown("**Override the controller**")
        c1, c2 = st.columns(2)
        c1.selectbox("Action", options, key="sel_action")
        if ss["sel_action"] == Action.BACKTRACK.value:
            # only steps whose batch can still be expanded to K' are valid targets
            targets = [j for j, why in session.backtrack_targets().items() if why is None]
            if ss.get("sel_target") not in targets:
                ss["sel_target"] = targets[0]
            c2.selectbox("Back to step", targets, key="sel_target")
        st.text_input("Reason (recorded in the audit log)", key="txt_reason")
        st.button("Apply the override", key="btn_override", on_click=_queue_override)
        if blocked:
            st.caption("not available now: " + "; ".join(blocked))


def _check_step_view(pause: Pause, values: Any) -> None:
    p = pause.payload
    st.subheader(f"The agent asks you to check step {p['step']} of {p['n_steps']}")
    st.markdown(f"**Context:** `{p['current']}` | **committed answer:** `{p['committed']}`")
    st.dataframe(render.clusters_from_payload(p), hide_index=True, width="stretch")
    with st.expander("Prompt sent to the model"):
        st.code(render.prompt_text(values["prompt"]))
    st.text_input("Your answer for this step", key="txt_answer")
    c1, c2 = st.columns(2)
    c1.button(
        "Submit answer",
        key="btn_answer",
        type="primary",
        on_click=_queue_answer,
        args=("txt_answer",),
    )
    c2.button("Keep the committed answer", key="btn_keep", on_click=_queue, args=("answer", ""))


def _take_over_view(pause: Pause) -> None:
    p = pause.payload
    st.subheader("The agent hands the case over to you")
    st.markdown(f"**Question.** {p['question']}")
    chain = " -> ".join(p["history"]) if p["history"] else "(no committed step)"
    st.markdown(f"**Chain so far:** {chain} | **pending answer:** `{p.get('pending')}`")
    st.text_input("Your final answer", key="txt_final")
    st.button(
        "Submit final answer",
        key="btn_final",
        type="primary",
        on_click=_queue_answer,
        args=("txt_final",),
    )


def _done_view(session: InteractiveSession, values: Any) -> None:
    worker = ss["worker"]
    instance = ss["instance"]
    trajectory = worker.to_trajectory(
        values, instance.instance_id, session.thread_id, float(ss.get("seconds", 0.0))
    )
    st.subheader("Finished")
    st.markdown(
        f"**Final answer:** `{trajectory.final_answer}` | correct per the task's truth "
        f"(analysis only): **{trajectory.final_correct}** | escalated: {trajectory.escalated}"
    )
    cols = st.columns(4)
    cols[0].metric("Total cost", f"{trajectory.total_cost:.2f}")
    cols[1].metric("LLM calls", trajectory.llm_calls)
    cols[2].metric("Interventions", trajectory.n_interventions)
    cols[3].metric("Seconds", f"{trajectory.duration_s:.1f}")
    st.dataframe(render.audit_frame(trajectory), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download trajectory JSON",
        trajectory.model_dump_json(indent=1),
        file_name=f"{instance.instance_id}.json",
        mime="application/json",
        key="btn_download",
    )
    c2.button("New run", key="btn_new_run", on_click=_queue, args=("new_run",))


def _timeline(values: Any) -> None:
    st.markdown("**Committed steps**")
    st.dataframe(render.steps_frame(values), hide_index=True, width="stretch")
    st.markdown("**Decisions so far**")
    st.dataframe(render.decisions_frame(values["interventions"]), hide_index=True, width="stretch")
    with st.expander("Event log"):
        for event in ss["events"]:
            st.text(event)


def _body() -> None:
    st.title("ucagent: a supervised agent run")
    if ss.get("error"):
        st.error(ss["error"])
    session = ss.get("session")
    if session is None or not session.started:
        _intro()
        return
    pause = session.pause
    values = session.values
    _header(values)
    if pause.kind == "control":
        _control_view(session, values)
    elif pause.kind == "check_step":
        _check_step_view(pause, values)
    elif pause.kind == "take_over":
        _take_over_view(pause)
    else:
        _done_view(session, values)
    _timeline(values)


def main() -> None:
    """Render the page (Streamlit re-runs this on every interaction)."""
    st.set_page_config(page_title="ucagent supervised run", layout="wide")
    bootstrap.ensure_cwd_on_path()
    _init_defaults(
        os.environ.get("UCAGENT_UI_CONFIG") or None,
        os.environ.get("UCAGENT_UI_RESULTS_DIR") or "results",
    )
    _sidebar()
    _process_pending()
    _body()


if __name__ == "__main__":
    main()
