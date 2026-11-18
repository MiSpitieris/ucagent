"""Executable API facts for the pinned LangGraph / LangChain versions.

Each test encodes one assumption that ucagent's graph code relies on, as a
toy graph. If a dependency bump breaks an assumption, this file fails before
any real code path does. ``docs/API_NOTES.md`` mirrors these facts in prose.
"""

from __future__ import annotations

import inspect
import operator
from typing import Annotated, Any, Literal, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, Send, interrupt
from pydantic import BaseModel


class Rec(BaseModel, frozen=True):
    k: int
    text: str


def merge(left: dict[str, list[Any]], right: dict[str, list[Any]]) -> dict[str, list[Any]]:
    out = {k: list(v) for k, v in left.items()}
    for k, v in right.items():
        out.setdefault(k, []).extend(v)
    return out


class S(TypedDict, total=False):
    n: int
    k_requested: int
    samples: Annotated[dict[str, list[Rec]], merge]
    estimate_runs: Annotated[int, operator.add]
    log: Annotated[list[str], operator.add]
    i: int
    rec: Rec | None


class Payload(TypedDict):
    key: str
    k: int


# F1: Send fan-out runs all targets in one superstep; a plain edge target runs once after.
def test_send_fan_out_and_fan_in() -> None:
    def prepare(state: S) -> dict[str, Any]:
        return {"k_requested": state["n"]}

    def fan_out(state: S) -> list[Send]:
        return [Send("sample", {"key": "1:0", "k": k}) for k in range(state["k_requested"])]

    def sample(payload: Payload) -> dict[str, Any]:
        return {"samples": {payload["key"]: [Rec(k=payload["k"], text=f"s{payload['k']}")]}}

    def estimate(state: S) -> dict[str, Any]:
        return {"estimate_runs": 1}

    g = StateGraph(S)
    g.add_node("prepare", prepare)
    g.add_node("sample", sample, input_schema=Payload)
    g.add_node("estimate", estimate)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", fan_out, ["sample"])
    g.add_edge("sample", "estimate")
    g.add_edge("estimate", END)
    out = g.compile().invoke({"n": 7, "samples": {}, "estimate_runs": 0})
    assert sorted(r.k for r in out["samples"]["1:0"]) == list(range(7))
    assert out["estimate_runs"] == 1
    assert all(isinstance(r, Rec) for r in out["samples"]["1:0"])


# F2: a node may return Command(goto=...), and `destinations` shows those edges in Mermaid.
def test_command_goto_and_destinations_in_mermaid() -> None:
    def control(state: S) -> Command[Literal["left", "finish"]]:
        target: Literal["left", "finish"] = "left" if state["i"] == 0 else "finish"
        return Command(goto=target, update={"log": [f"control:{target}"], "i": state["i"] + 1})

    def left(state: S) -> dict[str, Any]:
        return {"log": ["left"]}

    def finish(state: S) -> dict[str, Any]:
        return {"log": ["finish"]}

    g = StateGraph(S)
    g.add_node("control", control, destinations=("left", "finish"))
    g.add_node("left", left)
    g.add_node("finish", finish)
    g.add_edge(START, "control")
    g.add_edge("left", "control")
    g.add_edge("finish", END)
    compiled = g.compile()
    out = compiled.invoke({"i": 0, "log": []})
    assert out["log"] == ["control:left", "left", "control:finish", "finish"]
    mermaid = compiled.get_graph().draw_mermaid()
    assert "control" in mermaid and "left" in mermaid and "finish" in mermaid
    assert "control -.-> left" in mermaid or "control --> left" in mermaid


# F3: interrupt() surfaces as result["__interrupt__"]; Command(resume=) re-runs the node.
def test_interrupt_resume_and_configurable_on_resume() -> None:
    runs: list[str] = []

    def ask(state: S, config: RunnableConfig) -> dict[str, Any]:
        policy = config["configurable"]["policy"]
        runs.append(policy)
        answer = interrupt({"kind": "check_step", "step": state["i"]})
        return {"log": [f"{policy}:{answer}"]}

    g = StateGraph(S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "t1", "policy": "a"}}
    first = compiled.invoke({"i": 3, "log": []}, cfg, durability="exit")
    intr = first["__interrupt__"]
    assert isinstance(intr[0], Interrupt)
    assert intr[0].value == {"kind": "check_step", "step": 3}
    assert hasattr(intr[0], "id")
    cfg2: RunnableConfig = {"configurable": {"thread_id": "t1", "policy": "b"}}
    second = compiled.invoke(Command(resume="X"), cfg2, durability="exit")
    assert "__interrupt__" not in second
    assert second["log"] == ["b:X"]  # node re-ran from the start under the new configurable
    assert runs == ["a", "b"]


# F4: recursion_limit counts supersteps; a loop of n node executions needs limit >= n + 1
# (measured: minimal limit = supersteps + 1; a prepare/sample/estimate/control step = 4).
def test_recursion_limit_counts_supersteps() -> None:
    def step(state: S) -> dict[str, Any]:
        return {"i": state["i"] + 1}

    def route(state: S) -> str:
        return "step" if state["i"] < state["n"] else END

    g = StateGraph(S)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, ["step", END])
    compiled = g.compile()
    n = 6
    assert compiled.invoke({"i": 0, "n": n}, {"recursion_limit": n + 1})["i"] == n
    with pytest.raises(GraphRecursionError):
        compiled.invoke({"i": 0, "n": n}, {"recursion_limit": n})


# F5: update_state(as_node=) forks a past checkpoint; invoke(None, forked) continues
# under the configurable given at resume time; the original thread tip is unchanged.
def test_checkpoint_forking_with_update_state() -> None:
    def step(state: S, config: RunnableConfig) -> dict[str, Any]:
        tag = config["configurable"]["tag"]
        return {"i": state["i"] + 1, "log": [f"{tag}:{state['i'] + 1}"]}

    def route(state: S) -> str:
        return "step" if state["i"] < state["n"] else END

    g = StateGraph(S)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, ["step", END])
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "fork", "tag": "orig"}}
    final = compiled.invoke({"i": 0, "n": 4, "log": []}, cfg, durability="sync")
    assert final["log"] == ["orig:1", "orig:2", "orig:3", "orig:4"]
    history = list(compiled.get_state_history(cfg))
    snap = next(s for s in history if s.values.get("i") == 2 and s.next == ("step",))
    forked = compiled.update_state(snap.config, {"log": ["forked"]}, as_node="step")
    forked_cfg: RunnableConfig = {
        "configurable": {**forked["configurable"], "tag": "fork"},
    }
    out = compiled.invoke(None, forked_cfg, durability="sync")
    assert out["i"] == 4
    assert out["log"] == ["orig:1", "orig:2", "forked", "fork:3", "fork:4"]
    assert compiled.get_state(cfg).values["log"][-1] in {"orig:4", "fork:4"}
    assert len(list(compiled.get_state_history(cfg))) > len(history)


# F6: pydantic models in state survive the checkpoint round trip across an interrupt.
def test_pydantic_models_survive_checkpoint() -> None:
    def write(state: S) -> dict[str, Any]:
        return {"rec": Rec(k=1, text="x")}

    def ask(state: S) -> dict[str, Any]:
        interrupt("q")
        return {"log": ["resumed"]}

    def read(state: S) -> dict[str, Any]:
        rec = state["rec"]
        assert isinstance(rec, Rec), type(rec)
        return {"log": [rec.text]}

    g = StateGraph(S)
    g.add_node("write", write)
    g.add_node("ask", ask)
    g.add_node("read", read)
    g.add_edge(START, "write")
    g.add_edge("write", "ask")
    g.add_edge("ask", "read")
    g.add_edge("read", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "pyd"}}
    compiled.invoke({"log": []}, cfg, durability="exit")
    out = compiled.invoke(Command(resume="ok"), cfg, durability="exit")
    assert out["log"] == ["resumed", "x"]
    assert isinstance(out["rec"], Rec)


# F7: model kwargs on invoke() reach BaseChatModel._generate(**kwargs).
class _KwargsModel(BaseChatModel):
    seen: list[int] = []

    @property
    def _llm_type(self) -> str:
        return "kwargs-probe"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(int(kwargs.get("sample_index", -1)))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def test_model_kwargs_reach_generate() -> None:
    llm = _KwargsModel()
    llm.invoke([HumanMessage(content="hi")], sample_index=3)
    llm.invoke([HumanMessage(content="hi")], sample_index=5)
    assert llm.seen[-2:] == [3, 5]


# F8: checkpointer housekeeping and backend entry points exist with the expected names.
def test_checkpointer_and_backend_entry_points() -> None:
    saver = InMemorySaver()
    assert callable(getattr(saver, "delete_thread", None))
    from langchain.chat_models import init_chat_model

    params = inspect.signature(init_chat_model).parameters
    assert "model_provider" in params
    from langgraph.checkpoint.sqlite import SqliteSaver

    assert hasattr(SqliteSaver, "from_conn_string")


# F9: forking at a node that routes with Command needs a Command as the update value;
# a plain dict leaves no next node (measured on 1.2.20).
def test_update_state_with_command_at_a_command_routing_node() -> None:
    def prepare(state: S) -> dict[str, Any]:
        return {"log": [f"prep{state['i']}"]}

    def control(state: S) -> Command[Literal["prepare", "finish"]]:
        nxt = state["i"] + 1
        return Command(goto="finish" if nxt > state["n"] else "prepare", update={"i": nxt})

    def finish(state: S) -> dict[str, Any]:
        return {"log": ["fin"]}

    g = StateGraph(S)
    g.add_node("prepare", prepare)
    g.add_node("control", control, destinations=("prepare", "finish"))
    g.add_node("finish", finish)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "control")
    g.add_edge("finish", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "cmd"}}
    compiled.invoke({"i": 1, "n": 3, "log": []}, cfg, durability="sync")
    snap = next(
        s
        for s in compiled.get_state_history(cfg)
        if s.values.get("i") == 2 and s.next == ("prepare",)
    )
    plain = compiled.update_state(snap.config, {"log": ["plain"]}, as_node="control")
    assert compiled.get_state(plain).next == ()
    forked = compiled.update_state(
        snap.config, Command(update={"log": ["fork"]}, goto="prepare"), as_node="control"
    )
    assert compiled.get_state(forked).next == ("prepare",)
    out = compiled.invoke(None, forked, durability="sync")
    assert out["log"] == ["prep1", "fork", "prep2", "prep3", "fin"]


# F16: interrupt_before is a per-call option of invoke/stream; the static pause before a node
# leaves no `__interrupt__` entry in invoke's result and shows up as get_state().next.
def test_per_call_interrupt_before_pauses_before_a_node() -> None:
    def estimate(state: S) -> dict[str, Any]:
        return {"i": state["i"] + 1, "log": ["estimate"]}

    def control(state: S) -> Command[Literal["estimate", "finish"]]:
        nxt = "estimate" if state["i"] < 2 else "finish"
        return Command(goto=nxt, update={"log": [f"control:{nxt}"]})

    def finish(state: S) -> dict[str, Any]:
        return {"log": ["finish"]}

    g = StateGraph(S)
    g.add_node("estimate", estimate)
    g.add_node("control", control, destinations=("estimate", "finish"))
    g.add_node("finish", finish)
    g.add_edge(START, "estimate")
    g.add_edge("estimate", "control")
    g.add_edge("finish", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "static"}}
    out = compiled.invoke({"i": 0, "log": []}, cfg, durability="sync", interrupt_before=["control"])
    assert "__interrupt__" not in out and out["log"] == ["estimate"]
    assert compiled.get_state(cfg).next == ("control",)
    out = compiled.invoke(None, cfg, durability="sync", interrupt_before=["control"])
    assert out["log"] == ["estimate", "control:estimate", "estimate"]
    assert compiled.get_state(cfg).next == ("control",)
    out = compiled.invoke(None, cfg, durability="sync")  # no pause: runs to the end
    assert out["log"][-1] == "finish" and compiled.get_state(cfg).next == ()


# F17: stream(stream_mode="updates") yields {node: update} per node run and ends with an
# {"__interrupt__": (...)} chunk whose tuple is empty for a static pause (interrupt_before)
# and holds Interrupt(value, id) for a dynamic interrupt(); no such chunk when the run ends.
def test_stream_updates_surfaces_static_and_dynamic_pauses() -> None:
    def estimate(state: S) -> dict[str, Any]:
        return {"i": state["i"] + 1, "log": ["estimate"]}

    def control(state: S, config: RunnableConfig) -> Command[Literal["ask", "finish"]]:
        if config["configurable"].get("force") == "ask":
            return Command(goto="ask", update={"log": ["control:ask"]})
        return Command(goto="finish", update={"log": ["control:finish"]})

    def ask(state: S) -> dict[str, Any]:
        answer = interrupt({"kind": "check_step", "step": state["i"]})
        return {"log": [f"ask:{answer}"]}

    def finish(state: S) -> dict[str, Any]:
        return {"log": ["finish"]}

    g = StateGraph(S)
    g.add_node("estimate", estimate)
    g.add_node("control", control, destinations=("ask", "finish"))
    g.add_node("ask", ask)
    g.add_node("finish", finish)
    g.add_edge(START, "estimate")
    g.add_edge("estimate", "control")
    g.add_edge("ask", "finish")
    g.add_edge("finish", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "stream"}}

    def run(inp: Any, conf: RunnableConfig) -> tuple[list[str], tuple[Any, ...] | None]:
        names: list[str] = []
        intr: tuple[Any, ...] | None = None
        for chunk in compiled.stream(
            inp, conf, durability="sync", stream_mode="updates", interrupt_before=["control"]
        ):
            for name, update in chunk.items():
                if name == "__interrupt__":
                    intr = tuple(update)
                else:
                    names.append(name)
        return names, intr

    names, intr = run({"i": 0, "log": []}, cfg)
    assert names == ["estimate"] and intr == ()  # static pause: empty tuple
    forced: RunnableConfig = {"configurable": {**cfg["configurable"], "force": "ask"}}
    names, intr = run(None, forced)
    assert names == ["control"] and intr is not None and len(intr) == 1
    assert isinstance(intr[0], Interrupt) and intr[0].value == {"kind": "check_step", "step": 1}
    names, intr = run(Command(resume="X"), cfg)  # resume under the base configurable
    assert names == ["ask", "finish"] and intr is None
    assert compiled.get_state(cfg).values["log"][-2:] == ["ask:X", "finish"]
    assert len(list(compiled.get_state_history(cfg))) >= 5  # "sync" keeps every superstep
