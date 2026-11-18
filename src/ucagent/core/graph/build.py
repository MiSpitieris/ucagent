"""The explicit LangGraph control loop. Read top to bottom.

::

    START -> prepare_step -> [Send x K] sample -> estimate -> control
    control --proceed--> prepare_step | finish
    control --resample--> resample -> prepare_step
    control --backtrack--> backtrack -> prepare_step
    control --ask_human--> ask_human (interrupt) -> prepare_step | finish
    control --escalate--> escalate (interrupt) -> finish -> END

No prebuilt agent, no hidden control flow. The diagram in ``docs/DESIGN.md``
is generated from this function by ``scripts/export_graph.py``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ucagent.core.graph.nodes import GraphNodes, route_after_commit
from ucagent.core.graph.state import SamplePayload, TrajectoryState
from ucagent.core.protocols import Task

Graph = CompiledStateGraph[Any, Any, Any, Any]

# Record types that live in checkpoints; registered explicitly with the msgpack
# serializer so deserialisation stays allowed under LANGGRAPH_STRICT_MSGPACK.
RECORD_TYPES: list[tuple[str, str]] = [
    ("ucagent.core.records", name)
    for name in ("Action", "Sample", "Cluster", "StepRecord", "Intervention", "Decision")
]


def make_serializer() -> JsonPlusSerializer:
    """The checkpoint serializer with ucagent's record types allow-listed."""
    return JsonPlusSerializer(allowed_msgpack_modules=RECORD_TYPES)


def make_checkpointer(sqlite_path: str | None = None) -> BaseCheckpointSaver[Any]:
    """In-memory checkpointer by default; sqlite-backed when a path is given."""
    if sqlite_path is None:
        return InMemorySaver(serde=make_serializer())
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    saver: BaseCheckpointSaver[Any] = SqliteSaver(conn, serde=make_serializer())
    return saver


NODE_NAMES = (
    "prepare_step",
    "sample",
    "estimate",
    "control",
    "resample",
    "backtrack",
    "ask_human",
    "escalate",
    "finish",
)


def build_graph(
    task: Task,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Graph:
    """Compile the control loop for ``task`` and ``llm``.

    Policy, estimator, ``K`` and costs are not arguments: they are read from
    ``config["configurable"]`` at run time (see ``runtime.make_config``).
    """
    nodes = GraphNodes(task, llm)
    g: StateGraph[TrajectoryState, None, TrajectoryState, TrajectoryState] = StateGraph(
        TrajectoryState
    )
    g.add_node("prepare_step", nodes.prepare_step)
    g.add_node("sample", nodes.sample, input_schema=SamplePayload)
    g.add_node("estimate", nodes.estimate)
    g.add_node(
        "control",
        nodes.control,
        destinations=("prepare_step", "resample", "backtrack", "ask_human", "escalate", "finish"),
    )
    g.add_node("resample", nodes.resample)
    g.add_node("backtrack", nodes.backtrack)
    g.add_node("ask_human", nodes.ask_human)
    g.add_node("escalate", nodes.escalate)
    g.add_node("finish", nodes.finish)

    g.add_edge(START, "prepare_step")
    g.add_conditional_edges("prepare_step", nodes.fan_out, ["sample"])
    g.add_edge("sample", "estimate")
    g.add_edge("estimate", "control")
    g.add_edge("resample", "prepare_step")
    g.add_edge("backtrack", "prepare_step")
    g.add_conditional_edges("ask_human", route_after_commit, ["prepare_step", "finish"])
    g.add_edge("escalate", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer or make_checkpointer())


def mermaid(graph: Graph) -> str:
    """Mermaid source of the compiled graph (used for ``docs/DESIGN.md``)."""
    return graph.get_graph().draw_mermaid()
