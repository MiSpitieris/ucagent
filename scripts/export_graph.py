"""Write the Mermaid diagram of the control loop into docs/DESIGN.md (make docs)."""

from __future__ import annotations

import re
from pathlib import Path

from ucagent.core.graph.build import build_graph, mermaid
from ucagent.core.llm.mock import MockChainLLM
from ucagent.tasks.synthetic_chain import SyntheticChainTask, SyntheticConfig

BEGIN, END = "<!-- BEGIN GRAPH -->", "<!-- END GRAPH -->"


def render() -> str:
    task = SyntheticChainTask(SyntheticConfig(n_steps=2, n_instances={"test": 1}))
    graph = build_graph(task, MockChainLLM(task=task))
    return "```mermaid\n" + mermaid(graph).strip() + "\n```"


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "DESIGN.md"
    text = path.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{render()}\n{END}"
    new = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    print(f"updated {path}")


if __name__ == "__main__":
    main()
