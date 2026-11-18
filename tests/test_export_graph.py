import importlib.util
from pathlib import Path


def test_render_contains_all_nodes() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_graph.py"
    spec = importlib.util.spec_from_file_location("export_graph", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.render()
    assert text.startswith("```mermaid") and "control" in text and "escalate" in text
