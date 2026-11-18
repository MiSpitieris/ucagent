"""The core library must never import the research harness."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "ucagent" / "core"
FORBIDDEN = ("ucagent.tasks", "ucagent.experiments", "ucagent.cli", "ucagent.api", "ucagent.ui")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_core_does_not_import_harness() -> None:
    files = list(CORE.rglob("*.py"))
    assert files, "core package not found"
    offenders = [
        (str(p), name) for p in files for name in _imports(p) if name.startswith(FORBIDDEN)
    ]
    assert not offenders, f"core imports harness modules: {offenders}"
