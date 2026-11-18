"""Parquet I/O for trajectories (one row per trajectory, records as JSON)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ucagent.core.records import Trajectory


def to_frame(trajectories: Sequence[Trajectory]) -> pd.DataFrame:
    """Rows of :meth:`Trajectory.to_row`."""
    return pd.DataFrame([t.to_row() for t in trajectories])


def write_trajectories(path: str | Path, trajectories: Sequence[Trajectory]) -> Path:
    """Write a parquet file; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(trajectories).to_parquet(path, index=False)
    return path


def read_frame(path: str | Path) -> pd.DataFrame:
    """Read the parquet file as a DataFrame."""
    return pd.read_parquet(path)


def read_trajectories(path: str | Path) -> list[Trajectory]:
    """Read the parquet file back into :class:`Trajectory` objects."""
    frame = read_frame(path)
    rows = frame.to_dict(orient="records")
    return [Trajectory.from_row({str(k): v for k, v in row.items()}) for row in rows]
