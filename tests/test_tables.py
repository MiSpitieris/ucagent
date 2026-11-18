"""Run-time tables: calibrators are selected by batch size."""

from __future__ import annotations

import pytest

from ucagent.core.tables import Tables


def test_eps_routes_by_sample_count_and_rejects_unknown_counts() -> None:
    tables = Tables(calibrators={2: lambda u: 0.9, 4: lambda u: 0.1})
    assert tables.eps(0.0, 2) == 0.9 and tables.eps(0.0, 4) == 0.1
    with pytest.raises(ValueError, match="no calibrator fitted for batches of 3"):
        tables.eps(0.0, 3)
    assert Tables().eps(0.3, 7) == 0.3  # identity before calibration, any batch size
