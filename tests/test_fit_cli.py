from typer.testing import CliRunner

from tests.helpers.configs import small_config
from ucagent.cli import app
from ucagent.core.uq.calibration import load_artifact


def test_fit_cli_writes_artifact(tmp_path) -> None:
    cfg = small_config(task={"n_instances": {"train": 8}}, llm={"K": 2, "K_prime": 4})
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    out = tmp_path / "art.json"
    result = CliRunner().invoke(
        app, ["fit", "--config", str(path), "--out", str(out), "--workers", "1"]
    )
    assert result.exit_code == 0, result.output
    art = load_artifact(out)
    assert art.K == 2 and art.K_prime == 4 and art.n_trajectories == 8
