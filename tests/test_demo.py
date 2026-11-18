from typer.testing import CliRunner

from ucagent.cli import app


def test_demo_auto_runs_one_trajectory() -> None:
    result = CliRunner().invoke(app, ["demo", "--auto", "--policy", "per_step_u", "--knob", "0.3"])
    assert result.exit_code == 0, result.output
    assert "final answer" in result.output and "step 1" in result.output


def test_demo_terminal_reads_answers() -> None:
    result = CliRunner().invoke(
        app, ["demo", "--policy", "per_step_u", "--knob", "0.0"], input="\n" * 20
    )
    assert result.exit_code == 0, result.output
    assert "[review]" in result.output
