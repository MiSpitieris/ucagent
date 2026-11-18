"""Command-line interface: ``ucagent run | calibrate | figures | demo | ui``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from ucagent import __version__

app = typer.Typer(
    help="Uncertainty-controlled multi-step LLM agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ucagent {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Entry point for all subcommands."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Experiment YAML.")],
    split: Annotated[str | None, typer.Option(help="Override the split.")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes.")] = None,
    results_dir: Annotated[Path, typer.Option(help="Where results/<run_id>/ goes.")] = Path(
        "results"
    ),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan, run nothing.")
    ] = False,
) -> None:
    """Run one split through the graph and write trajectories.parquet."""
    from ucagent.experiments.config import ExperimentConfig
    from ucagent.experiments.runner import run_experiment

    cfg = ExperimentConfig.from_yaml(config)
    result = run_experiment(
        cfg, split=split, workers=workers, dry_run=dry_run, results_dir=results_dir
    )
    if dry_run:
        typer.echo("dry run only; see log for the plan")
        return
    n = len(result.trajectories)
    errors = sum(1 for t in result.trajectories if not t.final_correct)
    cost = sum(t.total_cost for t in result.trajectories) / max(n, 1)
    interventions = sum(t.n_interventions for t in result.trajectories) / max(n, 1)
    typer.echo(
        f"{cfg.run_id}: {n} trajectories, error {errors / max(n, 1):.3f}, "
        f"mean cost {cost:.3f}, mean interventions {interventions:.2f} "
        f"({result.seconds:.1f}s) -> {result.out_dir}"
    )


@app.command()
def fit(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Experiment YAML.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Artifact JSON path.")],
    workers: Annotated[int | None, typer.Option(help="Worker processes.")] = None,
) -> None:
    """Fit the calibration artifact (isotonic u->eps, resample effect, future factor)."""
    from ucagent.api import fit_calibrators
    from ucagent.experiments.config import ExperimentConfig

    artifact = fit_calibrators(ExperimentConfig.from_yaml(config), workers=workers, out_path=out)
    typer.echo(
        f"fitted on {artifact.n_trajectories} train trajectories "
        f"(estimator={artifact.estimator}, K={artifact.K}, K'={artifact.K_prime}) -> {out}"
    )


@app.command()
def suite(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Suite YAML.")],
    workers: Annotated[int | None, typer.Option(help="Worker processes.")] = None,
    results_dir: Annotated[Path, typer.Option(help="Where results/<suite_id>/ goes.")] = Path(
        "results"
    ),
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Rerun existing runs.")] = False,
) -> None:
    """Run a suite: every policy over its knob grid, every split and seed; write manifest.csv."""
    from ucagent.experiments.suite import SuiteConfig, run_suite

    root = run_suite(
        SuiteConfig.from_yaml(config),
        workers=workers,
        results_dir=results_dir,
        resume=not no_resume,
    )
    typer.echo(f"suite written to {root}")


@app.command()
def figures(
    suite_dir: Annotated[Path, typer.Option("--suite", "-s", exists=True, help="Suite directory.")],
    out: Annotated[
        Path | None, typer.Option(help="Figure directory (default <suite>/figures).")
    ] = None,
    split: Annotated[str, typer.Option(help="Split to plot.")] = "test",
) -> None:
    """Regenerate figures from a suite's manifest and parquet files."""
    from ucagent.experiments.figures import fig2, fig3, fig4, plot_risk_vs_knob

    paths = fig2(suite_dir, out, split=split) + fig3(suite_dir, out, split=split)
    paths += fig4(suite_dir, out, split=split)
    if (suite_dir / "ltt.csv").exists():
        paths += plot_risk_vs_knob(suite_dir, out)
    for path in paths:
        typer.echo(f"wrote {path}")


@app.command()
def demo(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Experiment YAML.")] = None,
    instance: Annotated[int, typer.Option(help="Instance index of the split.")] = 0,
    policy: Annotated[str, typer.Option(help="Policy name.")] = "per_step_u",
    knob: Annotated[float | None, typer.Option(help="Policy knob (tau or lambda).")] = None,
    auto: Annotated[
        bool, typer.Option("--auto", help="Answer with the oracle, no terminal.")
    ] = False,
) -> None:
    """Run one trajectory interactively: you answer the step checks and escalations."""
    from ucagent.core.graph.human import TerminalHuman, run_with_human
    from ucagent.core.graph.runtime import init_state, make_config
    from ucagent.experiments.config import ExperimentConfig
    from ucagent.experiments.runner import Worker

    cfg = ExperimentConfig.from_yaml(config) if config else ExperimentConfig(run_id="demo")
    cfg = cfg.with_updates(control={"policy": policy, "knob": knob})
    worker = Worker.from_config(cfg)
    inst = list(worker.task.instances(cfg.split))[instance]
    human = worker.human if auto else TerminalHuman()
    typer.echo(f"instance {inst.instance_id}: {inst.question}")
    state = init_state(inst, worker.settings)
    run_config = make_config(
        f"demo:{inst.instance_id}", worker.settings, worker.tables, inst.n_steps
    )
    final = run_with_human(worker.graph, state, run_config, human)
    traj = worker.to_trajectory(final, inst.instance_id, "demo", 0.0)
    for row in traj.audit():
        if row["committed"]:
            typer.echo(
                f"step {row['step']}: answer={row['answer']!r} u={row['u']:.2f} "
                f"eps={row['eps']:.2f} source={row['source']}"
            )
        else:  # the case was escalated at this step, or a backtrack discarded it
            typer.echo(f"step {row['step']}: not committed")
        for d in row["decisions"]:
            typer.echo(f"    -> {d['action']}: {d['reason']}")
    typer.echo(
        f"final answer {traj.final_answer!r} correct={traj.final_correct} "
        f"escalated={traj.escalated} cost={traj.total_cost:.2f}"
    )


@app.command()
def ui(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Experiment YAML to preload.")
    ] = None,
    port: Annotated[int, typer.Option(help="Port of the Streamlit server.")] = 8501,
    results_dir: Annotated[
        Path, typer.Option(help="Where results/<suite>/ live (artifacts, LTT knobs).")
    ] = Path("results"),
    headless: Annotated[
        bool, typer.Option("--headless/--no-headless", help="Do not open a browser.")
    ] = True,
) -> None:
    """Open the interactive supervision page (needs the ``ui`` extra: ``uv sync --extra ui``)."""
    import importlib.util
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        typer.echo(
            "streamlit is not installed: run `uv sync --extra ui` (or `pip install ucagent[ui]`)",
            err=True,
        )
        raise typer.Exit(code=1)
    script = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.headless",
        "true" if headless else "false",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    env = {**os.environ, "UCAGENT_UI_RESULTS_DIR": str(results_dir)}
    if config is not None:
        env["UCAGENT_UI_CONFIG"] = str(config.resolve())
    typer.echo(f"serving the supervision page on http://localhost:{port} (Ctrl-C to stop)")
    raise typer.Exit(code=subprocess.call(cmd, env=env))


@app.command()
def calibrate(
    suite_dir: Annotated[Path, typer.Option("--suite", "-s", exists=True, help="Suite directory.")],
) -> None:
    """Learn-then-Test: select each policy's knob from the calibration split of a suite."""
    from ucagent.experiments.figures import plot_risk_vs_knob
    from ucagent.experiments.suite import calibrate_suite

    table = calibrate_suite(suite_dir)
    cols = [
        c
        for c in [
            "policy",
            "seed",
            "feasible",
            "knob_star",
            "calib_risk_at_star",
            "test_error_at_star",
            "test_cost_at_star",
        ]
        if c in table.columns
    ]
    typer.echo(table[cols].to_string(index=False))
    for path in plot_risk_vs_knob(suite_dir):
        typer.echo(f"wrote {path}")
    typer.echo(
        f"wrote {suite_dir / 'ltt.json'}, {suite_dir / 'ltt.csv'}, {suite_dir / 'coverage.json'}"
    )


@app.command()
def pipeline(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Pipeline YAML.")],
    workers: Annotated[int | None, typer.Option(help="Worker processes.")] = None,
    results_dir: Annotated[Path, typer.Option(help="Where results go.")] = Path("results"),
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Rerun existing runs.")] = False,
) -> None:
    """Run every suite and stage behind the README figures, then draw them (make figures)."""
    from ucagent.experiments.stages import PipelineConfig, run_pipeline

    out = run_pipeline(
        PipelineConfig.from_yaml(config),
        workers=workers,
        results_dir=results_dir,
        resume=not no_resume,
    )
    for path in out["figures"]:
        typer.echo(f"wrote {path}")
    typer.echo(f"timings (s): {out['timings_s']}")
