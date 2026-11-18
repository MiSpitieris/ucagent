"""ucagent: uncertainty-controlled multi-step LLM agents on LangGraph.

The package has two halves. ``ucagent.core`` is the reusable library (LLM
backends, uncertainty estimators, calibration, risk propagation, controllers,
the LangGraph control loop). ``ucagent.tasks`` and ``ucagent.experiments`` are
the research harness (synthetic task, benchmark adapters, runner, metrics,
figures). The core never imports the harness.
"""

__version__ = "0.2.0"
