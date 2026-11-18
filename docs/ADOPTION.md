# Plugging in your own task

`ucagent` controls any multi-step workflow that can be expressed as a `Task`: a sequence of steps where each step's output feeds the next. You implement the task and a human interface; the library provides sampling, uncertainty, calibration, the controllers, the guarantee and the audit log. The worked example below is `examples/toy_task.py` (an arithmetic chain), exercised by `tests/test_adoption_example.py`.

## 1. Implement `Task`

The protocol is `ucagent.core.protocols.Task`. Three ideas matter:

- **Context.** Step `i` sees the committed output of step `i-1` (`current`). Your `step_prompt` must start with the header `render_header(instance_id, i, current, attempt)`; the mock model reads it, real models ignore it.
- **Local versus global truth.** `local_truth(instance, i, current)` is the right answer to step `i` *given the context the agent actually has*, even if that context is wrong; `global_truth(instance, i)` is the true chain. Local labels calibrate step error; global labels decide final correctness. A human reviewing one step is local too.
- **Distractors.** `distractor_names(instance, i, current)` lists the wrong answers present in the step's context (empty when the step is unambiguous). The mock draws its errors from them; real models do not need them, but keeping the method honest keeps your calibration labels honest.

```python
from ucagent.core.llm.mock import render_header
from ucagent.core.protocols import Instance
from ucagent.tasks import register_task

class ArithmeticChainTask:
    name = "arithmetic"
    def instances(self, split): ...            # yield Instance objects
    def get_instance(self, instance_id): ...   # cheap lookup by id (cache it)
    def n_steps(self, instance): return instance.n_steps
    def initial_context(self, instance): return str(instance.start)
    def step_prompt(self, instance, i, history, attempt):
        current = history[-1].answer if history else str(instance.start)
        header = render_header(instance.instance_id, i, current, attempt)
        ...                                     # return [SystemMessage, HumanMessage]
    def parse_answer(self, text): ...           # "Answer: 12" -> "12"
    def local_truth(self, instance, i, current): ...
    def global_truth(self, instance, i): ...
    def distractor_names(self, instance, i, current): ...
    def random_entity(self, instance, rng): ...
    def is_final_correct(self, instance, answer): ...
    def is_locally_correct(self, instance, i, current, answer): ...

register_task("arithmetic", lambda block: ArithmeticChainTask(**{k: v for k, v in block.items() if k != "name"}))
```

The full version is about 150 lines: `examples/toy_task.py`.

**Tasks without a knowledge base.** When the context of a step is already wrong, a benchmark may hold no answer for the step's sub-question (MuSiQue: `src/ucagent/tasks/musique.py`). Implement the optional method `has_local_truth(instance, i, current) -> bool`; steps where it is False are excluded from calibration, count as locally wrong, and the local oracle confirms the committed answer there instead of repairing the chain. The MuSiQue adapter is the worked example of local-versus-global truth on real data.

## 2. Implement `HumanInterface`

Two methods, both receive a plain dict and return a string:

```python
class ReviewQueue:
    def check_step(self, payload):   # payload: instance_id, step, current, committed, clusters, question
        ...                          # return the corrected answer for this one step
    def take_over(self, payload):    # payload: instance_id, question, history, pending
        ...                          # return the final answer for the whole case
```

In experiments `tasks.base.make_human` builds a simulated human from the task's truths (with optional noise `p_h`, `p_e`); `ucagent demo` uses a terminal; a production queue would enqueue the payload and block, or return a placeholder and resume the thread later (the graph is checkpointed at the interrupt).

## 3. Run it

**In code**, with any LangChain chat model (the mock below, or `get_chat_model("ollama:llama3.1:8b")` for a cached real model):

```python
from ucagent.api import run_instances
from ucagent.core.graph.runtime import RunSettings
from ucagent.core.llm.mock import MockChainLLM

task = ArithmeticChainTask(n_steps=3, n_instances={"test": 20})
llm = MockChainLLM(task=task, seed=0)
settings = RunSettings(policy="marginal", knob=0.05, K=5, K_prime=10)
trajectories = run_instances(task, llm, settings, [i.instance_id for i in task.instances("test")], ReviewQueue())
for t in trajectories:
    print(t.final_answer, t.final_correct, t.total_cost, t.audit())
```

**From configs**, by registering the task through a plugin module:

```yaml
run_id: arithmetic_demo
plugins: [examples.toy_task]
task: {name: arithmetic, n_steps: 3, n_instances: {train: 100, calib: 100, test: 100}}
llm: {backend: mock, K: 5, K_prime: 10}
control: {policy: marginal, alpha: 0.1}
```

Then the usual commands apply unchanged: `ucagent fit` (artifact: isotonic `u -> eps`, resample effect, future factor, influence), `ucagent suite` + `ucagent calibrate` (Learn-then-Test knob with the `1 - delta` guarantee), `ucagent run`, `ucagent figures`.

`llm.model_kwargs` passes provider options to the chat-model class unchanged (for ChatOllama: `reasoning: low`, `num_ctx: 8192`, `num_predict: 1024`, as in `configs/musique_small_gptoss.yaml`); non-empty kwargs become part of the cache key, so runs with different options never share samples. In code: `get_chat_model("ollama:gpt-oss:20b", model_kwargs={"reasoning": "low", "num_ctx": 8192})`.

## 4. What you get back

Every trajectory is a `Trajectory` with per-step records (`u`, `eps`, `w`, answer, source: llm or human) and every decision with its reason and the numbers behind it (`Trajectory.audit()`); one parquet row per trajectory; calibration artifacts versioned with data and config fingerprints and refused when they do not match the run's estimator and `K`.

## 5. What to calibrate on

The artifact needs local-correctness labels on a `train` split and the guarantee needs an i.i.d. `calib` split from the deployment distribution. For a real workflow that means a few hundred labelled cases; `docs/MATH.md` Section 7 explains why the small configs use `alpha = 0.1` with 100 calibration cases and what the guarantee does and does not cover.

## 6. Supervising a run interactively

`InteractiveSession` (`ucagent.core.graph.session`) drives one trajectory pause by pause, for interfaces that cannot block on a person. With `supervise=True` it stops before every controller decision; `ask_human` and `escalate` stop it with their interrupt payload; `done` ends it.

```python
from ucagent.core.graph.build import build_graph
from ucagent.core.graph.session import InteractiveSession
from ucagent.core.records import Action, Decision
from ucagent.experiments.runner import trajectory_from_state

session = InteractiveSession(build_graph(task, llm), settings, tables, supervise=True)
pause = session.start(instance)
while pause.kind != "done":
    if pause.kind == "control":                  # before the controller decides
        proposal = session.proposal()            # what the policy would do, with its reason
        effects = session.effects()              # expected risk reduction and cost per action
        if reviewer_disagrees(proposal, effects):
            pause = session.override(Decision(action=Action.ASK_HUMAN, reason="looks wrong"))
        else:
            pause = session.accept()
    elif pause.kind == "check_step":             # the agent asks for one step to be checked
        pause = session.answer(review_queue.check_step(pause.payload))
    else:                                        # "take_over": the agent hands the case over
        pause = session.answer(review_queue.take_over(pause.payload))
trajectory = trajectory_from_state(task, settings, "review", session.values, instance.instance_id)
```

`allowed_actions()` says which overrides the caps and the batch-size rule allow (a resample expands the step's batch to `K'` samples, so a step that already holds `K'` cannot be resampled or be a backtrack target; `backtrack_targets()` lists the feasible targets). An override travels through `RunSettings.force_decisions` for that one decision round of the step and appears in the audit log as `human override: <action> (<your reason>); policy proposed <action>: <its reason>`, flagged with `numbers["override"] = 1`. The Streamlit page (`ucagent ui`) is this loop with widgets; point it at your task with a config that lists your plugin module (`ucagent ui --config my_task.yaml`).

## 7. Plug in your own uncertainty estimator or calibrator

Estimators and calibrators are interchangeable components; nothing downstream depends on how a step's score is computed.

**An estimator** turns the `K` samples of a step into a score `u` in `[0, 1]`, a clustering and a committed answer. Register it by name from a plugin module (listed under `plugins:` in the config, so every worker imports it) and declare which sample fields it reads. A run records each sample's parsed answer and stated confidence, and the fit re-estimates every step from them:

```python
from ucagent.core.uq.base import StepUncertainty, register_estimator
from ucagent.core.uq.equivalence import cluster_samples


class ShortAnswerEstimator:
    """Example: trust short committed answers more (a toy, not a recommendation)."""

    name = "short_answer"
    reads = frozenset({"answer"})  # add "confidence" if you use the stated confidence

    def __init__(self, judge):
        self.judge = judge

    def estimate(self, samples, context):
        clusters = cluster_samples(samples, self.judge, context)
        top = clusters[0]
        u = min(1.0, (1.0 - top.proportion) + len(top.representative) / 200)
        return StepUncertainty(u=u, entropy=0.0, clusters=clusters, committed=top.representative)


register_estimator("short_answer", ShortAnswerEstimator)
```

Then set `uq: {estimator: short_answer}`. The fit refuses an estimator that reads stated confidences when the training run recorded none, and one that declares a field a run does not record (the raw reply, `text`, is not recorded). A factory function may be registered instead of a class; what it declares is then read from the estimator it builds.

**A calibrator** maps named step features (`ucagent.core.uq.features.FEATURE_NAMES`: the estimator's `u`, entropy, disagreement, cluster count, stated-confidence summaries, step position) to the probability that the step is locally wrong. Four are registered: `isotonic` (one feature; the default, on `u`), `logistic`, `boosted` and `gp` (any feature subset):

```yaml
uq:
  estimator: semantic_entropy
  calibrator: logistic
  features: [u, conf_mean, conf_agreement, position]
```

A new calibrator is a class with a `fit(features, rows, labels)` class method, `predict(row)`, `to_data()` and a `from_data(data)` class method that inverts `to_data`, registered with `ucagent.core.uq.calibrators.register_calibrator` from a plugin module. The artifact stores the fitted calibrator per batch size and is refused by a run whose estimator, calibrator or features differ. `scripts/estimator_arms.py` compares estimator and calibrator arms on a finished suite at zero model calls.
