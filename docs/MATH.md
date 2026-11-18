# Math

Expanded definitions from Section 2 of `AGENT_INSTRUCTIONS.md`, the derivations behind the controllers, the Learn-then-Test bound, and the assumptions the method rests on. Symbols match the code: `u`, `eps`, `w`, `S`, `R`, `G`, `tau`, `lambda`, `alpha`, `delta`, `rho`, `q`, `K`, `K'`, `p_h`, `p_e`, `c_*`.

## 1. Local and global correctness

A task instance is a chain of `n` steps. At step `i` the agent's context holds the committed output of step `i-1`, `c_{i-1}` (with `c_0` the start context). The **local truth** `t_i = follow(c_{i-1}, r_i)` is the correct answer to step `i`'s sub-question given that context, even if `c_{i-1}` is wrong; the **global truth** is `e_i` on the true chain. Step `i` is *locally* correct if `a_i ≡ t_i`, the final answer is correct iff `a_n ≡ e_n`.

Everything the uncertainty estimators see is local, so `eps_i` is calibrated on local labels (`ucagent.api.observations_from`), and the propagation from local errors to final error is carried by `w_i`. The human oracle of `ask_human` is local (`tasks.base.LocalOracle`). A global oracle would make "ask at the last step" optimal with one intervention; it exists only for the degeneracy test in `tests/test_m1_acceptance.py`.

**Benchmarks without a knowledge base (MuSiQue, `tasks/musique.py`).** The decomposition gives the gold answer of each hop, which is the local truth while the previous committed answer is the gold one; off track the data holds no local truth. Such steps are excluded from calibration (the optional task method `has_local_truth`), count as locally wrong, and a local reviewer confirms the committed answer instead of repairing the chain; only backtrack or escalation can repair it. This keeps the degeneracy rule above on a real benchmark.

## 2. Step uncertainty and calibrated local error

`K` samples at temperature `T` are clustered by an equivalence judge into `C_1..C_m` with proportions `p_j`. The committed answer is a representative of the largest cluster.

- disagreement: `u = 1 - max_j p_j`
- semantic entropy: `u = H / log K`, `H = -sum_j p_j log p_j`
- verbalized: `u = 1 - mean stated confidence` (baseline)

`eps = P(locally wrong | u, batch size)` by isotonic regression on the `train` split (`core/uq/calibration.py`), fitted once for batches of `K` samples and once for `K'`; at run time the map is selected by the size of the step's batch (`Tables.eps(u, n_samples)`), and an unfitted size is an error rather than a fallback (review finding R1, 2026-11-08). Before calibration `eps = u`.

## 3. Influence, survival, projection

- `w_i` = the probability that a local error at step `i` is *fatal* (flips the final answer) when the remaining steps run under policy `none`; prior `w_i = 1`. Estimated by paired counterfactual replay (M6, revised after the review of report v0.3): the checkpoint after step `i` is forked once with a locally wrong answer and once with the gold answer (the base run stands in when its committed answer was the gold one); with failure rates `p_wrong` and `p_correct` of the two continuations, `w_i = 1 - (1 - p_wrong) / (1 - p_correct)` clipped to `[0, 1]` (`0` when `p_correct = 1`). Under independent local errors this is exactly the fatality probability, which makes the product below exact. The earlier *absolute* estimate `w_i = p_wrong` counted the later steps' own failures and the future factor `G_i` then counted them again: with a step-2 error rate of 0.2 that always overwrites step 1, the true final failure is 0.2, the absolute product gives `1 - (1 - 0.5 * 0.2)(1 - 0.2) = 0.28`, the paired estimate gives 0.2. `risk.influence_estimator: paired | absolute` selects the estimator (`paired` is the default; artifacts record `counterfactual:paired`). Until correlated errors are modelled, `R_i` below is best read as a calibrated risk score whose end-to-end consequences Learn-then-Test certifies. Position priors in both directions exist for ablations; no direction is assumed (in a chain with reconvergence, later errors have fewer chances to be corrected, so `w_i` may increase with `i`).
- Survival `S_i = prod_{j<=i} (1 - w_j eps_j)`, risk `R_i = 1 - S_i`.
- Future factor `G_i = E[prod_{l>i} (1 - w_l eps_l)]` under policy `none`, per `(n, i)`; default 1.
  - **Convention of the fit (stated 2026-11-18, after an independent review).** The mean is over the training questions, and a later step that is off track there has no label and enters the product as certain to survive. So `G_i` is not the survival of the later steps given that the chain is on track at `i`. On MuSiQue with confidence-weighted agreement 560 of the 600 second hops have a label; the stored `G_1` is 0.853, and one minus the mean estimated error of the labelled second hops is 0.843 (`scripts/rule_actions.py --future`). The rule and the budgeted controller therefore see slightly too little future risk. Changing the fit would change their decisions, so it waits for a replay with predictions; the gate does not use `G`.
- Projected final risk if step `i` is committed as is: `Rhat_i = 1 - S_{i-1} (1 - w_i eps_i) G_i`.

**Assumptions (flagged in code with `# OPEN-QUESTION:`):** local errors are independent across steps; influence is one number per step; `G` and the resample table are estimated under `none` although actions change later steps (the LTT guarantee covers the end-to-end effect, these tables do not).

## 4. Action effect models (`core/risk/effects.py`)

With `S = S_{i-1}`, `G = G_i`, `w = w_i`, `eps = eps_i`:

| action | post-action projection | `Delta` (reduction of `Rhat`) | cost |
|---|---|---|---|
| proceed | `1 - S (1 - w eps) G` | 0 | 0 |
| ask_human | `1 - S (1 - w p_h) G` | `S G w (eps - p_h)` | `c_h` |
| resample (expand to `K'`) | `1 - S (1 - w ehat_r(u, K')) G` | `S G w (eps - ehat_r)` | `c_r + (K' - K) c_llm` |
| backtrack(`j`) | `1 - S_new G` | `G (S_new - S (1 - w eps))` | `c_b + (K' - K) c_llm + (i - j) K c_llm` |
| escalate | `p_e` | `Rhat - p_e` | `c_e` |

with `S_new = S_{j-1} (1 - w_j ehat_r(u_j, K')) prod_{l=j+1..i} (1 - w_l ebar_l)`.

**What a resample is (2026-11-08, review findings R1 and R2).** A resample *expands* the step's batch: the same prompt is sampled at indices `K..K'-1`, the step is re-estimated on all `K'` samples, and `eps` comes from the map fitted for `K'`-sample batches. That is exactly the experiment the resample-effect table is fitted on (`observations_from` estimates `u` at `K` from the first `K` and at `K'` from all `K'` samples of one draw), so `ehat_r(u, K')` was, until 2026-11-17, the conditional expectation of the post-action `eps` given the pre-action `u`, held as a table of means over ranges of `u` (the paragraph after this one has the present estimate), and the action costs only the `K' - K` extra draws. A backtrack to `j` expands step `j` the same way (its context `history[:j-1]` is unchanged, so its prompt is), then redoes the later steps with `K` fresh samples each at their position means `ebar_l`. One expansion per batch is all the table describes: a step that already holds `K'` samples is neither resampled nor a backtrack target, and with `K' = K` neither action exists. Before this date the runtime drew a fresh `K'` batch at a new attempt and mapped it through the `K` map, which the fitted table did not describe; the runs recorded up to then are "as run" (see `DECISIONS.md`). The backtrack target is the committed step with the largest `w_j eps_j`, feasible only if its share of `R_{i-1}` is at least `backtrack_min_share` and its batch can still be expanded. `resample` is never chosen without a fitted resample table. The "committed projection" (same formulas with `G = 1`) is what the budgeted controller's model-based risk budget uses.

**How `ehat_r` is estimated (2026-11-17).** `ehat_r(u, K')` is an isotonic regression, on the training steps, of the error label of the batch expanded to `K'` samples on the score `u` of its first `K` samples (`resample_map` in the artifact). It is the twin of the `K` map `eps_K(u)`, which regresses the label at `K` on the same scores, so the expected reduction `eps_K(u) - ehat_r(u)` is the net rate at which the expansion turns a wrong answer into a right one. It is zero where the expansion cannot change the answer: a unanimous batch of `K` keeps its answer when `K' <= 2K`. Until this date `ehat_r` was the mean of the `K'`-map value over five fixed ranges of `u` (`resample_table`). That had two defects, found by a review of the recorded runs:
- One value per range overstated the effect on the steps a controller chose to expand, because those are the steps of a range with the highest `eps`.
- A score of `1 - 0.8` is `0.19999999999999996` in floating point and fell below the edge at 0.2, so four-to-one splits were counted with the unanimous batches, at the fit and at run time.

The table is still fitted, with its edges met up to `1e-9`, as a diagnostic. The artifacts of the runs recorded before this date carry only the table and are read as they were fitted (`schema_version` 1). The redone steps of a backtrack are still valued at their position means `ebar_l`, which is the average case and not the case of the step at hand.

## 5. The value-of-intervention controller (`marginal`)

Objective (A): minimise expected cost subject to `E[final error] <= alpha`. Lagrangian: minimise `E[cost] + mu E[final error]`. Under the independence assumption and the product form, the one-step-lookahead rule at step `i` chooses the action with the largest net value `Delta_a - tau c_a`, `tau = 1 / mu`, and proceeds when no action has positive net value. Acting iff `Delta_a / c_a > tau` reads: `tau` is the minimum risk reduction per unit cost worth paying for; smaller `tau` means more interventions.

For `ask_human` with small `eps` the rule reduces to gating on `S_{i-1} G_i w_i eps_i`, a fixed threshold on influence-weighted, survival-weighted calibrated error. Two consequences, both tested:

- The **sunk-risk property**: when `S_{i-1}` is already small, protecting step `i` is worth less (`tests/test_effects.py::test_sunk_risk_lowers_ask_value`), because fixing step `i` cannot repair earlier steps; only `backtrack` and `escalate` can.
- **Budgeting is predicted not to help for objective (A)**: a per-trajectory budget tightens the threshold as `R_{i-1}` grows, the opposite of the Lagrangian rule. This is hypothesis H3c; the experiment can falsify it, and if `budgeted` wins that points at dependence between steps.

The rule is myopic; the exact dynamic programme over `(i, S)` is an optional reference controller.

## 6. The budgeted controller (`budgeted`): a model-based per-output risk budget

Objective (B): every delivered output satisfies `R_n <= lambda alpha`. With a spending function `g(i, n)` (nondecreasing, `g(n, n) = 1`) the budget after step `i` is `allowed_i = lambda alpha g(i, n)`. If the committed projection `1 - S_{i-1}(1 - w_i eps_i)` is within budget the step proceeds; otherwise the cheapest feasible action whose committed projection is within budget is taken; if none, the case is escalated. Delivered outputs therefore satisfy the certificate by construction, conditional on the calibration of `eps` and on independence, which is why `lambda` is still calibrated with Learn-then-Test for objective (A).

Spending functions (`core/control/spending.py`): `flat` (`g = 1`), `linear` (`i / n`), `pocock` (`ln(1 + (e - 1) t)`), `obf` (Lan-DeMets O'Brien-Fleming shape normalised to `g(n, n) = 1`), `influence_proportional` (`cumsum(w) / sum(w)`). The analogy with alpha-spending in group-sequential trials is loose: there the budget is over dependent looks at one hypothesis, here step contributions are additive and independent.

## 7. Learn-then-Test calibration of the knob (`core/control/ltt.py`)

Grid `knob_1 < knob_2 < ... < knob_m`, ascending = most conservative first for every shipped policy. On the calibration split (`n_c` trajectories, i.i.d. from the deployment distribution) run the policy at each grid point and count errors `X_k`.

- Hypotheses `H_k: R(knob_k) > alpha`, where `R` is the true end-to-end error rate of the full adaptive pipeline at that knob.
- p-value: exact one-sided binomial `p_k = P(Bin(n_c, alpha) <= X_k)`. Under `H_k`, `X_k ~ Bin(n_c, R)` with `R > alpha` stochastically dominates `Bin(n_c, alpha)`, so `p_k` is a valid (conservative) p-value. Hoeffding `exp(-2 n_c (alpha - X_k / n_c)_+^2)` is available as an option.
- Fixed-sequence testing: test `H_1, H_2, ...` in the fixed order, each at level `delta`; stop at the first `p_k > delta`; select `knob* =` the last rejected grid point (the least conservative knob among the rejected ones). If `H_1` is not rejected, report "no feasible knob at this alpha" and the smallest achieved calibration risk.
- Guarantee (family-wise error control of fixed-sequence testing): `P(a knob is certified and R(knob*) > alpha) <= delta`, probability over the calibration data; equivalently, with probability at least `1 - delta` either no knob is certified (`feasible = False`, the smallest achieved calibration risk is reported) or the certified configuration's *population* end-to-end error on tasks exchangeable with the calibration tasks is at most `alpha`. This is a statement about the population error of one controller on a grid fixed in advance, not about the empirical error of any finite batch of future tasks, and certifying several controllers separately at level `delta` is not a simultaneous guarantee for picking the best of them.

**What it covers.** The end-to-end error of the whole pipeline at a knob, treated as a black box: resampling, backtracking, noisy humans (`p_h`, `p_e`) and escalation included, because each grid point is tested as its own hypothesis. No per-trajectory monotonicity of the loss in the knob is needed (it fails here: a smaller knob can turn a right trajectory wrong through resampling or a noisy oracle), which is why conformal risk control is not used.

**What it does not cover.** Conditional guarantees (per task category, per instance), distribution shift between calibration and deployment, the per-output budget of the `budgeted` controller (a model-based risk budget of `lambda alpha` that rests on the calibration of `eps` and on independence, not a finite-sample bound per output), any configuration not on the tested grid, and any grid, prompt, label rule or policy revised after looking at the calibration data (the development runs of `EXPERIMENTS.md` did this; the confirmatory pass certifies a pre-specified grid on fresh calibration questions). Human behaviour and supervisor overrides at deployment must match what was calibrated.

**Finite test sets.** A held-out test error is itself an estimate; the acceptance test (`tests/test_m5_acceptance.py`) compares it with `alpha + 2 sqrt(alpha (1 - alpha) / n_test)` across seeds and requires the `1 - delta` fraction to hold. With `n_c = 100` and `alpha = 0.05`, rejection at `delta = 0.1` needs at most one error (`P(Bin(100, 0.05) <= 1) = 0.037`, `<= 2` gives `0.118`), which is why the small configs use `alpha = 0.1`.

## 8. KnowNo-style per-step conformal gate (`knowno`, a baseline)

After Ren et al. (2023), adapted from a planner's enumerated options to sampled answer clusters. At a step with clusters `C_j` and proportions `p_j`, the nonconformity of a candidate is `s_j = 1 - p_j`. The train split provides the scores of the locally correct cluster, `s* = 1 - p(correct cluster)` (`1` when no cluster is correct); with miscoverage `eps` the quantile is the `ceil((m + 1)(1 - eps))`-th smallest of the `m` scores (`1` when that rank exceeds `m`; `Tables.qhat`). The prediction set is the top cluster plus every `j` with `s_j <= qhat`; the step proceeds iff the set is a singleton, otherwise `ask_human`. The knob is `eps`: smaller `eps` gives larger sets and more asks, so the ascending grid is less conservative, as Section 7 requires. Without fitted scores the gate asks when `1 - p_max > eps` and flags the decision as uncalibrated.

This is a KnowNo-style *sampled-cluster adaptation*, not KnowNo. Same: the conformal set over candidates and "ask when the set is not a singleton". Different: candidates are sampled clusters, not enumerated options scored by the model, and there is no outside option (KnowNo's option "an option not listed here" lets the set express an answer the candidates missed; a sampled-cluster set cannot, so when every sample agrees on a wrong answer the set is a confident singleton and the gate proceeds: the floor seen on MuSiQue is a property of this adaptation, not of KnowNo); the quantile is fitted on the train split so that the calibration split is spent, as for every controller, on Learn-then-Test for the end-to-end error; the gate ignores costs, influence and the risk accumulated so far, which is exactly what its comparison with `marginal` isolates. KnowNo itself provides per-step coverage and, in its Section 3.3, a sequence-level task-completion guarantee for multi-step plans; the distinction with this pipeline is therefore not "per-step versus end-to-end" but what is calibrated (a coverage level of candidate sets versus the error of an adaptive pipeline with several priced actions) and how (a sequence-level conformal quantile versus Learn-then-Test over a knob grid).

## Open questions

Code locations are marked with `# OPEN-QUESTION:` comments.

- Correlated local errors: the product form assumes independence (`core/risk/propagation.py`). Should propagated risk include a correlation term, and does dependence explain any case where `budgeted` beats `marginal`?
- Downstream error correction: reconvergence `rho` is the synthetic stand-in; does counterfactual `w_i` capture correction on real tasks?
- Spending shape versus marginal rule: under what dependence structure would a spending function help?
- Sunk risk in deployment: the marginal rule protects clean trajectories and gives up on doomed ones unless backtracking or escalation pays; is objective (B) the right default in applied settings?
- Myopic versus exact: how much does the one-step rule lose against the dynamic programme on the mock?
- The guarantee under adaptivity: Section 7 states what Learn-then-Test covers; conditional guarantees and shift are open.
- Noisy humans (`p_h`, `p_e`) and the choice of `K`, `K'`.
