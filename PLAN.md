# ASLA Enhancement Plan — toward an AAAI submission

This is the working plan for enhancing ASLA (Algorithm-Selection Leaderboard
Audit). It is a living document: checkboxes are updated as work lands. Every
workstream must keep `pytest` green and add tests for new behavior.

## 1. What exists today (baseline)

The repo audits whether small-compute scaling-law projections select the same
training intervention that measured target-budget runs would select:

- **Schema & IO**: canonical runs table (`intervention`, `intervention_class`,
  `compute`, `seed`, `bpb`, optional `downstream`/`params_n`/`tokens_d`),
  parquet IO, W&B harvesting, HPC manifest tooling.
- **Fits**: 3-parameter compute power law `E + A·C^(-α)` and 5-parameter
  Chinchilla `E + A·N^(-a) + B·D^(-b)`, both unit-invariant, via
  `scipy.curve_fit`.
- **Decision rules**: plain projection ranker, largest-single-scale ranker,
  and a bootstrap **uncertainty gate** that escalates the top-2 comparison to
  an intermediate budget when the projected gap is within noise (`τ` threshold
  on a bootstrap z-like statistic).
- **Audit**: cell-wise seed bootstrap CIs on decision metrics (top-1 accuracy,
  top-k recall, pairwise accuracy, Kendall τ, Spearman, regret), seed-noise
  band, crossover detection (projected vs. measured order disagreement),
  fitted crossover-budget estimates with bootstrap CIs.
- **Synthetic scenarios**: `clean_crossover`, `saturation_crossover`
  (misspecification: saturating truth fools the power law), `noise_close_call`,
  `negative_controls_only`; Monte Carlo comparison of plain/largest/gate.
- **71 passing tests.**

### Honest assessment of the gap to AAAI

What exists is a solid *audit harness* but its methods are standard (power-law
fits + bootstrap + a heuristic gate). The paper needs (a) a crisper formal
framing, (b) at least one genuinely new method with an argument for why it
works, (c) a reusable benchmark artifact, and (d) a thorough controlled
empirical study. That is what the workstreams below add.

## 2. Novelty thesis for the paper

> **Claim.** Leaderboards and method papers increasingly justify training
> interventions by small-compute scaling-law extrapolation, yet extrapolation
> is used as a *decision procedure* without decision-level guarantees. ASLA
> formalizes intervention selection as a fixed-budget decision problem,
> quantifies when extrapolation-based selection fails (noise close-calls,
> functional-form misspecification, late crossovers), and contributes
> selection rules that are *misspecification-aware* and *compute-cost-aware*,
> evaluated on a controlled benchmark with known ground truth.

Three methodological contributions, in order of importance:

1. **Cost-aware sequential selection ("racing with scaling laws")** — WS4.
   Instead of one-shot projection from a fixed ladder, treat selection as
   budget-constrained successive elimination: advance interventions up the
   budget ladder only while their projected-target confidence intervals
   overlap the leader's. Report the full **compute–regret Pareto frontier**,
   not a single operating point. This connects scaling-law practice to
   best-arm identification / racing literature but exploits curve structure
   (extrapolation) rather than treating budgets as independent arms — that
   hybrid is the novel part.
2. **Misspecification-aware ensemble projection** — WS2. Fit a small dictionary
   of curve families (power law, saturating, power-law-with-floor …), weight
   them by leave-largest-budget-out extrapolation loss (pseudo-BMA), and
   report an **extrapolation reliability score** from cross-family
   disagreement at the target. Turns the current qualitative "fit is blind"
   message into a quantitative, actionable diagnostic.
3. **Calibrated decision gate** — WS3. Replace the heuristic bootstrap-z gate
   with a conformal (leave-one-budget-out residual) interval gate that has a
   distribution-free coverage interpretation; compare calibration of both.

Plus one artifact contribution:

4. **ASLA-Bench** — WS5. A parameterized generator of selection problems with
   controllable difficulty (crossover location vs. max fit budget, gap-to-noise
   ratio, curvature/saturation strength, ladder geometry), so any selection
   rule can be evaluated with known ground truth. Positioned as the reusable
   benchmark the community lacks.

### Related work to position against (for the paper's intro)

- Scaling-law fitting & pitfalls: Chinchilla (Hoffmann et al.), "Scaling laws
  are not reliable for downstream tasks" lines of work, replication critiques
  of Chinchilla fits.
- Best-arm identification / racing: successive halving, Hyperband, LCB-based
  racing — none exploit cross-budget curve structure.
- Model selection under extrapolation: Bayesian model averaging, stacking;
  conformal prediction for regression.
- Benchmark-design papers (e.g., HPO benchmarks) as precedent for the
  artifact contribution.

## 3. Workstreams

### WS0 — Repo health (DONE unless noted)

- [x] Fix packaging: `pip install -e .` failed (setuptools auto-discovery saw
  `scripts/`+`asla/` as multiple top-level packages). Added explicit
  `[tool.setuptools.packages.find] include = ["asla*"]`.
- [x] Baseline test suite green (71 passed).
- [ ] Add `ruff` config + CI workflow (GitHub Actions: install, ruff, pytest).
- [ ] Add `CITATION.cff` and expand README with the decision-problem framing.

### WS1 — Statistical core hardening

Fixes that make the existing audit defensible under review:

- [x] **Multiple-comparison correction in `detect_crossovers`**: replaced the
  single pooled noise band with per-pair Welch t-tests at the target +
  Benjamini–Hochberg FDR control (default q=0.05) via
  `detect_crossovers_fdr`; legacy band path kept for
  backward compatibility.
- [x] **Heteroscedastic fitting**: `fit_power_law`/`fit_chinchilla` accept
  per-point `sigma` (seed-count-aware SEs) and `asla audit --weighted` fits
  cell means weighted by `sd/√n_seeds`; falls back cleanly when seeds are missing
  (`cell_means_and_sigma`, `fit_all(weighted=True)`).
- [x] **Fit diagnostics**: `FitDiagnostics` (R², RMSE, max |residual|,
  n_points, dof) surfaced via `fit_power_law_diagnostics` and in audit
  reports (`fit_diagnostics` block).
- [x] **Gate ranker reproducibility**: `make_gate_ranker` takes a seed and
  spawns a fresh child RNG per call instead of mutating a shared generator
  across rankers/bootstrap replicates.
- [x] **Truth-ranking uncertainty**: `truth_ranking_with_se` reports the seed
  SE of each target mean and flags statistically tied top groups; audit
  output includes `truth_ties`.

### WS2 — Misspecification-aware ensemble projection (novel method)

- [ ] Promote the saturating form to a first-class fit:
  `fit_saturating(compute, bpb)` for `floor + drop/(1 + C/c_half)` with
  unit-invariant rescaling (same treatment as the power law).
- [ ] Add a third family: power law with breakpoint-free curvature —
  `E + A·C^(-α)·exp(-C/C_sat)` (`fit_damped_power_law`) to cover
  slow saturation. Keep the dictionary small and defensible.
- [ ] **Ensemble projection** (`asla/analysis/ensemble.py`):
  - leave-largest-budget-out (LLBO) extrapolation loss per family;
  - pseudo-BMA weights `w_m ∝ exp(-loss_m/T)` with tie-safe temperature;
  - ensemble target projection = weighted mean; disagreement = weighted SD
    across family projections.
  - **Reliability score** `ρ = disagreement / noise_band` — ρ ≫ 1 means "the
    data cannot distinguish families that disagree at the target" (the
    quantitative version of "fit is blind").
- [ ] `ensemble_ranker` registered alongside existing rankers; included in
  `audit_with_ci` comparisons and the demo.
- [ ] Tests: on `saturation_crossover` the ensemble must (a) down-weight the
  pure power law for the saturating intervention and (b) produce ρ above
  threshold; on `clean_crossover`/`negative_controls` ρ stays small.

### WS3 — Conformal projection intervals & calibrated gate

- [ ] Leave-one-budget-out conformal residuals per intervention: refit without
  each budget, collect scaled extrapolation residuals, use their quantile to
  wrap the target projection in a distribution-free interval
  (`conformal_projection_interval`).
- [ ] **Conformal gate**: escalate to the intermediate budget iff the top-2
  conformal intervals overlap (`conformal_gate_pick`); same escalation
  contract as the bootstrap gate.
- [ ] Calibration study in the benchmark (WS5): empirical coverage of
  bootstrap vs. conformal intervals vs. nominal level across scenario
  families (`scripts/run_calibration_study.py`).

### WS4 — Cost-aware sequential selection (headline method)

- [ ] **Cost accounting**: total compute spent by a selection procedure =
  Σ (runs launched × budget). All rules report `compute_spent` alongside the
  pick, enabling equal-cost comparisons (`asla/analysis/racing.py`).
- [ ] **Racing policy** (`race_pick`): start all interventions at the lowest
  rungs; at each rung, fit each survivor on its accumulated rungs, project to
  target with uncertainty, eliminate interventions whose LCB exceeds the
  leader's UCB (configurable interval source: bootstrap or conformal);
  advance survivors to the next rung; stop at ladder end or single survivor.
- [ ] **Monte Carlo comparison** extended: plain / largest-single / gate /
  racing evaluated on wrong-pick rate, regret, and compute spent;
  `monte_carlo` returns all four rules with cost columns.
- [ ] **Pareto frontier figure**: regret vs. compute-spent across rules and
  τ/α sweeps (`scripts/run_pareto_study.py`, figure `pareto_regret_cost`).
- [ ] **Theory note** (`docs/THEORY.md`): detectability lemma — for two
  power-law curves crossing beyond the fit range, a lower bound on the number
  of seeds/budget span needed for any projection rule to order them correctly
  with probability ≥ 1-δ; plus the elimination-correctness argument for the
  racing policy under valid intervals. Written as paper-ready sketches.

### WS5 — ASLA-Bench: parameterized benchmark generator

- [ ] `asla/data/benchmark.py`: scenario *families* sampled by difficulty
  knobs: `gap_to_noise` (target gap ÷ seed noise), `crossover_position`
  (crossover budget ÷ max fit budget, on log scale), `saturation_strength`,
  `n_interventions`, ladder geometry, seeds per cell. Every generated frame
  carries `ScenarioTruth` so noiseless regret is always available.
- [ ] Difficulty sweep API: `benchmark_grid(...)` yields (config, generator)
  pairs; `asla benchmark` CLI runs a rule set across the grid and writes a
  tidy results parquet + summary JSON.
- [ ] Reference results + figures: wrong-pick-rate heat maps over
  (gap_to_noise × crossover_position) per rule (`benchmark_heatmap`).
- [ ] Tests: generator determinism (same seed → same frame), knob monotonicity
  smoke checks (harder knob settings → no easier measured difficulty on
  average over a small grid).

### WS6 — Reporting, docs, DX

- [ ] `asla report`: single-command markdown report (fits table, diagnostics,
  reliability scores, crossovers with FDR q-values, gate/racing decisions,
  embedded figure references) written next to the audit JSON.
- [ ] Figures upgrades: fit curves extended past data with CI bands and
  measured target points overlaid; ensemble disagreement band figure.
- [ ] Docs: `docs/THEORY.md` (WS4), `docs/BENCHMARK.md` (WS5 usage),
  README rewrite around the decision-problem framing.
- [ ] `scripts/run_paper_experiments.py`: one entry point that regenerates
  every number/figure the paper needs (respects `--fast`).

## 4. Execution order & status

| Phase | Content | Status |
|-------|---------|--------|
| 0 | Packaging fix, baseline tests, PLAN.md | ✅ done |
| 1 | WS1 statistical hardening | ✅ done |
| 2 | WS2 ensemble projection + reliability score | ⬜ pending |
| 3 | WS4 racing policy + cost accounting (headline) | ⬜ pending |
| 4 | WS5 benchmark generator + sweep CLI | ⬜ pending |
| 5 | WS3 conformal gate + calibration study | ⬜ pending |
| 6 | WS6 reporting/docs/CI polish | ⬜ pending |

Rationale for the order: WS1 makes existing claims defensible (reviewers will
check the stats first); WS2 and WS4 are the paper's methods and need the most
iteration time; WS5 is the evaluation substrate for both; WS3 is a strong
add-on but the paper survives without it; WS6 last since it packages results.

## 5. Out of scope (explicitly)

- `mechanism_crossover_budget` (muP-derived predictor) stays a stub — it
  requires width-scaling theory and real training runs; noted as future work.
- Real-model training runs / W&B harvesting improvements — the HPC scaffold
  already covers collection; the paper's controlled study is synthetic +
  any run tables the lab collects with the existing tooling.
- Downstream-metric (non-BPB) selection — schema supports `downstream`; a
  short discussion section, not a workstream.
