# Pre-registration: common random numbers (CRN) for recipe selection, gated program

Written 2026-09-27, after the Task 0 search and before any measurement for
Tasks 1 and 2. The analysis of the paper stays frozen. Outputs: code, tests,
JSON, figures and this pre-registration. No paper prose, no documents. The
program stops at the gate; Task 3 is not started.

## Task 0: novelty (done before this file; recorded in `results/crn/novelty.json`)

CRN (shared seeds across compared alternatives) is established in general:
- Pearce, Poloczek & Branke, *Bayesian Optimization Allowing for Common Random
  Numbers*, Operations Research 70(6), 2022, including machine-learning
  hyperparameter tuning as a use case;
- Sharma, arXiv:2512.24145, paired seeds for learning-based multi-agent
  simulators, where paired evaluation resolves differences that independent
  evaluation leaves inconclusive;
- Sellam et al., MultiBERTs, 2021: a paired Multi-Bootstrap over shared
  pretraining seeds.

Related, but not CRN:
- seed-variance decompositions (Dodge et al. 2020; Bouthillier et al. 2021;
  PolyPythias);
- a paired multi-seed protocol on small vision and text benchmarks (Du,
  arXiv:2511.19794);
- covariate adjustment from training logs (Huang, arXiv:2608.02705);
- an initialisation confound under shared seeds across architectures (Yang
  et al., arXiv:2609.10702).

Not found: a quantification of the cross-recipe correlation of seed deviations
in language-model pretraining, the implied reduction in gap variance, or its
effect on how many recipe pairs a calibrated candidate-level procedure can
resolve. The search was bounded (web search plus reading the closest papers);
unindexed or industry practice cannot be excluded.

## Definitions

- y_{r,s} is a run's final metric, for recipe r and seed index s.
- Matched gap: y_{a,s} - y_{b,s}. Mismatched gap: y_{a,s} - y_{b,s'} with s != s'.
- With common seed variance, the correlation r of seed deviations across
  recipes gives:
  - Var(matched gap) = (1 - r) Var(mismatched gap);
  - gap-variance reduction = r;
  - reduction factor = 1 / (1 - r).
- **Estimator of r.** Two-way layout (recipe x seed index). r is the intraclass
  correlation of the seed-index effect:
  - r = s2_S / (s2_S + s2_E), where s2_S is the seed-index main-effect variance
    component and s2_E the residual;
  - its 95% interval comes from the F distribution with (S - 1, (R - 1)(S - 1))
    degrees of freedom.
  - Also reported per recipe pair: the matched-gap variance (2 df) against the
    mismatched-gap variance, and their ratio averaged over pairs.
- **rho_D, the data-order share of seed variance.** rho_D =
  s2_D / (s2_D + s2_W), where s2_D is the between-run variance of the
  data-seed arm (data order varies, initialisation fixed) and s2_W that of the
  weight-seed arm (initialisation varies, data order fixed). Its 95% interval
  comes from F(2, 2): three runs per arm, two degrees of freedom each.

## Task 1: compute-multipliers (7 recipes x 3 seeds, 1e19 FLOPs)

- **First step.** Determine from run.json and the released protocol whether a
  seed index fixes the data order (and the initialisation) across recipes.
  - If it does not, this is reported and Task 1 stops there.
- **Metric.** The primary metric is the final evaluation metric used for this
  suite in the repository. Other harvested per-run metrics are secondary.
- **P1.** If the seed index is shared, r lies in [0.2, 0.7] (point estimate).
  - With 3 seeds its 95% interval is wide, spanning at least 0.6 in width.
  - Stated now: three seeds make it noisy.

## Task 2: PolyPythias, upper bound

- **Models.** `EleutherAI/pythia-160m-data-seed{1,2,3}` and
  `pythia-160m-weight-seed{1,2,3}`, final checkpoint (main).
  - Downloaded one at a time (375 MB each), evaluated, then deleted.
  - Run in an isolated scratch virtual environment (transformers, reusing the
    installed torch) at low priority.
- **Held-out sample, fixed now.** Text written after the Pile (2020), so it is
  not in the training data:
  - the full text of arXiv:2504.11393v2 (DataDecide, 2025), followed by
    arXiv:2609.10702v1 (2026), as extracted to plain text;
  - the first 20 chunks of 1,024 GPT-NeoX tokens, 20,480 tokens in all.
- **Metric.** Mean token cross-entropy (nats). Per-chunk losses are also kept.
- **Secondary, conditional.** If the model cards state that the data-seed
  models share the original pythia-160m initialisation and the weight-seed
  models its data order, `pythia-160m` is added to both arms as a fourth run,
  giving 3 degrees of freedom each.
- **P2.** rho_D is in [0.3, 0.7] (point estimate). Its F(2, 2) 95% interval
  covers most of [0, 1], so rho_D is essentially not identified by three runs
  per arm; that is reported as such.
- **As an upper bound.** rho_D bounds CRN's benefit if only the data order is
  shared, and assumes perfect cross-recipe correlation of the data-order effect.
  If initialisation can also be shared (same architecture, with parameters
  copied explicitly, per arXiv:2609.10702), the bound is the total seed
  variance.

## Gate

- **Threshold.** CRN is not worth pursuing if the gap-variance reduction is under
  20%, i.e. r < 0.2, a reduction factor below 1.25.
- **Decision rule, fixed now.**
  - Recommend the controlled experiment (Task 3) if the Task 1 point estimate of
    r is at least 0.2.
  - If Task 1 is unavailable, recommend it if the Task 2 bound (rho_D) is at
    least 0.2.
  - Do not recommend it if every available point estimate is below 0.2.
  - Because every interval will be wide, the recommendation states that
    Task 3 is the experiment that measures r precisely. Tasks 1 and 2 only
    screen.
