# Pre-registered predictions: checkpoint-augmented fitting (Task 1)

**Status: pre-registration. Written and committed before any checkpoint-augmented
ranker was implemented or evaluated.** The only quantities computed beforehand are
the descriptive counts in section 2, which characterise the available data and do
not involve any decision metric.

Date: 2026-08-27. Commit this file, then compute.

## 1. What is being tested

Choshen, Zhang & Andreas (*A Hitchhiker's Guide to Scaling Law Estimation*, ICML
2025, arXiv:2410.11840) find that "fitting scaling laws to intermediate
checkpoints of training runs (and not just their final losses) substantially
improves accuracy", and that discarding roughly the first 10% of checkpoints (or
the first 10B tokens) matters: ARE drops from >15% to 4-10% for some families
when early checkpoints are excluded. Verified against the paper text.

That result is about **fit** accuracy. It has never been evaluated for **decision**
accuracy. Our H1 mechanism says projection's excess mis-selection over
single-scale ranking is fit/extrapolation variance, not crossover bias
(AUDIT_ADVERSARIAL.md: 289/289 specifications, 100% of 300 seed-bootstrap
resamples). If that is right, any technique that reduces fit variance must
improve decision accuracy, and must help most where fit variance is worst.

This is the third arm of the same story:

| lever | direction | status |
|---|---|---|
| add flexibility (3-family ensemble) | worse decisions | confirmed: 18.67% vs 5.33%, +43 flips |
| pool across interventions (shared exponent) | better decisions | Task B: 5.33% -> 2.00% |
| **pool across checkpoints** | **better decisions** | **this task** |

## 2. The data, and the statistical trap (measured before predicting)

DataDecide's released `eval_macro_avg` artifact contains intermediate-checkpoint
evaluations. Counted directly:

| quantity | value |
|---|---|
| (scale, step) cells with all 25 recipes x 3 seeds present | 272 of 272 |
| distinct fitting points per intervention, final checkpoints only | 11 |
| distinct fitting points per intervention, all checkpoints, scales 4M-530M | 225 |
| after discarding the first 10% of each run's steps (the paper's rule) | **208** |
| ratio | **18.9x more fitting points** |

**The trap.** Checkpoints within a run are serially correlated and are not
independent observations. Measured on one 530M run (C4, default seed), the lag-1
autocorrelation of log-residuals about the fitted line is **0.242**, giving an
AR(1) effective sample size of `n (1 - rho) / (1 + rho)` = 25.0 from 41
checkpoints, a deflation factor of **0.61**. Treating 208 checkpoints as 208
independent points would therefore overstate the effective information by roughly
1.6x, understate every standard error, and make the fit over-confident.

We will implement both a naive version (checkpoints as independent) and a
correlation-aware version (AR(1) effective-sample-size weighting, with the
autocorrelation estimated per run from fit residuals), and report both.

## 3. Predictions

Primary evaluation: C4 bits/token, 25 recipes, 1B target, `pairwise_decisions`
estimand, seed-bootstrap CIs, frozen splits, exactly as in Task B. Reference
values from `results/task_b/task_b.json`: plain projection 5.33% [4.00, 8.67],
single-scale 1.33% [0.67, 1.67], shared-exponent 2.00% [1.33, 3.67].

### C1 - Direction and magnitude versus plain projection

Checkpoint-augmented fitting beats plain projection on pairwise mis-selection.

- Point prediction: 5.33% -> **2.0-4.0%**, a relative reduction of **25-62%**,
  centred on ~45%.
- Reasoning: 18.9x more fitting points deflated by the 0.61 ESS factor is an
  effective 11.5x, so the standard error of the fitted slope should fall by
  roughly sqrt(11.5) ~ 3.4x if checkpoints were exchangeable draws. They are not
  - they are concentrated at low compute and add little leverage at the top of
  the ladder - so the realised gain must be smaller than that bound. A reduction
  comparable to pooling's (62%) is the optimistic end; a quarter of the excess is
  the pessimistic end.
- **CONFIRMED** at >= 25% relative reduction.
- **UNDERPOWERED** if the improvement is under 0.5 points absolute and the CI
  upper bound does not fall below plain projection's point estimate.
- **REFUTED** otherwise, including any case where it is worse than plain.

### C2 - The hard bar: versus single-scale ranking

Checkpoint augmentation does **not** beat single-scale ranking (1.33%).

- Confidence: ~65%. Stated deliberately against our own method, as with P2.
- Reasoning: checkpoints add points along the *existing* ladder rather than
  extending it, so they shrink parameter uncertainty without shortening the lever
  arm (12.5x here). The Task B result showed even complete pooling stopping at
  2.00%, short of 1.33%.
- **REFUTED** if the checkpoint-augmented CI lower bound falls below
  single-scale's point estimate. If refuted, that is the headline of this task.

### C3 - Interaction with the lever arm

The gain from checkpoint augmentation is larger at long lever arms.

- Define `rho_C = I(52.5x) / I(4.7x)`, with `I(L)` the flips removed relative to
  plain projection at that arm, bootstrapped as for P4.
- Point prediction: **`rho_C >= 1.5`**. Weaker than P4's 2.0 because checkpoints
  reduce parameter variance uniformly rather than targeting the extrapolation
  specifically.
- **FAILED** only if the point estimate is below 1.5 *and* the CI upper bound is
  below 1.5. **UNDERPOWERED** if the CI includes 1.5 or `I(52.5x) < 3`.

### C4 - Composition with pooling

Checkpoints and pooling reduce **overlapping** variance, so their combination is
better than either alone but **sub-additive**.

- Let `G_ckpt`, `G_pool`, `G_both` be the relative reductions versus plain.
- Point prediction: `G_both > max(G_ckpt, G_pool)` **and**
  `G_both < G_ckpt + G_pool` (sub-additive).
- Quantitatively: `G_both` in **60-85%** relative reduction.
- **REFUTED (redundant)** if `G_both <= max(G_ckpt, G_pool)` within CIs, i.e.
  adding checkpoints to pooling buys nothing.
- **REFUTED (independent)** if `G_both >= G_ckpt + G_pool`, i.e. the two reduce
  disjoint variance components, which the shared-mechanism story does not predict.
- **UNDERPOWERED** if the CIs cannot separate `G_both` from `max(G_ckpt, G_pool)`.

### C5 - The correlation correction matters in a known direction

The naive (independent-checkpoints) and correlation-aware versions differ, and
specifically the naive version is **over-confident**: its seed-bootstrap
intervals are narrower, and it weights checkpoint-dense scales more heavily than
their information content justifies.

- Point prediction: naive CI width < corrected CI width on the projected target
  value, and the two produce different point estimates on at least one design.
- **REFUTED** if the corrected version is indistinguishable from naive on every
  design, which would mean the correlation correction is a no-op here and the
  trap was not a trap.
- This is a methods check, not a claim about decisions: either outcome is
  reportable, and a no-op result would be worth stating plainly.

## 4. What would refute the mechanism

The mechanism claim (H1: the excess is fit variance) is damaged if **C1 is
REFUTED** - a technique that demonstrably reduces fit error failing to improve
decisions would mean decision error is not dominated by fit variance after all.

It is damaged more weakly if C1 is confirmed but **C3 FAILED**: a variance
reduction that does not interact with the lever arm is hard to reconcile with the
dose-response already measured (Spearman +0.564 on log L, p = 3e-28).

As throughout this project, UNDERPOWERED is its own outcome and never counts
toward or against the mechanism.

## 5. Protocol fixed in advance

- Discard the first 10% of each run's checkpoints, per the paper's rule. The
  undiscarded variant is reported as a sensitivity row, not as the headline.
- Both naive and correlation-aware weighting are implemented; the
  **correlation-aware version is the headline**, the naive one is the comparison
  for C5.
- Checkpoint augmentation changes only the *fitting* data. The truth at the
  target remains the final-checkpoint 1B measurement, unchanged, so the decision
  being scored is identical to every other ranker's.
- No checkpoint at or beyond the target compute is ever used for fitting; this is
  asserted in code, not assumed.
- Frozen splits and hyperparameter provenance exactly as in Task B.
