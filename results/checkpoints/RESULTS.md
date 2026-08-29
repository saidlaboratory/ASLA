# Checkpoint-augmented fitting: results

Generated from `results/checkpoints/checkpoint_study.json` (n_boot=80, metric `c4_en_bits_per_token`, seed 1729). Predictions were fixed in `PREDICTIONS_TASK_CHECKPOINTS.md` before implementation.

## 1. Technique critique (primary result)

Choshen, Zhang & Andreas (arXiv:2410.11840, ICML 2025) recommend fitting scaling laws to intermediate checkpoints and specify discarding roughly the first 10%. They do not address serial correlation between checkpoints, the resulting information deflation, ladder imbalance, or the over-confidence of intervals computed as if checkpoints were independent.

Four properties of the technique, measured here, that change how it must be applied:

### 1.1 Checkpoints carry far less information than their count suggests

| metric | ladder points | checkpoints | mean rho | max rho | effective points | deflation |
|---|---|---|---|---|---|---|
| `c4_en_bits_per_token` | 11 | 63 | 0.427 | 0.592 | 25.4 | **2.5x** |
| `olmes_macro_error` | 11 | 182 | 0.797 | 0.871 | 20.8 | **8.8x** |

Serial correlation between checkpoints of the same run reaches rho = 0.87. Treating 182 checkpoints as 182 independent observations overstates their information by up to **8.8x**.

### 1.2 The obvious correction is a mathematical no-op

Down-weighting a correlated run by one constant factor cancels in the least-squares normal equations:

| weighting | max relative change in fitted parameters |
|---|---|
| uniform, sigma = 2.0 | 0.00e+00 |
| uniform, sigma = 10.0 | 5.66e-08 |
| uniform, sigma = 50.0 | 1.34e-03 |
| **varying across groups** (downweight bottom half) | **9.99e-01** |
| **varying across groups** (downweight top half) | **9.77e-01** |

That is, a per-run uniform weight changes the fit only to optimizer tolerance; a weight that varies across groups (here, across scales) changes it by orders of magnitude more. A correlation correction must therefore act *between* groups (here, between scales), not within a run.

### 1.3 Real ladders are badly unbalanced, so naive fitting silently reweights them

| metric | fewest checkpoints at a scale | most | imbalance |
|---|---|---|---|
| `c4_en_bits_per_token` | 1 | 16 | **16x** |
| `olmes_macro_error` | 4 | 33 | **8x** |

Scales contribute checkpoints in proportion to how often they were evaluated, not to how much they constrain the curve, so an unweighted fit tilts toward whichever scales happen to be densely logged.

### 1.4 Naive intervals are over-confident by roughly the deflation factor

Seed-bootstrap CI width, naive: **0.0067**; correlation-aware: **0.0201** (3.0x wider). The correction changes the fit materially: maximum projected-value shift **3.27e-02**, and the induced ordering differs (True).

*(A first version of this check compared summary mis-selection rates, which coincided at 4.00% for both weightings, and wrongly read as a no-op. Two different fits can share a summary rate; the comparison must look at what the correction actually changes.)*

## 2. Decision result (secondary)

The decision-level effect is modest and, on the primary design, not statistically separated from the baseline it improves on. It is reported here with that caveat attached.

| ranker | pairwise mis-selection | net flips removed vs plain |
|---|---|---|
| `single_scale` | 1.33% [0.66%, 2.00%] | +12 |
| `shared_exponent` | 2.00% [1.33%, 3.67%] | +10 |
| `checkpoint_plus_pooling` | 3.33% [3.00%, 4.67%] | +5 |
| `checkpoint_ar1` | 4.00% [3.33%, 5.33%] | +4 |
| `checkpoint_naive` | 4.00% [3.33%, 4.00%] | +4 |
| `plain_projection` | 5.33% [4.00%, 8.67%] | +0 |
| `ensemble` | 18.67% [12.66%, 21.67%] | -40 |

### Pre-registered verdicts

| prediction | verdict | qualification |
|---|---|---|
| C1 beats plain projection | **CONFIRMED** | point estimate improved 25.00% relative (1.33 points), exactly at the 25% threshold, but the CI is **not separated from plain** (separated = False): the improvement is not statistically distinguishable from baseline |
| C2 does not beat single-scale | **CONFIRMED** | single-scale 1.33% [0.66%, 2.00%] versus checkpoint-augmented 4.00% [3.33%, 5.33%] |
| C3 lever-arm interaction | **UNDERPOWERED** | the short arm removed -2 flips (net negative), so the ratio is undefined and 0 bootstrap draws were valid |
| C4 composition with pooling | **UNDERPOWERED** | G_both = 37.50% against pooling alone 62.50%; CIs overlap the best single lever |
| C5 correction matters | **CONFIRMED** | see section 1.4 |

**Natural reading of C3 and C4, explicitly not established at these intervals:** checkpoints add points along the existing ladder without shortening the lever arm, so they reduce parameter uncertainty but not extrapolation distance, and buy less than pooling does. The confidence intervals cannot distinguish this from the alternatives, so it is stated as an interpretation awaiting more seeds, not a result.

Mechanism damaged by these results: **False**.

