# Pre-registration: the corrected recipe model, the excess curve, and an untuned replication spec

Committed before the validation runs, the curve analysis, and the design runs below. Analysis only: no paper
edits. Data: DataDecide C4 bits per token, 750M excluded. Simulator: `asla/analysis/sim_design.py`, run
through `scripts/sim_runner.py` with one niced worker, per-job timeouts and checkpoints.

## Task 1: the corrected recipe model, and its validation

### The model (candidates = "corrected")

- **25 recipes or fewer:** draw the recipes without replacement from DataDecide's 25, unperturbed.
- **More than 25:** draw independent recipes from a Gaussian fitted to the 25. Each recipe is a 16-vector of
  E, log A, alpha and its recipe-specific residual at the 13 sizes. The Gaussian uses the sample mean and
  covariance (ddof 1).
- **Shared misfit:** the common-mode residual is identical across recipes, so every recipe gets it.
- **Noise:** the median relative seed SD at the nearest DataDecide size.
- **Seeds:** 3.

### Validation, decided now

**Common settings for V1 and V2:**

- the full 13-size ladder;
- targets 150M, 300M, 530M and 1B, all admitted, as in the real subset analysis;
- multiplier 1;
- 200 worlds per check.

**V1.** With 25 recipes, the mean true fixed-target slope is within 2 combined Monte Carlo SEs of +11.96. The
combined SE takes the simulation's SE and the reference's 0.05 in quadrature.

**V2.** On random subsets of 10 and of 15 recipes, the simulated distribution of the estimated slope must match
the real-data subset distribution from `results/target_scoring/recipe_dependence.json`:

| Subset size | Real mean | Real SD |
|---|---|---|
| 10 | +11.61 | 1.71 |
| 15 | +11.72 | 0.97 |

Two conditions define a match:

- the simulated mean lies within ±1.0 point per decade of the real mean;
- the simulated SD lies between 0.67 and 1.5 times the real SD.

The simulated SD also includes noise variation, which the real subset SD does not.

**If V1 or V2 fails,** report it and stop.

### Reported alongside, not gates

- **Plateau check.** The mean true slope at multipliers 0.5, 1, 2, 3, 4 and 6, on 60 worlds each, for:
  - the full ladder with 25 recipes;
  - 8 DataDecide sizes up to 300M with 15 recipes;
  - 8 DataDecide sizes up to 1B with 15 recipes.

  Report the maximum and the multiplier at which it occurs.
- **Generative sanity check.** 25 recipes drawn from the Gaussian (not from the 25 themselves), on the full
  ladder, 100 worlds: report the mean true slope next to +11.96.

## Task 2: the excess-versus-lever-arm curve on the real data

### Designs

At each fixed target with at least 50% of pairs determined (150M, 300M, 530M and 1B):

- every fit window of at least 3 consecutive rungs below the target, of any rung count;
- the design excess is the mean expected-error kernel, in points;
- the lever arm L is log10 of target compute over top-rung compute.

### Binned curve

- **Bins in L (decades):** [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0), [2.0, 3.0), [3.0, 4.5).
- **Per target:** the mean excess of the designs in each bin.

### Saturating form

- **Form:** excess(L) = s·tau·(1 − exp(−L / tau)).
  - s is the initial slope, in points per decade as L goes to 0.
  - tau is the saturation scale, in decades.
- **Fit:** least squares over each target's designs.
  - Bounds: s in [−100, 100], tau in [0.1, 20].
  - A common fit uses all four targets' designs together.
- **tau at its upper bound** is reported as "no saturation detectable over the lever arms covered".

### Inference

- **Standard errors:** every statistic's SE is the leave-one-recipe-out jackknife.
- **Calibration:** the critical value per statistic is the 95% quantile of |estimate − true| / SE. It comes
  from 200 worlds of the corrected model with 25 recipes on the real ladder and targets.
- **Intervals:** estimate ± max(calibrated value, 1.96) × SE.

### Coincidence test

- **Primary (binned).** Take every bin covered by two or more targets. In each, contrast each target's bin mean
  with the mean across those targets.
  - The test is a Wald test with the jackknife covariance, and degrees of freedom equal to the rank of the
    contrast covariance.
  - The covariance is inflated by max(1, (median calibrated critical value of the bin means / 1.96)^2).
- **Secondary.** A Wald test that the four initial slopes s are equal.
- **The curves coincide if both hold:**
  - the primary test's p > 0.05;
  - no target's bin mean departs from the bin's across-target mean by more than 3 points.

### What each outcome means

- **Coincide.** The curve is the claim. A fixed-target slope is a design-dependent summary of it, and a
  replication tests the curve.
- **Differ.** Report how: which bins and targets carry the departures, whether the initial slopes differ (the
  secondary test), and whether tau differs. The claim then has to name the target, or model how the curve
  shifts with target size.

### Predictions

- **C1:** the initial slopes fall with target size. The fixed-target slopes per target were 14.7, 13.5, 11.7
  and 10.3 from 150M to 1B.
- **C2:** the binned coincidence test rejects (p < 0.05).
- **C3:** tau is below its upper bound for at least two of the four targets (saturation is detectable).

## Task 3: an untuned replication spec

All at multiplier 1, with the corrected model and 3 seeds. Ladders use DataDecide's own sizes:

- 6, 8 or 10 sizes up to a top target of 300M, 530M or 1B;
- sizes chosen as in `scripts/run_direct_power.py` `node_ladder`;
- targets are rungs at or above 150M compute with at least 5 rungs below;
- a target counts in a world when at least half its pairs are determined (the frozen admission rule).

Recipes: n in {5, 10, 15, 20, 25, 30, 40}, with 150 worlds per design. Within a geometry, recipe counts are run
in increasing order. The search stops once a valid design reaches 0.8 power in both framings.

### Per design

- **Predicted slope:** the mean true fixed-target slope over worlds. Report its Monte Carlo 95% interval, and
  the 2.5% to 97.5% spread across recipe draws.
- **Curve-implied slope:** the Task 2 curve evaluated at the design's own windows, summarised as the pooled
  within-slice slope. Use the common curve if the curves coincide, otherwise each target's own curve.
- **Random-candidate power:** the share of worlds whose frozen test rejects (estimate − 1.96·SE > 0).
- **Fixed-candidate power:** the same share among worlds whose own true slope lies within 20% of the predicted
  slope.

**Validity, as last round.** A design counts only if:

- at least 90% of its worlds run a test;
- fixed-candidate power rests on at least 40 worlds.

**At 25 recipes,** drawing without replacement always gives DataDecide's 25. Random-candidate power there has
no recipe variation, which is stated.

### Outputs

- the cheapest valid design reaching 0.8 power in each framing, with:
  - power and its Monte Carlo SE;
  - the predicted slope;
  - training FLOPs;
  - H100-hours and RTX 3090-hours, under the same utilization assumptions as last round (989 TFLOPS at 40%,
    and 71 TFLOPS at 30%).
- **If no design reaches 0.8 within 40 recipes,** report the best one and stop.

**Prediction D1:** a valid design reaches 0.8 random-candidate power within 25 recipes, with a 300M top target.

### Files

- `results/replication/curve_validation.json`
- `results/target_scoring/excess_curve.json`
- `results/replication/untuned_spec.json`
- `scripts/run_curve_round.py`
