# Pre-registration: separating lever arm from top-rung size and rung count

Written and committed before any code for this task exists and before any outcome
is computed. This task deliberately breaks the DataDecide analysis freeze and is the
only new analysis in this round.

## Why

The paper's one claim that survives family-wide correction is that, on DataDecide C4
bits per token, projection's excess mis-selection over single-scale ranking rises
with the lever arm (Spearman 0.87 over 55 designs). In those 55 designs every fit
ladder starts at the smallest model (4M), so the number of fitted rungs is fixed by
the top rung. Long lever arms come only from small top rungs. Lever arm, top-rung
size and rung count are therefore confounded. This task separates them.

## Definitions (fixed)

- Scales: the 13 DataDecide sizes from 4M to 1B, with 750M excluded as in every
  other analysis. Indices 0 to 12 in compute order.
- Design (s, f, t): fit window of rungs s..f (contiguous, at least 3 rungs, so
  s <= f - 2), target t > f. There are 220 designs; the 55 used in the paper are
  those with s = 0.
- Lever arm L = log10(C_t / C_f). Top-rung compute log10 C_f. Rung count k = f - s + 1.
- Metric: C4 bits per token only (c4_en_bits_per_token in data/datadecide_runs.parquet).
- Per design kernel: for every recipe pair, the expected-error charge of projection
  (offset power law fitted by asla.models.fit_power_law to the seed-mean values on
  the window, evaluated at C_t) minus that of single-scale ranking at rung f, scored
  against the target's own three-seed evidence (asla.analysis.moderation.
  evidence_from_cells, the moderated model) with asla.analysis.target_scoring.
  expected_charges, in points. The design's excess is the mean kernel over pairs.
  This is the kernel of asla.analysis.lever_arm.designs, generalised to windows.
- Determined share at each target: the fraction of the 300 recipe pairs whose
  `determined` flag is true in the target's evidence, the same flag the paper uses.

## Statistics (real data)

1. Slice (i), lever arm only: within each (s, f) with at least 3 targets
   (36 slices, 192 designs), top rung and rung count are exactly fixed. Statistic:
   the pooled within-slice OLS slope of excess on L (slice fixed effects).
2. Slice (ii), top-rung size only: lever arm cannot be held exactly fixed because the
   ladder is irregular. Designs with the same k and L in the same 0.2-decade bin
   (round(L / 0.2)), with at least 3 distinct top rungs spanning at least one decade,
   form a slice (29 slices, 100 designs; L varies within a slice by at most 0.163
   decades). Statistic: the pooled within-slice OLS slope of excess on log10 C_f,
   with L as a covariate to absorb the residual lever-arm variation.
3. Slice (iii), rung count only: within each (f, t) with at least 3 window starts
   (36 slices, 192 designs), top rung, target and lever arm are exactly fixed.
   Statistic: the pooled within-slice OLS slope of excess on k.
4. Joint model over all 220 designs: OLS of excess on L, log10 C_f and k with an
   intercept. Coefficients beta_L (points per decade of lever arm), beta_top
   (points per decade of top-rung compute), beta_k (points per rung).

Inference for every statistic: recipes are the unit. The standard error is the
leave-one-recipe-out jackknife (drop a recipe's pairs from every design at once,
recompute every design's excess, refit), the candidate-level procedure the paper
uses for the lever-arm test. (The request's "leave-one-ladder-out" is read as this
leave-one-recipe-out jackknife; the ladder is not the unit of resampling anywhere in
the paper's lever-arm test.) Critical values are calibrated: in 200 random-candidate
semi-synthetic worlds (asla.analysis.calibration.random_truth with bandwidth 0.5 on
the shrunk C4 truth, simulate for noise, the same construction as the lever-arm
calibration), each statistic and its jackknife standard error are computed exactly
as on the real data, together with the statistic computed from the worlds' true
excess (mis-selection against the noiseless target). The true value of each
statistic is the statistic of the true excess averaged over worlds. The calibrated
critical value is the 95% quantile over worlds of |estimate - true value| / SE. The
95% interval is estimate +/- critical value x SE. The interval at 1.96 is reported
alongside. Worlds are cached one per line and resumable; the run uses at most two
niced workers.

## Semi-synthetic factor isolation (part b)

Fixed-candidate worlds with continuous ladders, so each factor can vary with the
other two exactly fixed.

- Truth per recipe: its power-law fit to the shrunk C4 curve (the arm A truth of
  run_lever_arm_arms.py) plus the recipe-specific misspecification of arm B2
  (residual minus its common mode), linearly interpolated in log10 compute between
  the DataDecide scales and held at the end value outside them.
- Noise: each recipe's moderated C4 noise sd, interpolated the same way. Three seeds
  per rung and at the target; seed means for fitting and the top rung.
- Rungs equally spaced at 0.35 decades. Base design: k = 6, top rung 10^19 FLOPs,
  L = 1 decade.
- Lever arm only: L in {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, top and k at base.
- Top rung only: log10 C_f in {17.75, 18.25, 18.75, 19.25, 19.75}, L and k at base.
- Rung count only: k in {3, 4, 5, 6, 7, 8}, top and L at base.
- 300 worlds. Per design: true excess (projection minus single-scale mis-selection
  against the noiseless target, in points), averaged over worlds. Each factor's effect
  is the OLS slope of the world-mean true excess on the factor, with a Monte Carlo
  standard error from the spread of per-world slopes. The same designs are also run
  once without noise.

## Predictions

Real data (calibrated 95% intervals):

- P1: beta_L > 0 and its interval excludes 0.
- P2: beta_top < 0 (point estimate).
- P3: beta_k's interval includes 0.
- P4: slice (i) slope on L > 0 and its interval excludes 0.
- P5: slice (ii) slope on log10 C_f < 0 (point estimate).
- P6: slice (iii) slope on k: interval includes 0.
- P7: the determined share at the smallest target (10M) is below the share at 1B.

Semi-synthetic (Monte Carlo 95% intervals):

- P8: the lever-arm effect is positive and excludes 0.
- P9: the top-rung effect is negative (a smaller top rung gives a larger excess).
- P10: the rung-count effect, per added rung, is smaller in magnitude than the
  lever-arm effect per half decade.

## What the paper says in each case (decided now)

The verdict uses the real-data joint model and slices together.

- **Lever arm survives** if beta_L's calibrated interval excludes 0 and is positive,
  and the slice (i) slope is positive. The paper keeps the claim, scoped to C4,
  reports the joint coefficients and slice slopes as the confound test, and reports
  beta_top and beta_k whatever their verdict. If beta_top's interval also excludes 0,
  the paper says both contribute.
- **Top-rung size explains it** if beta_L's interval includes 0 while beta_top's
  excludes 0, and the slice (ii) slope has the same sign as beta_top. The paper
  restates the claim as a top-rung-size effect and renames Section 4 to match.
- **They cannot be separated on real data** in every other case (for example both
  intervals include 0, or a slice contradicts the joint model in sign). The paper
  restates the claim as an association on the 55 nested designs that the confound
  test could not attribute, and renames Section 4 to match.

The semi-synthetic results are reported as a description of the simulator; they do
not change the verdict, which rests on the real data.

The joint coefficients test a confound of an existing claim; they are not added to
the paper's 38-claim correction family.

## Outputs

- results/target_scoring/confound.json (real data, calibration, semi-synthetic,
  predictions with verdicts, the outcome).
- scripts/run_confound.py, asla/analysis/confound.py, tests/test_confound.py.
