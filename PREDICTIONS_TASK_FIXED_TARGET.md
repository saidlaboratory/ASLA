# Pre-registration: the lever-arm effect at a fixed target

Committed before any code for this task and before any outcome is computed. This is
the last new analysis before the preprint, in the same family as
PREDICTIONS_TASK_CONFOUND.md, and on the same data: DataDecide C4 bits per token,
13 sizes with 750M excluded.

## Why

In the confound test, with the top rung fixed, lever arm changes only through the
target, so the target-varied slope (+7.7 points per decade) cannot be separated from
target size, and target size carries reference determinacy (the share of target
pairs whose three-seed order is determined runs from 4% at 20M to 86% at 1B). At a
fixed target, determinacy is constant by construction.

## What is already known when this is written

The determined share at each target (from the confound run): 10M 29%, 14M 45%,
16M 27%, 20M 4%, 60M 20%, 90M 36%, 150M 60%, 300M 54%, 530M 82%, 1B 86%. The
confound test's joint model, rewritten in log target and log top-rung compute,
implies a slope of +11.8 points per decade of lever arm at a fixed target (its
top-rung coefficient, -11.8, with the sign reversed, since at a fixed target a
decade more lever arm is a decade less top-rung compute). No fixed-target slope has
been computed.

## Task 1a: fixed-target slices

- Admitted targets: determined share at least 0.5, which admits 150M, 300M, 530M and
  1B. The threshold is set here; no other value is reported as primary.
- Slice: a target t and a rung count k >= 3; the fit window of k consecutive rungs
  slides from the bottom of the ladder to the rung just below the target. A slice is
  kept if it has at least 3 windows. This gives 26 slices and 152 designs.
- Kernel and scoring: unchanged from the confound test (expected-error charge of
  projection minus single-scale ranking at the window's top rung, against the
  target's three-seed evidence, in points).
- Statistics:
  - per target: the pooled within-slice OLS slope of excess on log10 lever arm over
    that target's slices (slice fixed effects);
  - pooled (primary): the same over all 26 slices; slice fixed effects nest target
    fixed effects, so target size and determinacy are absorbed.
- Note, stated now: at a fixed target, lever arm and top-rung compute are the same
  variable (a decade more lever arm is a decade less top-rung compute). The
  fixed-target slope is therefore "excess against how far below the target the
  fitted range ends", not lever arm separated from top-rung size.

## Task 1b: determinacy check on the target-varied slice

Slice (i) of the confound test (window fixed, target varied; 36 slices, 192 designs),
rescored with each design's excess the mean kernel over only the pairs whose order is
determined at that design's target (the `determined` flag). Statistic: the pooled
within-slice slope on log10 lever arm.

## Inference

The confound test's procedure: leave-one-recipe-out jackknife standard errors;
critical values calibrated in 200 random-candidate worlds (seeds new to this task)
as the 95% quantile of |estimate - true value| / SE, where the true value is the
statistic of the world-mean true excess. True excess for the determined-only
statistic uses the same pairs as its estimate (pairs determined in that world's
observed target evidence). The reported interval uses max(calibrated value, 1.96):
when the calibrated value is below the normal value, the normal interval is quoted.

## Task 1c: the simulation reference

The confound test's semi-synthetic factor isolation already scores against the
noiseless truth at the target (`synthetic_excess` computes the truth from the
interpolated curve without noise), not against a noisy three-seed reference. No
rerun is needed; this is reported as such.

## Task 3 (power for the allocation comparisons)

The allocation comparisons fit projection on each optimised design's supported
budgets and target 1B (rescoring.json). In 300 random-candidate worlds built as the
calibration benchmark, each allocation comparator is run against single-scale
ranking at the ladder's top; power is the share of worlds where |estimate| / U
standard error exceeds the C4 calibrated random-candidate critical value (2.99), the
procedure the restated claims use. The benchmark true effect is the world-mean true
difference. This is a power calculation, not a test, and has no prediction.

## Predictions

- F1: the pooled fixed-target slope is positive and its interval excludes 0.
- F2: the pooled fixed-target slope is below +11.8.
- F3: each of the four per-target slopes is positive.
- F4: the determined-only target-varied slope is positive and its interval excludes 0.
- F5: the determined-only target-varied slope is below the all-pairs one (+7.7).

## What the paper says (decided now)

- **Fixed-target slope positive, interval excluding 0:** the fixed-target form becomes
  the primary claim: on C4 bits per token, at a fixed target, projection's excess grows
  as the fitted range ends farther below it. The target-varied slope becomes secondary,
  labelled as possibly including a determinacy component. Section and title stay.
- **Fixed-target slope indistinguishable from 0 (interval includes 0):** the claim is
  restated as a target-size or determinacy effect, Section 4 is renamed to match, and a
  new title is proposed.
- **Fixed-target slope negative, interval excluding 0:** treated as the second case,
  with the sign reported.

## Outputs

results/target_scoring/fixed_target.json; scripts/run_fixed_target.py; additions to
asla/analysis/confound.py with tests; results/target_scoring/allocation_power.json and
scripts/run_allocation_power.py.
