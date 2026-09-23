# Pre-registration: final calibration round

Written 2026-09-23, before any computation in this round.

## Disclosure carried into this round

Two decisions in the previous round were **data-dependent deviations**, made
after the coverage results and not pre-registered as decisions:

- the switch from 1.96 to simulation-calibrated critical values, with the
  jackknife chosen over the U-statistic;
- the rule selecting the expected-error weight (moderated SE, raw Welch degrees
  of freedom).

Addenda 1 and 2 of `PREDICTIONS_TASK_CALIBRATED_INFERENCE.md` were written before
those rules were applied, but after the results that prompted them. The paper
will say so.

## Task 1: calibrate the lever-arm test

- **Test, made identical for real and simulated data.** For each of the 55
  designs (all fit/target pairs on the full ladder, 750M excluded), the kernel is
  the per-pair difference in expected-error charge, projection minus
  single-scale, with calibrated evidence at the design's target. The statistic
  is Spearman rho between log lever arm and the per-design mean kernel.
  Leave-one-recipe-out drops the recipe's pairs from every design's kernel;
  evidence is not refitted. The standard error is the jackknife of Fisher z.
  The real-data value is recomputed with this function and replaces the
  previous refit-everything version.
- **Worlds.** The full ladder, truth from Task 2's shrunk curves. There are 300
  random-candidate worlds.
- **True rho.** Spearman between log lever arm and each design's true excess:
  mis-selection against the noiseless target, averaged over random worlds.
- **Correlated-design null.** In every world a replicate comparator ranks by the
  top rung of an independent noise draw of the same truth. Its true excess is
  zero at every design, while designs sharing a rung share its noise. The
  test's rejection rate at 0.05 on it measures null behaviour under design
  overlap.
- **P1.** Held-out coverage of true rho by the LORO interval is between 0.90 and
  0.97. The null rejection rate is at most 0.08. True rho exceeds 0.5. The claim
  survives. If coverage is below 0.90 or the null rate above 0.08, the claim is
  restated with the simulation-calibrated critical value. If it then fails
  correction, it is withdrawn.

## Task 2: shrunk truth

- **Truth.** Per scale and checkpoint, each recipe's observed mean is shrunk
  toward the across-recipe mean by normal-normal empirical Bayes:
  mu = m + tau^2/(tau^2 + v_r)(y_r - m). Here v_r is the recipe's moderated
  variance over 3, and tau^2 = max(var_r(y) - mean v, 0). Simulated candidate
  spread then matches the estimated true spread rather than the noisy one.
- **Rerun.** The C4 benchmark (150 fixed worlds with 50 bootstrap draws, 300
  random) with this truth becomes the primary calibration. The observed-truth
  benchmark is kept for comparison.
- **Candidates required.** For a two-point difference at power 0.8 under the
  calibrated test: the adopted SE, scaled with n as the U-statistic variance,
  with the calibrated critical value, for each truth.
- **P2.** Under shrunk truth, calibrated power at the planted comparators changes
  by less than 0.10 each. The candidates required for two points under the
  calibrated test exceed the known-SE figure of 82, and lie between 90 and 160.

## Task 3: estimator choice by coverage

For every comparator, held-out (cross-fit) coverage of the jackknife and of the
U-statistic, each at its own calibrated critical value. **Rule:** the jackknife
is kept only if its held-out coverage is at least the U-statistic's for every
comparator. Otherwise the U-statistic is adopted and affected claims are
restated.

- **P3.** The jackknife fails the rule, at least at one comparator, and the
  U-statistic is adopted. No claim's corrected verdict changes.

## Task 4: per-metric calibration

- **OLMES macro error rankers.** A separate benchmark (shrunk truth; 120 fixed
  worlds with 40 bootstrap draws; 300 random).
- **Correct Prob against accuracy.** Joint Gaussian noise for the two metrics on
  the same run, with the within-run correlation of seed deviations estimated
  per scale. Truth is shrunk per metric. The target is 1B accuracy, three-seed
  mean. The kernel is observed-target agreement difference, as published.
- **DataDecide scaling-law variants.** The released predictions are held fixed
  (they cannot be regenerated). The target and single-scale 750M rankings get
  noise. In random-candidate worlds, recipes are resampled with a coherent
  per-recipe shift applied to every quantity of that recipe, including its
  released prediction.
- Each family gets its own calibrated critical value and held-out coverage, and
  those claims are restated under it.
- **P4.** OLMES-error critical values exceed C4's. No claim outside the lever
  arm (primary family) and the ensemble (fixed family) survives correction.

Refutations are reported first. After this round: rebuild, verify, compile,
freeze.
