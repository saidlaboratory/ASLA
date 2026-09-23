# Pre-registration: a calibrated inference procedure for comparing selection rules

Written 2026-09-23, before any computation in this task. It covers seed coupling,
moderated variance, the two estimands, end-to-end calibration, and restating
claims under the calibrated procedure.

## Task 0: seed coupling across scales

**Documentation.** DataDecide describes its three seeds as "3 random seeds for
initialization and data order". Below 1B they are labelled `default`, `small aux 2`
and `small aux 3`; at 1B they are `default`, `large aux 2` and `large aux 3`. Only
`default` carries the same label at every size. The paper does not say whether a
label fixes the same data order or initialization across sizes.

**Measurement.** For each recipe, scale and seed label, the deviation is the
seed's value minus its cell mean. For a pair of scales, the statistic is the
correlation of deviations across (recipe, label). The null permutes labels
within each recipe at one scale (6 permutations per recipe, 5000 random draws).
It is run for every pair of scales on the primary design (fit 4M to 300M,
target 530M) and for 1B, on C4 and on OLMES macro error. The labels are matched
by name, which means index 0 = `default`, and index 1 and 2 are the same label
below 1B.

**Prediction T0.** No coupling for the auxiliary labels (|rho| < 0.15). For
`default`, a small positive coupling (0 < rho < 0.3) is possible if the data
order is shared, and would be undetectable at this size.

**Decision rule.** If the median permutation p across scale pairs is below 0.05
with a positive median rho, the noise model gets a seed random effect: a label
effect correlated across scales at the measured rho. The bias this induces
toward single-scale ranking is quantified by simulation. Otherwise the
coupling is stated as not detected and the noise model is independent across
scales.

## Task 1: moderated variance

- **Method.** limma's `fitFDist` / `squeezeVar` (Smyth 2004, *Statistical
  Applications in Genetics and Molecular Biology* 3(1); verified against the
  limma source). Estimation is by moment matching on log s^2, per scale, across
  recipes, with d = 2 (three seeds). The posterior variance is
  (d s^2 + d0 s0^2)/(d + d0) with d + d0 degrees of freedom.
- **Pair test.** The gap's standard error uses each recipe's posterior variance.
  Its degrees of freedom are Welch-Satterthwaite, with each variance carrying
  d + d0. The expected-error weight becomes the Student-t CDF at those degrees
  of freedom.
- **Homogeneity.** Bartlett's and the Brown-Forsythe test, per scale, across
  recipes, are reported with d0 and s0^2.
- **Prediction M1.** At most scales on C4, homogeneity is not rejected at 0.05.
  Three seeds give these tests little power. Estimated d0 lies between 2 and 50
  at most scales.
- **Known-answer test on PolyPythias.** Units are tasks within each (size, step)
  with 10 seeds, deduplicated to the earliest evaluation pass. Three of the ten
  seeds are subsampled, 2000 times per (size, step). The target is the 10-seed
  standard deviation.
- **Prediction M2.** The moderated estimate has lower RMSE in log sigma than the
  raw 3-seed estimate at a majority of (size, step) cells. Its median ratio to
  the 10-seed sigma is within 5% of 1, where the raw ratio is near the c4 bias
  of 0.886 or below.
- **Prediction M3 (determined fractions).** Moderation adds degrees of freedom, and
  at a Bonferroni far tail that dominates. Determined fractions therefore
  **increase** on C4 and on OLMES macro error. Tiny raw variances are shrunk
  up, which removes some spuriously determined pairs, but less than the gain.

## Task 2: two estimands

- **Fixed-candidate.** Conditional on these 25 recipes, and asking whether this
  difference is real for these recipes. Uncertainty is seed noise only. It is
  computed by parametric resampling of every cell mean, at the fit scales and
  the target, from the moderated noise model around the observed means. Rankers
  and scoring are re-run on each draw; there is no seed bootstrap.
- **Random-candidate.** Generalizing to new recipes: the U-statistic test as now.
- **Prediction E1.** The fixed-candidate interval for the C4 difference excludes
  zero. The random-candidate interval does not, as already reported.

## Task 3: end-to-end calibration

**World construction.**

- **Primary design.** C4, fit 4M to 300M, target 530M.
- **Truth.** For fixed-candidate worlds, the 25 recipes' observed cell means,
  with their checkpoint trajectories. For random-candidate worlds, 25 recipes
  drawn with replacement, and each shifted by a smooth perturbation
  dE + dA C^-alpha. The (dE, log dA) are drawn from a normal fitted to the
  recipes' power-law parameters, scaled by a smoothed-bootstrap bandwidth of 0.5.
- **Noise.** Normal, with the moderated per-recipe, per-scale variance divided by
  3, and a seed random effect only if Task 0 finds coupling. Checkpoint noise is
  AR(1) within a run, at the checkpoint study's rho.
- **Estimands.** The true difference is the mis-selection rate against the
  noiseless target ordering. Its fixed-candidate value is the mean over noise
  draws; its random-candidate value is the mean over candidate draws.
- **Power.** Planted effects are single-scale ranking at lower rungs against the
  top rung, whose true differences span several sizes.

**Predictions C1-C4.**

- **C1.** Random-candidate U-statistic coverage is between 0.90 and 0.97 for
  projection, ensemble, checkpoint-augmented and the planted comparators.
- **C2.** For shared-exponent and EB shrinkage, whose fits couple candidates,
  random-candidate U-statistic coverage is below 0.90. A delete-one-candidate
  jackknife that refits the whole ranker restores coverage to at least 0.90.
- **C3.** Fixed-candidate parametric coverage is between 0.90 and 0.97 for every
  ranker.
- **C4.** Expected-error scoring is approximately unbiased for the true
  mis-selection difference: the mean error is under 0.5 points for every ranker.

## Task 4: restating claims

Every inferential claim in the paper is listed, restated under the calibrated
procedure, and corrected across the family. Holm is used, and
Benjamini-Yekutieli where dependence is plausible, which covers every claim on
the same DataDecide candidate pool.

**Prediction R1.** The survivors are the ensemble's inferiority on C4, the
lever-arm dependence, allocation at the smallest budget fraction, and Correct
Prob at 4M. The C4 projection difference does not survive under the
random-candidate estimand.

**Decision rule.** Every refutation is reported alongside the confirmations. If a
result invalidates the procedure itself, work stops and it is reported.
