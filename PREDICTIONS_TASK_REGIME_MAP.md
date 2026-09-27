# Pre-registration: the lever-arm mechanism, the regime map, the ensemble dose-response, DataDecide consequences

Written 2026-09-26, before any computation in this round. No paper prose changes.
Outputs: JSON, figures, generated macros.

## Task 1: the mechanism

### 1a. Definition of the noiseless column, and error counts

**Definition, as implemented in `scripts/run_lever_arm_arms.py:noiseless`.**
- **Truth.** The arm's truth table mu(r, s) is used as the data: one value per
  recipe and scale, no seeds and no noise.
- **Designs.** For every design (top fitted rung f, target rung t, with t >= 3
  and 2 <= f < t), the same 55 designs as the lever-arm test:
  - projection fits E + A C^-alpha to mu(r, rungs 0..f) per recipe and evaluates
    it at C_t;
  - single-scale ranks by mu(r, f).
- **Scoring.** Each rule's mis-selection rate is the fraction of the 300 recipe
  pairs whose predicted sign differs from the sign of the true gap
  mu(r, t) - mu(r', t). Pairs with a zero predicted or true gap, or a failed fit,
  are dropped.
- **Excess** is 100 x (projection rate - single-scale rate).
- **The column.** It reports the Spearman rho, and the OLS slope, of excess on log
  lever arm, log(C_t / C_f), across the 55 designs.

**Readout.** Per design: projection's and single-scale's error counts (pairs
wrong, out of pairs scored) in noiseless arm A.

**P1a.**
- In noiseless arm A, projection makes at most 1 wrong pair at every design
  (the truth is exactly a power law; any error is numerical).
- Single-scale's error count has a Spearman correlation of at least 0.8 with log
  lever arm.
- The negative excess is therefore crossover capture.
- **Refuted** if projection has 2 or more wrong pairs at any design.

### 1b. Regime map

- **Truth.** mu = mu_P + mean(a) m + lambda x [(a - mean(a)) m + E]. The common
  mode is held at its measured value; lambda scales the recipe-specific part.
- **Noise.** Each per-run seed sd is multiplied by s.
- **Grid.** s and lambda each take the values 0, 0.25, 0.5, 1 and 2. Cell (1, 1)
  is the real shrunk truth (arm B3 x 1), and cell (1, 0) is arm B1.
- **Worlds.**
  - Each noisy cell has 150 fixed-candidate worlds.
  - Common random numbers: world i uses the same standard-normal draws in every
    cell, so the noise is s times one draw and cells differ only in s and lambda.
  - The s = 0 row is computed exactly by the noiseless procedure of 1a.
- **Designs reported.**
  - primary: 300M -> 530M, L = 2.64;
  - short: the smallest-L design, 14M -> 16M, L = 1.24;
  - long: the largest-L design, 8M -> 1B, L = 16,132.
  - As a robustness readout, also the mean over the lowest- and highest-L
    terciles of the 55 designs.
- **Per cell and design.**
  - The true excess (mean over worlds) with a 95% interval from a bootstrap over
    worlds.
  - The sign, "wins" if the interval lies below 0 and "loses" if above 0. At
    s = 0 the sign is exact.
  - Per cell, also the slope of excess on log L.
- **Contour.**
  - Along each s-row, the lambda where the excess crosses zero is found by linear
    interpolation between adjacent grid values. Along each lambda-column, the s
    where it crosses zero is found the same way.
  - Each crossing has a bootstrap interval, resampling world indices jointly
    across cells.

**P1b-1 (as given).** Projection wins only in the low-noise, low-recipe-specific
corner: every cell whose excess is significantly below 0 has s <= 0.5 and
lambda <= 0.5.

**P1b-2 (as given).** The winning region shrinks as L grows: the number of
winning cells is non-increasing from short to primary to long. Winning is
judged on the point estimate (excess < 0), so noisy cells are not
under-counted.

**P1b-3 (mine; partly contradicts P1b-2).** In the noise-free row (s = 0) the
winning region does not shrink with L. The noiseless-A result (excess falling
with L) predicts that at s = 0 and small lambda, projection's advantage grows
with L. Shrinkage with L appears only for s >= 0.25.

### 1c. Placing real metrics on the map

Both coordinates are quantities a practitioner can measure, expressed relative
to the true between-recipe spread so that metrics in different units share one
map.

- **Between-recipe spread.** T = sqrt(mean over scales of tau^2(s)), where
  tau^2(s) is the across-recipe variance of the shrunk recipe means at scale s.
- **Noise coordinate.** x = sqrt(mean over recipes and scales of sigma^2) / T,
  where sigma is the moderated per-run seed sd estimated from the replicates.
- **Recipe-specific coordinate.** y = sd(a) x rms(m) / T.
  - a are the loadings and m the shape of the rank-one SVD of the residuals from
    per-recipe power-law fits to the shrunk curves. The product is loading
    dispersion times residual-shape magnitude.
  - It scales exactly with lambda, so cell (s, lambda) has coordinates
    (s x_C4, lambda y_C4).
  - Also reported: CV(a) x RMS(R), and the RMS of the whole recipe-specific
    residual including E.
- **Placement.** C4 sits at (1, 1) by construction. OLMES macro error, on its
  full 13-rung ladder (750M excluded, 3 seeds per cell), is placed at
  (x_OLMES / x_C4, y_OLMES / y_C4) in C4 multiples.
- **Distance from the boundary.** At each design:
  - the ratio of the metric's s to the crossing s* at its lambda, interpolated
    on the map;
  - the ratio of its lambda to lambda* at its s.
- **Off-grid rule.** If OLMES lies outside the grid on an axis, the grid is
  extended on that axis by doubling (4, 8) until the point is bracketed.
- **Transfer check.** 150 OLMES fixed-candidate worlds (its own shrunk truth and
  measured noise) give its true excess at the three designs. This is compared
  with the map's value at OLMES's coordinates. The map holds C4's other
  structure (curvature, spread of alpha) fixed, so this is a check on
  transferring it.

**P1c-1.** At the primary design, both C4 and OLMES are in the projection-loses
region.

**P1c-2.** OLMES is farther from the boundary than C4 on the noise axis:
x_OLMES / x_C4 > 1.

**P1c-3.** The sign in the OLMES worlds matches the map's sign at OLMES's
coordinates at all three designs.

## Task 2: ensemble dose-response

- **Intervals for the existing arms.** Every existing arm gets a 95% interval on
  its ensemble excess: mean +/- 1.96 SE over its 150 independent worlds.
- **Refined grid.**
  - Truth mu_P + lambda R, for lambda in 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.
  - 150 worlds per amplitude, with common random numbers across amplitudes.
    This is a fresh set, so 0.5, 1 and 2 are re-estimated; the earlier values
    are reported beside them.
- **Readouts at the primary design.**
  - The ensemble's true excess over single-scale, with an interval.
  - Per recipe: the mean pseudo-BMA weight of each family (power law,
    saturating, damped power law) and the modal top-weighted family, over
    worlds.
  - The same ensemble run once on the truth itself (noiseless weights and
    excess).
- **Verdict rule, fixed now.**
  - **Family switching** if both hold:
    - (i) some adjacent step has a jump in excess larger than half the range of
      the excess over the grid and larger than 4 combined SEs;
    - (ii) across that step, 3 or more recipes change their modal top family, or
      3 or more recipes change some family's mean weight by at least 0.3.
  - **Smooth peak** if no adjacent jump exceeds half the range and no recipe's
    mean weight changes by 0.3 or more between adjacent amplitudes.
  - **Indeterminate** otherwise, reported as such.

**P2.** Family switching: the excess at lambda = 1 exceeds both neighbours
(0.75 and 1.25) by more than 4 combined SEs, and condition (ii) holds there.

## Task 3: candidates needed against misspecification

- **Per-cell N.** At s = 1, across the lambda grid of 1b, N is computed from the
  world-averaged Hoeffding components (zeta1, zeta2) of the
  projection-versus-single-scale kernel at the primary design. N is the number
  of candidates for 2 points at power 0.8 with the calibrated random-candidate
  critical value, k = 2.99.
- **Per-world N.** Also computed in each world, with its distribution.
- **What produced each figure.**
  - 128 (`calibration.json`, `\candShrunkTwo`) came from zeta1 and zeta2 of the
    real C4 kernel. That is one realisation of the real data: the observed
    evidence at 530M and the 4M-300M fits.
  - 97 (arm B3 x 1) came from zeta averaged over 150 simulated worlds with
    shrunk truth.

**P3-1.** N rises monotonically with lambda at s = 1.

**P3-2.** 128 lies inside the central 95% of the per-world N distribution under
B3 x 1, so the difference is estimation noise in zeta from a single
realisation (together with the averaging of zeta before inverting).

**Rule for which figure the paper quotes.**
- **If P3-2 holds:** quote 128 as the estimate from the data the paper analyses,
  with no dependence on the simulation model. Give the simulated per-world 95%
  range as its uncertainty; 97 is the model's central value.
- **If 128 falls outside:** the simulation misstates the pair-level variance.
  Quote 128, and report the discrepancy.

## Task 4: DataDecide consequences

`compute_decision_accuracy` and `compute_2_class` are run unmodified. The only
stated modification is the tie order of the target list.
- **Stable.** A mergesort of `get_perf_size_simple`'s input order, replacing its
  default sort.
- **Reversed.** Stable order reversed within each tie block.
- **Randomized.** 1,000 random permutations within tie blocks.

**Efficiency.** For speed, the predicted lists are built once exactly as their
function builds them (`group.sort_values('stacked_pred')`), and
`compute_2_class` is called per row. This path is first checked to reproduce
their full function's output exactly at the default order.

### 4a. Tie-breaking sensitivity

**Readouts.**
- For every row: the minimum, maximum and range of decision accuracy across all
  tie orders.
- Whether a conclusion changes. "Conclusion" is operationalised on the 8
  variants x 11 tasks (primary metric) of the paper's comparison:
  - **C-A.** "No scaling-law variant exceeds single-scale ranking": on the macro
    average, no variant's decision accuracy exceeds single-scale at 750M.
    - The single-scale comparator ranks by each of the three 750M seeds (as in
      `run_published_comparisons.py`), scores each with `compute_2_class`
      against the same target list, and takes the mean over seeds.
    - Per task, the number of variants exceeding single-scale is also reported.
  - **C-B.** The best variant on the macro average, and the ordering of the 8
    variants by mean decision accuracy across the 11 tasks.

**P4a-1.** The macro average has no target ties, so its rows are invariant to tie
order.

**P4a-2.** A row's range is at most 100 x (tied target pairs / 300) points, with
a maximum of 4/3 points.

**P4a-3.** C-A and C-B do not change under any tie order.

### 4b. As described versus as implemented

The paper (arXiv:2504.11393v2 §2.3) defines the target as "mean downstream
performance over 3 random seeds". Their code uses seed `default` only.

- **Rerun.** Their function is rerun unmodified on an input whose 1B rows are the
  three-seed mean (`default`, `large aux 2`, `large aux 3`) at the final step
  common to the three seeds.
- **Readouts.** Per-row changes, and C-A and C-B under the three-seed target.

**P4b-1.** The median absolute per-row change is at least 1 point.

**P4b-2.** C-A holds under the three-seed target. C-B may change.

### 4c. The 9 rows of 5,280 that do not reproduce

**P4c.**
- All 9 are in (task, metric) groups whose target list has ties.
- For each, some tie order reproduces the released value, so the residual is
  platform-dependent quicksort ordering.
- The tie orders are enumerated exhaustively when there are at most 100,000,
  and randomized otherwise.
- **Refuted** if any of the 9 has no target ties, or no tie order reproduces it.
  Such rows are then explained individually.

## Task 5: attribution bases

No prediction; this is arithmetic on `lever_arm_arms.json`. The readouts are:
- the components as shares of the B3 x 1 model slope (summing to 100);
- the model slope as a share of the real slope;
- the components re-expressed as shares of the real slope (summing to the
  model's share), with the unexplained remainder.
