# Pre-registration: noise or genuine reordering on Marin; the leaderboard at 1e22

Written 2026-10-12, after the Task 1(a) search and before any computation. No
paper prose. Laptop safety: at most 2 workers under `nice -n 19`. Refutations
are reported first.

## Task 1(a): replicates (searched before this file)

No repeated runs between 130M and 1.2B were found under any recipe:
- **Speedrun board.** `runs.json`, 118 records: no prefix-and-size repeats.
- **Delphi's public tables.** `delphi-ladder` and `lucky-seeds` have seeds 42 and
  62746 only at 1e21 and 1e22, which are 3.4B and 9.7B parameters. The other
  tables have single seeds.
- **Marin repositories.** `marin`, `marin-experiments` and `marin-logbooks`: no
  seed-sweep experiments.
- **W&B.** Not accessible: no API key on this machine.

The borrowed values and the empirical bound are therefore used throughout.

## Task 1(b): empirical noise bound

- **Fits.** For each of the 21 submissions:
  - a two-parameter log-log fit over the 4 budgets (2 residual degrees of
    freedom each, 42 pooled);
  - a three-parameter fit (`asla.models.fit_power_law`; 1 residual degree of
    freedom each, 21 pooled).
- **Relative residuals.** (y - y_hat) / y_hat.
- **Pooled bound.** sigma_rel = sqrt(sum of squared relative residuals / pooled
  degrees of freedom).
- **Comparison.** Against the borrowed relative SDs (0.044%, 0.051%, 0.075%).
- **It is an upper bound on seed noise,** because it includes lack of fit.

## Task 1(c): both-rung determination

- **Determination.** A pair is determined at a rung if |gap| > 1.96 sqrt(sigma_a^2
  + sigma_b^2):
  - Marin: sigma = relative SD x BPB, under each borrowed value and each
    empirical bound;
  - DataDecide C4: the moderated SE of seed means (3 seeds).
- **A genuine reordering** is a pair determined at both the fit rung and the
  target with opposite signs.
- **Designs.** Marin 520m -> 1.2b; DataDecide 300M -> 1B and 530M -> 1B.
- **Readouts.** The count; the rate among pairs determined at both rungs; and how
  many of Marin's 40-41 target-determined reorderings are also determined at
  520m.

## Task 1(d): sign pattern across the ladder

- **Marin.** For every pair, the gap sign at 130m, 300m, 520m and 1.2b, and the
  number of sign changes (0 to 3).
- **Crossing-like pattern.** For pairs with exactly one change: whether the gap
  (in log BPB) moves monotonically across the four rungs.
- **Noise null.**
  - Each submission's smooth fit (two-parameter log-log fitted values) plus
    Gaussian relative noise at the empirical bound (two-parameter, pooled),
    2,000 simulations.
  - The null contains the fitted lines' own smooth crossings, so "noise" means
    the extra changes noise adds.
  - Statistics: the distribution of the change count over 210 pairs, the mean
    change count, and the share of pairs with 2 or more changes, each compared
    with the null's 95% band.
- **DataDecide at matched rungs.** 150M, 300M, 530M and 1B, on C4 seed means, with
  the same change-count distribution.

**Predictions (as given).**
- **PN1.** Under the empirical noise bound, most of the 40 reorderings are not
  determined at both rungs.
- **PN2.** Marin's sign-change distribution matches the noise null: the
  observed mean change count and share with 2 or more changes lie inside the
  null's 95% band.

## Task 2: the leaderboard at 1e22

- **Reference.** The true least-squares optimum (Python, converged in all 21).
- **(a) Shift decomposition.** shift_s = optimum_s - as-implemented_s.
  - Common component: the mean shift over submissions. Differential: shift_s
    minus the mean.
  - Each relative to the submission's nearest-neighbour gap in the as-implemented
    ranking (median and maximum).
  - **Statement.** The common component explains why a median 22x shift changes
    only 11 orders if the median differential / gap is at most 1 while the common
    / gap exceeds 1.
- **(b) The 11 changed pairs** and the top-3 change, listed.
- **(c) Initialisation range,** defined from the data:
  - c0 in {0, 0.25, 0.5, 0.7, 0.9, 0.99} x min y (the floor must lie in [0, min
    y]);
  - b0 in {0.5, 1, 2} x the log-log slope of (y - c0), clipped to [0.01, 2];
  - 18 settings at their cap (200), plus each at 100,000.
  - **Readout.** The share of settings whose top 1 and top 3 differ from the
    as-implemented default.
- **(d) Rank agreement at 1e22 across forms.** As implemented; converged
  three-parameter (the least-squares optimum); two-parameter log-log; shared
  exponent (common log-log slope). Pairwise Kendall tau, the top 3 under each,
  and whether the leaderboard's top 3 is stable across forms.
