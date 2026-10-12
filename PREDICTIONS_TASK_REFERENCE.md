# Pre-registration: class-dependent crossover, the leaderboard's convergence defect, family-level inference

Written 2026-10-12, before any computation for these tasks. No paper prose.
Laptop safety: at most 2 workers under `nice -n 19`. Refutations are reported
first.

## Task 1: class comparison at matched conditions

**Determined-pair reorder rate.** Among pairs determined at the target, the
share whose order at the top fitted rung disagrees with the target order (ties
dropped).

**Suites, designs and noise.**
- **Marin.** 21 submissions, recipe axis, Paloma C4-EN BPB, 520m -> 1.2b
  (parameter ratio about 2.3). Single runs.
  - Noise: the borrowed relative seed SD (DataDecide C4 at 1B, Delphi 1e21,
    Delphi 1e22) times each 1.2b BPB.
  - A pair is determined if |gap| > 1.96 sqrt(sigma_a^2 + sigma_b^2).
- **DataDecide C4 bits per token.** 25 recipes, 300M -> 1B (ratio 3.3) and
  530M -> 1B (ratio 1.9).
  - The order at the fit rung is the seed mean.
  - A pair is determined at 1B if |gap of seed means| > 1.96 times the
    moderated SE of the difference (3 seeds), the same 1.96 rule as Marin.
  - The raw Welch test at 0.05 is reported alongside.
- **Signal and Noise C4 bits per byte.** The same designs, single runs, with noise
  borrowed as before (C4 moderated SD x the bits-per-byte / bits-per-token
  ratio) and the Marin determination rule.

**Noise sensitivity.** Marin's rate with the borrowed SD multiplied by m in
{1, 1.5, 2, 3, 4, 5, 7, 10}, for each borrowed value. The readout is the
smallest m at which Marin's rate falls to DataDecide C4's (530M -> 1B, and
300M -> 1B).

**Gap control.** The rate in bins of target gap over its SE, [1.96, 4), [4, 8),
[8, 16) and [16, inf), with counts, for every suite and design.

**Mechanism.**
- **Level.** The log BPB at the fit rung (520m / 530M).
- **Exponent (primary).** The two-parameter log-log slope over each suite's top
  four rungs:
  - Marin: 130m, 300m, 520m, 1.2b;
  - DataDecide and S&N: 150M, 300M, 530M, 1B.
- **Exponent (secondary).** The three-parameter alpha: Marin's four-point fit
  (`asla.models.fit_power_law`), and the existing full-ladder fits for
  DataDecide.
- **Crossover index.** SD(slope) x ln(C_target / C_fit) / SD(level), the spread
  of slope-driven drift over the design relative to the spread of levels.
  Raw SD(slope) / SD(level) is reported alongside.
- **Reordered against non-reordered pairs (Marin, determined pairs).** |Delta
  slope|, by medians and a one-sided Mann-Whitney test.

**Structure.** Marin's reorder rate within families and across families, with
the mechanical family rule from last round.

**Predictions (as given).**
- **PR1.** Marin's determined-pair reorder rate is at least 5x DataDecide C4's,
  at matched designs, and survives noise multiples up to 3x.
- **PR2.** Marin's crossover index (exponent dispersion relative to level
  dispersion) is larger.
- **PR3.** On Marin, reordered pairs have larger |Delta slope|.

**Caveats recorded with the result.** Curated submissions (not a designed
sample); possibly architecture-dependent noise; single runs at both rungs.

## Task 2: the leaderboard's convergence defect

- **Reimplementation.** The leaderboard's fit is re-implemented in JS with the
  same function, bounds, damping, gradient difference, tolerance and
  initialisation, plus configurable iteration cap and initial values. At the
  defaults it must reproduce their unmodified `regression.ts` output exactly,
  or the comparison stops.
- **Forecasts at 1e22,** fit on all four budgets:
  - iteration caps 50, 100, 200 (as implemented), 500, 1,000, and converged
    (100,000 iterations; also cross-checked against a Python least-squares fit);
  - initialisations: their default, and the four combinations of
    c0 in {0.5 min y, 0.9 min y} and b0 in {0.05, 0.3}.
- **As implemented against converged:**
  - Kendall tau over the 21;
  - pairwise order changes among the 210 pairs;
  - whether the top 1 and top 3 change;
  - each forecast's shift divided by the gap to its nearest neighbour in the
    as-implemented ranking.
- **Held-out.** Form (i)'s wrong pairs at 1.2b (fit on three points) as a
  function of iteration cap.
  - If the converged count is close to the exact fit's 95, the 57 comes from
    stopping early. "Close" means within 10 pairs.

## Task 3: family-level inference

- **Units.** The 13 families. For each family pair, the kernel is the mean over
  member pairs of 100 x (projection wrong - single-scale wrong); within-family
  pairs are excluded.
- **Test.** The candidate-level U-statistic over the 13 units, per fit form, with
  a delete-one-family jackknife SE alongside.
- **Critical values.** 1.96 and 2.99 (the most conservative earlier calibrated
  value). There is no Marin calibration.
- **Power.** 13 units give low power, and that is stated.

## Task 4: small items

- **Lever-arm designs.** Each Marin design is mapped to its lever arm and
  excess. The trend test is recorded as uninformative, since three of four arms
  are nearly identical (4.9, 5.1, 5.1). It is not a refutation.
- **Flat-profile metrics analysed with C4's critical values.**
  - Search result before this file: the only such case is last round's Marin
    U-statistic tests (k = 2.99 from C4).
  - Marin has no replicates, so its profile is not estimable and no Marin worlds
    exist. It is flagged, not recalibrated.
  - Every other reported interval was calibrated in its own metric's worlds:
    OLMES; the published re-tests on OLMES macro worlds; the lever-arm test per
    metric; C4 on C4.
