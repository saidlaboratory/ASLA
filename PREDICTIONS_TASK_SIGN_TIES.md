# Pre-registration: sign-change ties; form uncertainty vs deployed fragility; LM non-convergence

Written 2026-10-12, before any computation for these tasks. No paper prose. Laptop
safety: at most 2 workers under `nice -n 19`. Refutations are reported first.

**What was read before writing (code only, no results).**
- The noiseless row in `run_marin_noise.sign_pattern` evaluates submission i's
  two-parameter fit at **its own** FLOPs at each rung: `fitted[i] = exp(intercept +
  slope * log(x[i]))`. Submissions sit at different model FLOPs under the same size
  label, so the rung-r "gap" compares two lines at two different x values. It is
  not linear in a single log-compute axis and can change sign more than once.
- Sign changes are counted between consecutive rungs; an exact zero is skipped.
- Their `computePowerLawWithOffset` fits raw x. At the 1e22 leaderboard x is
  `model_flops`, about 1e18 to 1e20, so ln x spans only about 41 to 46.
- Their solver: ml-levenberg-marquardt, damping 1.5, absolute gradientDifference
  1e-6, 200 iterations, a in [1e-10, 1e30].

## Task 1: sign-change ties

**Diagnosis tested two ways.**
- **Your hypothesis (ties).** The two-change pairs in the noiseless row come from
  near-identical variants whose fitted gaps are at numerical zero. They would
  vanish at a numerical tolerance of 1e-9 in |log gap|.
- **The code hypothesis (misaligned x).** They come from evaluating each fit at its
  own FLOPs. They would survive the 1e-9 tolerance and vanish when every fit is
  evaluated at a common x per rung: the geometric mean of that rung's FLOPs across
  the 21 submissions. That gap is linear in log x, so it has at most one change by
  construction.
- For each two-change pair, the record lists:
  - the pair;
  - the minimum |log gap| on each side of each change;
  - the ratio of their FLOPs at each rung.

**Counting rule with a tolerance.**
- The gap is the log-BPB difference, which has the same sign as the BPB
  difference.
- Rungs with |gap| <= tau are dropped. A sign change is counted between
  consecutive retained rungs with opposite signs, so both sides exceed tau.

**Tolerances.**
- **Primary:** tau = 1.96 * sqrt(2) * 0.000511 = 0.00142. This is the smallest pair
  difference determined under the smallest borrowed noise (DataDecide C4 at 1B).
- **Numerical tie:** tau = 1e-9.
- **Sweep:** 1e-9, 1e-4, 5e-4, 1e-3, 0.00142, 0.003, 0.006, 0.012.

**Rows recomputed at every tau.**
- Marin observed.
- Noiseless fits at their own x.
- Noiseless fits at a common x (diagnostic).
- Each noise null: 1.36% and 0.43%, with the borrowed 0.0511% added and labelled
  as added. 2,000 simulations each, same seed. For every null, the 95% band of the
  mean and of the share with 2 or more changes.
- DataDecide C4 seed means at 150M, 300M, 530M and 1B.

**Predictions.**
- **PT1.** At tau = 1e-9 the noiseless own-x row still has 21 two-change pairs (10%).
  The tie hypothesis is refuted.
- **PT2.** At a common x the noiseless row has 0 pairs with 2 or more changes.
- **PT3.** At the primary tau:
  - Marin's mean changes per pair falls from 0.55 but stays at least 0.30;
  - DataDecide's stays at most 0.04;
  - Marin remains at least 5x DataDecide.
- **PT4.** At the primary tau, the observed Marin mean stays below the 1.36% null's
  band.

## Task 2: form uncertainty vs deployed fragility

**(a) Form uncertainty.** Three rankings, each at its unique optimum:
- the three-parameter global optimum (well-scaled solver: log a, tolerances
  1e-15, the minimum-SSE fit over the 18 starts, which must match `python_marin`);
- two-parameter log-log;
- shared exponent.

Readouts: robust pairs (of 210), the largest invariant top-k, and the top-1.

**(b) Deployed fragility.** Their LM as implemented, capped at 200 iterations.
- **Reasonable starts:** c0 in {0.25, 0.5, 0.7, 0.9} x min y, crossed with b0
  factors {0.5, 1, 2} x the log-log slope of (y - c0), clipped to [0.01, 2]. That is
  12 starts, plus their default (c0 = min(0.7, 0.7 min y), slope factor 1).
- **Excluded a priori:** c0 = 0, which removes the floor the form exists to
  model; and c0 = 0.99 min y, which starts within 1% of the floor's bound.
- **Degenerate fits** are reported separately, not ranked. A fit is degenerate if
  its final floor c >= (1 - 1e-4) x 0.999 min y (at its bound), or b >= 2(1 - 1e-4),
  or b <= 1e-5.
- **Ranking:** a start enters only if no submission is degenerate under it. The
  number excluded is listed per start.
- **Readouts:** robust pairs, the largest invariant top-k, top-1 by start, and the
  share of starts with each top-1.

**Predictions.**
- **PF1.** (a) has at least 140 robust pairs, the largest invariant top-k is 1, and the
  top-1 is muonh_qwen3_scaling.
- **PF2.** (b) has fewer robust pairs than (a), and the top-1 is not invariant: at
  least 2 distinct top-1s across the non-degenerate reasonable starts.

## Task 3: root cause of non-convergence

**Check.** Their fit takes raw model FLOPs. This was confirmed by reading:
`computePowerLawWithOffset` uses `p.x` directly, and the forecast is made at
x = 1e22.

**Rescaled rerun.**
- Their solver, unmodified except that x is replaced by x / 1e18 and the forecast
  is taken at 1e22 / 1e18 = 1e4. Their initialisation adapts automatically: a0
  comes from the log-log intercept in the new units.
- The bounds on a stay [1e-10, 1e30]; they are only rescaled if a fit hits one.
- **Default start:** their unmodified `computePowerLawWithOffset` on the rescaled
  points.
- **The reasonable-start range** (Task 2b): the options runner, which matches their
  function exactly at the defaults.
- **Reaches the global optimum:** the forecast is within 1e-5 BPB of the
  well-scaled least-squares optimum. Rescaling x does not change the optimum
  forecast.
- **Secondary scale** (if the primary fails): x divided by each fit's own minimum
  x.

**Prediction.**
- **PR1.** With x / 1e18 at the 200-iteration cap:
  - all 21 submissions reach the global optimum from their default start;
  - at least 90% of (submission, reasonable start) fits do;
  - the resulting ranking from the default start equals the optimum ranking
    (Kendall tau = 1).
- If every fit from every reasonable start reaches it, the change is recorded as
  the fix.
