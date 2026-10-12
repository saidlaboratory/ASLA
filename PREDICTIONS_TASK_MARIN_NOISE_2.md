# Pre-registration: single-run noise at speedrun scales; the robust core of the 1e22 leaderboard

Written 2026-10-12, before any search or computation for these tasks. No paper
prose. Laptop safety: at most 2 workers under `nice -n 19`. Refutations are
reported first.

## Correction to the record

In `marin_noise.json`, the "smooth fits alone" row used **two-parameter log-log
fits** (2 residual degrees of freedom each), not three-parameter fits; they do
not interpolate the four points.
- Its two-change pairs arise because submissions sit at different model FLOPs
  for the same size label, so two fitted lines are compared at misaligned x.
- Its match to the observed pattern is still not evidence of crossing. The fits
  are made to the noisy observations and inherit their noise.

This is recorded in the output.

## Task 1: a noise estimate at speedrun scales

**(a) Fantastic Optimizers.**
- The paper (Wen et al., "Fantastic Pretraining Optimizers and Where to Find
  Them", arXiv:2509.02046) and its appendix are searched for reported seed
  variance at 130M to 1.2B.
- The harvested sweep is searched for repeated configurations: identical
  optimizer, hyperparameters, size and token budget across separate runs. The
  SD is estimated from any found.
- **Unit.** Relative SD (SD / mean) of the final C4-EN BPB (or the loss the
  sweep reports), so it transfers to Marin's BPB.

**(b) W&B.** A script holds the exact query, runnable in one step once an API key
exists:
- the projects;
- filters on finished runs at 130M to 1.2B;
- the fields: config with seed and run-identity keys stripped, and the final
  Paloma C4-EN BPB;
- grouping by the stripped config to find repeats.

It is unit-tested on synthetic records; it cannot be run here.

**(c) If an estimate sigma is found,** both-rung determination (Marin 520m ->
1.2b) and the sign-change null are rerun with it. If none is found, the bracket
(about 0.05% to under 0.43%) stands, and one replicate pair at 520m or 1.2b would
settle it.

**Predictions.**
- **PM1.** Any estimate from repeated configurations at these scales lies in
  [0.1%, 0.4%] relative: above the borrowed 1B/1e21 values, below the
  three-parameter bound.
- **PM2.** Under that estimate, the genuine reorder rate (among pairs determined
  at both rungs) is at most 10%, and most of the 40-41 target reorderings are
  not determined at both rungs.
- **PM3.** If no repeated configurations exist, the bracket stands.

## Task 2: the robust core of the leaderboard at 1e22

**Rankings,** 56 combinations in all:
- the three-parameter form from each of 18 starting values (the c0 x b0 range
  of last round) under three convergence settings:
  - their LM capped at 200;
  - their LM at 100,000;
  - the least-squares optimum (Python, trust region), started from the same
    values;
- the two-parameter log-log form, which is closed-form;
- the shared-exponent form, which is closed-form.

**Readouts.**
- The number of pairs (of 210) whose order is identical in every combination.
- The largest k whose top-k set is the same in every combination.
- Interchangeable submissions: the connected groups of non-invariant pairs.

**Least-squares check.** The optimum from all 18 starts. It is the same optimum
if the forecasts agree within 1e-6 BPB and the SSE within 0.1%; if so, the
ranking is the same.

**Monotone distortion.**
- Spearman of the as-implemented forecasts against the optimum's.
- Where monotonicity breaks: the as-implemented rank distance of each of the 11
  discordant pairs.
- **The reading,** if the map is approximately monotone (Spearman >= 0.95 and
  discordant pairs mostly adjacent, rank distance <= 2): a monotone
  transformation of forecasts preserves every pairwise order, so large
  differential shifts that are close to monotone change few orders. That is
  recorded as the reason.

**PM4.** The invariant core is small. Top-1 (muonh_qwen3) is invariant, and top-3
is not.
