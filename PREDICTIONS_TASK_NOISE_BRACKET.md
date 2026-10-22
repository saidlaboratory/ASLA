# Pre-registration: narrowing the noise bracket; leaderboard resolution at 1e22; power for the class comparison

Written 2026-10-22, before any fitting or computation for these tasks. No paper
prose. Laptop safety: at most 2 workers under `nice -n 19`. Refutations are
reported first.

## Known before writing (data inspection only)

- **W&B:** no key on this machine (no WANDB_API_KEY, no netrc entry), so
  `wandb_replicate_query.py` cannot run. The bounds below stand in for real
  replicates.
- **Delphi isoFLOP sweeps** (`data/raw/delphi/{delphi-ladder,first-ladder}`,
  harvested from marin-community/delphi-blog-data):
  - two recipes: AdamH (Delphi Scaling Suite) and Cautious AdamC (Initial Scaling
    Suite);
  - seven budgets in [3e18, 3e20] (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20),
    with 5, 10, 15, 9, 14, 14 and 14 runs per sweep: 14 sweeps, 162 runs;
  - batch size varies within a sweep, which shows up as lack of fit, so every
    estimate here is an upper bound.
- **Fantastic Optimizers:** the ablations are one-parameter-at-a-time slices
  around each file's best configuration, with 1 to 10 values per parameter. A
  "response surface" here is therefore a 1-D slice through the baseline.
- **Speedrun:** 25 prefixes in all.
  - 21 share 130m, 300m, 520m and 1_2b.
  - `hackable_transformer_attn_sink/std_attn` has the same ladder but labels its
    top rung "1.2b", so earlier rounds excluded it. Its 130m FLOPs (2.107e18)
    match the attn_sink group.
  - `parallel_llama/old_scaling/{gpu,tpu}` sit on a 75m/150m/300m(/520m) ladder.
- **Missed replicates.** `parallel_llama/old_scaling/gpu` and `.../tpu` have the
  same model size, model FLOPs and description ("on TPU" added) at 75m, 150m and
  300m. Earlier replicate searches matched on run-name prefix and missed them.
  - At 75m: BPB 1.3114 and 1.3109. At 150m: 1.2307 and 1.2320. The 300m values
    have not been looked at.
  - They are a **cross-hardware pair, configuration identity unverified**. They
    may differ in precision or kernels as well as in random trajectory.

## Task 1: narrow the noise bracket

**(a) Delphi isoFLOP leave-one-out.**
- **Per sweep:** y = loss, x = log(params).
- **Flexibilities:**
  - quadratic;
  - cubic;
  - smoothing spline: `scipy.interpolate.make_smoothing_spline`, lambda by
    GCV, refitted on each leave-one-out subset.
- **Leave-one-out:** for each run, refit without it and predict it. Relative
  error e = (y - yhat) / yhat. A flexibility enters a sweep only if the
  leave-one-out fit keeps at least k+1 points (k = parameter count; the spline
  needs 5).
- **Per budget, pooled over the two recipes:**
  - LOO RMS relative error and its N (runs);
  - residual SD with df = sum(n - k) for the polynomials.

**(b) Fantastic Optimizers 1-D slices.**
- **Slice:** for each (optimizer, size, budget multiple, parameter), the baseline
  plus the ablation values.
- **Coordinate:**
  - log(value) if every value is positive;
  - log(1 - value) for beta1 and beta2;
  - the raw value otherwise (weight decay and max_grad_norm include 0).
  - Non-numeric parameters are dropped.
- **"Near the optimum":** points whose loss is within 2% (relative) of the slice
  minimum.
- **Flexibilities:** linear (needs 3 or more points), quadratic (4 or more) and
  cubic (5 or more).
- **Readouts by size, pooled over optimizers and budgets:** LOO RMS relative error
  with N, and residual SD with df.
- **The baseline is shared** by all of a file's slices. It enters each slice's
  leave-one-out, and counts once per slice in N. This is stated as a correlation
  caveat.

**(c) Calibration on DataDecide.**
- **Curves:** each recipe (25) and each seed (3) separately, single-run curves of
  BPB against log(params) over the 7 largest sizes: 60M, 90M, 150M, 300M, 530M,
  750M and 1B. This matches the 5-15 point density of the Delphi sweeps.
- **The same flexibilities and leave-one-out rule as (a).**
- **True seed SD:** the per-cell SD over 3 seeds, relative to the cell mean,
  pooled as sqrt(mean var_rel) over the 25 recipes, at each of the 7 sizes.
- **Calibration ratio** per flexibility: LOO RMS relative error / pooled true
  relative seed SD. Over all 7 sizes, and at 300M, 530M and 1B separately.

**(d) Combining into a bound.**
- **Mapping to speedrun sizes:** Delphi budgets nearest in log10 to the median
  speedrun model FLOPs:
  - 130m (2.1e18) -> 3e18;
  - 300m (1.0e19) -> 9e18;
  - 520m (4.9e19) -> 3e19;
  - 1.2b (2.1e20) -> 1.8e20.
  - Fantastic Optimizers sizes map directly.
- **Per source and size,** the flexibility with the smallest pooled LOO error is
  used (a mild optimism, disclosed). All flexibilities are reported.
- **Upper bound per source:** the LOO RMS times sqrt(N / chi2_{0.05, N}), the
  one-sided 95% limit treating N as df. That is anti-conservative under
  correlation, which is stated.
  - If that flexibility's DataDecide ratio is below 1 (leave-one-out
    underestimates seed noise), the bound is divided by the ratio.
- **Smallest defensible upper bound** at each size: the minimum over the two
  sources. The cross-hardware pair is not used in the bound (identity
  unverified); it is reported as a check.
- **Calibrated point estimate:** LOO / DataDecide ratio, shown next to the
  bound.
- **Caveat stated:** speedrun architectures may be noisier than Delphi's or
  Fantastic Optimizers'.

**(e) Reruns under the bound.**
- **Both-rung genuine reorder rates,** with the 520m bound at the fit rung and the
  1.2b bound at the target, for:
  - named rungs; W; W-LQ; C2; C3;
  - the near-duplicate groups separately: the within-group pairs of attn_sink
    plus parallel and of attn_gate, 31 pairs.
- **The sign-change null** at the per-rung bound vector, over the 210 pairs at
  named rungs and in C2, primary tolerance.

**Cross-hardware pair.** Relative SD from each pair = |d| / (sqrt(2) x mean),
pooled over the three sizes (3 df).

**Predictions.**
- **PB1.** The smallest defensible upper bound lies in [0.10%, 0.50%] at every
  speedrun size.
- **PB2.** The DataDecide calibration ratio (best flexibility, all 7 sizes) lies in
  [2, 10]: leave-one-out is loose, dominated by lack of fit. So the calibrated
  point estimate is 0.02-0.20%.
- **PB3.** The cross-hardware pooled relative SD lies in [0.02%, 0.15%].
- **PB4.** Under the bound, the near-duplicate pairs give at most 3 genuine
  both-rung reorderings, and the Llama/Qwen group at most 1. Most near-duplicate
  pairs are not determined at both rungs.
- **PB5.** Under the bound, the observed Marin mean sign changes lie inside or
  below the null's 95% band.

## Task 2: leaderboard resolution at 1e22

**(a) Noise propagation.**
- **Draws:** each submission's four points are multiplied by (1 + sigma
  N(0, 1)), with sigma at:
  - 0.05% (DataDecide 1B, 0.0511%);
  - the Task 1 bound (per rung);
  - 0.43%.
- **Refit** with the corrected solver: their `computePowerLawWithOffset` with
  the committed patch (x / 1e18, errorTolerance 1e-15), default start. Forecast
  at 1e22.
- **1,000 draws per level,** seed 20261022.
- **Readouts:**
  - per-submission forecast SD;
  - pairs whose order (as in the noiseless fixed ranking) holds in at least 95%
    of draws;
  - the largest top-k whose set holds in at least 95% of draws;
  - the intersection with the 175-pair form-robust core (three-parameter optimum
    and log-log).
- **Stated:** this is noise only, a lower bound on total uncertainty.

**(b) Delphi forecast check.**
- **Points:** for each recipe, the compute-optimal run at the four budgets nearest
  the speedrun ladder (3e18, 9e18, 3e19, 1.8e20).
  - Primary: the lowest-loss actual run in each sweep, x = its total FLOPs.
  - Secondary: Delphi's fitted loss_star at C.
- **Fit** the corrected leaderboard form as if a submission. Forecast at 1e21, 1e22
  and 1e23.
- **Compare with Delphi's held-out runs:**
  - AdamH: seed means of 3 at 1e21 and 1e22, single run at 1e23.
  - Cautious AdamC: single runs at 1e21 and 1e22.
  - Its "1e23" run is excluded with reason: C = 6.9e22, and its loss is above
    its own 1e22 run.
- **Readouts:** relative error, and that error in units of the median
  nearest-neighbour gap of the fixed 1e22 leaderboard. To convert, relative
  error x the median leaderboard forecast.
- **Labelled:** absolute error, largely common across recipes. It bounds forecast
  accuracy, not ranking accuracy.

**PL1.** At the Task 1 bound, fewer than 100 of 210 pairs hold in at least 95% of
draws, and the largest top-k that holds is 0 or 1.

**PL2.** The Delphi 1e22 forecast error exceeds 10 median neighbour gaps.

## Task 3: more power for the class comparison

**Listing:** every speedrun prefix with its budgets and FLOPs (the 118 links).

**Designs:** every (fit rung, target rung) with fit < target among the shared
ladder.
- 130m->300m, 130m->520m, 130m->1.2b, 300m->520m, 300m->1.2b and 520m->1.2b,
  over the 22 four-rung submissions (std_attn included).
- The 75m/150m/300m ladder has only parallel_llama's two hardware variants, so it
  contributes no genuine-recipe pair.

**Genuine-recipe, identical-compute pairs:** both submissions in the same FLOPs
group (identical model FLOPs at both rungs) and different families (text before
the first "/").

**Comparison per design:**
- determined-pair reorder rate under the DataDecide-1B SD, and under the Task 1
  bound;
- against DataDecide C4 at the matched rungs (130m:150M, 300m:300M, 520m:530M,
  1.2b:1B), with one-sided Fisher exact tests.
- **Pooled over designs:** a pooled count is reported, but pairs repeat across
  designs and are not independent, so no pooled test is claimed.

**Power:** the number of Marin determined pairs needed to detect a 4-point
difference over DataDecide's matched rate at 80% power.
- One-sided Fisher exact at alpha 0.05, DataDecide n fixed at its determined
  count.
- Simulated: 2,000 draws per n.

**PP1.** The required n is at least 150 determined pairs. No set of designs here
reaches it with independent pairs.
