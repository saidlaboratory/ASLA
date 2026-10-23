# Pre-registration: inclusion audit and recount; reordering with the per-size protocol held fixed

Written 2026-10-22, before the recount, the selection-bias fits, the pair classification and
the sensitivity fits. No paper prose. Compute-free (public data only). Laptop safety: at most
2 workers under `nice -n 19`. Refutations are reported first.

## Known before writing (the inclusion data step)

**The pin.** The official displayed leaderboard is marin-community/speedrun at 31fe8028 (the
current main). Our `runs.json` is byte-identical to it (sha256 957feb...).

**What the website does** (App.tsx, useSpeedrunData.ts, scaling.ts at the pin):
- The scaling track takes every run whose name contains "/".
- Runs are grouped by folder (`getScalingGroupName`).
- A group ranks only if one of its runs has more than 3e20 training hardware FLOPs.
- Groups are sorted by the 1e22 forecast.

**Result at the pin:**
- 22 groups qualify, and they are exactly our 22.
- parallel_llama/old_scaling/{gpu,tpu} and nano_arch_ablations are present but do not qualify.

**muonremez and prism:**
- They are on the official displayed board.
- They entered `runs.json` through a manual commit, 04e37e78e4 by kaiyue (2026-05-06), "Add
  Qwen3 optimizer sweep winners". The other entries came through the automated pipeline.
- Their code PRs (#4932, #4933) were closed unmerged.

**Instance 11.** The condition, a non-official submission included, is not met. Instance 11 is
therefore not recorded unless the per-submission entry audit below finds one.

**The reverse defect.** The official submission `hackable_transformer_attn_sink/std_attn` was
omitted from every 21-member analysis, because its top-rung label is "1.2b", not "1_2b". It is
reported as an omission. Whether it becomes a practice-record instance is left to you, since
it does not meet the stated condition.

## Task 1

**(a) Entry audit.** For each of the 22: the first `runs.json` commit containing the group, and
whether that commit is an "automated update" or a manual one, with its author.

**(b) Recount on the official set,** the 22 four-rung submissions with std_attn's "1.2b" mapped
to 1_2b. Each readout is reported against its previous (21-member) value:
1. **Determined-pair reorder rate,** 520m -> 1.2b, under the DataDecide 1B SD and the Task-1
   bound, plus the class comparison against DataDecide 530M -> 1B (one-sided Fisher).
   - The genuine-recipe designs of the class-power round already used the 22, so they are
     unchanged.
2. **Held-out projection of 1.2b from 130m-520m,** for all four forms against single-scale: pair
   errors and the U test.
3. **The fix's effect on the displayed ranking:** changed pairs, Kendall tau and top-3, displayed
   against patched.
4. **The form-robust core:** the three-parameter optimum against two-parameter log-log, robust
   pairs out of 231, and the invariant top-k.
5. **Noise-propagated resolution at 1e22** at the Task-1 bound: pairs holding in at least 95% of
   1,000 draws, and the contiguous top-k.
6. **Certified pairs at 1e22** at sigma* = 0.58%.

**(c) Selection bias** (muonremez, prism; the 5-run LR sweeps in `selection_summary.json`).
- **Per size:** BPB against log LR. A quadratic (3 parameters, 2 df) gives the fitted optimum,
  the minimum of the quadratic within the swept range.
  - Realized bias = picked run's BPB - fitted optimum. Negative means optimistic.
  - The picked run is checked to equal the submitted run.
- **Varying with size:** the bias difference between consecutive rungs, compared with the median
  absolute gap between this submission and others at those rungs.
- **Crossings:** pairs involving muonremez or prism whose gap sign at 300m or 520m flips when the
  picked value is replaced by the fitted optimum.
- **Sweep residuals:** pooled relative residual SD (2 df per sweep, 16 df), and the leave-one-out
  RMS.

## Task 2

**(a) Pair classification.**
- **Protocol fields:** batch, warmup, schedule, decay, min_lr_ratio, weight_decay, seq_len,
  hardware and devices.
  - LR is excluded: per-size LR tuning is the norm, and every submission does it.
  - tokens_per_param is excluded: it is set by the ladder's nominal parameter counts and is
    identical within a FLOPs group.
- **A pair is config-matched at a rung step** if, for every protocol field, both submissions
  make the same change: the same ratio for numeric fields, the same equal/changed status for
  categorical ones. Otherwise it is mismatched.
- **Pairs:** genuine-recipe pairs (identical FLOPs at both rungs, different families), 22
  submissions. Steps: 130m->300m, 300m->520m, 520m->1.2b.
- **Per class and step, under the Task-1 bound:**
  - the determined-pair reorder rate;
  - its candidate-level SE (delete-one-candidate jackknife over the class's pairs; the U
    statistic needs complete pair sets).
- **Comparisons:**
  - matched against DataDecide at matched rungs (its jackknife SE; z test), with one-sided
    Fisher as secondary;
  - matched against the Fantastic Optimizers fitted optima (break round: 0/14 and 0/12);
  - matched against mismatched (z test).

**(b) Validation on config-consistent submissions.**
- **Primary:** submissions whose protocol fields are constant across 130m, 300m, 520m and 1.2b.
- **Prediction PV0:** there are none, because every submission changes batch size somewhere.
- **If none:** the pre-registered fallback is to restrict validation to pairs that are
  config-matched at all three steps, same rule (2-parameter primary, 3-parameter alternative,
  the grid, 400 draws), and report sigma* on that subset.
- **Reading:** if sigma* lies in [0.21%, 0.35%], the earlier 0.58% was the config break.

**(c) Forecast sensitivity at 1e22,** per official submission.
- **A:** the corrected three-parameter forecast (the leaderboard's, patched).
- **B0:** two-parameter log-log on 4 points (2 df).
- **B:** two-parameter log-log plus a step at the rung where batch size first changes (1 df).
  The 1e22 forecast is on the post-change curve.
- **C:** two-parameter log-log on the post-change rungs only, with at least 2 points (0 df).
  Otherwise "not identified".
- **Reported:** B - B0, C - B0 and B - A, in BPB and in median neighbour gaps (0.0026 BPB). The
  SE of B - B0 comes from the WLS covariance under the bound.
- **Marked "cannot be separated from the step"** if |B - B0| exceeds one median neighbour gap.
  That means the forecast at leaderboard resolution depends on how the step is treated.

## Predictions

- **PI1.** All 22 entered through automated-update commits except muonremez and prism, which
  entered through the manual 04e37e78e4.
- **PI2.** Adding std_attn moves each recount by little:
  - reorder rates within 2 percentage points;
  - projection still no better than single-scale for every form;
  - the fix still changes the top 3;
  - form-robust core at least 190/231;
  - at most 100 pairs hold at the bound;
  - at most 20 certified pairs.
- **PS1.**
  - Realized selection bias is negative at every size, between -0.3% and -0.02%.
  - The bias difference between rungs is smaller than the median pair gap, so selection alone
    does not create crossings (0 flips).
  - Sweep residual SD lies in [0.05%, 0.5%].
- **PM1.** The config-matched reorder rate at 300m->520m under the bound is at most 5%, and not
  significantly different from DataDecide's (|z| < 1.96). The mismatched rate is higher.
- **PV0.** No submission is config-consistent across the four rungs.
- **PV1.** On config-matched pairs, sigma* is below 0.58%.
- **PF1.** At least half of the submissions are marked "cannot be separated from the step".
