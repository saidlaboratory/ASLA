# Pre-registration: Marin analyses at aligned compute; two-form uncertainty; the leaderboard fix

Written 2026-10-12, before any computation for these tasks. No paper prose. Laptop
safety: at most 2 workers under `nice -n 19`. Refutations are reported first.

**Known before writing** (from `sign_ties.json` and the gate's FLOPs table):
- **Three FLOPs groups** share an identical FLOPs ladder: Llama/Qwen (10
  submissions, 45 pairs), attn_gate (3, 3 pairs) and attn_sink plus parallel_llama
  (8, 28 pairs). That is 76 within-group pairs.
- **Relative to the per-rung geometric mean,** the groups sit at:
  - Llama/Qwen: 0.913, 0.997, 0.920, 0.906;
  - attn_gate: 1.094, 1.020, 1.111, 1.131;
  - attn_sink: 1.083, 0.996, 1.066, 1.080.
- **Implication.** The cross-group FLOPs ratio is nearly the same at 520m and 1.2b
  (about 0.86 and 0.84) but near 1 at 300m. An offset that is the same at the fit
  and target rungs largely cancels in a 520m -> 1.2b comparison. The 300m rung is
  misaligned in the opposite direction, which affects the held-out projections
  that fit through 300m and the sign-change pattern.

**Practice record.** Instance 10 is recorded after computing. It cites its numbers
from the output JSON so they regenerate.

## Task 1: Marin at aligned compute

**(a) Size of the offset.**
- **FLOPs table:** per submission at each rung.
- **Each submission's two-parameter log-log slope** beta_s (4 rungs) converts its
  FLOPs offset to BPB. For a pair (a, b) at rung r:
  - delta_ab,r = y_a,r [(x_b,r / x_a,r)^beta_a - 1], the BPB change in a from
    moving it to b's compute;
  - this equals 0 within groups.
- **Units, over cross-group pairs** (median and 90th percentile per rung):
  - |delta| over each borrowed per-run SD (cv x BPB);
  - |delta| over the observed |gap| at that rung;
  - at 1.2b, the share of cross-group pairs whose |delta| exceeds |gap|.

**(b) Aligned designs.**
- **W: within FLOPs groups,** the 76 pairs. Observed cells are unchanged, because
  alignment is exact.
- **W-LQ: the Llama/Qwen group alone,** 10 submissions and 45 pairs.
- **C: all 210 pairs at common compute.**
  - **Target x\*_r:** the median FLOPs over the 21 at rung r, clipped into the
    intersection of all submissions' observed ranges, [max_s min_r x, min_s max_r
    x], so no cell is extrapolated. How far clipping moved each target is reported.
  - **Aligned value:** y_s,r x f_s(x\*_r) / f_s(x_s,r), where f_s is the
    submission's own fit. The fit is used only to move along the curve, so each
    observation keeps its residual noise.
  - **C2:** f_s is the two-parameter log-log fit.
  - **C3:** f_s is the three-parameter least-squares global optimum (well-scaled
    solver, 18 starts, minimum SSE).
  - **Held-out projection only:** the fit rungs (130m, 300m, 520m) are aligned
    using a fit on those three rungs alone, so the 1.2b target does not leak into
    them. The target is aligned with the four-rung fit.
  - **Sensitivity C-drop:** the unclipped median, with out-of-range cells dropped.
    Readouts: coverage, and the reorder rate on pairs complete at 520m and 1.2b.

**(c) Rerun under W, W-LQ, C2 and C3.**
1. **Determined-pair reorder rate,** fit 520m -> target 1.2b (`reorder`), under
   each borrowed SD (DataDecide 1B 0.0511%, Delphi 1e21 0.0440%, Delphi 1e22
   0.0752%) and the 0.43% bound.
2. **Both-rung genuine reorderings** (`both_rung`) under the same four SDs.
3. **Held-out projection** of 1.2b from 130m, 300m and 520m, for every fit form:
   their LM, exact three-parameter, log-log and shared exponent.
   - Pair errors against single-scale (the 520m value).
   - **Candidate-level test:** the U statistic on the kernel 100 x (projection
     wrong - single wrong).
     - C2, C3 and W-LQ: every pair is scored.
     - W: the U test needs complete pair sets, so it is run per group with at
       least 4 submissions (Llama/Qwen, attn_sink). The groups are combined with
       pair-count weights: z = sum w_g m_g / sqrt(sum w_g^2 se_g^2). The 3
       attn_gate pairs enter the error counts only.
   - The shared exponent is fitted over all 21 submissions in every design, and
     pairs are then restricted.
4. **Class comparison against DataDecide at matched rungs:** the item-1 rate
   against DataDecide C4 300M -> 1B (1.0%) and 530M -> 1B (0.3%).
   - DataDecide recipes at one scale share the architecture and token count, so
     they are compute-aligned already.
5. **Sign-change table with tolerance,** at tau = 1e-9 and 0.00142:
   - observed;
   - the noiseless two-parameter fit at the design's x;
   - nulls at 0.0511%, 0.43% and 1.36% (2,000 simulations, the same seed);
   - DataDecide.

**The class difference survives** (pre-registered threshold) if the aligned Marin
rate under the DataDecide-1B SD meets both conditions:
- it exceeds the higher DataDecide rate (300M -> 1B, 1.0%) by at least 5
  percentage points;
- a one-sided Fisher exact test of Marin (reordered / determined) against
  DataDecide 300M -> 1B gives p < 0.05.

If either fails, it does not survive.

**Your predictions, recorded as given.**
- **UA1.** At aligned compute, Marin's reorder rate falls substantially. This is
  read as: the aligned rate is at most half the named-rung rate (21.6% under
  DataDecide-1B), in each design.
- **UA2.** The class difference shrinks: (Marin rate - DataDecide 300M->1B rate)
  is smaller aligned than at named rungs, in each design.

**My predictions.** They follow from the cancellation at 520m and 1.2b described
above.
- **MA1.** The 520m -> 1.2b reorder rate falls by less than a third relative and
  stays above 10% in W, W-LQ, C2 and C3. UA1 is refuted.
- **MA2.** The class difference survives under the threshold above in every
  design.
- **MA3.** Cross-group |delta| at 1.2b has a median of at least 5 borrowed SDs (all
  three borrowed values), and its median ratio to the observed |gap| lies between
  0.3 and 3.
- **MA4.** In every design, every form's held-out projection makes at least as
  many pair errors as single-scale. No prediction is made on significance.
- **MA5.** At tau = 0.00142, the observed mean sign changes per pair is at least 5x
  DataDecide's in W, W-LQ, C2 and C3. The share with 2 or more changes is at most
  5% in W and W-LQ.

## Task 2: form uncertainty without the shared exponent

**Reason for the distinction.** The shared exponent imposes one scaling exponent
on every submission, so it assumes the very differences in scaling that an
architecture and optimizer leaderboard exists to detect. It is a test of offsets
given a common slope, not an alternative estimate of each submission's curve.

**Readouts.** Robust pairs and the invariant top-k for:
- the converged three-parameter global optimum and two-parameter log-log alone;
- next to the three-form result (130/210, k = 1).

**MF1.** The two-form result has at least 150 robust pairs (it is necessarily at
least 130), the invariant top-k is between 1 and 3, and muonh_qwen3 is top-1.

## Task 3: record the fix

**Record.**
- A unified diff against their `regression.ts`: fit x / 1e18 and forecast at
  x / 1e18; errorTolerance 1e-15.
- The equivalent variant: x / 1e18 with y x 100 (forecast / 100).

**Verification.**
- The patched `regression.ts` is applied to a temporary copy and imported. From
  its default start, it must reach the global optimum (within 1e-5 BPB) in all 21
  submissions.
- The options runner, set to the same change, must reach it in 273 of 273 fits
  (13 starts) at 200 iterations, for both variants.

**Displayed against fixed.**
- **Displayed:** their unmodified function on raw model FLOPs, the leaderboard's
  1e22 forecast.
- **Fixed:** the patched function.
- **Readouts:** the changed pairs (listed), Kendall tau, and the top-3 under each.
  Restricted to qualifying submissions if their qualification rule excludes any.

**MX1.**
- 11 changed pairs, the same as the as-implemented vs least-squares-optimum count
  of earlier rounds.
- Top-1 (muonh_qwen3) unchanged.
- Top-3 changes: adamh_llama is replaced by attn_sink/splash/default_x4.
