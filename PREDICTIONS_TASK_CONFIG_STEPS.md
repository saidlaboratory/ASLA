# Pre-registration: do discrete config steps in DataDecide's ladder contribute to the lever-arm effect?

This is committed before any slope, misfit statistic or step-term fit for this task is computed. The data are
DataDecide C4 bits per token, with 750M excluded. It is analysis only, with no paper edits. The only numbers
looked at so far are the training configs below and window counts, which come from window structure alone.

## (a) Audit: DataDecide's per-size training config

### Sources

- **[T]** DataDecide's released runs table, `data/datadecide_runs.parquet`. It gives N, D, the final step (the
  same for all seeds at a size) and tokens per parameter. In every case D = steps × batch × 2048.
- **[P]** the paper's Table 2 (arXiv 2504.11393, App. A).
- **[L]** the OLMo `DataDecide` branch, `scripts/ladder.py` (commit 7094aab).

### Settings shared by every size [L]

- architecture: SwiGLU, mlp_ratio 8, RoPE, RMSNorm;
- AdamW with betas (0.9, 0.95) and weight decay 0.05;
- LR schedule: cosine with warmup to 1% of peak, with warmup tokens equal to N;
- sequence length 2048.

### Per-size config

| Size | Layers | d_model | Heads | Batch (seqs) | Peak LR | Warmup steps | Steps (released) | Tokens/param |
|---|---|---|---|---|---|---|---|---|
| 4M | 8 | 64 | 8 | 32 | 1.4e-2 | 57 | 5,725 | 100.2 |
| 6M | 8 | 96 | 8 | 32 | 1.2e-2 | 92 | 9,182 | 100.1 |
| 8M | 8 | 128 | 8 | 32 | 1.1e-2 | 130 | 13,039 | 100.1 |
| 10M | 8 | 144 | 8 | 32 | 1.0e-2 | 151 | 15,117 | 100.1 |
| 14M | 8 | 192 | 8 | 32 | 9.2e-3 | 219 | 21,953 | 100.0 |
| 16M | 8 | 208 | 8 | 32 | 8.9e-3 | 244 | 24,432 | 100.0 |
| 20M | 16 | 192 | 8 | 64 | 8.4e-3 | 146 | 14,584 | 100.1 |
| 60M | 16 | 384 | 12 | 96 | 5.8e-3 | 290 | 29,042 | 100.0 |
| 90M | 16 | 528 | 12 | 160 | 4.9e-3 | 299 | 29,901 | 100.0 |
| 150M | 12 | 768 | 12 | 192 | 4.2e-3 | 381 | 37,500 | 97.1 |
| 300M | 16 | 1024 | 16 | 320 | 3.3e-3 | 458 | 45,000 | 92.2 |
| 530M | 16 | 1344 | 16 | 448 | 2.8e-3 | 578 | 51,250 | 88.7 |
| 1B | 16 | 2048 | 16 | 704 | 2.1e-3 | 694 | 69,369 | 85.0 |

Warmup steps are computed as round(N_rule / (batch × 2048)); they are not published.

### Smooth settings

These follow formulas in N and change at every size without a discrete step:

- peak LR: 0.0047 (N/108e6)^(-1/3);
- warmup tokens: N;
- batch above its floor: round(160 (N/108e6)^(2/3) / 32) × 32.

### Discrete steps

**At 20M:**
- the batch leaves its floor of 32 sequences, which held from 4M to 16M, and doubles to 64;
- total steps fall by 40% (24,432 to 14,584), and warmup steps fall from 244 to 146;
- depth doubles from 8 to 16 layers;
- width falls from 208 to 192.

**At 150M:**
- the token budget switches from the code's N to a nominal N (150e6, 300e6, ...), so tokens per parameter fall
  from 100 to 97.1, then to 92.2, 88.7 and 85.0;
- the ladder code at the branch head rejects sizes of 150M and above, so these sizes were launched from code
  not in the public branch;
- depth falls from 16 to 12;
- the released 150M to 530M runs stop before the paper's step counts (for example 51,250 against 57,786 at
  530M).

**At 300M:** depth goes from 12 to 16.

**Head changes:** the number of heads changes at 60M (8 to 12) and 300M (12 to 16). These are not used to
define regimes.

## Regimes, defined now

**Primary (batch rule and LR schedule):**
- A = {4M, 6M, 8M, 10M, 14M, 16M}: batch at its floor and 100 tokens per parameter;
- B = {20M, 60M, 90M}: scaled batch and 100 tokens per parameter;
- C = {150M, 300M, 530M, 1B}: scaled batch and a nominal-N token budget, which shortens the schedule.

The boundaries are at 20M and 150M.

**Secondary (depth):**
- D8 = {4M to 16M};
- D16a = {20M, 60M, 90M};
- D12 = {150M};
- D16b = {300M, 530M, 1B}.

The boundaries are at 20M, 150M and 300M.

## Scope fixed by the ladder

**Every target (150M, 300M, 530M, 1B) lies in regime C.** So a window "within one regime" still projects
across one or two boundaries to its target. Part (c) therefore tests steps inside the fit window. A step
between the window and the target cannot be separated from lever arm on this ladder, because no slice has
three windows that share the target's regime. This is stated as a limit, not tested.

**Within-regime windows:**
- At each target there are 8 within-regime windows in admitted slices of at least 3 windows:
  - the 3-rung slice has 5 windows, 4 in A plus 20M–60M–90M in B;
  - the 4-rung slice has 3 windows, all in A.
- At 1B the 3-rung slice also contains 150M–300M–530M under the primary definition only.
- The A windows' tops span only 8M to 16M, which is 0.55 decades of lever-arm variation within a slice.

**A constant-crossing version (c2):** use only windows inside regime A. Every one of them crosses both
boundaries to reach its target, so lever arm varies while the number of steps crossed is held fixed.

## (b) Misfit at steps

**Per-size misfit:**

1. Fit E + A·C^(−α) to each recipe's seed means on the full 13-size ladder, using the frozen `fit_power_law`.
2. Take the residuals in log loss and remove the across-recipe mean at each size (the common mode).
3. Take the SD across recipes. This is the raw misfit.

**Noise-corrected misfit (primary):**
- sqrt(max(0, raw² − v̄)), where v̄ is the mean over recipes of the variance of the seed mean in log loss, that
  is, the seed variance divided by the number of seeds;
- this removes the part of the misfit that seed noise alone would produce.

**Step sizes:** the two sizes on either side of each primary boundary, {16M, 20M, 90M, 150M}. The other 9 sizes
are the within-regime comparison.

**Test:**
- statistic: the mean misfit at the step sizes minus the mean at the other sizes;
- p-value: one-sided, by exact enumeration of all C(13, 4) = 715 four-size subsets;
- this is run for both the noise-corrected misfit (primary) and the raw misfit (reported);
- the depth definition's step sizes are {16M, 20M, 90M, 150M, 300M}, over C(13, 5) = 1,287 subsets, and are
  reported.

## (c) Within-regime against spanning windows

The frozen kernel and frozen pooled fixed-target slope (`asla.analysis.geometry_scan`) are restricted to
window subsets, with slices of at least 3 windows recomputed within each subset. All uncertainty comes from the
leave-one-recipe-out jackknife, with intervals of ±1.96 SE. The contrasts use the same LORO replicates.

**At each target, and pooled over the four targets:**
- W, the slope from within-regime windows (primary definition);
- W2, the slope from regime-A windows only (constant crossing);
- S, the slope from spanning windows;
- the contrast S − W.

**Reference:** F, the fixed-target slope from all windows at all four targets on the full ladder (the frozen
statistic).

**Ratio:** r = W / F, with a jackknife interval.

**Depth definition:** the same quantities are reported.

**Across geometries:** for every one of last round's 5,682 geometries, the within-regime slope is computed on
that geometry's windows that lie inside one primary regime. A geometry is tested if it has at least one slice of
at least 3 such windows. Report:
- the number tested;
- the share positive;
- the share with an interval excluding zero on the positive side;
- the share significantly negative.

## (d) Step term

**Model:** for a window that spans a boundary,
E + A·C^(−α) + Σ_b δ_b·1[C ≥ C_b].
- There is one free δ_b per primary boundary b inside the window, that is, one where the window has sizes on
  both sides of b.
- The fit is unweighted least squares, with the same bounds and starting values as `fit_power_law`, and
  δ in [−0.5, 0.5] bpb starting at 0.

**Degrees of freedom:** the fit is used only when rungs ≥ 3 + (number of boundaries inside) + 1, which leaves at
least one residual degree of freedom.

**Projection:**
- the projection to the target includes every fitted δ_b, because the target lies above all of them;
- a boundary between the window's top and the target has no data in the window, so it gets no term.

**Statistic:**
- D, the fixed-target slope over within-regime windows (ordinary fit, unchanged) plus spanning windows with
  enough degrees of freedom (step fit);
- F_same, the frozen slope on exactly the same window set.

**Reported:** the share 1 − D / F_same, with a jackknife interval. Spanning windows without enough degrees of
freedom (the 3-rung and 4-rung spanning windows) are excluded from both D and F_same, and their number is
reported.

## (e) Outcomes, decided now

These are applied to W (primary), with F as the reference:

| Outcome | Condition | Meaning |
|---|---|---|
| **O1. Persists at a similar size** | W − 1.96 SE > 0 and r ≥ 0.75 | Steps inside the fit window are not the driver. |
| **O2. Persists but shrinks** | W − 1.96 SE > 0 and r < 0.75 | Steps contribute. Report the share 1 − r with its interval. |
| **O3. Vanishes** | W + 1.96 SE < 0.5 F | The effect is largely a ladder-protocol artifact, at least for steps inside the window. |
| **O4. Inconclusive** | anything else | For example, the interval includes zero but does not exclude half of F. The within-regime design cannot decide, and the report says so. |

Under O4, W2, the contrast S − W and part (d) are reported as the remaining evidence, and none is promoted
to the primary verdict.

**"Steps contribute"** for Task 2's protocol change means O2 or O3, or a share from part (d) whose 95% interval
lies above zero.

## Predictions

- **M1:** the noise-corrected misfit is not concentrated at step sizes (p ≥ 0.05).
- **M2:** the raw misfit is not concentrated either (p ≥ 0.05). 20M alone is high, but 16M and 90M are typical.
- **W1:** the primary outcome is O4 (inconclusive), because of the 0.55-decade lever variation inside the regime-A
  slices.
- **W2:** W2's interval includes zero.
- **D1:** the share from part (d) has an interval that includes zero.
- **G1:** among tested geometries, the within-regime slope is positive in 60% to 90%.

## Outputs

- `results/target_scoring/config_steps.json`
- `scripts/run_config_steps.py`
- `asla/analysis/step_fit.py`
- `tests/test_step_fit.py`
