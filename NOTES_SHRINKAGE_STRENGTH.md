# Note: reconciling the reported shrinkage strengths

Three different values of the empirical-Bayes shrinkage strength
`lambda = within / (within + between)` appear in this project's history. They are
not instability; they differ for two separable and measurable reasons. Because
`lambda` anchors the magnitude prediction in `PREDICTIONS_TASK_B.md`, the
reconciliation is recorded here rather than left to be inferred.

## The three numbers

| context | interventions used | bootstrap draws | lambda |
|---|---|---|---|
| contamination check, full set | all 25 | 25 | **0.1605** |
| contamination check, ad-hoc subset (first 17 alphabetically) | 17 | 25 | **0.1477** |
| corrected pipeline, frozen split | 17 (seeded selection) | 40 | **0.2271** |

## What explains the gap

Measured directly (`between_sd` / `within_sd` at each setting):

| setting | between_sd | within_sd | lambda |
|---|---|---|---|
| all 25, n_boot=25 | 0.00700 | 0.00306 | 0.1605 |
| ad-hoc 17, n_boot=25 | 0.00779 | 0.00324 | 0.1477 |
| all 25, n_boot=40 | 0.00700 | 0.00319 | 0.1715 |
| **frozen 17, n_boot=40** | **0.00648** | **0.00344** | **0.2196** |
| ad-hoc 17, n_boot=40 | 0.00779 | 0.00340 | 0.1600 |

Two effects, in order of size:

1. **Subset composition dominates.** The frozen split and the ad-hoc
   alphabetical subset share only **12 of 17** interventions. Their
   between-intervention spreads genuinely differ (`between_sd` 0.00648 vs
   0.00779), because a different set of 17 data recipes really does have a
   different spread of decay exponents. At matched `n_boot`, that alone moves
   `lambda` from 0.1600 to 0.2196.
2. **Bootstrap noise in the within-component is secondary.** Raising the inner
   bootstrap from 25 to 40 draws moves `lambda` by roughly 0.01 at fixed subset
   (0.1605 -> 0.1715 on the full set).

**The leakage itself was the smallest of the three effects**: at matched
settings, going from all 25 interventions to a training subset moves `lambda` by
about 0.01-0.05. Fixing it was still correct - a hyperparameter must not see the
data it will be scored on - but the fix should not be credited with the
difference between 0.1605 and 0.2271, which is mostly subset composition.

## What is reported now

`scripts/run_task_b.py` records, for every design:

- `eb_strength` - the value actually used, estimated on training interventions;
- `strength_estimated_from` - one of `fixed_constant`, `separate_strength_rows`,
  `same_rows_as_evaluation`, so contamination is visible rather than inferred;
- `strength_source_interventions` - the exact names it was estimated from;
- `descriptive_full_set_*` - the full-set components, explicitly labelled
  descriptive, used only to ground the pre-registered prediction;
- `shrinkage_strength_sensitivity` - `lambda` recomputed over repeated
  same-size random training subsets, giving its mean, sd, and range.

The last of these is the honest summary: `lambda` is a ratio of two estimated
variances from a subset of 25 interventions, so it carries real sampling
uncertainty, and any single value should be read against that spread rather than
treated as a constant. The pre-registered prediction was anchored at a noise
share of **0.33** measured on one recipe; `PREDICTIONS_TASK_B.md` states in
advance that if the pooled analysis finds a share below 0.10 or above 0.70, the
magnitude predictions are to be judged against the measured share instead.
