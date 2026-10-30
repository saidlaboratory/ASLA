# Pre-registration: how much the DataDecide fixed-target result depends on its recipes

Committed before any slope is computed for this task. Data: DataDecide C4 bits per token,
750M excluded, as in PREDICTIONS_TASK_FIXED_TARGET.md. No paper edits follow this round.

## The statistic (frozen)

The fixed-target pooled slope:

- targets 150M, 300M, 530M and 1B (the targets admitted by the 50% determinacy rule
  on the full set, fixed here);
- slices of target and rung count with at least 3 windows;
- the pooled within-slice OLS slope of the excess on log10 lever arm, in points per
  decade.

The machinery is unchanged:

- design excess comes from `asla.analysis.confound.window_designs`, run on the
  recipes in the set, so the moderation prior and the pairs are those of the set;
- slices and the statistic come from `asla.analysis.fixed_target`;
- the leave-one-recipe-out jackknife gives the standard error.

The interval is estimate ± 1.96 × SE: every calibrated critical value for this
statistic was below 1.96, so the normal value is used.

The full-set value is +11.67 [9.24, 14.09].

## Families (fixed here)

DataDecide's `intervention_class` field is "data" for all 25 recipes, so families
are taken from the lineage that DataDecide's recipe names record:

| Family | Recipes | Count |
|---|---|---|
| DCLM-Baseline | DCLM-Baseline; DCLM-Baseline (QC 10%), (QC 20%), (QC 7%, FW2), (QC 7%, FW3), (QC FW 10%), (QC FW 3%) | 7 |
| Dolma | Dolma1.6++; Dolma1.7; Dolma1.7 (no Flan), (no Reddit), (no code), (no math, code) | 6 |
| Falcon | Falcon; Falcon+CC; Falcon+CC (QC 10%), (QC 20%), (QC Orig 10%), (QC Tulu 10%) | 6 |
| DCLM/Dolma mixes | 25/75, 50/50, 75/25 | 3 |
| Other | C4; FineWeb-Edu; FineWeb-Pro | 3 |

## (a) Families

- **Within family.** For each family with at least 5 recipes (DCLM-Baseline, Dolma,
  Falcon), the fixed-target slope is computed on that family's recipes alone, with
  its interval.
- **Leave one family out.** For each of the five groups, the pooled slope on the
  other recipes, with its interval.

## (b) Random subsets

For k in {5, 10, 15, 20}, draw 200 random subsets of k recipes without replacement,
with a seed fixed in the script. For each subset, compute the slope and its interval
on that subset alone. Report:

- the mean and SD of the slope;
- the share of subsets whose slope is positive with an interval excluding zero.

The full-set designs, slices and targets are used for every subset.

## Predictions

**(a) Families**

- **F1:** all three within-family slopes are positive.
- **F2:** at most one of the three within-family intervals excludes zero, because
  6 or 7 recipes give wide intervals.
- **F3:** every leave-one-family-out slope lies in [+8, +15] and its interval
  excludes zero.

**(b) Random subsets**

- **S1:** the mean slope is in [+10, +13.5] at every k.
- **S2:** the SD of the slope falls with k. The predicted bands are:

  | k | Predicted SD |
  |---|---|
  | 5 | 6 to 15 |
  | 10 | 3 to 8 |
  | 15 | 2 to 5 |
  | 20 | 1 to 3 |

- **S3:** the share of subsets that are positive and exclude zero rises with k. The
  predicted bands are:

  | k | Predicted share |
  |---|---|
  | 5 | 10% to 50% |
  | 10 | 40% to 90% |
  | 15 | 70% to 100% |
  | 20 | 90% to 100% |

## What each outcome means (decided now)

These readings are not mutually exclusive.

- **Holds within every family:** all three within-family slopes are positive with
  intervals excluding zero, and every leave-one-family-out interval excludes zero.
  The effect is not carried by one lineage of recipes.
- **Driven by one family:** leaving out one family makes the pooled interval include
  zero, or brings the slope below +5.85 (half the full-set value). The effect depends
  on that family and is stated as such.
- **Too variable at small k to call general:** fewer than half of k = 10 subsets are
  positive with an interval excluding zero. The effect is then a property of
  DataDecide's set as a whole, which typical sets of 10 recipes cannot detect, and it
  is reported as a population average over this set.
- **Otherwise:** the effect is robust to which recipes are in the set, at the stated
  k.

## Outputs

- `results/target_scoring/recipe_dependence.json`
- `scripts/run_recipe_dependence.py`
