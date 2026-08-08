# Mis-selection estimand options

ASLA requires paper-facing audit and demo commands to select an estimand
explicitly. The software does not designate either option as the paper's
headline. That scientific choice must be made and documented before results are
generated or interpreted.

## `single_design_seed_sensitivity`

Replication unit: one leaderboard decision over the complete intervention set
in one runs table. The point `mis_selection_rate` is therefore a binary
indicator, not a population percentage. Regret is the measured target-BPB
penalty of that one selected winner.

Uncertainty resamples seeds with replacement inside every observed
`(intervention, compute)` cell. The candidate set, budget design, and cell sizes
remain fixed. This reports the sensitivity of the one decision to the observed
seed variation.

The calculation assumes seed rows are exchangeable within cells. Cells are
resampled independently; it is unsuitable without revision when the same seed
is a paired experimental unit whose dependence must be preserved across
interventions or budgets.

It licenses a statement about whether this fixed candidate-set decision was
correct and how sensitive it is to within-cell seed resampling. It does not
estimate the fraction of future leaderboards, interventions, tasks, or systems
that will be mis-selected.

## `pairwise_decisions`

Replication unit: one unordered intervention pair. ASLA evaluates every pair
in the supplied table and averages each decision metric over that fixed finite
set. The point `mis_selection_rate` is the observed pairwise error frequency;
regret is averaged over the same pair decisions.

Uncertainty resamples seeds inside cells and recomputes the complete set of
pairs. Pairs themselves are not resampled. Pairs share interventions and are
dependent, so the reported pair count is not presented as an independent
sample size.

It licenses a statement about pairwise decisions among interventions in this
particular table. It does not establish a population rate for unseen
interventions, classes, datasets, target budgets, or training systems.

## Human decision required

Before paper analysis, pre-register the primary estimand, whether seeds are
paired across cells, what population (if any) the intervention set represents,
and which analyses are descriptive versus inferential. Both estimands may be
reported when separately labeled; they must not be pooled into one headline
percentage.
