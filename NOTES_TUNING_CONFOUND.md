# The tuning confound

## The threat

Lourie, Cho, Ullrich and Lotfi, "Small-Scale Experiments: Are We There Yet?"
(arXiv:2608.11859) argue that scaling laws have looked unreliable at small
scale because small models are far more sensitive to hyperparameters than
large ones. Studies that did not tune each scale to its own frontier therefore
saw inconsistent trends, and with adequate tuning small-scale experiments do
recover large-scale results (demonstrated on pre- versus post-normalisation).
The first reviewer question for any crossover claim is therefore: **are the
crossovers you report just under-tuning at small scale?** If a recipe looks
worse at 130M only because its 130M hyperparameters were off, its "crossover"
against a better-tuned recipe is an artifact of tuning effort, not of the
intervention's scaling behaviour.

## How the design controls for it

1. **A `tuning_quality` column in the schema** (nullable). Every harvested row
   records what is known about how its hyperparameters were tuned at its
   scale. Values in use:
   - `shared_scale_heuristic_hparams` (DataDecide): one configuration per
     model scale from the OLMo ladder heuristics (Porian et al., 2024), shared
     by all 25 data recipes. Nothing was tuned per recipe. This is the axis on
     which per-intervention re-tuning is least expected to matter, and it is
     the predicted scale-stable negative control.
   - `coordinate_descent_tuned` (Fantastic Optimizers): the paper's three-phase
     per-optimizer, per-scale coordinate-descent optimum; value = best observed
     loss in the sweep for that cell, which is what the authors plot.
   - `single_hp_ablation_median` (Fantastic Optimizers): the median final loss
     over the released one-hyperparameter ablations around that optimum - a
     concrete, artifact-backed "one hyperparameter off" condition at the same
     scale. (A "worst ablation" condition was rejected: the maximum is
     dominated by diverged runs, which is a different failure.)
2. **Headline optimizer-axis numbers use the tuned condition only.** The
   Fantastic Optimizers grid is used precisely because it tuned per optimizer
   and per scale; the size ladders to 1.2B (`fantastic_optimizers_size_ladder_*`)
   contain only `coordinate_descent_tuned` rows.
3. **A stratified analysis path** (`asla.analysis.tuning.stratify_by_tuning`,
   tested in `tests/test_tuning.py`, run by `scripts/run_first_audit.py`):
   single-scale pairwise agreement between a small size and the largest size
   that has ablations (520M), computed separately for the tuned and the
   one-hyperparameter-off condition on the same eleven optimizers.

## What the available metadata shows (1x Chinchilla, 11 optimizers, 55 pairs, target 520M)

| small scale | condition | pairwise agreement with 520M order | disagreeing pairs |
|---|---|---|---|
| 130M | `coordinate_descent_tuned` | 87.3% | 7 |
| 130M | `single_hp_ablation_median` | 76.4% | 13 |
| 300M | `coordinate_descent_tuned` | 98.2% | 1 |
| 300M | `single_hp_ablation_median` | 85.5% | 8 |

(Numbers are regenerated into `results/first_audit/first_audit.json` and
FIRST_AUDIT.md by the script; the table here is a copy of that run.)

Reading: mis-tuning by one hyperparameter roughly doubles the number of order
flips between small and larger scale, in the direction Lourie et al. predict.
Among well-tuned runs, 300M already orders 54 of 55 optimizer pairs the way
520M does; at 130M, 7 pairs still flip. Whether those 7 are true crossovers or
residual tuning slack cannot be decided from this artifact: it has one run per
cell and no seeds, so no flip on the optimizer axis can be tested against
run-to-run noise.

On the four-optimizer ladders to 1.2B, the only flips relative to the 1.2B
order involve pairs whose 1.2B losses differ by 0.0000 to 0.0025 nats (see
FIRST_AUDIT.md), i.e. ties at any plausible seed-noise level. The tuned
optimizer axis in this artifact shows convergence toward equal loss at scale,
not decisive crossovers.

## What can and cannot be concluded

Can:
- Report crossover rates stratified by tuning condition, and state that in the
  one grid with per-scale tuning metadata, tuning quality measurably changes
  the number of small-to-large order flips.
- Use only per-scale-tuned runs for the optimizer-axis headline, and label
  every optimizer-axis flip as untestable for significance.
- Use DataDecide's data axis as the control where per-intervention tuning is
  not a mechanism for crossovers (all recipes share hyperparameters), with
  seed-noise-tested significance.

Cannot:
- Claim that any optimizer-axis crossover is a mechanism effect rather than
  residual tuning, because "tuned to the coordinate-wise optimum with
  threshold 3e-3" is not "tuned to the frontier" in Lourie et al.'s sense and
  there are no seeds to bound the noise.
- Extend the data-axis tuning statement beyond "hyperparameters were shared";
  DataDecide's shared per-scale configuration may be closer to optimal for
  some recipes than others, and that differential is not measured.
- Say anything about tuning for sources without tuning metadata; the column
  is null there and analyses must not impute it.

Closing the confound properly needs seeds on the optimizer axis (the W&B
project `marin-community/optimizer-scaling` may hold more runs than the
released JSON summaries; harvesting it requires an API key and is not done
here) or a controlled run of our own with a per-scale tuning budget recorded
in `tuning_quality`.
