# Pre-registration: replicating the fixed-target lever-arm effect on an independent suite

Committed before any computation that involves a ranking on the suite below. The only
loss values read so far are one model's eight held-out losses, printed while
checking the file schema (c4_original, 411.6M parameters, multiplier 0.25), which
is not a ranking of anything. No paper edits follow from this task.

## The inventory result (Task 1)

Suites were checked from their sources, with metadata only, against five criteria:

- at least 3 recipes on a shared scale schedule;
- a per-model loss on a common held-out corpus;
- a licence that permits use and publication of derived results;
- no model shared with DataDecide;
- a fixed target with at least three windows of equal rung count below it.

The window rule follows the frozen confound code: windows of at least 3
consecutive rungs, so it needs at least 5 rungs below a target.

**One suite qualifies:** Gadre et al. 2024, "Language models scale reliably with
over-training and on downstream tasks" (github.com/mlfoundations/scaling, MIT).

Not qualifying, with reasons (the full inventory is in the round's report):

| Suite | Reason |
|---|---|
| Loss-to-loss (Kempner) | Structurally qualifies, but its data CSVs carry no licence |
| Porian et al. 2024 | Values could not be verified: they exist only in a pickle, which this session was not permitted to load |
| Farseer | No licence; no metric common to all recipes; held-out set not released |
| Signal and Noise, OLMo ladder | One recipe |
| Delphi | Two recipes |
| Fantastic Optimizers | At most 4 rungs and no seeds |
| Compute multipliers | 5 budgets, and only the top one has public losses |
| DCLM | At most 5 scale points |
| Datablations contours | Recipes not compute-matched; checkpoints not verifiable |
| gzip PCFG ladders | Each recipe's loss is on its own synthetic held-out set |

## The suite

**Recipes.** Three training corpora with an identical model, optimizer and
schedule at each size: `c4_original`, `rpj` (RedPajama) and `rw_original`
(RefinedWeb). These are the files' `dataset_name` values; their 104 run files are
`exp_data/models/*.json`.

**Ladder.** The Chinchilla multiplier 1 runs, where tokens are 20 times parameters
for every recipe. The sizes (`hyperparameters.params`) are 10.6M, 78.9M, 153.7M,
411.6M, 1.44B and 6.89B. Every recipe has every size. Compute is 6 N D from the
file's params and tokens.

**Metric.** Loss in nats per token on the `c4_val` held-out set (11 shards,
174,077,952 tokens). This is the `results` entry whose `val_data` is
`training/eval_data/c4_val/...`. Every model is scored on the same tokens.

**Seeds.** None: every run has seed 124 and there are no replicates.

## Target and windows

The target is the largest model, 6.89B. It is the only target with at least three
windows: with 5 rungs below it, the 3-rung windows are 10.6M to 153.7M, 78.9M to
411.6M, and 153.7M to 1.44B. Windows of 4 and 5 rungs give fewer than three
designs, so the frozen rule drops them. That leaves one slice of three designs.

Lever arms, as log10 of target compute over top-rung compute, are about 3.30,
2.45 and 1.36 decades.

## Noise model (no seeds)

**The single-run SD.** For each target model, the seed noise of one run is the
target loss times DataDecide C4's relative seed SD at the DataDecide size nearest
in log parameters (capped at 1B):

- the relative SD is the median over recipes of the moderated seed SD divided by
  the cell mean, from `data/datadecide_runs.parquet`, 750M excluded;
- it is combined in quadrature with the evaluation-sampling SE of that model's
  `c4_val` loss, which is (loss minus `loss_sequences_lower_95`) / 1.96.

**Target evidence.** The frozen evidence functions need at least two seeds, so
they are replaced by their known-variance limit:

- gap = difference of the two recipes' target losses;
- standard error = root sum of the two SDs squared;
- a pair is determined when |gap| / SE exceeds the normal quantile at
  0.05 / (2 x 3);
- the probability that the observed order is correct is the normal CDF at
  |gap| / SE.

This is the only change to the scoring. `PairEvidence` objects are built
directly and passed to the frozen `expected_charges`.

**Determinacy threshold.** At least 50% of the three target pairs (two of three)
must be determined. If not, the suite cannot test the claim and no slope is
reported as a replication result.

## The frozen code path

Everything else is unchanged from `asla/analysis/confound.py` and
`asla/analysis/fixed_target.py`, with no re-tuning:

- projection is `asla.models.fit_power_law` on each recipe's window values,
  evaluated at target compute;
- single-scale ranking orders recipes by their value at the window's top rung;
- the kernel is the per-pair expected-error charge of projection minus
  single-scale, in points;
- the design excess is the mean kernel;
- the statistic is the pooled within-slice OLS slope of excess on log10 lever
  arm.

**Inference.** The leave-one-recipe-out jackknife with 3 recipes: each deletion
leaves one pair. No calibration benchmark exists for this suite, so the frozen
interval rule, max(calibrated critical value, 1.96), uses 1.96.

**Sensitivity.** Not used for the verdict, but reported:

- the same interval at the t quantile with 2 degrees of freedom (4.30), because
  a jackknife over 3 recipes has 2;
- the verdict with the noise model's SD halved and doubled.

## Prediction

- **Sign:** I predict a slope near zero.
- **Band:** -10 to +10 points per decade, with the interval including zero.
- **Why:** with three corpora scored on C4 validation, the C4-trained recipe is
  likely to lead by a wide margin at every size, so at most one pair (RedPajama
  against RefinedWeb) is contested. A single contested pair can move the excess
  by up to a third of 100 points. That could produce a large slope of either sign
  or none, but the most likely outcome is that both rules order the pairs alike.
- **Predicted verdict:** does not replicate (interval includes zero).

## Success criterion

The slope is positive and its 1.96 interval excludes zero.

## What each outcome means for the claim

- **Not testable** (determined share below 50%): nothing changes. The
  fixed-target claim remains a DataDecide-only finding.
- **Replicates** (positive, interval excluding zero): one independent suite agrees
  in sign. With 3 recipes, 3 pairs and 3 designs this is weak support, and
  stronger only if the t(2) interval also excludes zero.
- **Indistinguishable** (interval includes zero): the suite has too few contested
  pairs to test the claim. That does not count against it, but nor does it extend
  the claim beyond DataDecide.
- **Opposite sign** (negative, interval excluding zero): evidence against
  generality. The claim would have to be stated as specific to DataDecide C4,
  with the failed replication reported.

## Outputs

- `results/replication/gadre_fixed_target.json`
- `scripts/run_replication_gadre.py`
- `data/gadre_scaling_runs.csv`: the derived per-run table (MIT, with attribution),
  built from the run files in `data/raw/gadre_scaling/`
