# Pre-registration: candidate-level tests of published comparisons, and the pair-resampling correction curve

Written 2026-09-22, before any comparison below has been computed. The analysis
freezes after this task.

## What the sources report (verified against the arXiv HTML, not predicted)

- **DataDecide (arXiv:2504.11393v2).** Decision accuracy is "an accuracy over all
  pairs of data recipes" over 25 recipes. The only uncertainty shown is "the average
  decision accuracy of a given method over 3 prediction attempts using small
  models with different random seeds, and shading shows standard deviation". For
  scaling laws, "we make only one prediction attempt". The text contains no
  bootstrap, confidence interval or significance test on any method difference.
- **Signal and Noise (arXiv:2508.13144v1).** Method comparisons (Table 1, checkpoint
  averaging; the Figure 2 table, BPB against the primary metric) are point
  estimates. The only resampling is over checkpoints ("we sample one of the final
  5 checkpoints"). No interval is reported on a method difference.

Neither paper resampled pairs or candidates. The question is therefore not which
unit their intervals used, but whether their conclusions hold at the candidate
level at all.

## Comparisons attempted

Candidate count 25 throughout. The pair kernel is the difference, between the two
methods, in agreement with the target ordering. Agreement is averaged over the
three small-model seeds where the paper averages over seeds. Each comparison
gets the U-statistic test (`asla.analysis.target_scoring.u_statistic_test`).
Targets are scored against the observed ordering, as the papers do.

- **D1. Scaling-law variants against single-scale ranking** (DataDecide section 3.2).
  Data: the released `scaling_law_fit` table (per-recipe predictions and targets).
  *Known-answer gate:* decision accuracy recomputed from the released predictions
  and targets must equal the released `decision_acc` column to 1e-6 for every
  setup used. The eight variants are the `*-default` setups. The comparator is
  single-scale ranking at the largest size those setups fit on, using the
  paper's per-seed protocol. Primary: `olmes_10_macro_avg`, `primary_metric`.
  Secondary: the ten tasks.
- **D2. Correct Prob against accuracy for single-scale ranking** (DataDecide section 3.3).
  Data: the released macro-average table. Single-scale ranking with
  `correct_prob_per_char` against `primary_metric`, at each small scale, against
  the 1B `primary_metric` ordering. Primary: `olmes_10_macro_avg`. Secondary: the
  ten tasks.
- **S1. BPB against the primary metric** (Signal and Noise Figure 2 table, 150M to 1B,
  "All 30-Task Avg. 77.0 | 83.7"). Attempted only if the released raw evaluations
  reconstruct both published arms to within 1.5 points. Otherwise it is listed as
  not reconstructed.

**Listed as uncheckable, not attempted:** Signal and Noise Table 1 (checkpoint
averaging), because the paper does not state how the source sizes 60M to 750M are
combined; its SNR subset selection, whose selections are not released;
DataDecide's intermediate-checkpoint comparison, which is figure-only; and any
comparison whose method outputs were not released.

## Predictions

- **P1 (gate).** The D1 recomputation matches the released `decision_acc` exactly
  for every setup used.
- **P2 (D1).** On `olmes_10_macro_avg`, no scaling-law variant's difference from
  single-scale ranking excludes zero at the candidate level. The paper's claim
  that no variant exceeds the frontier is then consistent with the data but not
  a demonstrated equivalence.
- **P3 (D2).** On `olmes_10_macro_avg`, Correct Prob's advantage over accuracy
  excludes zero at no more than 2 of the small scales.
- **P4 (S1, if reconstructed).** The BPB advantage in the 30-task average excludes
  zero at the candidate level.

## The correction curve

The pair-resampled variance is `zeta2 / C(n,2)`. The candidate-level variance is
`2 (2(n-2) zeta1 + zeta2) / (n(n-1))`. Their standard-error ratio is therefore

    R(n) = sqrt(1 + 2 (n - 2) zeta1 / zeta2).

Report `R(n)` for n from 5 to 200 at the `zeta1 / zeta2` measured on every
comparison above, and at the C4 value.

- **P5.** Subsampling m of the real candidates (m in 8, 12, 16, 20; finite
  population corrected) reproduces `R(m)` to within 15% on the C4 comparison.

After this task: regenerate everything, run the verifier, compile the paper,
stop.
