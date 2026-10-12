# Pre-registration: review risks (lever-arm generality, compute-multipliers replication, Marin speedrun, mechanism robustness, calibration under a wrong simulator)

Written 2026-10-12, before any computation for these tasks. The analysis stays
frozen otherwise.

- **Outputs.** Code, tests, JSON, figures and this file; no paper prose.
- **Hardware.** At most 2 workers under `nice -n 19`, one long job at a time,
  every job resumable.
- **Reporting.** Refutations are reported first.

## Task 1: is the lever-arm regularity general?

**Existing claim.** It was tested on **C4 bits per token** only
(`results/target_scoring/lever_arm_calibration.json`). The test is the
leave-one-recipe-out (LORO) Fisher-z test of rho(log lever arm, projection's
excess over single-scale), across 55 designs. Its critical value, 1.49, was
calibrated on 300 random-candidate worlds.

**Metrics.** Every DataDecide macro-average metric with a full 13-rung,
3-seed ladder, each made lower-is-better:
- C4 bits per token;
- OLMES error (1 - accuracy);
- correct-prob deficit;
- 1 - margin_per_char;
- 1 - norm_correct_prob_per_char;
- 1 - total_prob_per_char.

**Calibration per metric.**
- The same procedure as for C4, with that metric's own shrunk truth and noise:
  150 random-candidate main worlds (bandwidth 0.5) and 1,000 null worlds.
- Coverage of the true rho at the 95th percentile gives k. The null worlds give
  p against the replicate comparator.
- C4 reuses its existing calibration (300 main worlds, 8,000 null worlds).

**Verdict per metric.**
- Holds: the calibrated interval lies above 0.
- Reverses: it lies below 0.
- Otherwise: inconclusive.

**Chance-rung exclusion.**
- **Definition.** A rung is at or below chance if its recipe-mean OLMES macro
  accuracy is at most chance accuracy, 0.348. That is the mean of 1/k over the
  10 tasks: ARC-C 4, ARC-E 4, BoolQ 2, CSQA 5, HellaSwag 4, MMLU 4, OBQA 4,
  PIQA 2, SIQA 3, WinoGrande 2.
- **Why accuracy.** It is a property of the models and tasks. Correct-prob,
  margin, norm correct prob and total prob have no closed-form chance level, so
  the accuracy criterion is applied to every OLMES-family metric. C4 and S&N
  bits have no chance level and get no exclusion.
- **What is excluded.** Designs whose top fitted rung is a chance rung. The test
  is recalibrated on the remaining designs in the same worlds.
- **Sensitivity.** The threshold raised to chance + 0.01.

**Signal and Noise and compute-multipliers.** Descriptive only.
- S&N has one run per cell. Its per-design excess is scored against the observed
  1B order, with a LORO jackknife interval on Fisher z and no calibration.
- Compute-multipliers is covered in Task 2.

**Predictions.**
- **P1a.** Holds on C4.
- **P1b.** Fails or reverses on correct-prob with chance rungs included.

## Task 2: replication on compute-multipliers

- **Inputs.** All seven extracted charts are listed with their content. The full
  five-budget ladder (1e17, 3.16e17, 1e18, 3.16e18, 1e19, each with +/- 1 sd)
  exists only for OLMES10 accuracy, on the recipe axis (`djBI7`, 7 recipes on
  FineWeb-Edu) and the corpus axis (`lm6Jz`, 7 corpora under OLMo 2).
- **Designs.** At least three consecutive fitted budgets and a later held-out
  target:
  - [b0..b2] -> b3;
  - [b0..b2] -> b4;
  - [b0..b3] -> b4;
  - [b1..b3] -> b4.
- **Rankers.** Single-scale (top fitted budget), projection (per-candidate power
  law) and shared-exponent, on 1 - accuracy.
- **Readouts.** Mis-selection against the observed target means. The three-way
  decomposition (introduced, corrected, shared) of each comparator against
  single-scale. The pairs resolved at the target under the calibrated
  determination rule.
  - **Lever-arm trend.** Spearman of excess on log lever arm over the 4 designs.
    Descriptive.
  - **Chance exclusion.** OLMES10 chance accuracy is 0.348; no budget falls at or
    below it, which is reported.
- **Uncertainty.** 4,000 parametric resamples of every cell mean, as
  mean + (sd / sqrt 3) x t_2.
  - The assumptions are stated: normal seed noise, independence across cells, no
    per-seed data, and sd read from chart bands.
- **No candidate-level inference.** With 7 candidates it is meaningless, so
  results are descriptive with seed-level uncertainty only, and that is stated.

**P2.** Projection does not beat single-scale on either axis. The recipe axis
leaves most pairs unresolved: its spread is about 5.4 seed SDs, against 20.7
for corpora.

## Task 3: Marin speedrun (gated)

- **Source.** `marin-community/speedrun` `data/runs.json` (118 run records,
  commit 31fe8028, 2026-05-06). The metric is `eval_paloma_c4_en_bpb`.
- **Units.**
  - A submission is the run-name prefix before the final "/".
  - A budget is the nominal size label after it, for example "130m".
  - Budgets are shared when the labels match.
- **Gate.** At least 8 submissions sharing the same at least 4 budgets.
  - If it passes: hold out the top shared budget, project from the rest with a
    per-submission power law in model FLOPs, and compare against single-scale
    ranking at the next rung down, with the same decomposition.
  - If it fails: report the counts and stop this task.

## Task 4: mechanism robustness

- **Noiseless error counts.** Per design, for both rankers, on the fixed-candidate
  real shrunk curves (C4, 55 designs). Arm A's projection exactness holds by
  construction (the truth is the projection's own functional form), and that
  is recorded.
- **Attribution variants.** The variance / common-mode / recipe-specific /
  interaction attribution of the B3 x 1 slope is recomputed with a lighter world
  (true excess at all 55 designs) under:
  - the baseline decomposition, 60 worlds per arm;
  - (a) a rank-two common mode (mean loadings on the first two SVD components),
    60 worlds per arm;
  - (b) a Chinchilla-form base fit, E + A / N^alpha + B / D^beta per recipe,
    instead of the compute power law, 60 worlds per arm;
  - (c) leave-one-recipe-out (25 variants on 24 recipes), 15 worlds per arm.
  - Common random numbers are used across variants.
- **Readout.** The range of each component across variants.

## Task 5: calibration when the simulator is wrong

- **Scope.** The calibrated procedure for projection against single-scale (the
  paper's primary comparison): C4, fit to 300M, target 530M.
- **Estimands and critical values.**
  - Random-candidate: U-statistic interval with k = 2.99.
  - Fixed-candidate: parametric bootstrap SE (50 draws, under the procedure's own
    Gaussian model) with k = 2.24.
- **Conditions,** each with 200 random worlds and 80 fixed worlds:
  - **Control:** unperturbed.
  - **Student-t noise:** df drawn from {3, 4, 5} per world, scaled to the
    measured variance.
  - **Profile flattened:** sigma(r, s) x (geometric-mean profile / profile(s))^0.5,
    renormalised to the measured RMS.
  - **Profile steepened:** the inverse exponent.
  - **Misspecification 0.5x and 1.5x:** truth = mu_P + lambda R. Both are inside
    the physical region (it ends at 1.79x on this axis).
  - **Candidate spread 2x:** recipe deviations from the mean curve doubled.
- **True values.** theta per condition is the mean true difference over that
  condition's worlds.
- **Readout.** Coverage per condition and estimand, with a binomial interval.
  Coverage below 0.90 is recorded as a limit of the procedure.
