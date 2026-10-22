# Pre-registration: the 300M→520M break; the low end of the noise bracket; a certified leaderboard

Written 2026-10-22, after the config files were fetched and before any audit, fit or test. No
paper prose. Compute-free (public data only). Laptop safety: at most 2 workers under
`nice -n 19`. Refutations are reported first.

## Known before writing

**Sources.**
- Per-rung `speedrun_results.json` and the sweep scripts come from marin-community/marin at
  commit cdd4e25a8d (2026-03-25): 87 files.
  - `prism_berkeley_qwen3_scaling` and `muonremez_qwen3_scaling` were never merged. Their files
    are from the heads of closed PRs #4933 and #4932, labelled "archived submission,
    agent-generated", and include a `selection_summary.json`.
  - The commit that first added each results file is recorded in `file_commits.tsv`.
- **Seen while checking file structure:** `adamw_sweep.py` states that its optimizer configs
  were "searched & provided by Kaiyue Wen" (the Fantastic Optimizers sweep). It sets batch size
  128 at 130m and 300m and 256 at 520m and 1.2b, and changes LR, weight decay and warmup at
  520m. No other submission's config has been read.
- **Recorded fields:** total_tokens, train_batch_size, train_seq_len, num_train_steps, LR,
  warmup, lr_schedule, decay, min_lr_ratio, weight_decay, tpu_type and device, data_seed,
  tokenized_dataset, and the completion timestamp.
  - There is no code-commit field. The commit that added the file stands in for it.
  - No model-init seed field is recorded.

**Status of the two tasks.**
- **Task 2 asks to "tighten the low end".** Both methods it names produce upper bounds, which
  narrow the top of the bracket, not the bottom. The low end (about 0.05%) rests on borrowed
  values and the cross-hardware pair. If (a) relabels that pair as same-seed hardware
  divergence, the low end loses its speedrun-specific support. This is stated in advance.
- **Task 1(d) dichotomy, recorded as given.** A config change aligned with the break means
  artifact. Fantastic Optimizers reproducing the break with no config change means genuine
  scale-dependent reordering.
  - **Third case.** If Fantastic Optimizers shares the same config schedule as the
    submissions, its reproduction of the break cannot separate the two readings. The shape test
    (b) then decides.

## Task 1: diagnose the 300M→520M break

**(a) Config audit.**
- **Submissions:** every one in the 53 genuine-recipe pairs (Llama/Qwen 10, parallel_llama,
  and the 8 attn_sink submissions including std_attn).
- **Per rung:** the recorded fields above, plus tokens per parameter (total_tokens /
  model_size) and steps.
- **The change from 300m to 520m** is the 520m value over the 300m value for numeric fields,
  and changed or unchanged for categorical ones.
- **A setting is flagged** if its 300m→520m change differs across the submissions in any
  genuine pair. This is computed over all pairs, and within Llama/Qwen.
- **Provenance:**
  - derived from Fantastic Optimizers, if the sweep script or description cites it or Kaiyue
    Wen's report;
  - tuned per size, if optimizer hyperparameters differ across rungs; otherwise transferred;
  - submitted through an unmerged agent-generated PR, as recorded above.

**(b) Shape test.**
- **Flip classes.** A pair's gap sign is taken per rung at the primary tolerance (0.00142 in
  log BPB).
  - Persistent: exactly one sign change, located between 300m and 520m.
  - Reverting: two or more changes.
  - Other: no change, or one change elsewhere.
- **Submission-level step,** under the Task-1 bound (per rung, relative):
  - fit log y = a + b log x + d·1[rung >= 520m] by weighted least squares (1 residual df),
    giving d_hat and its SE from the known sigma;
  - a step is "detected" at |d_hat / se| > 1.96;
  - also reported: the spread of d_hat across submissions against the SE. A common step
    cancels in pairs; only a submission-specific step can reorder.
- **Pair-level shape.** For each pair that changes sign between 300m and 520m, the log-gap
  series g at the four rungs is fitted two ways, both with 2 parameters and 2 residual df:
  - step: g = c + d·1[>= 520m];
  - steady trend: g = c + b·log x (at that pair's mean log FLOPs per rung).
  - The model with the lower SSE is preferred. Readouts: the count preferring the step, and a
    binomial test against 1/2.

**(c) Fantastic Optimizers corroboration.**
- **Fitted optimum** per (optimizer, size), at budget multiple 1 (the speedrun's 20x params):
  - a quadratic in the slice coordinate on each one-parameter slice with at least 3 points
    inside the 2% window;
  - pooled baseline estimate: the mean of the slices' fitted values at the baseline
    coordinate;
  - optimum = pooled baseline + the sum over slices of min(0, min over the observed range of
    [fit(z) - fit(z_base)]);
  - an optimizer-size cell enters only if at least 2 slices qualify.
- **Pairs:** optimizers present at both 300m and 520m.
- **Under the bound** (300m and 520m values, used as if each optimum were a single run, which
  is conservative):
  - determined-pair reorder rate, determined at 520m with the sign taken at 300m;
  - both-rung genuine count.
- **Same for 130m→300m and 520m→1.2b,** where present, as the comparison.
- **Independence,** stated from (a): the speedrun configs are copies of Fantastic Optimizers'
  configs if (a) finds that. Fantastic Optimizers' runs are separate runs, so they are
  independent in run noise but not in config schedule.

**(d) Predictions.**
- **PA1.** In at least 8 of the 10 Llama/Qwen submissions, batch size changes between 300m and
  520m, and optimizer hyperparameters are tuned per size.
- **PA2.** At least one setting is flagged: its 300m→520m change differs across genuine pairs
  (at minimum between the Llama/Qwen and the hackable-transformer submissions).
- **PA3.** Fantastic Optimizers' fitted optima reproduce the break: a reorder rate across
  300m→520m of at least 15% under the bound. Fantastic Optimizers shares the config schedule
  (the third case), so this does not by itself separate the readings.
- **PA4.** Among pairs that change sign between 300m and 520m, the step is preferred in more
  than half. At least 3 submissions have |d_hat / se| > 1.96. The overall reading is the
  artifact one: a config schedule aligned with the break.

## Task 2: the low end

**(a) The GPU/TPU pair.**
- Every recorded field of `parallel_llama/old_scaling/{gpu,tpu}` is diffed at each size,
  including data_seed. The sweep scripts are searched for seed settings.
- **Relabelling:** if the seeds are equal, or both unset (so the same default), the value
  becomes "same-seed hardware divergence", not seed noise.
- **PG1.** The seeds are equal or unset (the default), so the pair is relabelled.

**(b) Delphi forecast bound.**
- **Points:** the fitted isoFLOP optima (loss_star at C) at 3e18, 9e18, 3e19 and 1.8e20 for
  AdamH and Cautious AdamC. Fit with the corrected leaderboard form; forecast 1e21 and 1e22.
- **Truths:**
  - AdamH: seed means of 3 at 1e21 and 1e22.
  - Cautious AdamC: single runs at 1e21 and 1e22.
- **Sensitivity:** s_i = d ln(forecast) / d ln(y_i), by finite difference through the
  corrected solver.
- **Conservative bound (valid):** all realized error is attributed to the truth's noise and
  misspecification. Fitted optima pool 5-15 runs each, so their noise is at most a single
  run's.
  - sigma^2 <= n_t e_t^2 in expectation.
  - Per fit, q_j = sum over t of n_t e_t^2 / 2, so each fit counts once.
  - sigma_up^2 = sum_j q_j / (chi2_{0.05, m} / m), with m = 2 independent fits. It is one-sided
    95%.
- **Equal-noise version (secondary, not a valid upper bound):** the points carry the same
  sigma as single runs, so Var(e_t) = sigma^2 (sum_i s_i^2 + 1/n_t). Same df rule.
  - Not valid because fitted optima are less noisy than single runs, which makes it too
    tight.
- **PG2.** The conservative bound exceeds 0.4%, so it does not narrow the bracket. The
  equal-noise version lies in [0.03%, 0.2%].

## Task 3: certified leaderboard

**(a) Rule.** A pair's order at the target is certified at noise level sigma only if both hold:
1. **Noise propagation:** each submission's points are multiplied by (1 + sigma N(0,1)) and
   refitted with the design's primary form. The order holds in at least 95% of draws.
2. **Form-robust:** the order is the same under the primary form and the alternative form
   (three-parameter optimum against two-parameter log-log) at the target.

The nominal false-certification rate is 5% per certified pair; that is the upper limit implied
by the 95% rule.

**(b) Validation where truth exists.**
- **Noise grid:** sigma in {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.43, 0.7, 1.0}%.
- **Draws:** 400 per sigma, seed 20261022.
- **DataDecide:**
  - each recipe and seed index is a pseudo-submission, one seed per rung, with fit rungs 150M,
    300M, 530M and 750M (x = compute);
  - primary form: the corrected three-parameter fit (the patched solver);
  - alternative: two-parameter log-log;
  - target: 1B, with the 3-seed mean as truth;
  - three replicate validations (seed index 0, 1, 2), 300 pairs each.
- **Marin:**
  - fit 130m, 300m and 520m, forecast 1.2b at each submission's own 1.2b FLOPs, with the
    observed 1.2b as truth;
  - primary form: two-parameter log-log; alternative: the three-parameter fit (patched solver);
  - 210 pairs.
- **Readouts per sigma:**
  - certified pairs;
  - realized false-certification rate: certified pairs whose order is opposite to the truth's,
    over all certified pairs.
- **Noise estimate:** sigma* is the smallest grid sigma at which the realized rate is at most 5%,
  interpolated log-linearly between grid points.
  - It absorbs misspecification and extrapolation error, so it is an effective noise level for
    this forecasting distance, an upper-bound-like quantity.

**(c) Applied at 1e22.**
- The corrected leaderboard, with primary form the patched three-parameter fit and the
  alternative log-log. The core is the 175 pairs.
- Run at sigma* from Marin's validation (primary) and from DataDecide's (secondary), with 1,000
  draws.
- **Reported:** certified pairs and the certified partial order. Per submission: the
  submissions it is certified above and below, plus the maximal antichains at the top.

**Predictions.**
- **PC1.** DataDecide sigma* lies in [0.05%, 0.3%].
- **PC2.** Marin sigma* is at least DataDecide's.
- **PC3.** At Marin's sigma*, at most 120 pairs are certified at 1e22, and no submission is
  certified above all others.
