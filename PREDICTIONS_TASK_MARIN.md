# Pre-registration: the Marin speedrun comparison made rigorous

Written 2026-10-12, before any computation in this round. No paper prose.
Laptop safety: at most 2 workers under `nice -n 19`. Refutations are reported
first.

## Why last round's Marin result is not yet usable

- A three-parameter law through three single-run points has no residual degrees
  of freedom.
- Projection's 95 of 210 wrong pairs (45%) is near chance, which a degenerate fit
  would produce.
- Projection corrects only 1 of single-scale's 46 errors, which suggests many of
  them are noise in the single-run 1.2B reference.

## Task 1: Marin's own procedure (found before this file was written)

The source is `marin-community/speedrun` @ 31fe8028: `website/src/utils/regression.ts`,
`website/src/hooks/useSpeedrunData.ts` and `website/src/App.tsx`.

- **Grouping.** A submission is the run-name prefix (`getScalingGroupName`).
- **Axis.** x = `model_flops` by default (`App.tsx`); y = `eval_paloma_c4_en_bpb`.
- **Form.** L = a x^(-b) + c, fitted to all of a submission's runs, unweighted.
- **Fitting.**
  - Levenberg-Marquardt (`ml-levenberg-marquardt` ^5.0.0): damping 1.5,
    gradient difference 1e-6, 200 iterations, error tolerance 1e-8.
  - Bounds: a in [1e-10, 1e30], b in [1e-6, 2], c in [0, 0.999 min y].
  - Initial values: c0 = min(0.7, 0.7 min y); a0 and b0 from a log-log
    regression of y - c0.
- **Forecast.** The fitted law at x = 1e22, used to rank submissions.
- **Qualification.** A submission qualifies only if a run exceeds 3e20 training
  hardware FLOPs.

The leaderboard computes these in the browser from `runs.json`; no projection
values are stored or published elsewhere.

**Reproduction.**
- Their TypeScript is run unmodified under Node with `ml-levenberg-marquardt` v5
  (installed in a scratch directory), on the 21 submissions.
- A Python reimplementation is checked against it.
- "Matches the published projections" means it matches their code's output,
  since that is what the site displays.

## Task 2: held-out 1.2B under several fits

- **Setup.** Hold out 1.2b, fit on 130m, 300m and 520m, and project each
  submission to its own 1.2b `model_flops`. Compare against single-scale ranking
  at 520m, on 210 pairs.
- **Fit forms.**
  - (i) Marin's procedure as implemented (their code, three points).
  - (ii) A two-parameter power law, linear in log-log (log L = a - alpha log x),
    least squares: 1 residual degree of freedom.
  - (iii) A shared exponent across submissions: log L = a_s - alpha log x with
    one alpha for all submissions and a per-submission intercept. It is
    identifiable with two or more points per submission.
  - (iv) The exact three-parameter fit (`asla.models.fit_power_law`), last
    round's result.
- **Readouts per form.** Wrong pairs against the observed 1.2b order;
  introduced, corrected and shared errors against single-scale.

**P-T2.** Projection loses under every form, with the gap smallest under (ii)
and (iii).

## Task 3: lever arms on Marin

- **Designs.**
  - {130m, 300m} -> 520m;
  - {130m, 300m} -> 1.2b;
  - {300m, 520m} -> 1.2b;
  - {130m, 300m, 520m} -> 1.2b.
- **Fits and comparator.** Forms (ii) and (iii); single-scale at the top fitted
  budget.
- **Lever arm.** The median over submissions of target `model_flops` over top
  fitted `model_flops`.
- **Readout.** Excess against log lever arm, descriptive over 4 designs.
- **Context.** The leaderboard itself extrapolates to 1e22. That is stated
  without claiming any error at 1e22.

## Task 4: the reference

- **Repeated runs.** Any submission with repeated runs at a budget (same prefix
  and label) gives a direct seed-SD estimate.
- **Borrowed noise, otherwise.** Seed SD is borrowed as a relative SD (SD / mean,
  since the units differ) and applied to each 1.2b BPB:
  - DataDecide C4 bits per token at 1B: the moderated per-run SD over the cell
    mean, averaged over recipes;
  - Delphi 1e21 (3 seeds) and 1e22 (3 seeds) held-out loss.
- **Determination.** A pair is determined if |gap| > z sqrt(2) sigma_pair, where
  sigma_pair is the borrowed SD at each submission's 1.2b BPB.
  - Primary: z = 1.96, with no multiplicity correction, so it is an upper bound
    on the determined fraction.
  - Also: Bonferroni over 210 pairs.
- **Expected-error scoring.** A prediction agreeing with the observed order costs
  1 - p, and one disagreeing costs p, where p = Phi(|gap| / (sqrt 2 sigma)) is
  the probability that the observed order is the true one.
- **Readouts.**
  - Task 2 rescored with expected-error scoring and on determined pairs only;
  - how many of single-scale's errors sit on undetermined pairs.

## Task 5: candidate structure and inference

- **Families,** by a mechanical rule: the first path component for nested names
  (`hackable_transformer_attn_gate/*`, `hackable_transformer_attn_sink/*`); every
  other submission is its own family.
- **Within- and cross-family pairs.** Reported separately.
- **One representative per family.** Reported for every combination of
  representative choices (mean and range), so no choice is cherry-picked.
- **Exclusions,** counted:
  - BPB missing or not finite (failed or diverged runs);
  - fewer than the 4 shared budgets;
  - a rise in BPB between consecutive budgets of more than 2 sqrt(2) sigma, with
    sigma the largest borrowed value.
- **U-statistic test.** The candidate-level test on the kernel (projection error
  minus single-scale error per pair), for each fit form.
  - No Marin calibration exists, so it is reported under the most conservative
    earlier calibrated U critical value (2.99, C4 random-candidate) and under
    1.96, and that is said.

## Small items

- **Mechanism record.** `mechanism_record.json` is updated:
  - the real-curve noiseless counts (projection 3,888 against single-scale 744
    wrong pairs, over 55 designs) become the primary mechanism evidence;
  - the attribution is recorded as dependent on the base functional form, with
    the power-law and Chinchilla values.
- **Noise profiles.** The seed-noise profile across scale is estimated for every
  metric analysed:
  - DataDecide: C4, OLMES error, correct-prob, margin, norm correct prob;
  - compute-multipliers: OLMES10 on both axes, from the chart SDs.
  - **Profile measure.** log(max RMS sigma / min RMS sigma) across scales.
  - **Flags.**
    - "Flatter than C4": the log-range is below C4's.
    - "As flat as the tested flattened profile": the log-range is at most half
      of C4's. That flattened condition halved C4's log-range, and there
      fixed-candidate coverage was 0.86.
- **Practice record.** The Task 5 perturbation bug becomes a new instance, with
  its detection path: a check that the estimate tracks theta.
