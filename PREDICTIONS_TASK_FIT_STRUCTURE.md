# Pre-registration: fit-structure results, and the floor profile-likelihood fix

Written 2026-09-22, before `scripts/run_floor_profile.py` exists or has been run.

## Why

The fit-structure findings (misspecification, amplitude sharing, monotone
distortion, floor identifiability, checkpoint critique, resolvability, metric
instability, the design theorem) were argued in part as causes of a
decision-accuracy gap. The expected-error round showed that gap is not
significant once candidate dependence is respected (E1 refuted). Each finding
must therefore stand on fit-level evidence alone, or be reframed.

The floor-identifiability result (`audit/floor_identifiability.py`) has three
defects in its "profile":

1. At each fixed floor, `(A, alpha)` come from a regression in log space, so the
   reported SSR is not minimised in the response scale. It is not a profile.
2. The admissibility band is `n * sigma_run^2`, but the fitted points are
   three-seed cell means, whose variance is `sigma_run^2 / 3`.
3. The band is read off a 60-point grid, not solved for.

## Method, fixed now

For each metric and each intervention, on the full-ladder design (fit 4M..300M,
n = 11 cell means, target 1B):

- **Profile.** For each floor `E`, minimise the response-scale SSR over
  `(A, alpha)` by nonlinear least squares, started from the log-space solution.
- **Primary interval (F-based, Bates & Watts).**
  `{E : (SSR(E) - SSR_min) / (SSR_min / (n - 3)) <= F(1, n - 3; 0.95)}`, with the
  endpoints found by root-finding on the profile, and `E` restricted to
  `[0, min y)`. This uses residual variance and so remains honest under the
  misspecification already documented on C4.
- **Secondary interval (known sigma).**
  `{E : (SSR(E) - SSR_min) / sigma_mean^2 <= chi2(1; 0.95)}` with
  `sigma_mean^2` the pooled seed variance divided by the seeds per cell. It is
  valid only if the model is correct and is reported, not used for verdicts.
- **Conditioning.** The condition number of `J^T J` at the joint least-squares
  optimum, in the parameterisation `(E, log A, alpha)`.

## Predictions

- **F1 (C4 floor identified).** The primary interval excludes `E = 0` for at
  least 23 of 25 interventions on `c4_en_bits_per_token`.
- **F2 (accuracy floor unidentified).** The primary interval includes `E = 0`
  for at least 20 of 25 interventions on `olmes_macro_error`.
- **F3 (conditioning).** The median condition number on `olmes_macro_error` is
  at least 100 times the median on `c4_en_bits_per_token`.
- **F4 (the fix tightens C4).** The median primary-interval width on C4 is
  narrower than the committed median admissible width.

Any refutation is reported as such, in the results JSON and in the paper.

## The other fit-structure findings

Each is re-read from its committed JSON by `scripts/run_fit_structure_check.py`,
against the criterion stated in its own source. For each, the output records
whether its evidence uses the target-scale reference labels at all. A finding
that does not is unaffected by the expected-error re-scoring. Any sentence that
presents a finding as the cause of a decision-accuracy gap is reframed as a
statement about the fit.
