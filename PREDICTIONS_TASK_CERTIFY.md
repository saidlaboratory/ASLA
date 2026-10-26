# Pre-registration: Stage 2, the certifying selector

Written 2026-10-26, before any Stage 2 code is written or run. The DataDecide and Marin analyses are frozen; this is
the only new analysis. Laptop safety: `nice -n 19`, `.venv/bin/python`, no job over 30 minutes without
checkpointing (per-design results are cached). Refutations are reported first.

## The selector (one specification, applied unchanged to both datasets)

**Algorithm (CERTIFY).** Every free choice below is fixed here.

```
input:  candidates i = 1..n; fit rungs r with compute C_r, per-rung values y_ir and standard errors s_ir;
        target compute C_T; form set F; level alpha; misfit coefficients kappa_f (from CALIBRATE)
for each form f in F and candidate i:
    yhat_if   <- forecast at C_T from the form-f fit to (C_r, y_ir)
    g_if      <- d yhat_if / d y_ir   (finite differences, relative step 1e-4)
    L_i       <- log10(C_T / max_r C_r)           (lever arm, decades)
    v_if      <- sum_r g_ifr^2 s_ir^2 + (kappa_f * L_i * yhat_if)^2
for each pair (i, j):
    certify "i before j" iff for every f in F:
        yhat_if < yhat_jf  and  Phi((yhat_jf - yhat_if) / sqrt(v_if + v_jf)) >= 1 - alpha
    otherwise abstain
output: the certified partial order, and the possibly-best set = candidates certified worse than no one

CALIBRATE(form f, calibration predictions P):
    each p in P: a fit window W, a held-out rung h strictly above W and strictly below the evaluation target,
                 forecasts yhat_ih from the form-f fit to W, observed y_ih with standard error s_ih
    r_ih  <- (yhat_ih - y_ih) / y_ih - mean_i[(yhat_ih - y_ih) / y_ih]       (common mode removed)
    u_ih  <- [sum_r g_ifr^2 s_ir^2 + s_ih^2] / y_ih^2                        (seed part, relative)
    kappa_f^2 <- max(0, sum (n/(n-1) r_ih^2 - u_ih) / sum L_ih^2)
```

**Fixed choices.**
- **Level:** alpha = 0.05.
- **Forms:** F = {two-parameter log-log in compute; three-parameter y = a C^-b + c}.
  - Three-parameter: least squares in (log a, b, c) with b in [1e-6, 2], c in [0, 0.999 min y], tolerances 1e-12,
    starts c0 in {0, 0.5, 0.9} x min y, keeping the minimum SSE.
- **Draws.** "Holds in at least 1 - alpha of draws" is evaluated in its exact Gaussian limit under first-order
  propagation: Phi(margin / sd) >= 1 - alpha.
  - **Check:** on Marin and on one DataDecide design per metric, the certification decisions are compared with
    400 full-refit draws, and the agreement is reported.
- **Misfit model:** recipe-specific relative SD kappa_f x L, linear in lever arm.
  - The common mode is removed because it cancels in pair orders.
  - kappa_f is estimated per form, only from held-out rungs below the evaluation target, never from the target or
    the pairs being evaluated.
- **Seed noise.**
  - **DataDecide:** standard error of the three-seed mean. The per-cell variance is moderated across the 25
    recipes at each scale (`asla.analysis.moderation`, the paper's noise model).
  - **Marin:** relative SD at the Task-1 bound per rung, the upper end of the bracket.
    - Sensitivity: the low end (DataDecide 1B, 0.0511%).

## Data, designs and leave-one-ladder-out calibration

**DataDecide.**
- **Metrics (committed tables):** c4_en_bits_per_token, olmes_macro_error and
  olmes_macro_correct_prob_per_char_deficit; 25 recipes and 3 seeds.
- **Ladders:** the nine fit windows of the specification curve.
  - Start at 4M, 14M or 60M; end at 150M, 300M or 530M.
  - Every scale between start and end inclusive, excluding 750M.
- **Designs:** (window, target) for targets 1B and 530M, with target compute above the window's top: 15 designs
  per metric, 45 in all.
- **Leave-one-ladder-out.** For design (window l, target T), the calibration predictions are all (window l', rung
  h) with l' != l and C_top(l') < C_h < C_T, same metric. Rungs are ordered by compute; 750M (less compute than
  530M) qualifies as a rung h when it lies in that interval.
- **Truth:** the three-seed mean at T, with moderated standard error, for the reference-uncertainty treatment.

**Marin.**
- **Design:** the 22 official submissions; fit 130m, 300m and 520m; target each submission's own 1.2b compute.
- **Calibration:** fit {130m, 300m}, held-out 520m. One ladder only, so leave-one-ladder-out is not possible;
  stated.
  - Sensitivity: kappa transferred from DataDecide C4 (all its calibration predictions pooled).
- **Truth:** the observed 1.2b BPB with standard error = the 1.2b bound x BPB.

## Baselines

- **B1 single-scale ranking:** order by the top fit rung. Always decides.
- **B2 projection:** three-parameter point forecast. Always decides.
- **B3 seed-SE certification:** CERTIFY with kappa_f = 0, same form set.
- **B4 bootstrap-interval projection.**
  - Three-parameter form; 2,000 seed resamples per cell (with replacement within cell), propagated through the
    same first-order map. Certify if the order holds in at least 1 - alpha of resamples.
  - Marin has no seeds: parametric draws at the bound, labelled.
- **B5 conformal projection.**
  - Split conformal on the same leave-one-ladder-out calibration predictions: score s = |relative residual| / L;
    q = the ceil((n+1)(1-alpha))-th smallest score.
  - Interval yhat (1 +/- q L), three-parameter form. Certify "i before j" iff upper_i < lower_j.
  - Marin: calibrated on its 520m hold-out.

## Evaluation

**Per design and method:**
- **Coverage:** certified pairs over all pairs.
- **Realized false-certification rate (reference-uncertainty treatment).** Each certified "i before j" is charged
  p_ij = Phi((m_i - m_j) / se_ij), the probability that the truth is the other way given the target data. FCR =
  sum of charges / number certified. Always-deciding methods are charged over every decided pair.
- **The possibly-best set:** its size; whether it contains the observed best; and P(true best in set), from
  4,000 draws of the truth.

**Pooled per metric and over all DataDecide designs:**
- FCR = sum of charges / sum certified; coverage = mean coverage.
- **Candidate-level interval (the Stage-1 test).** Kernel k_ij = mean over designs of cert_ij (p_ij - alpha); the
  U-statistic mean, with an interval at the calibrated conservative critical value (calibration.json
  random-candidate critical value).

**By lever arm:** every quantity per design against L, and in bins L < 0.5, 0.5 to 1, and 1 or more decades.

## Success criterion (pre-registered as given)

Over all DataDecide designs, leave-one-ladder-out:
- **S1:** the selector's pooled realized FCR <= alpha.
- **S2:** the selector's pooled coverage is higher than that of every baseline whose pooled FCR <= alpha.

**Confirmed iff S1 and S2.** Per-metric results are reported, but the criterion is the pooled one. Marin is
reported and is not part of the criterion (no leave-one-ladder-out).

**What failure looks like:**
- **F1, miscalibration:** pooled FCR > alpha.
  - F1a: FCR rising with lever arm, meaning the linear misfit model is too flat at long extrapolation.
- **F2, no advantage:** a baseline meets nominal with equal or higher coverage.
- **F3, degenerate:** S1 and S2 hold but the selector's pooled coverage is below 10%. This is reported as
  practically useless.

## Predictions

- **PS1.** Selector pooled FCR <= alpha, at about 1-3%.
- **PS2.** On C4 alone, single-scale ranking meets nominal at full coverage, so the per-metric version of S2 fails
  on C4.
  - Pooled over metrics, single-scale FCR exceeds alpha, driven by the OLMES metrics, so it is excluded from the
    pooled S2.
- **PS3.** Projection (B2) pooled FCR > alpha.
- **PS4.** Seed-SE certification (B3) and bootstrap (B4) exceed alpha pooled, with FCR rising in lever arm.
- **PS5.** Conformal (B5) meets alpha with lower coverage than the selector.
- **PS6.** S1 and S2 pooled: confirmed (about 55% confidence; the decisive comparison is the selector against
  conformal).
- **PS7.** Marin: the selector certifies few pairs (at most 30 of 231) with FCR <= alpha. Single-scale FCR > alpha.
