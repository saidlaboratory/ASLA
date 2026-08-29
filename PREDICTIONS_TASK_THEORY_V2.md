# Pre-registered extension: three-parameter variance (Task 2, version 2)

**Status: pre-registration. Written and committed before computing the extension.**
The two-parameter result and its three misses stand unchanged in
`PREDICTIONS_TASK_THEORY.md` and `results/theory/theory_check.json`; this file
extends that work rather than replacing it. A theory that failed in a
pre-registered direction for an identified reason is a result, and deleting it
would destroy that.

Date: 2026-08-28.

## 1. The floor is one object appearing in three places

The parameter the two-parameter theory omits is the irreducible-loss floor `E`.
The same parameter is already the subject of two other findings in this project,
and stating them as one thread is a stronger claim than three separate
observations:

| where it appeared | what happened | measured value |
|---|---|---|
| **alpha-bounds defect** (`NOTES_FIT_BOUNDS_DEFECT.md`) | freeing the exponent bound moved the pathology onto the floor: 25/25 OLMES fits then pinned `E` at zero | 25/25 interventions at the bound |
| **identifiability result** (`AUDIT_ADVERSARIAL.md`) | the likelihood is *exactly flat* in `E` on accuracy metrics, so three-parameter extrapolation is unidentifiable there | `SSR(E=0)/SSR(best) = 1.0000` vs 10.03 on C4 |
| **theory failure mode** (`PREDICTIONS_TASK_THEORY.md`, A1) | the derivation assumes `E` known; the pipeline fits it, and the theory under-states projection variance | 3.2x variance shortfall |

**The single statement:** the floor is simultaneously the parameter a projection
theory must model to get magnitudes right, and the parameter that cannot be
estimated at all on a whole metric family. Where it is identifiable (C4 bits per
token) it is the missing term in the variance; where it is not (downstream
accuracy) no amount of theory repairs the extrapolation, because the quantity the
theory needs is not in the data. These are not three problems with a fit; they are
one property of the functional form meeting one property of the metrics.

## 2. Decision 1: analytic or numeric — ANSWERED EMPIRICALLY, and it changed the plan

I intended to attempt the delta method: `Var(g(theta)) ~= grad_g^T Cov(theta)
grad_g` with `Cov` from the three-parameter fit's Jacobian. **I computed it first
as a feasibility check, and it does not work**, which redirects the whole
extension:

| quantity | mean log-space variance |
|---|---|
| two-parameter formula `sigma^2 h*` | 1.840e-06 |
| **three-parameter delta method** | **1.060e-06** |
| empirical seed-bootstrap | 5.329e-06 |

The delta method gives **0.58x** the two-parameter value and leaves **5.0x**
unexplained — it moves in the *wrong direction*. Adding the third parameter does
not close the gap, so **the missing variance is not the parameter count.**

Diagnosis, measured: the power law **does not fit the DataDecide cell means within
seed noise**. Residual scatter of cell means about the fitted curve is **0.03655**
against a seed standard error of **0.00866** — a ratio of **4.22x**, i.e. a
**17.8x** variance inflation. The dominant error term is *model misspecification*,
not parameter estimation. Substituting the misfit residual scale for the seed SE
in the same formula gives 3.182e-05, which **overshoots** the empirical value by
5.97x.

So the truth is bracketed: seed-SE noise under-states projection variance by 3.2x,
misfit-scale noise over-states it by 6.0x. The seed bootstrap perturbs only seeds,
so it captures neither pure limit — it holds the misfit pattern fixed while
resampling within cells.

**Decision: numeric characterisation with an analytic limit check.** Specifically:

1. Characterise projection variance numerically under a noise model that mixes the
   two components, `sigma_eff^2 = sigma_seed^2 + phi * sigma_misfit^2`, with `phi`
   estimated from data rather than chosen.
2. Validate the numeric machinery against the analytic two-parameter case as a
   known limit: with `phi = 0` and a floor held fixed, the numeric result must
   reproduce `sigma^2 h*` to within Monte Carlo error. **If it does not, the
   numeric implementation is wrong and no result from it is reportable.**

Reason for numeric over analytic: the failure is misspecification, which has no
closed form in general. A delta-method expression would be exact algebra about the
wrong quantity.

## 3. Decision 2: handle A2 at the same time — YES, and here is why

A2 (homoscedasticity) is violated 7.9x, with noise falling systematically with
scale. It must be handled jointly rather than sequentially, because the two defects
interact through the same weights: misfit residuals are *also* scale-dependent
(the curve fits the mid-ladder better than the ends), so a variance model that
corrects one while assuming the other is uniform will mis-attribute the residual
between them. Fixing them separately risks fitting `phi` to absorb
heteroscedasticity, or vice versa.

Concretely, the noise model is per-budget rather than pooled:

```
sigma_eff,i^2 = sigma_seed,i^2 + phi * r_i^2
```

with `sigma_seed,i` the measured per-budget seed SE (already known to span 7.9x)
and `r_i` the per-budget misfit residual. `phi` is a single scalar estimated once
per design, so the model adds exactly one free parameter over the two-parameter
theory.

## 4. Pre-committed predictions

Target to beat: the two-parameter theory's **3.2x** variance shortfall (empirical
5.329e-06 vs predicted 1.666e-06).

### V1 - Variance calibration

The extended model reproduces the empirical projection variance within a factor of
**1.5x** (i.e. ratio in **[0.67, 1.5]**), against the two-parameter model's 0.31x.

- **CONFIRMED** if the ratio falls in [0.67, 1.5].
- **PARTIAL** if it falls in [0.5, 2.0] but outside [0.67, 1.5] — an improvement on
  3.2x but not calibration.
- **FAILED** if outside [0.5, 2.0], or if it is *further* from 1.0 than the
  two-parameter model's 0.31x.

### V2 - T1 re-test (lever-arm ratio)

With calibrated variance, the predicted excess-flip ratio between the 52.5x and
4.7x arms should approach the measured **5.28**.

- Committed prediction: **3.5-7.0**, versus the two-parameter prediction of 2.545
  which missed low by 2.07x.
- **CONFIRMED** if the new prediction lands in [3.5, 7.0]; **MISSED** otherwise,
  with direction reported.

### V3 - T3 re-test (pooling retrodiction)

The two-parameter theory over-predicted the pooling benefit: 67.2% [64.9, 69.8]
against a measured 62.5%. Pooling reduces only the *slope* variance; if a large
share of the total is misspecification, which pooling does not remove, the
predicted benefit must **fall**.

- Committed prediction: the extended model predicts a reduction of **50-63%**,
  i.e. strictly lower than 67.2% and bracketing the measured 62.5% from below.
- **CONFIRMED** if the point prediction lies in [50, 63] and is closer to 62.5 than
  67.2 was.
- **FAILED** if the extension predicts a *higher* reduction than the two-parameter
  version, which would mean misspecification variance is being attributed to the
  slope.

### V4 - What would count as the extension failing overall

Stated in advance:

1. **V1 FAILED** — the extended model is not better calibrated than the one it
   replaces. Then the extra parameter buys nothing and should be reported as a
   dead end, not tuned further.
2. **V1 CONFIRMED but V2 and V3 both MISSED** — variance is calibrated in aggregate
   yet the decision-level predictions do not improve. That would mean the route
   from variance to ordering errors (equation 5, assumption A5) is where the
   theory actually breaks, not the variance model, and the next work is there.
3. **The analytic limit check in section 2 fails** — the numeric machinery does not
   reproduce `sigma^2 h*` at `phi = 0`. Then nothing computed from it is
   reportable and the implementation must be fixed first.

`phi` is estimated per design from the fit residuals, never chosen to make V1-V3
land. Its estimation procedure is fixed here: least-squares match of the model's
per-budget residual variance to the observed per-budget residual variance, which
uses only fitting-range data and no target information.
