# Curvature gate: 1B target

Target 7.062e+20 FLOPs, 12 fitting rungs, 25 recipes. Scored against `PREDICTIONS_TASK_CURVATURE.md`.

## Pre-registered predictions

| prediction | verdict | evidence |
| --- | --- | --- |
| C1 (quadratic consistently signed) | **CONFIRMED** | 25/25 share a sign (sign test p = 6.0e-08); mean coefficient -1.02e-03 |
| C2 (curvature predicts overshoot) | **CONFIRMED** | Spearman -0.887 (p = 3.5e-09) |
| C3 (quantitative recovery) | **REFUTED** | observed overshoot +0.0877, quadratic implies -0.0335; -38% recovered, i.e. the wrong sign |
| C4 (gap centring error) | **NOT EVALUATED** | gate stopped after 4a (MECHANISM REFUTED, PHENOMENON REAL); 4b-4d were pre-registered as conditional on it |
| C5 (gap intervals valid and narrower) | **NOT EVALUATED** | gate stopped after 4a (MECHANISM REFUTED, PHENOMENON REAL); 4b-4d were pre-registered as conditional on it |
| C6 (abstention falls) | **NOT EVALUATED** | gate stopped after 4a (MECHANISM REFUTED, PHENOMENON REAL); 4b-4d were pre-registered as conditional on it |
| C7 (loading CV predicts scope) | **NOT EVALUATED** | gate stopped after 4a (MECHANISM REFUTED, PHENOMENON REAL); 4b-4d were pre-registered as conditional on it |

## Why the gate stopped despite C1 and C2 passing

C1 and C2 establish that residual curvature is systematic and tracks the overshoot. C3 refutes it as the *mechanism*: the fitted quadratic extrapolates to an undershoot where an overshoot is observed. Comparing rival explanations on how much absolute projection error each removes settles it.

| fit | mean signed error | mean absolute error | overshooting |
| --- | --- | --- | --- |
| power law (compute only) | +0.0877 | 0.0928 | 24/25 |
| quadratic in log-log | +0.0691 | 0.0724 | 24/25 |
| + log(tokens/param) | -0.0363 | 0.0371 | 1/25 |

Taking the curvature hypothesis at face value removes **22%** of the absolute error; adding tokens-per-parameter removes **60%**.

## The actual source: a ladder confound

DataDecide's tokens-per-parameter is not constant along the ladder. It holds near 100 across 9 rungs and then drifts over the remaining 3, reaching **85.0** at the target. On the fitting ladder it correlates with log compute at **-0.80**, so a compute-only fit attributes one effect to the other and mis-extrapolates.

The confound is common-mode. The compute-only and tokens-aware projections agree at Spearman **0.9977** over 300 pairs, with mis-selection 3.00% against 2.00%. It shifts the projected level without reordering recipes, so selection results are unaffected and the coverage and centring results are what it explains.

The log-log quadratic coefficient is positive in 25 of 25 recipes, so the curves genuinely are convex. Convexity is simply not what drives the error.
