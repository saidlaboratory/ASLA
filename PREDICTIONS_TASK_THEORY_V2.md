# Pre-registered extension: three-parameter variance (Task 2, version 2)

**Status: pre-registration. Written and committed before computing the extension.**
The two-parameter result and its three misses stand unchanged in
`PREDICTIONS_TASK_THEORY.md` and `results/theory/theory_check.json`; this file
extends that work rather than replacing it. A theory that failed in a
pre-registered direction for an identified reason is a result, and deleting it
would destroy that.

Date: 2026-08-28.

## 0. What the misfit measurement changes about H1

H1 has said that projection's excess mis-selection over single-scale ranking is
"fit variance". The misfit measurement says **which kind**, and the distinction
carries opposite practical implications:

| kind of error | shrinks with more seeds? | shrinks with more budgets? | measured share here |
|---|---|---|---|
| parameter estimation noise | yes, as `1/s` | yes, via `S_uu` | seed SE 0.00866 |
| **functional-form misspecification** | **no** | only by changing the fitted range | residual 0.03655 |

Residual scatter of cell means about the fitted power law is **4.22x** the seed
standard error, a **17.8x** variance inflation. So the dominant term in projection
variance is **misspecification, which more seeds would not shrink**, not parameter
uncertainty, which they would.

**This is a refinement of H1, not a side observation.** The practical
recommendation changes: for a practitioner deciding between recipes by
extrapolation on this kind of ladder, buying more seeds per cell mostly does not
help - the curve is in the wrong family, and the error it induces is a property of
the family and the ladder, not of the sampling. What helps is reducing the
extrapolation distance, constraining the fit (pooling), or changing the functional
form. The pre-registered predictions in section 4 are the test of whether that
refined account also predicts the decision-level numbers.

**Two claims that must not be conflated.** "The excess is fit variance" is
established (289/289 specifications, 100% of 300 seed-bootstrap resamples).
"The dominant component of that variance is misspecification" is measured here on
one suite and one metric, and is what the extension tests.

**Scope of the practical inversion.** "More seeds mostly do not help" is the
sharpest actionable claim in this document and the most attackable, so its scope is
fixed here: it is established for **DataDecide's 5xC ladder, C4-EN bits per token,
and a power-law fit** - one suite, one metric, one functional form. It is *not*
established for other suites, other metrics, other ladders, or other fitted
families, and it is not established at all until the extension's V1-V3 verdicts
land. Until then it is a measured property of one grid, stated as such.

**It does not contradict our own compute ask, and here is the distinction before a
reviewer draws it.** FIRST_AUDIT.md asks for 9 seeds x 4 optimizers x 1.2B to
resolve optimizer orderings. That is not in tension with "more seeds mostly do not
help here", because the two use seeds for different questions:

| question | what seeds buy | our instance |
|---|---|---|
| **identifiability**: are two systems distinguishable *at the scale where they were measured*? | everything - the gap is compared against seed noise directly, and with one run per cell there is no noise estimate at all | the 1.2B/8xC result: noise is 207x the smallest adjacent gap, so the ordering is unresolvable at any feasible seed count. Seeds are the *only* thing that answers this |
| **extrapolation**: does an ordering fitted at small scale hold at a target nobody trained? | little, once misspecification dominates - more seeds shrink the 0.00866 term while the 0.03655 term is untouched | the DataDecide projection result: seed noise is ~19% of the residual scale |

Both are legitimate uses; they are different uses. The compute ask buys a noise
estimate where none exists, so that the *identifiability* question can be answered
at all. The inversion says that once such an estimate exists and the fit is
misspecified, buying still more of it does not fix the *extrapolation* question.
A single sentence version: **seeds tell you whether a measured difference is real;
they do not tell you whether a fitted trend will hold.**

## 0b. Three threads, one statement about DataDecide's geometry

Three separate results in this project turn out to describe the same fact:

| result | what it showed |
|---|---|
| **synthetic `saturation_crossover` scenario** (from the original harness) | a truth placed *outside* the fitted family makes the naive power-law fit confidently wrong at the target - the "fit is blind" case, constructed |
| **ensemble result** (Task B, and the sanity check) | a *more flexible* family is worse (18.67% vs 5.33%), because with realistic noise the extra freedom fits noise rather than the missing structure |
| **alpha-bounds defect** (`NOTES_FIT_BOUNDS_DEFECT.md`) | a *more rigid* fit was also wrong - exponents pinned three orders of magnitude from truth - but wrong **monotonically**, so the induced ordering survived exactly |

The single statement: **on DataDecide's ladder the power law is misspecified by a
factor of 4.22 in residual scale, and the interventions differ mainly in level
rather than curvature.** Every one of the three follows. Misspecification is why
the naive fit is blind and why the residual is 4.22x seed noise; the level-not-
curvature geometry is why a rigid, badly-biased fit still orders correctly (the
bias is common to all interventions and cancels in comparisons), and why extra
flexibility buys nothing but variance. The curvature demonstration
(`audit/curvature_boundary_demo.py`) is the control: change the geometry so
interventions differ in curvature, and the same defect reorders 69x more pairs.

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
2. **GATE (hard stop, evaluated first, before any V1-V3 number is computed).**
   With `phi = 0` and the floor held fixed, the numeric machinery must reproduce
   the analytic `sigma^2 h*` to within Monte Carlo error, defined in advance as
   **a ratio in [0.9, 1.1]** at 2000 draws. This is not a sanity check to be
   noted and moved past: **if the gate fails, the implementation is wrong, no
   V1-V3 result is computed or reported, and the failure is what gets
   reported.** The gate result is written to the output JSON before anything
   else, so a passing V1 can never be shown without it.

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

### V0 - Is `phi` a model, or a fudge factor? (falsification of V1)

A single free scalar multiplying squared residuals can fit almost anything at
these scales, so **V1 could pass by construction**. V1 is therefore not
interpretable unless V0 passes first. Two independent falsifications, both fixed
here:

**V0a - stability across designs.** If `phi` is capturing a real property of the
misspecification, one value should serve every design; if it is absorbing whatever
each design needs, it will move. Measured in advance (this is descriptive, not an
outcome): `phi` estimated per design is **0.797** (fit to 150M), **0.876** (300M),
**0.905** (530M) - a spread of 0.108.

- **PASS** if the per-design estimates span **less than 0.20**, and a *single*
  `phi` fixed at the primary design's value still satisfies V1 on the other two
  designs to within a factor of 2.
- **FAIL** if a shared `phi` cannot satisfy V1 elsewhere, i.e. each design needs
  its own value. Then `phi` is a fudge factor and V1's result is void.

**V0a-trend (added before computing, because the spread test alone is too weak).**
The three values are inside the threshold but they are **monotone**: 0.797 <
0.876 < 0.905, ordered by ladder length (10, 11, 12 budgets) and inversely by
lever arm (52.5x, 12.5x, 4.7x). A `phi` that is stable *and unstructured* is
evidence of a real property; a `phi` that is stable *but trending with design
geometry* may be partially absorbing the scale-dependence that A2 describes,
which would mean the noise model is double-counting rather than decomposing.

- **Test:** regress the per-design `phi` on `log L` and on `k`, and report the
  slope, sign, and whether the trend is monotone in each.
- **REPORTED AS WEAKER EVIDENCE** if `phi` is monotone in either geometry
  variable, even when V0a's spread test passes. In that case V1 is reported with
  an explicit caveat that `phi` co-varies with design geometry and may be partly
  absorbing heteroscedasticity.
- With only three designs this cannot be a formal significance test, and it is not
  presented as one; the direction and monotonicity are what get reported.

**V0b - held-out structure.** `phi` is estimated only from fitting-range
residuals. If it models real misspecification, the resulting variance model should
also predict residual structure it never saw: specifically, the observed residual
at the *intermediate* held-out budget (530M, excluded from the primary design's
fit and never used to estimate `phi`).

- **PASS** if predicted and observed held-out residual variance agree within a
  factor of **2**.
- **FAIL** otherwise: `phi` then describes the fitting range only and does not
  generalise, which is exactly what a fudge factor does.

**V1 is reported as CONFIRMED only if V0a and V0b both pass.** If either fails,
V1's verdict is recorded as `UNINTERPRETABLE` regardless of how well the variance
matches.

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
   yet the decision-level predictions do not improve. That locates the break at
   the variance-to-ordering step (equation 5, assumption A5), which we already
   flagged as semi-empirical because `f(Delta)` is an empirical property of which
   interventions happen to be in the suite rather than anything derived.

   **What we would do, stated in advance rather than left as a direction.** A5
   assumes the flip probability depends on the gap only through
   `Phi(-|Delta| / sd_gap)` with gaps drawn independently of the fit errors. The
   specific, testable suspicion is that **`Delta` and the projection error are not
   independent**: interventions whose curves are hardest to fit may also be the
   ones with unusual target values, so pairs with small gaps could be
   systematically the pairs with large fit errors. That correlation would make the
   integral in (5) wrong in a way no variance calibration can repair.

   The concrete next step is therefore not more theory but a measurement: compute,
   per pair, the realised projection error against the true gap, and estimate their
   correlation directly by seed bootstrap (the same machinery that measured
   `rho_ab = -0.004` for A4). If that correlation is near zero, A5's independence
   holds and the break is in the *shape* of `f(Delta)`, which we would then replace
   with the empirical joint distribution rather than the marginal. If it is not
   near zero, equation (5) must be re-derived conditioning on the gap. Either way
   the next artefact is a measured correlation, not a new assumption.
3. **The analytic limit check in section 2 fails** — the numeric machinery does not
   reproduce `sigma^2 h*` at `phi = 0`. Then nothing computed from it is
   reportable and the implementation must be fixed first.

`phi` is estimated per design from the fit residuals, never chosen to make V1-V3
land. Its estimation procedure is fixed here: least-squares match of the model's
per-budget residual variance to the observed per-budget residual variance, which
uses only fitting-range data and no target information.
