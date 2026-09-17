> **Post-hoc correction (recorded, not silently edited).** The constants table
> below defines `sigma = mean relative SE` = 0.00145. That definition is the
> origin of what the methods practice records as instance 6: equation (1) and
> everything downstream consume `sigma^2`, and squaring a linear mean of scales
> understates the mean of their squares --- here by 2.06x, the same factor at
> which the independent `phi = 0` gate in the v2 work failed. The correct pooling
> is `sqrt(mean(relative^2))` = 0.00208, now computed in
> `scripts/run_theory_check.py` and shipped beside the mean with the
> understatement factor.
>
> **The defect was in this specification, not in the implementation** --- the code
> did faithfully what this file asked for, which is why no test caught it and why
> a sweep for the mathematical pattern was needed to find it.
>
> No verdict changes. T1 is sigma-free (a ratio of leverages); T2 and T3 compare
> a measured quantity against a band committed here, and all three remain MISSED.
> What weakens is the diagnosis: the empirical-over-theory variance ratio falls
> from 3.20 to 1.55, so roughly half of what was attributed to the unmodelled
> third parameter was this pooling error. The omission remains real and
> directionally right, but it no longer accounts for the T2 miss on its own.

# Pre-registered theory and predictions: variance of projected selection (Task 2)

**Status: pre-registration. The derivation and every numeric prediction below were
written and committed before comparing against the measured dose-response.** The
only quantities computed beforehand are the design constants in section 1, which
describe the ladder and are inputs to the algebra, not outcomes.

Date: 2026-08-28.

Ye et al. (arXiv:2409.15156) document a scaling-law extrapolation misplacing a
crossover by an order of magnitude and state that naive extrapolation for model
comparison lacks a solid theoretical foundation. This is an attempt at that
foundation for the decision problem specifically. **Verify their claim before
citing it; it is used here only as motivation, not as evidence.**

Notation: `k` fitted budgets `C_1..C_k`, `s` seeds per cell, per-run noise `sigma_run`,
target `C*`, lever arm `L = C* / C_max`. Write `u = log C`, `u_i = log C_i`,
`ubar = mean(u_i)`, `S_uu = sum_i (u_i - ubar)^2`.

## 1. Measured design constants (inputs, not predictions)

For the primary DataDecide design (C4 bits/token, 11 budgets 4M-300M, target 1B):

| quantity | value |
|---|---|
| `k` | 11 |
| `ubar` | 40.448 |
| `S_uu` | 81.037 |
| `log L` | 2.524 (L = 12.5x) |
| OLS leverage at target `h* = 1/k + (u* - ubar)^2 / S_uu` | 0.0909 + 0.7051 = **0.7960** |
| log-space noise on a cell mean, `sigma = mean relative SE` | **0.00145** |
| true target gaps between interventions (300 pairs) | median 0.1754, mean 0.2383, sd 0.2081 |
| fraction of pairs with gap < 0.01 | 0.040 |

## 1b. SCOPE OF THE THEORY (stated up front, not as a closing caveat)

**This theory applies to metrics whose irreducible-loss floor is identifiable. On
DataDecide that is C4 bits/token and not the downstream accuracy metrics.**

The derivation linearises in log space, which requires the floor `E` to be known
(assumption A1 below). Our own identifiability result measures exactly how far
that fails on accuracy metrics: profiling the likelihood in `E` gives
`SSR(E=0)/SSR(best) = 1.0000` on OLMES macro error - the likelihood is *flat*, so
`E` is not merely imprecise but unidentified. On C4 bits/token the same ratio is
10.0x and the floor is well determined.

The two results interlock, and each strengthens the other: **the parameter the
theory needs known is precisely the parameter that is unidentifiable on accuracy
metrics.** So the scope limit is not an inconvenience of this derivation, it is a
restatement of a measured property of those metrics. Any attempt to extend this
theory to accuracy metrics must first supply a floor from outside the data.

## 2. Derivation

### 2.1 Step 1: variance of the projected value (DERIVED, standard OLS algebra)

Work with the two-parameter form in log space, which is exact:

```
log(y - E) = log A - alpha u        [linear in (log A, alpha)]
```

**Assumption A1 (LOAD-BEARING):** the floor `E` is known. This is where the
three-parameter form breaks, and it is not a technicality - our own
identifiability result shows the likelihood is *exactly flat* in `E` on accuracy
metrics (`SSR(E=0)/SSR(best) = 1.0000`), so on those metrics this derivation does
not apply at all. On C4 bits/token the floor is identified (ratio 10.0x), so the
approximation is defensible there and nowhere else in our data.

Under A1, write `z_i = log(y_i - E)`, model `z_i = beta_0 + beta_1 u_i + eps_i`
with `Var(eps_i) = sigma^2` (see A2). Standard OLS prediction variance of the
fitted mean at a new point `u*`:

```
Var(zhat*) = sigma^2 [ 1/k + (u* - ubar)^2 / S_uu ]                       (1)
```

Both terms are derived, not assumed. Now substitute `u* - ubar = log C* - ubar`.
Writing `log C* = log C_max + log L`:

```
(u* - ubar)^2 = (log C_max - ubar + log L)^2
              = (log C_max - ubar)^2 + 2 (log C_max - ubar) log L + (log L)^2   (2)
```

So **the projected-value variance is quadratic in `log L`**, with leading
coefficient `sigma^2 / S_uu`. That is the derivation's central claim:

```
Var(zhat*) ~= sigma^2 [ 1/k + (d + log L)^2 / S_uu ],   d = log C_max - ubar    (3)
```

**Assumption A2 (MEASURED, and it is VIOLATED):** homoscedastic errors in log
space. Independence *across* budgets is reasonable (cell means at different
budgets come from different runs). Homoscedasticity is not: the measured
log-space noise per budget ranges from 0.00047 to 0.00366, a **7.9x spread**, and
it falls systematically with scale - the largest, most informative budgets are
also the quietest.

The consequence is directional and predictable. Equation (1) uses one `sigma` for
all budgets, so it under-weights the precise high-compute anchors and over-weights
the noisy small ones. A design that drops the top of the ladder (the long-arm
design, fit only to 150M) loses its *quietest* points, so its true variance rises
by more than the leverage term alone predicts. **The theory should therefore
under-predict the long-arm penalty**, i.e. under-predict the excess-flip ratio
between long and short arms. That is a pre-committed statement about the direction
of the T1 miss.

A2 also fails for checkpoint-augmented fitting (rho = 0.43-0.80 serial
correlation), which the theory does not cover at all.

**Assumption A3:** `sigma^2` scales as `sigma_run^2 / s` with `s` seeds per cell.
Standard, and it is why more seeds and more budgets both help, but only `k` and
the spacing enter `S_uu`.

### 2.2 Step 2: from variance to ordering errors (PARTLY ASSUMED - flagged)

A pair `(a, b)` is ordered wrongly when the projected gap has the opposite sign to
the true gap `Delta`. With independent projections,

```
Var(gap) = 2 Var(zhat*)     [A4: the two interventions' fit errors are independent]
P(flip | Delta) = Phi( -|Delta| / sqrt(2 Var(zhat*)) )                        (4)
```

**Assumption A4 (MEASURED, and it holds):** independence of the two
interventions' fit errors. I expected this to fail - interventions share a budget
ladder, so their fit errors ought to be positively correlated, which would make
(4) over-predict flips. **Measured, it does not fail.** Seed-bootstrapping the
projections (120 draws, 25 interventions, 300 pairs) gives a mean cross-intervention
correlation of projection errors of **rho_ab = -0.004** (median -0.002, range
-0.234 to +0.312), and the ratio of the actual `Var(gap)` to its
independence value is **1.002**. Independence inflates the gap SD by a factor of
0.999, i.e. not at all.

This matters for how the results below must be read: **A4 cannot be used to
explain away a miss.** If T1 or T2 misses, the cause lies elsewhere, and the
candidate is A2 (see below), which *is* violated.

**Assumption A5 (LOAD-BEARING, and the step the human specifically asked to see
flagged):** to get an *expected number* of ordering errors we must integrate (4)
over the distribution of true gaps `Delta`:

```
E[flips] = N_pairs * Integral P(flip | Delta) f(Delta) dDelta                 (5)
```

`f(Delta)` is **not derived from anything** - it is an empirical property of which
interventions happen to be in the suite. Here we use the *observed empirical
distribution* of the 300 true target gaps (median 0.1754) rather than a parametric
family, which avoids inventing a shape but means (5) is a semi-empirical
prediction, not a purely theoretical one. Any claim resting on (5) inherits that.

### 2.3 Step 3: the regime boundary (DERIVED from 4 and 5, given A1-A5)

Single-scale ranking has zero fit variance but is wrong whenever the true order at
`C_max` differs from the order at `C*` - the crossover rate `p_x`, an empirical
property of the suite. Projection is better exactly when

```
Integral Phi( -|Delta| / sqrt(2 Var(zhat*)) ) f(Delta) dDelta   <   p_x        (6)
```

Substituting (3), projection wins when

```
sigma^2 [ 1/k + (d + log L)^2 / S_uu ]  <  threshold(p_x, f)                  (7)
```

which is decreasing in `k` and `s`, increasing in `sigma` and in `log L`
**quadratically**. Since `p_x` does not depend on `L` at all, the boundary is
crossed by increasing the lever arm - matching the measured dose-response
qualitatively before any number is compared.

## 3. Pre-committed quantitative predictions

### T1 - Excess flips grow quadratically in log L

From (3), the *ratio* of projected-value variance at two lever arms is
`[1/k + (d + log L_2)^2/S_uu] / [1/k + (d + log L_1)^2/S_uu]`. With the measured
constants (`k = 11`, `S_uu = 81.037`, `d = log C_max - ubar`), predict:

- **T1a:** the variance ratio between the 52.5x arm (fit to 150M) and the 4.7x arm
  (fit to 530M) is **1.5-4.0x**.
- **T1a (exact, from the committed constants):** the long-arm design (fit to 150M,
  `k = 10`, `S_uu = 53.147`) has leverage `h* = 1.3231`; the short-arm design (fit
  to 530M, `k = 12`, `S_uu = 114.111`) has `h* = 0.5199`. Predicted variance ratio
  = **1.3231 / 0.5199 = 2.545**, inside the pre-stated 1.5-4.0 band.
- **T1b:** consequently mean excess flips should be **~2.5x** larger at the long
  arm than the short arm. Measured value to be compared: the dose-response table in
  AUDIT_ADVERSARIAL.md gives 27.9 versus 5.3 flips, a ratio of **5.3** - roughly
  **2x larger than predicted**, i.e. the theory **under-predicts** the long-arm
  penalty. Per A2 above, that is the direction the measured heteroscedasticity
  predicts, since the long-arm design loses the quietest anchors.
- **CONFIRMED** if the measured ratio falls in the predicted band; **MISSED** otherwise,
  with the direction of the miss reported.

### T2 - Predicted versus observed slope of excess flips against log L

The measured slope is approximately 3x more excess flips per decade of `L`.
Predict from (3) and (5), using the empirical `f(Delta)`:

- **T2:** predicted slope within **a factor of 2** of the measured 3x per decade.
- **MISSED** otherwise. Because A4 biases (4) upward, a predicted slope *steeper*
  than measured is the expected failure direction.

### T3 - Retrodiction of the pooling gain (the strongest available test)

Pooling the exponent across `K` interventions replaces `K` separately estimated
slopes with one estimated from `K` times as much data. Under A2 the slope variance
falls by a factor of `K`; the intercept remains per-intervention. Since only the
slope multiplies `log L`, the target-variance reduction is

```
Var_pooled / Var_plain = [1/k + (u*-ubar)^2/(K S_uu)] / [1/k + (u*-ubar)^2/S_uu]  (8)
```

With `k = 11`, `K = 25`, and the measured leverage `h* = 0.0909 + 0.7051 = 0.7960`:

```
h*_pooled = 1/k + (u* - ubar)^2 / (K S_uu) = 0.0909 + 0.7051/25 = 0.1191
variance ratio = 0.1191 / 0.7960 = 0.1496
```

- **T3a:** predicted variance reduction factor **0.1496** (an 85.0% reduction in
  projected-value variance). This is a point value from the committed constants,
  with no band.
- **T3b (SHARPENED to a point prediction with propagated uncertainty).** Feeding
  `h*_plain` and `h*_pooled` through (4) and (5) with the empirical log-space gap
  distribution and the measured `sigma = 0.00145` gives a predicted flip rate of
  0.0109 plain versus 0.0036 pooled, hence a predicted **relative reduction in
  mis-selection of 67.2%**. Bootstrapping the `sigma` estimate (2000 draws) gives
  a **95% interval of [64.9%, 69.8%]**.
- Measured value to be compared: shared-exponent achieved **62.5%** relative
  (5.33% -> 2.00%), measured days before this theory was written.
- **CONFIRMED** only if 62.5% falls inside **[64.9%, 69.8%]**. It does not, so this
  is pre-committed as a **MISS of about 4.7 points, with the theory over-predicting
  the benefit of pooling.** Reported as such rather than widened.
- The interval propagates uncertainty in `sigma` only. It does not propagate
  uncertainty in the empirical gap distribution or in `K`, so it is narrower than a
  fully propagated interval would be. Stated so the miss is not overstated either.

### T4 - Where the theory must fail

Stated in advance so failures are not rationalised afterwards:

- **On accuracy metrics.** A1 fails outright (the floor is unidentified), so the
  theory should mispredict OLMES designs. If it happens to fit them well, that is
  evidence the derivation is capturing something other than what it claims.
- **On checkpoint-augmented fitting.** A2 fails (rho = 0.43-0.80), so predicted
  variance should be too small by roughly the measured deflation factor
  (2.5-8.8x). Predict the theory **over-predicts** checkpoint augmentation's
  benefit relative to what was measured.
- **Systematic over-prediction of flips** from A4, as noted.

## 4. Protocol

- Predictions are computed from the committed constants in section 1 and the
  formulae above, in code, with no free parameters fitted to the outcome.
- Comparison is against numbers already in `results/adversarial/a2_dose_response.json`
  and `results/task_b/task_b.json`, both computed before this file existed.
- Agreement is reported per prediction, and disagreements are reported with the
  direction of the miss and which assumption is the likely cause.
