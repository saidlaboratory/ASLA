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

**Assumption A2:** homoscedastic, independent errors in log space. Seeds within a
cell are independent by construction (different initialisations), but cell means
at different budgets come from *different* runs in DataDecide, so independence
across budgets is reasonable here. It would fail for checkpoint-augmented fitting,
where we measured rho = 0.43-0.80 - the theory as written does not cover that case.

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

**Assumption A4 (LOAD-BEARING):** independence of the two interventions' fit
errors. This is *false in a specific direction*: interventions share the same
budget ladder and the same evaluation, so their errors are positively correlated,
which makes `Var(gap)` smaller than `2 Var(zhat*)` and (4) an **over-estimate** of
the flip probability. Predictions below therefore give an upper bound on the
expected number of flips, and systematic over-prediction is the expected failure
mode rather than evidence against the mechanism.

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
- **T1b:** consequently mean excess flips should be **1.5-4.0x** larger at the long
  arm than the short arm. Measured value to be compared: the dose-response table in
  AUDIT_ADVERSARIAL.md gives 27.9 versus 5.3 flips, a ratio of 5.3 - **outside the
  predicted band**, which would be a partial miss to report as such.
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

With `k = 11`, `K = 25`, and the measured leverage (0.0909 + 0.7051):

- **T3a:** predicted variance reduction factor **0.11-0.16** (i.e. an 84-89%
  reduction in projected-value variance).
- **T3b:** translating through (5) with the empirical gap distribution, predicted
  reduction in *mis-selection rate* of **40-75%** relative.
- Measured value to be compared: shared-exponent achieved **62.5%** relative
  (5.33% -> 2.00%). This was measured before the theory was written, so T3b is a
  genuine retrodiction of an effect the theory was not fitted to.
- **CONFIRMED** if 62.5% falls inside the predicted band; **MISSED** otherwise.

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
