# Pre-registration: curvature, cancellation, and gap-based intervals (Task 4)

Written before computing. Task 3 is committed at `84ac114`; nothing below has
been run.

Frozen before computing: the pairwise estimand, the cell-wise seed bootstrap, the
1B target and its 12-rung ladder, the DataDecide 750M exclusion, and the
loading-CV / cancellation-ratio values already committed in
`results/common_mode/common_mode_gate.json`. Any hyperparameter introduced here
records its provenance as `lambda` does in `NOTES_SHRINKAGE_STRENGTH.md`.

## 0. The tension this task resolves

Two committed measurements look contradictory:

- **P5 (Task 2).** The signed mean residual is a *wave, not a trend*: it
  oscillates across the ladder and correlates with `log C` at only **-0.022**
  (1B) and **+0.007** (60M). Read as "the bias channel is empty."
- **A1 (Task 3).** The projection sits **+0.0877** from the observed 1B target
  with **24 of 25** recipes overshooting, a one-sided error **119x the seed SE**.

**Hypothesis (H-curv2): the deviation is curvature, not slope.** Fitting a line
in log-log space to a gently convex truth produces in-range residuals that
oscillate with near-zero *linear* correlation, while the extrapolation is
systematically one-sided. A second-order in-range deviation becomes a
first-order out-of-range error. If this holds, the bias channel is not empty ---
it is second-order, which is precisely what a linear-in-log fit cannot see, and
what a correlation against `log C` cannot detect.

This supersedes the earlier reading. The two measurements are not in conflict;
the first used a diagnostic blind to the structure the second exposes. Recording
that explicitly because "the bias channel is empty" was stated as a finding and
would have sent Task 4 after the wrong target.

## 1. TASK 4a --- GATE. Is the deviation curvature?

Fit a quadratic in `log C` to each recipe's residuals from its own power-law fit.
**Nothing in 4b-4d is computed until this reports.**

### C1 --- The quadratic coefficient is consistently signed
- Point prediction: at least **20 of 25** recipes share the sign of the quadratic
  coefficient on the C4 metric.
- Refuted if: fewer than 15 of 25 share a sign (i.e. not distinguishable from
  an even split at roughly the 0.05 level under a sign test).
- UNDERPOWERED if: 15-19 share a sign.

### C2 --- Curvature magnitude predicts the target overshoot
- Point prediction: Spearman correlation between the per-recipe quadratic
  coefficient and the per-recipe target overshoot has **|rho| > 0.5** with
  **p < 0.01**.
- Refuted if: |rho| < 0.3, or p > 0.05.
- UNDERPOWERED if: 0.3 <= |rho| <= 0.5.

### C3 --- Curvature explains the overshoot quantitatively, not just directionally
Extrapolating the fitted quadratic to `C*` should predict the overshoot's size,
not merely its sign.
- Point prediction: the quadratic-implied overshoot recovers **> 50%** of the
  observed mean overshoot of +0.0877.
- Refuted if: < 20% recovered.
- This is the strictest of the three and the one most likely to come back
  UNDERPOWERED; a quadratic is itself an approximation to whatever the true
  deviation is.

**Gate rule.** Proceed to 4b-4d only if C1 and C2 both confirm. If either is
refuted, **stop and report**: the overshoot has another source, and candidates to
hunt then are (i) the floor parameter `E` being unidentified and absorbing the
extrapolation, (ii) a systematic difference between the 1B cell and the ladder
(tokens-per-parameter drift), (iii) seed-mean bias at the target itself. Do not
build 4b-4d on a refuted premise.

## 2. TASK 4b --- Does the overshoot cancel in gaps?

Only if 4a passes.

### C4 --- Gap centring error is far smaller than projection centring error
- Point prediction: mean absolute centring error of the *gap* estimate is
  **below 30%** of the mean absolute centring error of the individual
  projections (0.0927 on C4 at 1B). The committed cancellation ratio of
  **0.1474** on C4 predicts roughly `sqrt(0.1474) = 0.38` in standard-deviation
  terms if the ratio transfers; 30% is the stricter reading and is what is
  predicted.
- Refuted if: gap centring error exceeds 60% of projection centring error.
- UNDERPOWERED if: between 30% and 60%.
- If confirmed, this explains the otherwise strange pairing of a
  119x-seed-SE projection error with **Spearman 1.000000** orderings: the error
  is common-mode and the decision never sees it.

## 3. TASK 4c --- An interval on the quantity the decision needs

Only if 4a passes. Build the abstention interval directly on the pairwise gap
rather than on individual projections, and compare three interval sources on
width, empirical coverage at nominal delta, and resulting abstention rate:

1. bootstrap on projections (Task 3: coverage 0.000, effective delta 0.500 --- invalid)
2. conformal on projections (Task 3: coverage 0.880, median width 0.2841 --- valid, 15x wide)
3. **conformal on gaps** (new)

### C5 --- Gap intervals are valid and substantially narrower
- Point prediction: empirical coverage **>= 0.85** at nominal 0.90, with median
  width **below 40%** of the projection-conformal width.
- Refuted if: coverage < 0.80 (invalid), or width >= 80% of projection-conformal
  (no meaningful gain).
- UNDERPOWERED if: coverage in [0.80, 0.85) or width in [40%, 80%).

### C6 --- Narrower valid intervals lower the abstention rate
- Point prediction: at delta = 0.05 and 3 seeds, abstention using gap intervals
  is **lower** than using projection-conformal intervals by at least 10
  percentage points.
- Refuted if: abstention is equal or higher.
- Note this is a *conditional* improvement: it only matters if C5 holds, since a
  narrower invalid interval is worthless.

**On resurrecting the differenced estimator.** We killed it in the common-mode
work because its advantage was largest where single-scale ranking already had no
headroom. Certification is a different task and it is not competing with
single-scale there --- single-scale certifies nothing, it only ranks. If 4c
works, the honest statement is that **the earlier negative was task-specific, not
a property of the estimator**, and both results stand: differencing does not help
you select, and it does help you certify.

## 4. TASK 4d --- Scope, using the diagnostic we already have

Only if 4a passes. The committed loading CV separates the suites cleanly, with no
overlap:

| type | suite / metric | loading CV | cancellation | loadings same sign |
| --- | --- | --- | --- | --- |
| offset | DataDecide C4 bits/token | 0.156, 0.219 | 0.147, 0.234 | yes |
| offset | DataDecide OLMES macro error | 0.145, 0.182 | 0.130, 0.192 | yes |
| scaled | DataDecide OLMES correct-prob | 1.398, 2.032 | 0.707, 0.772 | no |
| scaled | Signal-and-Noise C4 bits/byte | 1.370, 2.153 | 0.817, 0.916 | no |

### C7 --- The diagnostic predicts where gap intervals help
- Point prediction: C5's width reduction holds on **both** offset-type metrics
  and fails on **both** scaled-type metrics, giving a clean 2-2 split that the
  pre-computable loading CV predicts in advance.
- Refuted if: the split is not clean --- any offset-type metric fails or any
  scaled-type metric succeeds.
- UNDERPOWERED if: a metric cannot be evaluated (too few rungs, fit failures).
- **A method that works on some metrics with a pre-computable test saying which
  is a legitimate contribution; silently working on one suite is not.** If C7
  refutes, the honest report is that the loading CV does not predict this
  particular benefit, even though it predicted cancellation.

## 5. Honest confidence

- C1: 80%. Convexity in log-log is the common shape for these curves.
- C2: 70%. Correlation across recipes needs the curvature to vary enough.
- C3: 45%. A quadratic may capture the direction without the magnitude.
- C4: 75%. This largely restates the committed cancellation ratio in a new form.
- C5: 60%.
- C6: 55%, conditional on C5.
- C7: 50%. This is a genuine coin-flip and the most interesting of the set.

## 6. What would make me abandon this task

If 4a refutes, the premise is wrong and 4b-4d are not built. If 4a passes but C4
refutes --- curvature is real but does *not* cancel in gaps --- then gap
intervals will not be narrower and the contribution reduces to a mechanism for
the overshoot, which is worth reporting but is not a method.
