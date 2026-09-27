# Pre-registration: physicality, structure swap, dimensionless transfer, DataDecide's own single-scale baseline

Written 2026-09-27, before any computation in this round. No paper prose changes.
Outputs: JSON, figures, generated macros.

## Reframe, as given

In OLMES's own calibrated worlds, projection truly wins at the primary design:
-0.72, with interval [-0.94, -0.50]. The earlier independent calibration also
gives theta_fixed = -0.72. Two consequences follow:
- the descriptive claim that no tested alternative beats top-rung ranking holds
  on C4, but must be scoped on OLMES;
- a rule that predicts which regime a metric is in would be a method
  contribution.

## Task 1: physicality of the extended grid

**Curves checked.**
- C4's map axis: mu_P + common + lambda x specific.
- C4's ensemble axis: mu_P + lambda x R.
- The same map axis built on OLMES.
- For every metric, its own shrunk truth (lambda = 1).
- Seed noise does not enter these truth tables, so the noise axis is physical
  by construction.

**Conditions.** Each is scanned for lambda from 0 to 8 in steps of 0.01, and the
first failing lambda is reported.
- **Strict monotonicity.** Every recipe's truth is non-increasing across the rungs
  in compute order.
- **Tolerant monotonicity.** No increase between adjacent rungs exceeds that
  recipe's per-run seed sd at the higher rung. An increase of that size is not a
  shrinkage artifact.
- **Valid range.** Bits per token or byte must be > 0. OLMES error and the
  correct-prob deficit must be in [0, 1].
- **Operative rule.** The tolerant condition, together with the valid range.
  Strict failures are reported beside them.

**Restricted map.** Cells with lambda at or beyond the first operative failure
are unphysical. On the physical region the following are re-evaluated:
- the short-design winning region above 6x;
- the fall in candidates needed above 2x;
- whether OLMES's placement at 4.1x lies inside C4's physical region.

**Predictions.**
- **P1-1.** C4's operative failure on the map axis comes below 4. The short-design
  winning region (6x and above) and the cells of the candidates fall (4x, 8x)
  are unphysical, and OLMES's placement at 4.1x lies outside C4's physical
  region.
- **P1-2.** C4's own truth is strictly monotone. OLMES's own truth has strict
  violations at small scales, where accuracy is near chance, but no tolerant
  ones.

## Task 2: structure swap (C4 and OLMES, primary design)

### Representation

- **Shared layout.** Both metrics use the same 25 recipes and the same 13 rungs
  (750M excluded).
- **Truth and fit.** mu_M is the shrunk truth. P_M is the per-recipe power-law fit
  over the full ladder, R_M = mu_M - P_M is the residual, and sigma_M(r, s) is
  the moderated per-run sd.
- **Unit.** g_M is the median absolute true gap over recipe pairs at 530M.
- **Normalization is division only**, with no centring:
  - P~ = P / g, R~ = R / g and sigma~ = sigma / g.
  - This keeps the power-law fit's bound E >= 0 meaningful: the fit is bounded,
    E in [0, y_min] and A <= 5, so it is not affine-invariant.
- **Where hybrids live.** Every hybrid is built in its base metric's units,
  y = g_base x y~.

### Components

A hybrid takes each component from C4 or from OLMES.
- **(i) Noise profile shape.**
  - The profile is pi_M(s) = RMS_r sigma~_M(r, s) / A_M, where
    A_M = RMS_{r,s} sigma~_M.
  - The hybrid noise is sigma = g_base x sigma~_base(r, s) x pi_(i)(s) /
    pi_base(s) x A_(v) / A_base.
  - The base supplies the per-recipe pattern.
- **(ii) Distribution of true gaps at the target.**
  - After the truth is assembled, each recipe's whole curve is shifted by a
    constant. The shift makes its 530M deviation from the recipe mean equal
    the (ii)-source's sorted deviation at the hybrid's own target rank.
  - A constant shift stays inside the power-law family and keeps the recipes'
    target order. Both sources have median gap 1 in g units, so (ii) changes
    the shape of the gap distribution, not its scale.
- **(iii) Curve shape and floor.** The parametric part P~ (mean curve, level
  relative to zero, and per-recipe exponents and amplitudes).
- **(iv) Misspecification loading pattern.** The residual R~, recipe-aligned.
- **(v) Noise amplitude A** (sigma relative to the median gap).
  - Added by me; it is not on the given list. Last round showed a 5x difference
    in this amplitude.
  - Components (i) to (iv) hold it at the base's value.

### Runs

- **Configurations (14).** At the primary design (300M -> 530M), 150 worlds each:
  - the base alone, C4 and OLMES (2);
  - C4 with each single component from OLMES (5);
  - OLMES with each single component from C4 (5);
  - each base with all five components swapped (2). This is a completeness check:
    it should reproduce the other metric.
- **Random numbers.** Within each base, world i uses the same draws in every
  configuration.
- **Readouts.**
  - The true excess, with a 95% bootstrap interval.
  - The share of the gap closed:
    - forward: (e - e_C4) / (e_OLMES - e_C4);
    - reverse: (e - e_OLMES) / (e_C4 - e_OLMES).
  - A physicality flag for each hybrid: positivity, valid range, and tolerant
    monotonicity.
- **"Moves the sign"** means the interval lies entirely on the other metric's
  side of zero.
- **"Identified,"** for use in Task 3: a component that closes at least 50% of the
  gap in either direction, or that moves the sign.

### Predictions

- **P2-1 (as given).** (i) is the primary driver. The forward (i) swap flips C4's
  sign to negative, the reverse (i) swap flips OLMES's to positive, and (i)
  closes the largest share of (i) to (iv).
- **P2-2 (as given).** (ii) is secondary, with the second-largest share.
- **Refutation (as given).** If neither (i) nor (ii) moves the sign in either
  direction, the driver is curve shape or floor.
- **P2-3 (mine).** (iii) or (v) closes the largest share, and (i) closes less
  than 50% in both directions.
- **P2-4.** Each all-swapped configuration reproduces the other metric's excess,
  with the interval containing the other base's point estimate. If not, the
  representation is incomplete, and what it misses (units interacting with the
  fit's bounds, or the per-recipe noise pattern) is reported.

## Task 3: dimensionless map and transfer test

### Metrics

- **C4 bits per token.** 13 rungs, 3 seeds.
- **OLMES macro error.** 13 rungs, 3 seeds.
- **OLMES correct-prob deficit** (1 - correct_prob_per_char). 13 rungs, 3 seeds.
- **Signal-and-Noise C4 bits per byte.**
  - 8 rungs (750M excluded for comparability) and **one run per cell**, so it has
    no replicates of its own.
  - Its noise is borrowed: sigma_SN(r, s) = sigma_C4(r, s) x kappa(s), where
    kappa(s) is the ratio of the mean bits per byte to the mean bits per token
    at that scale.
  - Its truth is shrunk with this sigma and one seed.

### Designs, per ladder

- primary: 300M -> 530M;
- short: the smallest-L design on that ladder;
- long: the largest-L design on that ladder.

### Coordinates

All are measured from each metric's shrunk truth and noise, per design.
- **q1.** Cell-mean seed noise at the top fitted rung, RMS_r sigma / sqrt(n),
  divided by g_t, the median absolute true gap at the design's target.
- **q2.** RMS over recipes and rungs 0..t of the recipe-specific residual (the
  residual minus the common mode, as in the earlier arms), divided by g_t.
- **q3.** RMS_r sigma at the smallest rung divided by RMS_r sigma at the top
  fitted rung.
- **q4.** The lever arm, which is fixed by the design.

**Conditional additions,** fixed now and applied by rule from Task 2's
"identified" components:
- (i) is covered by q3.
- (ii): import the target's normalized gap distribution by the rank-preserving
  shift.
- (iii): add q5 and q6.
  - q5 = mean target value / g_t, the floor level.
  - q6 = (mean curve at the smallest rung - at the target) / g_t, the decay.
- (iv) is covered by q2.
- (v) is covered by q1.

### Evaluating the map at a target's coordinates

The map is not interpolated. The source's structure is transformed to the
target's coordinates exactly, and 150 worlds are run on the target's rung set:
- **q2.** Scale the source's recipe-specific part by lambda, iterating so that q2
  matches, since g_t depends on lambda.
- **q3.** Tilt the noise by (C_s / C_top)^(-beta) below the top fitted rung,
  leaving it unchanged from the top up, so that q3 matches.
- **q1.** Scale all sd by k so that q1 matches.
- **q5, q6 (if added).** Replace the recipe-mean curve by
  mu(T) + kappa (mu(s) - mu(T)) + c.
- **(ii) (if added).** Apply the rank-preserving shift.

At the source's own coordinates this transformation is the identity; that is
checked.

### Ground truth, success and independence

- **Ground truth.** Each metric's own worlds (its truth, noise, ladder and seed
  count), 150 of them, with a separate random stream: the true excess and its
  95% interval at each design.
- **Success.** At every (source, held-out target, design) where the target's own
  sign is resolved, the predicted sign (the sign of the mean over 150 worlds)
  matches it. The match rate is also reported.
- **Independence, stated in advance.**
  - Signal-and-Noise bits per byte re-evaluates the same DataDecide models as C4.
    The user reports 93 to 99.7% pair agreement with it; that figure is not
    re-verified this round. Its noise is also borrowed from C4.
  - Correct-prob and OLMES error share a metric family and the same tasks.
  - So the four metrics are about two to three independent tests, not four.

**Predictions.**
- **P3-1.** With only q1 to q4 (no conditional additions), the map does not
  transfer: at least one resolved held-out sign is wrong.
- **P3-2.** With the conditional additions from Task 2, every resolved held-out
  sign matches.

## Task 4: DataDecide's own single-scale baseline

**Search, done before this file was written.**
- Found as code in `allenai/DataDecide` at commit 68abc49:
  - `single_scale/main.py`, `single_scale/evaluator.py` (binary accuracy) and
    `single_scale/config.yaml`.
  - Protocol: `task_aggregation: olmes`, `target_model: 1B`, raw values with no
    transform, and each small-scale seed scored separately.
  - The 1B target is the **mean over the three seeds** (`take_mean_over_seeds`).
    Their scaling-law decision accuracy (`scaling_laws/utils/stats.py`) instead
    uses the default seed only.
- Its outputs are not in the repository or the Hugging Face release.
  - `results/readme.md` points to a Google Drive folder. Access to it was denied
    in this environment, so those released outputs are unchecked.
  - Its input CSV comes from their S3 export, which is not released.

**Run.** Their evaluator is run unmodified: `Evaluator.calculate_metrics` with
`binary_accuracy`, `get_transformed_values_compute_and_seeds`,
`take_mean_over_seeds` and `Raw`.
- **Stated deviation.** The input is the released `macro_avg` table arranged in
  their input layout for the macro average (primary_metric of
  `olmes_10_macro_avg`), not their S3 CSV.
- **Readout.** Their single-scale binary accuracy at 750M (final checkpoint, mean
  over the three small-scale seeds), and at every scale.

**Comparisons on the macro average.** Each variant's decision accuracy against
their baseline:
- (a) as implemented: scaling law against the default-seed target, their
  single-scale against the three-seed target;
- (b) both against the three-seed target, the paper's §2.3 definition;
- secondary: their single-scale fed only the default seed at 1B.

**P4.** Under (b), `3_param-1_step` (82.67) still exceeds their single-scale
baseline at 750M, so the point-comparison flip persists.
