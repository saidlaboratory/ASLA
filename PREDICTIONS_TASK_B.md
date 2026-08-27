# Pre-registered predictions: pooled / shrinkage projection estimators (Task B)

**Status: pre-registration. Written and committed before any pooled estimator was
implemented or run.** The only pooled quantity computed before writing this file is
the variance decomposition of the *already-fitted* per-intervention exponents
reported in section 2, which uses the existing plain fit and no new estimator.

Author: analysis agent. Date: 2026-08-25. Commit this file, then run Task B.

---

## 1. The mechanism being tested

Phase 1 and the adversarial audit established, on DataDecide with a pipeline
validated against DataDecide's own published decision accuracy:

- Projection mis-selects more pairs than single-scale ranking (C4: 5.3% vs 1.3%).
- **All** of that excess is fit/extrapolation error, not crossover: 14 of 16 C4
  projection flips are on pairs the largest fit budget already ordered correctly;
  the conclusion holds in 289/289 testable specifications and in 100% of 300
  seed-bootstrap resamples.
- The excess scales with the lever arm `L = C_target / C_max fitted`
  (Spearman +0.56 on log L, p = 3e-28), while the no-fitting control tracks it
  about three times more weakly.
- Adding flexibility makes it worse: the three-family ensemble mis-selects 18.7%
  vs 5.3% for the plain fit, and on well-specified synthetic data at realistic
  noise 10.1% vs 4.5%.

**Mechanism claim (H1):** the dominant error term in scaling-law-based selection
is *variance in the fitted target projection*, not bias from mis-modelled
crossovers. Everything below follows from that single claim.

**The symmetric prediction.** If flexibility hurts because it adds variance, then
removing flexibility by pooling information across interventions must help, for
the same reason and by a predictable amount. Arm 1 (more flexibility -> worse) is
already observed. Arm 2 (less flexibility -> better) is what Task B tests.

---

## 2. The quantity that makes the magnitude prediction falsifiable

Pooling helps to the extent that the per-intervention exponents are *not* really
different. Measured on C4 bits/token, 25 recipes, the 11-budget ladder 4M-300M,
using the existing plain per-intervention fit:

| quantity | value |
|---|---|
| mean fitted `alpha` across recipes | 0.1520 |
| **between**-recipe sd of fitted `alpha` | 0.0070 |
| **within**-recipe sd of fitted `alpha` (200-replicate seed bootstrap of one recipe) | 0.0049 |
| implied noise share of the observed spread, `within^2 / (within^2 + between^2)` | **0.33** |

So roughly **one third of the apparent variation in decay exponents between data
recipes is estimation noise, not real difference.** A James-Stein / empirical-Bayes
shrinkage estimator should therefore pull each exponent about a third of the way
toward the pooled exponent, and the variance of the projected target value should
fall by roughly the same order. This is the number the magnitude predictions below
are derived from, and it is measured, not assumed.

Caveat stated in advance: the within-recipe bootstrap was run on one recipe
(alphabetically first) for cost reasons, and the shrinkage factor implied by a
per-recipe estimate may differ. If the pooled analysis finds the noise share is
below 0.10 or above 0.70, the magnitude predictions in section 4 should be judged
against that measured share rather than against 0.33.

---

## 3. The estimators, ordered by flexibility

| rank | estimator | free parameters over K interventions | pooling |
|---|---|---|---|
| 1 (most flexible) | ensemble over 3 curve families | 3K plus family weights | none, plus family averaging |
| 2 | plain per-intervention power law | 3K | none |
| 3 | hierarchical shrinkage, strength from data | between 2K+1 and 3K | partial |
| 4 (least flexible) | shared exponent, per-intervention coefficient | 2K+1 | complete on `alpha` |

---

## 4. Predictions

Each is stated with a direction, a magnitude, a band, and what would refute it.
Primary evaluation: C4 bits/token, 25 recipes, 1B target, `pairwise_decisions`
estimand, seed-bootstrap CIs, on the held-out split (Task B4).

### P1 — Direction against plain projection (the easy bar)

**Shared-exponent and hierarchical shrinkage both beat plain per-intervention
projection** on pairwise mis-selection.

- Point prediction: plain 5.3% -> **shrinkage 2.5-4.5%**, i.e. a relative
  reduction of **15-55%**, centred on ~35%.
- Reasoning: the noise share of 0.33 says a third of the exponent spread is
  noise; removing it removes roughly that share of the projection variance that
  the exponent contributes.
- **Refuted if** either pooled estimator's point estimate is >= plain projection's,
  or the improvement is under 10% relative.

### P2 — Direction against single-scale ranking (the hard bar)

**Neither pooled estimator beats single-scale ranking on C4 at the 1B target with
the 4M-300M ladder.** Predicted ordering: single-scale (1.3%) < shrinkage < plain
(5.3%) < ensemble (18.7%).

- Reasoning: pooling removes the *exponent* component of fit variance, but the
  per-intervention coefficient and floor remain freely fitted, and the lever arm
  here is 12.5x. The dose-response says excess is ~11 flips at this arm; a one-third
  variance cut should not close a 4-point gap.
- Confidence: this is the prediction I am least sure of, stated at ~70%.
- **Refuted if** either pooled estimator's mis-selection CI upper bound falls below
  single-scale ranking's point estimate. If that happens, the interesting result is
  that pooling turns a losing rule into a winning one, and it should be reported as
  the headline of Task B.

### P3 — Shape of the accuracy-vs-flexibility curve

**Monotone in flexibility over the range tested**: ensemble worst, then plain, then
hierarchical shrinkage, then shared exponent best or tied with shrinkage.

- I predict **no interior optimum between plain and shared-exponent**, because the
  between-recipe exponent differences that complete pooling discards are small
  (sd 0.0070 on a mean of 0.1520, a 4.6% coefficient of variation).
- I do predict the curve **flattens** between hierarchical shrinkage and shared
  exponent (difference under 1 point), since shrinkage at a noise share of 0.33
  already captures most of the available variance reduction.
P3 has **two named alternative outcomes**, and both are substantive results
rather than incidental details. Let `d = shrinkage_mis_selection -
shared_exponent_mis_selection`, in percentage points, with CIs from the seed
bootstrap.

- **P3a — interior optimum (shared-exponent clearly worse).** Triggered when
  `d <= -1.0` points and the two CIs do not overlap. Interpretation: there is real
  between-recipe exponent structure that complete pooling destroys. The
  accuracy-vs-flexibility curve has an interior optimum, the "less flexibility is
  always better" reading of H1 is wrong, and the correct claim becomes that there
  is an optimal flexibility level which the data can locate.
- **P3b — under-pooling (shared-exponent clearly better).** Triggered when
  `d >= +1.0` points and the two CIs do not overlap. Interpretation: the
  data-driven shrinkage is under-pooling, and the between-recipe exponent
  variation is essentially all noise — i.e. the measured noise share of 0.33 is an
  underestimate and the true share is near 1. This would say the empirical-Bayes
  shrinkage target is too conservative, and that the practitioner's fixed-exponent
  device (the Olmo Hybrid choice) is not merely defensible but optimal. It also
  predicts that a re-measured noise share on the full recipe set should come out
  well above 0.33; that check is required before P3b is claimed.
- **P3 as predicted (monotone, flattening)** requires `|d| < 1.0` or overlapping
  CIs, with both pooled estimators beating plain projection.

### P4 — Interaction with the lever arm and the number of budgets

**Pooling helps most where fit variance is worst**, i.e. at long lever arms and few
fit budgets.

Define the **interaction ratio** `rho_L = I(52.5x) / I(4.7x)`, where `I(L)` is the
number of excess flips removed by the best pooled estimator at lever arm `L`
(fit to 150M and to 530M respectively, both targeting 1B, same metric and ladder
start). `rho_L` is computed on every seed-bootstrap replicate, giving a point
estimate and a 95% CI.

- Point prediction: **`rho_L >= 2.0`**.
- At lever arm ~1-5x, where the audit shows excess flips of only ~4-5, I predict
  `I(4.7x) < 2` flips, because there is little variance left to remove.
- **P4 counts as FAILED** when the **point estimate `rho_L < 2.0` AND the 95% CI
  upper bound is below 2.0** — i.e. the data positively exclude the predicted
  interaction, rather than merely failing to establish it.
- **P4 counts as UNDERPOWERED** (neither confirmed nor failed, and it does not
  count toward the compound refutation in section 5) when the point estimate is
  below 2.0 but the CI includes 2.0, or when `I(52.5x) < 3` so the ratio is
  computed over too few removed flips to be stable.
- **P4 counts as CONFIRMED** when the point estimate is at or above 2.0.
- If `I(4.7x) = 0` the ratio is undefined; in that case P4 is confirmed if
  `I(52.5x) >= 3`, since an improvement present only at the long arm is the
  strongest possible form of the predicted interaction.

### P5 — Where the removed errors come from

**The flips that pooling removes are fit-error flips, not inherited-crossover
flips.** Applying the audit's decomposition to the pooled estimator, the reduction
in fit-error flips should account for **at least 80%** of the total reduction.

**Evaluability threshold (fixed in advance).** Let `R` be the total number of flips
removed by the best pooled estimator relative to plain projection, on the primary
C4 design. An 80% ratio is meaningless over a handful of flips.

- **P5 is evaluable only if `R >= 8`.** At `R = 8` the 80% threshold distinguishes
  "at least 7 of 8 removed flips are fit-error" from "6 or fewer", which a
  seed-bootstrap CI on the count can resolve at this sample size. (For reference,
  C4 has 14 fit-error flips and 2 inherited-crossover flips in total, so `R >= 8`
  means pooling removed a majority of the fit-error flips.)
- **If `R < 8`, P5 is reported as UNDERPOWERED — neither confirmed nor refuted** —
  and the observed ratio is reported with its bootstrap CI as descriptive only. It
  does not count toward or against the mechanism.
- **Refuted (when evaluable)** if the fit-error share of removed flips is below
  80% and its bootstrap CI upper bound is also below 80%, which would mean pooling
  is changing the small-scale ordering rather than stabilising the extrapolation —
  a different mechanism from the one claimed.

---

## 5. What would refute the mechanism itself

The mechanism claim is refuted, not merely dented, if **both** of these hold:

1. Neither pooled estimator improves on plain projection by more than 10% relative
   (P1 fails), **and**
2. P4 counts as FAILED under the sharp criterion above (point estimate `rho_L < 2.0`
   *and* CI upper bound below 2.0). A P4 result of UNDERPOWERED does **not**
   contribute to this compound condition.

That combination would mean flexibility reduction does not buy accuracy even though
flexibility increase demonstrably costs it, which is asymmetric in a way a pure
variance account cannot explain. In that case the ensemble result would need
re-interpretation as something other than a variance effect — most likely the
ensemble's family weights being actively misled by noisy leave-one-out losses,
which is a narrower claim than H1.

A weaker but still important negative: if P1 holds but P5 fails (pooling helps, but
by changing which pairs the small scales order correctly rather than by stabilising
extrapolation), then pooling is useful but the stated mechanism is wrong about
*why*, and the paper's mechanistic claim must be softened accordingly.

---

## 5b. Addendum (2026-08-27, before the final design completed)

Two protocol points fixed while the run was still in progress, recorded here so
they are not read as post-hoc choices:

- **Headline estimator: `shared_exponent`.** Complete pooling on the exponent has
  no free hyperparameter, so it is held out by construction and needs no
  argument about how a shrinkage strength was selected. `shrinkage_eb` is
  reported alongside it. P2 continues to be judged against whichever pooled
  estimator does *best*, which is the hardest version of that test for our own
  prediction.
- **The sweep optimum is diagnostic, never a method result.** Fixed strengths
  were evaluated to trace the flexibility curve; the best of them was identified
  by inspecting evaluation data and is labelled `sweep_optimum_diagnostic` in the
  output with an explicit "must not be quoted as the method's performance".
- **P1 gains an UNDERPOWERED path**, symmetric with P4 and P5: UNDERPOWERED when
  every improvement is under 0.5 points absolute *and* none is separated from
  plain projection. As with the others, UNDERPOWERED cannot contribute to the
  compound mechanism refutation in section 5.

## 6. Analysis protocol fixed in advance

- Primary metric: C4 bits/token. OLMES metrics reported as secondary, with the
  adaptive-bounds fit (the fixed 0.05 floor is a known defect, see AUDIT_ADVERSARIAL.md).
- Primary estimand: `pairwise_decisions`, as in FIRST_AUDIT.md.
- Shrinkage strength is chosen on **train** interventions/scales via the frozen
  split machinery (`asla.analysis.splits`) and never on the evaluation set.
- Confidence intervals: cell-wise seed bootstrap, same machinery as the audit.
- Significance for flip counts: Benjamini-Yekutieli (dependence-robust), per the
  A4 recommendation.
- Every estimator is evaluated through the existing `Ranker` interface so it sees
  exactly the same data the plain fit sees.
- Refutations are reported at the top of the Task B report, not in an appendix.
