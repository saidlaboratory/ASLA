# Pre-registration: expected-error significance and the published figure (round 3)

Written before computing. Committed state is `71c0b9a`.

## Correction to the record, carried forward

The previous round's headline --- "the single-scale advantage was largely label
noise" --- is **withdrawn**. It triggered on a determined subset in which both
rankers made zero errors, so the test had no power to detect a difference. That
is recorded in `PREDICTIONS_TASK_ROBUSTNESS.md` as R1 and is not relitigated
here. The primary rule from this point is **expected-error**, which is
non-vacuous: it charges every pair in expectation and retains 9.80 versus 3.84
expected errors on the primary C4 design.

## E1 --- The expected-error gap on C4 is significant under dependence-aware resampling

Pairs share candidates, so pair outcomes are dependent and a test treating the
300 pairs as independent is anti-conservative. The null is built by resampling
**candidates** (the independent experimental unit) with replacement and
recomputing both rankers' expected error on the induced pair set.

- Point prediction: the expected-error difference of **+1.99pp** is significant
  at the 0.05 level under candidate resampling, with a 95% interval **excluding
  zero**.
- Refuted if: the interval includes zero. That would mean even the non-vacuous
  rule cannot distinguish the rankers once candidate dependence is respected,
  and the C4 claim would reduce to a point estimate with no inferential support.
- UNDERPOWERED if: the interval excludes zero under naive pair resampling but
  includes it under candidate resampling. Both are reported either way, so the
  size of the dependence correction is visible.
- Honest confidence: 65%. Twenty-five candidates is few, and resampling them is
  a much coarser bootstrap than resampling 300 pairs.

## E2 --- The two published-figure estimators

The previous round reported "0.830 becomes 0.917 on determined pairs" as a
correction. That is **withdrawn as selection-biased**: determined pairs are the
well-separated ones, where any ranker scores near 1.0, so 0.917 is accuracy on an
easier population rather than a corrected estimate of the same quantity. The
determined fraction itself is kept as a finding.

Two principled estimators replace it, with different assumptions stated.

**Posterior expected accuracy.** Each pair is charged by the probability its
target order is correct. Pair-conditional, makes no claim about unresolved
pairs, which contribute near 0.5.

**Disattenuated accuracy.** Classical measurement-error correction: observed
agreement `a_obs = a*r + (1-a)*(1-r)`, where `r` is the reference's probability
of matching the true order, inverted for `a`. This extrapolates to unresolved
pairs under an **independence assumption** --- that ranker errors and reference
errors are independent. That assumption is doubtful here, because both the
ranker and the reference err preferentially on small true gaps, so the estimate
is reported conditioned on true-gap stratum and, pooled, as a bound.

- Point prediction: posterior expected accuracy falls **below** the observed
  0.830 (because undetermined pairs contribute ~0.5, pulling the average down),
  while disattenuated accuracy rises **above** it (because it credits the ranker
  for reference errors). Predicted posterior in **[0.78, 0.83]** and
  disattenuated in **[0.83, 0.95]**.
- Refuted if: they land on the same side of 0.830, which would mean the two
  corrections do not bracket the observed figure and my reading of what each
  does is wrong.
- **The disagreement between them is the result.** If they bracket 0.830 widely,
  the honest statement is that the data cannot pin the published figure more
  precisely than that interval, and the width is set by how much of the
  reference is undetermined.

## E3 --- Two regimes, split by determined fraction

- Point prediction: across the nine design-by-metric cells, determined fraction
  under Bonferroni separates the metrics with **no overlap**: C4 cells all above
  50%, accuracy-metric cells all below 20%.
- Refuted if: any C4 cell falls below 50% or any accuracy cell exceeds 20%.
- Consequence if confirmed: decision-accuracy comparisons are interpretable on
  low-noise metrics and dominated by reference noise on high-noise ones, and the
  published DataDecide figure sits in the second regime.

## E4 --- Dose-response survives under expected-error on C4

The lever-arm dose-response (Spearman +0.564 of excess flips against log L) was
measured under observed-target scoring.

- Point prediction: recomputed with expected-error excess, the Spearman against
  log lever arm stays **positive and above +0.35**, with p < 0.01.
- Refuted if: it falls below +0.20 or loses significance, which would mean the
  dose-response lived in reference noise.
- This is the most consequential of the re-scorings, because the lever-arm
  relationship is the mechanism behind several downstream claims.

## What would make the expected-error framing unsound

If E1 refutes, the C4 claim has a point estimate but no inferential support, and
the paper must say that the difference between rankers is not established at
this candidate count on any rule. That is a materially weaker paper than one
reporting a significant +1.99pp, and it must not be written as if the point
estimate carried the claim.
