# Pre-registration: robustness of the determined-subset result (Priority 3, 5)

Written before computing the threshold curve, the equivalence bound, or the
cross-design sweep. Committed state is `01f0c1b` plus the provenance fix.

## What is already known and is NOT being predicted

At alpha = 0.05 on the primary design, scored with per-pair Welch df:

| rule | determined | projection errors | single-scale errors |
| --- | --- | --- | --- |
| Bonferroni | 246/300 | **0** | **0** |
| Benjamini-Hochberg | 294/300 | 6 | 2 |
| uncorrected | 294/300 | 6 | 2 |

So the 0.00pp tie under Bonferroni is a comparison in which **neither method
makes any error at all**. That is a vacuous tie, not evidence of equivalence, and
it must be reported as such. Under BH the gap is 4 errors out of 294.

## R1 --- The vanishing is threshold-specific

- Point prediction: swept over the determination threshold from very strict to
  very permissive, the projection-minus-single-scale gap is **0.00pp only in the
  strict region where both methods make zero or near-zero errors**, and rises
  monotonically toward the observed-target value (+2.33pp) as the threshold
  relaxes. There is no interior threshold at which both methods err appreciably
  and the gap is zero.
- Refuted if: the gap is near zero across a broad interior range in which each
  method makes at least five errors. That would be genuine equivalence rather
  than an empty comparison.
- **This is the prediction I expect to confirm, and confirming it weakens the
  new headline.** The honest statement then becomes: the difference is carried by
  pairs near the resolution boundary, and the strictest subsets contain too few
  errors to distinguish the methods at all.

## R2 --- Equivalence bounds will be wide

- Point prediction: a two-one-sided-test equivalence bound on the
  determined-subset difference will fail to rule out differences smaller than
  **2 percentage points**, i.e. the data cannot establish equivalence at any
  margin narrower than roughly the effect originally claimed.
- Refuted if: the 90% CI on the difference lies inside +/- 0.5pp, which would be
  real evidence of equivalence.

## R3 --- The pattern holds across designs and metrics

- Point prediction: across the three designs and three metrics, the
  determined-subset gap is smaller than the observed-target gap in **at least
  7 of 9** design-metric cells.
- Refuted if: fewer than 5 of 9, which would mean the shrinkage is specific to
  the primary design.
- UNDERPOWERED if: more than 3 cells have fewer than five determined-subset
  errors in total, since those cells cannot distinguish the methods.

## R4 --- The published 0.80 figure moves

Applying the protocol to the DataDecide known-answer figure (single-scale
ranking at 150M predicting 1B, published ~0.80, reproduced here at 0.8033).

- Point prediction: the determined fraction of its 300 pairs is **above 70%**,
  and decision accuracy on the determined subset is **higher** than the overall
  figure by **1 to 6 percentage points**, because undetermined pairs are
  coin flips that drag a ranker toward 50%.
- Refuted if: determined-subset accuracy is *lower* than overall, or the
  determined fraction is below 50%.
- If confirmed, the published figure understates single-scale ranking's accuracy
  on the decisions that are actually determined, which is a correction to a
  widely cited number in the direction of the method looking better.

## Honest confidence

- R1: 85% confirm. The zero-error Bonferroni cell already points this way.
- R2: 80% confirm.
- R3: 60% confirm; the metrics differ enough that this may be design-specific.
- R4: 70% confirm the direction, less confident on the magnitude band.

## What would make the new framing unsound

If R1 confirms in its strongest form --- the gap is zero only where there are no
errors to compare --- then "they tie on determined pairs" cannot carry a paper on
its own. The defensible claim would narrow to: *reference uncertainty is a
material and unreported component of decision-accuracy comparisons, and on this
data the measured difference between two rules is concentrated in pairs the
reference cannot resolve.* That is still a real contribution about how such
comparisons are evaluated, and it is what the protocol is for, but it is weaker
than an equivalence claim and must not be written as one.
