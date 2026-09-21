# Pre-registration: scoring against a determined target (Task 0)

Written before computing 0a/0b. Committed state is `9c7d503`. Nothing in this
file has been run except the two diagnostics quoted below, which are reported as
inputs rather than results.

Frozen: the DataDecide C4 primary design, the 1B target, the 750M exclusion, the
25-candidate set and its 300 pairs.

## Inputs already measured (not predictions)

Target cells carry exactly 3 seeds each. The Welch-Satterthwaite degrees of
freedom for the 300 pairwise comparisons range **2.02 to 4.00 with median 2.96**
--- so the draft's assertion of 4 df for all three-seed comparisons was wrong;
4 is the maximum, attained only when the two cells have equal variance. That is
review point 6b and it is conceded.

## P0.1 --- Most target pairs are determined

The review's concern is that the target ranking is a 3-seed estimate and may be
"partly coin flips". It is not, on this design.

- Point prediction: **> 85%** of the 300 target pairs are determined under a
  Welch t test at Bonferroni-corrected alpha = 0.05/300, and **> 95%** under
  Benjamini-Hochberg at q = 0.05.
- Refuted if: the Bonferroni figure falls below 70%, which would mean the target
  label is noisy enough that the headline rates are substantially label error.
- Basis for the prediction, stated so it is not mistaken for a result: the median
  pair has a gap of 107 standard errors. The target gaps are large relative to
  seed noise because the candidate set spans very different corpora.

## P0.2 --- The projection-minus-single-scale gap does not close

This is the one that matters. Under both scoring rules, the direction must be
checked before anything downstream is touched.

- Point prediction: projection's excess over single-scale stays **positive and
  within 1.5 percentage points of the observed-target value** (projection
  5.33%, single-scale 1.33%, excess +4.00pp) under both the determined-subset
  rule and the expected-error rule.
- **Refuted, and a hard stop, if:** the excess changes sign under either rule, or
  falls below +1.0pp. Either would mean the headline comparison was substantially
  an artifact of scoring against a noisy label.
- UNDERPOWERED if: the excess stays positive but its bootstrap CI includes zero
  under either rule.

### What I would conclude in each case, stated in advance

- **Gap stays (predicted).** The headline comparison survives; the decomposition
  accounting still has to be corrected (Task 1) because that is a separate error.
- **Gap shrinks to near zero.** The single-scale advantage was largely label
  noise. The paper's central empirical claim would need restating as "no method
  distinguishably beats any other once the target's own uncertainty is
  propagated", which is a weaker and quite different paper.
- **Gap grows.** Scoring against a noisy label was *understating* projection's
  disadvantage. Report the larger number but lead with the correction, not the
  larger number.

## P0.3 --- Determined-subset and expected-error rules agree

- Point prediction: the two rules give excesses within **0.5pp** of each other.
- Refuted if: they differ by more than 1.5pp, which would mean the result depends
  on how the undetermined pairs are handled and neither rule should be reported
  alone.

## Models and assumptions, stated before use

**Determined-subset rule.** A pair is determined if a two-sided Welch t test on
the two target cell means rejects equality at Bonferroni alpha = 0.05/300, with
Welch-Satterthwaite df computed per pair. Rankers are scored only on determined
pairs. Assumption: cell means are approximately Gaussian with unknown, possibly
unequal variances. Three seeds is few; the t reference is what the Student-t
finding in this project says to use, and the df is computed rather than asserted.

**Expected-error rule.** For each pair, the probability that the observed target
order is the true order is modelled as `Phi(|delta| / se_diff)` under a Gaussian
approximation to the sampling distribution of the difference in cell means --- a
normal approximation used deliberately here, because the quantity needed is a
posterior-like weight rather than a tail test, and the t correction matters for
critical values rather than for a probability near the centre of the
distribution. A ranker predicting the observed order is charged `1 - p`; a ranker
predicting the opposite is charged `p`. This is a proper accounting of expected
error against the unknown true order, and it uses no prior beyond the
observed gap. Assumption: the target cell means are unbiased estimates of
expected target performance, which is exactly review point 3 and is NOT
addressed by this rule --- a systematic bias in the target measurement would
survive it.

## Scope limit recorded in advance

Neither rule addresses the deeper version of review point 3: that expected target
performance may differ from observed target means for reasons other than seed
noise (a different draw of the evaluation set, a different checkpoint). Both
rules treat the target cells as unbiased. That limitation is stated, not solved.
