# Pre-registration: validating the noise model on ten seeds (Task 8)

Written before computing. The PolyPythias gate is committed at `fd64285`;
nothing in 8a-8c has been run. Task 7 stopped at its gate and is not part of this.

Frozen before computing: the estimand (per-checkpoint accuracy on
`lambada_openai`), the 50-run panel and its common-step subset, and the
three-seed subsampling procedure described below. Any hyperparameter introduced
records its provenance as `lambda` does.

## Why this matters to us

Every resolvability number in this project estimates `sigma` from **three
seeds**: the 93.4% unresolvable figure, the seeds-required curves, the
minimum-detectable-gap diagnostic. Our own Student-t finding says three seeds is
exactly the regime where the naive reference distribution is anti-conservative by
3.63x. That was an argument from the sampling distribution of a variance
estimate. PolyPythias lets it become a measurement, on an independent suite, at
five model sizes, with no new compute.

## N1 --- The three-seed sigma estimate is badly dispersed

A variance estimated from `n = 3` has `2` degrees of freedom, so its sampling
distribution is wide and right-skewed.

- Point prediction: over repeated three-seed subsamples of the ten available, the
  ratio of the 90th to the 10th percentile of the estimated `sigma` is **> 2.5**.
  The chi-squared model gives `sqrt(chi2_{0.90,2} / chi2_{0.10,2})` =
  `sqrt(4.605 / 0.211)` = **4.67** if the ten seeds were exactly Gaussian and
  independent; the prediction is deliberately looser because they are neither.
- Refuted if: the ratio is **< 1.5**, i.e. three seeds is nearly as stable as ten.
- UNDERPOWERED if: between 1.5 and 2.5.

## N2 --- The three-seed estimate is biased low as a *typical* value

The sample standard deviation is a biased estimator of `sigma`, downward, and the
bias is largest at small `n` (the `c4` correction is 0.886 at `n = 3`).

- Point prediction: the **median** three-seed estimate is below the ten-seed
  estimate at a majority of (size, step) cells, by a median factor in
  **[0.80, 0.95]**.
- Refuted if: the median ratio exceeds 1.0, i.e. three seeds over-estimates.
- Note the direction matters for us specifically: under-estimating `sigma`
  makes gaps look *more* resolvable than they are, so our 93.4% unresolvable
  figure is, if anything, an **under**-statement of the problem.

## N3 --- Sigma does not shrink with model size

Our transfer of DataDecide's 1B seed SD to the Fantastic Optimizers grid assumed
seed noise is comparable across scales. That assumption is load-bearing for the
207x non-identifiability result and has never been tested on independent data.

- Point prediction: the Spearman correlation between model size and `sigma`
  across the five sizes is **> -0.5**, i.e. no strong shrinkage, and plausibly
  positive.
- Refuted if: Spearman **< -0.9** with sigma falling monotonically, which would
  mean the transfer inflated noise at the small scales and our non-identifiability
  result needs re-deriving.
- UNDERPOWERED if: five sizes prove too few to separate `> -0.5` from `< -0.9`,
  which is a real possibility and is why the bar is a correlation rather than a
  slope.
- *Prior, stated as a prior not a result:* the paper's Table 2 reports
  performance variance from 0.58 to 0.98 across sizes with 410M highest, which is
  consistent with no shrinkage. That is a different metric from ours and is not
  evidence for N3; it is why N3 is worth testing rather than assuming.

## N4 --- Sigma is *not* constant across training stages

The noise estimator our resolvability work uses pools across checkpoints, which
assumes across-seed variance is roughly stage-independent.

- Point prediction: `sigma` at an early checkpoint differs from `sigma` at the
  final common checkpoint by a factor **> 2** at a majority of sizes.
- Refuted if: the ratio is within **[0.7, 1.4]** everywhere, i.e. pooling is safe.
- If confirmed, pooling across stages is wrong and any estimate that does it
  reports a number belonging to no particular stage. That would be a defect in
  our own resolvability pipeline and must be reported as one.

## Honest confidence

- N1: 85% confirm. This is close to arithmetic.
- N2: 70% confirm. The `c4` bias is certain; whether it survives as a *median
  ratio* across real, possibly non-Gaussian cells is less so.
- N3: 60% confirm, and this is the one I most want to be wrong, because a
  refutation would invalidate a transfer we have already used.
- N4: 65% confirm.

## What would make me abandon this task

If N1 refutes --- three seeds turns out to be stable --- then the Student-t
correction is a technicality rather than a finding, and the resolvability
analysis needs no caveat. That would weaken a claim we have already made
prominently, and it would be the correct outcome to report.
