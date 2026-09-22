# Pre-registration: the candidate bootstrap, and how many candidates a comparison needs

Written 2026-09-22, before any of the computation below has been run.

## A defect in the candidate bootstrap

`candidate_resampled_difference` (scripts/run_expected_error.py) and
`paired_candidate_test` (scripts/run_rescoring.py) draw 25 candidates with
replacement and then score the pairs among the *unique* candidates drawn
(`set(subset)`). A with-replacement draw of 25 contains about 16 unique
candidates, so each replicate is the statistic at about 16 candidates, not 25.
The bootstrap spread then estimates the standard error at the smaller size, and
the reported intervals, including the C4 interval [-0.21, +4.98] with p = 0.12,
are too wide by an unknown factor.

## Method, fixed now

For a comparison between two rankers, let `h(a, b)` be the difference in
expected-error charge on the pair `(a, b)`. The statistic is the mean of `h` over
all pairs of distinct candidates, a U-statistic of order two.

- **Primary: multiplicity-weighted candidate bootstrap.** Draw candidates with
  replacement. Weight each pair of distinct candidates by the product of their
  draw counts; a candidate drawn twice does not pair with itself. Percentile
  interval, and two-sided p from the same helper as before. 4000 replicates.
- **Cross-checks.** (i) The U-statistic variance
  `Var = 2/(n(n-1)) * (2(n-2) zeta1 + zeta2)`, where `zeta1` is the variance of
  the candidate-level mean of `h` and `zeta2` the variance of `h`, with standard
  unbiased estimates. (ii) The delete-one-candidate jackknife.
- **Power.** Hold the estimated `zeta1` and `zeta2` fixed and evaluate the
  variance at `N` candidates. The candidates needed are the smallest `N` for which
  a two-sided test at 0.05 has power 0.8 against a true difference `delta`.
  Report this at the observed point estimate, and on a grid of `delta`.
- **Validation of the scaling.** Subsample `m` of the 25 candidates without
  replacement, for several `m`. Compare the empirical standard deviation of the
  statistic with the formula's finite-population value at `m`.

Everything that used the unique-set bootstrap is re-run with the corrected one:
the C4 test in expected_error.json and every candidate interval in
rescoring.json.

## Predictions

- **Q1.** The corrected bootstrap standard error of the C4 difference is smaller
  than the unique-set one by a factor between 1.1 and 1.4.
- **Q2.** The corrected C4 interval still includes zero.
- **Q3.** Power 0.8 at the observed C4 point estimate needs between 40 and 120
  candidates.
- **Q4.** The formula's standard error agrees with the subsampling standard
  deviation to within 20% at every `m` tested.

**Decision rule.** The paper reports the corrected p-value, whatever it is. The
claim that the C4 comparison is underpowered stands only if the corrected
interval includes zero. If it does not, the C4 claim is rewritten as a
difference established at this candidate count. Any verdict in the re-scoring
table that changes is reported as changed.
