# The reference distribution defect on small-seed leaderboards

A leaderboard that reports significance over a grid of recipes is making many simultaneous comparisons from very few seeds. Both halves of that sentence matter, and standard practice gets the second one wrong.

With `n` seeds per entry, the standard deviation is estimated from `n` points, so the reference distribution for a two-sided test is Student's t on `2n-2` degrees of freedom, not the normal. Under a Bonferroni correction for 300 comparisons at delta=0.05, the per-comparison level is 1.667e-04, and the two quantiles diverge sharply:

| seeds per entry | degrees of freedom | t quantile | normal quantile | ratio |
| --- | --- | --- | --- | --- |
| 3 | 4 | 13.653 | 3.765 | **3.63x** |
| 5 | 8 | 6.616 | 3.765 | **1.76x** |
| 10 | 18 | 4.731 | 3.765 | **1.26x** |
| 20 | 38 | 4.177 | 3.765 | **1.11x** |
| 50 | 98 | 3.916 | 3.765 | **1.04x** |

At three seeds --- the count DataDecide uses, and a common choice across published suites --- the Gaussian threshold is anti-conservative by **3.63x**. A procedure using it does not hold its stated error rate. The approximation only becomes adequate at seed counts nobody runs: the ratio is still 1.26x at ten seeds and reaches 1.04x only at fifty.

The multiplicity correction is what makes this bite. At a single comparison the t and normal quantiles differ by only 1.42x at the same degrees of freedom; it is pushing into the far tail, where the t distribution's heavier tail dominates, that opens the gap.

## How this was caught, and why it nearly was not

Not by inspection. The abstention rule reported 16.7% abstention on DataDecide at 1B where an independently written resolution diagnostic reported 54.2% on the same data. The pre-registration required any such discrepancy to be resolved before either number was reported. After the fix the two agree exactly, at 0.5417.

The rule's own test suite did not catch it, and the reason generalises. The test simulated the null and confirmed the family-wise error rate was held --- but it supplied the **true** standard error to the rule rather than estimating one from simulated seeds. A simulation that hands the estimator a known sigma cannot detect a wrong small-sample reference distribution, because the quantity the t correction exists to handle has been assumed away. The test passed for a rule whose guarantee did not hold.

The replacement draws seeds, estimates sigma from them exactly as the study does, and checks both variants: the t-based rule holds delta, the Gaussian one does not. This is the same pattern as the other measurement artifacts in this project --- the measurement could not see the thing it was supposed to check.
