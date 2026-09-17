# Where certification is free, and where it is unavailable

A delta-PAC rule that abstains rather than guessing was expected to be expensive: the pre-registered prediction was that it would decline to certify more than 60% of the comparisons a plain ranking gets right. At the top of the ladder that is wrong by a wide margin.

| delta | seeds | pairs | certified | certified errors | correct calls abstained |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 3 | 300 | 292 | **0** | 1.7% |
| 0.05 | 10 | 300 | 297 | **1** | 0.3% |

At the 1B target with 3 seeds, the rule certifies **292 of 300** pairs with **0 certified errors**, abstaining on only **1.7%** of the calls the baseline gets right. Correctness is close to free where gaps are wide.

## The complement: where nothing can be certified

The same rule abstains on everything at small scales, and that is the honest other half of the finding. Abstention at delta=0.05 with 3 seeds, by scale:

| scale | entries | abstention | median seeds required |
| --- | --- | --- | --- |
| 4M | 25 | 1.000 | 8 |
| 6M | 25 | 1.000 | 24 |
| 8M | 25 | 0.958 | 16 |
| 10M | 25 | 0.917 | 13 |
| 14M | 25 | 1.000 | 6 |
| 16M | 25 | 0.958 | 12 |
| 20M | 25 | 1.000 | 131 |
| 60M | 25 | 1.000 | 29 |
| 90M | 25 | 1.000 | 10 |
| 150M | 25 | 0.792 | 2 |
| 300M | 25 | 0.917 | 3 |
| 530M | 25 | 0.500 | 2 |
| 1B | 25 | 0.542 | 2 |

Across the 9 scales at or below 90M, abstention runs from **0.917 to 1.000**, and is total (1.000) at 6 of them. At those scales a 3-seed leaderboard supports at most a handful of certified orderings: nearly every adjacent gap is inside the noise. This is not a limitation of the rule --- it is a statement about where small-scale selection is possible at all, and on this suite the answer below 90M is essentially nowhere, at any delta a reader would accept.

Reported together, the two halves say something a single number would hide: certification is cheap exactly where the decision is easy, and unavailable exactly where small-scale proxies are most tempting to use.
