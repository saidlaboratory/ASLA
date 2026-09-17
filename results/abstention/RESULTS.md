# Abstention study: 1B target

Target 7.062e+20 FLOPs; 12 fitting rungs; 400 bootstrap replicates. Thresholds use Student's t on 2n-2 degrees of freedom, since sigma is estimated.

## Pre-registered predictions

| prediction | verdict | evidence |
| --- | --- | --- |
| A1 (widths undercover) | **PARTIALLY REFUTED (conformal calibrated)** | bootstrap 0.000 [0.000, 0.133] (n=25, median width 0.0189); conformal 0.880 [0.700, 0.958] (n=25, median width 0.2841); nominal 0.90. Both methods share the same point projection, so the same centring error (+0.0877, 24/25 overshooting); they differ only in width. Error-to-width ratio 4.73 for bootstrap versus 0.34 for conformal: conformal covers by being 15x wider, not by being better centred |
| A2 (abstention vs delta and budget) | **REFUTED (level)** | at delta=0.05, 3 seeds: 1B 0.5417, mean across scales 0.8910; monotonicity violations: 0 |
| A3 (seeds required) | **REFUTED** | median seeds required at 1B: 2; median across scales 10; range 2-131; 10 adjacent pairs unresolvable within the cap |
| A4 (cost of correctness) | **REFUTED** | delta=0.05, 3 seeds: 0.017 of 297 correct calls abstained, certified error rate 0.0000; delta=0.05, 10 seeds: 0.003 of 297 correct calls abstained, certified error rate 0.0034 |
| A5 (effective delta) | **CONFIRMED** | bootstrap: nominal 0.05 -> effective 0.5000; conformal: nominal 0.05 -> effective 0.0600 -- the guarantee is conditional on Task 4's width correction |

## Abstention rate by delta and seed budget

| delta | 1 seeds | 2 seeds | 3 seeds | 5 seeds | 10 seeds | 20 seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 1.000 | 0.792 | 0.417 | 0.208 | 0.125 | 0.125 |
| 0.1 | 1.000 | 0.875 | 0.500 | 0.250 | 0.125 | 0.125 |
| 0.05 | 1.000 | 0.875 | 0.542 | 0.292 | 0.167 | 0.125 |
| 0.01 | 1.000 | 0.958 | 0.667 | 0.292 | 0.167 | 0.125 |

Leaderboard: DataDecide at 1B, adjacent orderings.

## Abstention across scales at delta=0.05, 3 seeds

| scale | entries | abstention | median seeds required | never resolvable |
| --- | --- | --- | --- | --- |
| 4M | 25 | 1.000 | 8 | 2 |
| 6M | 25 | 1.000 | 24 | 0 |
| 8M | 25 | 0.958 | 16 | 1 |
| 10M | 25 | 0.917 | 13 | 1 |
| 14M | 25 | 1.000 | 6 | 2 |
| 16M | 25 | 0.958 | 12 | 1 |
| 20M | 25 | 1.000 | 131 | 1 |
| 60M | 25 | 1.000 | 29 | 1 |
| 90M | 25 | 1.000 | 10 | 0 |
| 150M | 25 | 0.792 | 2 | 0 |
| 300M | 25 | 0.917 | 3 | 1 |
| 530M | 25 | 0.500 | 2 | 0 |
| 1B | 25 | 0.542 | 2 | 0 |
