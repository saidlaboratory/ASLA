# Allocation study: 1B target

Target 7.062e+20 FLOPs, ladder of 12 rungs, lever arm 4.72x, 25 interventions, 400 bootstrap replicates.

## Pre-registered predictions

| prediction | verdict | evidence |
| --- | --- | --- |
| P1 (allocation shape) | **REFUTED (sparsity)** | support points 5-12; top-rung cost share 0.000 to 0.656 across budgets (0.01x:0.000, 0.03x:0.000, 0.1x:0.000, 0.3x:0.000, 0.5x:0.315, 1x:0.656); the >=3 support points refute the Caratheodory claim |
| P2 (variance reduction) | **PARTIALLY CONFIRMED** | 0.01x: 0.0884; 0.03x: 0.1446; 0.1x: 0.2511; 0.3x: 0.4929; 0.5x: 0.6790; 1x: 1.0000 (in the predicted band at some budgets only) |
| P3 (hard bar) | **CONFIRMED (as a negative)** | single-scale 0.0078 [0.0033, 0.0133]; excess by budget: 0.01x: +0.1169, 0.03x: +0.0954, 0.1x: +0.0778, 0.3x: +0.0591, 0.5x: +0.0534, 1x: +0.0277 |
| P4 (SL2 head-to-head) | **CONFIRMED** | max cost-share gap 0.00058 (0.01x: 0.00000; 0.03x: 0.00000; 0.1x: 0.00038; 0.3x: 0.00000; 0.5x: 0.00058; 1x: 0.00000) |
| P5 (bias-aware shift) | **REFUTED** | max |cost-share shift| 0.00000; 0.01x: top +0.0000, cheap +0.0000; 0.03x: top +0.0000, cheap +0.0000; 0.1x: top +0.0000, cheap +0.0000; 0.3x: top +0.0000, cheap -0.0000; 0.5x: top -0.0000, cheap +0.0000; 1x: top +0.0000, cheap -0.0000; signed deviation correlates with log C at -0.0217 -- the measured deviation is too small to move the design |

## Decision accuracy at matched compute

| budget | cost vs single-scale | decision-optimal | estimation-optimal (SL2) | bias-aware | uniform ladder |
| --- | --- | --- | --- | --- | --- |
| 0.01x ladder | 0.015x | 0.1247 | 0.1252 | 0.1241 | 0.1245 |
| 0.03x ladder | 0.046x | 0.1032 | 0.1032 | 0.1026 | 0.1032 |
| 0.1x ladder | 0.152x | 0.0856 | 0.0842 | 0.0850 | 0.0844 |
| 0.3x ladder | 0.457x | 0.0669 | 0.0665 | 0.0666 | 0.0671 |
| 0.5x ladder | 0.762x | 0.0612 | 0.0608 | 0.0610 | 0.0503 |
| 1x ladder | 1.524x | 0.0355 | 0.0353 | 0.0356 | 0.0352 |

Single-scale ranking at the top rung: **0.0078** [0.0033, 0.0133], costing 1.122e+22 FLOPs.


# Allocation study: 60M target

Target 1.955e+18 FLOPs, ladder of 7 rungs, lever arm 8.93x, 25 interventions, 400 bootstrap replicates.

## Pre-registered predictions

| prediction | verdict | evidence |
| --- | --- | --- |
| P1 (allocation shape) | **REFUTED (sparsity)** | support points 2-7; top-rung cost share 0.000 to 0.348 across budgets (0.03x:0.000, 0.1x:0.000, 0.3x:0.142, 1x:0.348); the >=3 support points refute the Caratheodory claim |
| P2 (variance reduction) | **REFUTED (band)** | 0.03x: 0.6119; 0.1x: 0.6119; 0.3x: 0.6681; 1x: 1.0000 (never in the predicted [0.25, 0.45] band) |
| P3 (hard bar) | **CONFIRMED (as a negative)** | single-scale 0.0714 [0.0467, 0.0933]; excess by budget: 0.1x: +0.2345, 0.3x: +0.0675, 1x: +0.0893 |
| P4 (SL2 head-to-head) | **CONFIRMED** | max cost-share gap 0.00269 (0.03x: 0.00269; 0.1x: 0.00269; 0.3x: 0.00000; 1x: 0.00000) |
| P5 (bias-aware shift) | **REFUTED** | max |cost-share shift| 0.00000; 0.03x: top +0.0000, cheap -0.0000; 0.1x: top +0.0000, cheap -0.0000; 0.3x: top -0.0000, cheap -0.0000; 1x: top -0.0000, cheap -0.0000; signed deviation correlates with log C at 0.0066 -- the measured deviation is too small to move the design |

## Decision accuracy at matched compute

| budget | cost vs single-scale | decision-optimal | estimation-optimal (SL2) | bias-aware | uniform ladder |
| --- | --- | --- | --- | --- | --- |
| 0.03x ladder | 0.086x | n/a | n/a | n/a | n/a |
| 0.1x ladder | 0.287x | 0.3060 | 0.3090 | 0.3031 | 0.3655 |
| 0.3x ladder | 0.862x | 0.1389 | 0.1446 | 0.1645 | 0.2247 |
| 1x ladder | 2.874x | 0.1607 | 0.1640 | 0.1638 | 0.1619 |

Single-scale ranking at the top rung: **0.0714** [0.0467, 0.0933], costing 1.643e+19 FLOPs.

Apparent non-monotonicity in budget: 0.3x -> 1x: 0.1389 [0.0867, 0.1967] to 0.1607 [0.1033, 0.2167] (intervals overlap; not a real reversal).
