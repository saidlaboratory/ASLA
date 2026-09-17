# Two ways confidence statements fail in scaling-law extrapolation

Measured on DataDecide, projecting a 12-rung ladder to the 1B target (7.062e+20 FLOPs), where the true target value is **observed** rather than assumed.

## Failure 1: precisely wrong, not merely narrow

| | bootstrap | conformal |
| --- | --- | --- |
| empirical coverage (nominal 0.90) | **0.000** [0.000, 0.133] | 0.880 [0.700, 0.958] |
| median width | 0.0189 | 0.2841 |
| mean centring error | +0.0877 | +0.0877 |
| error-to-width ratio | **4.73** | 0.34 |
| recipes overshooting | 24/25 | 24/25 |

The bootstrap interval covers the observed target **0% of the time** --- 0 of 25 recipes. That is not undercoverage that a modest inflation would fix. The band is **4.73x narrower than its own centring error**: it is confidently, precisely wrong. A band propagating only seed noise cannot cover a projection sitting 119x the seed standard error from truth, at any bootstrap count.

Both methods share the same point projection and therefore the same centring error (+0.0877, 24 of 25 overshooting). They differ only in width. Conformal achieves valid coverage by being **15x wider**, not by being better centred --- its error-to-width ratio is 0.34, comfortably below one. Validity here is bought with width, and the width is the honest price of extrapolating a misspecified curve.

## Failure 2: a nominal guarantee that is not the achieved one

Undercovering intervals inflate a rule's real error rate by `(1 - empirical) / (1 - nominal)`. Stating delta while using them does not make the procedure delta-PAC:

| interval | nominal delta | effective delta | degradation |
| --- | --- | --- | --- |
| bootstrap | 0.05 | **0.500** | 10.0x |
| conformal | 0.05 | **0.060** | 1.2x |

A rule quoting delta=0.05 on bootstrap widths is in fact running at **0.500** --- a 10-fold degradation, and a coin flip rather than a guarantee. Conformal widths hold up: 0.060 against a nominal 0.05.

Taken together these are two quantified failures of confidence statements in this setting: one where the interval measures the wrong thing entirely, and one where the number attached to a procedure is not the number it achieves.
