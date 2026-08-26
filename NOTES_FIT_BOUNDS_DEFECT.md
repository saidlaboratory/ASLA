# Defect note: the power-law `alpha` lower bound, and what the fix changed

## What the defect was

`asla.models.fit_power_law` constrained the decay exponent to `alpha >= 0.05`,
a constant chosen for BPB-like metrics. On DataDecide's OLMES macro-accuracy
error the true exponent is ~2.3e-5, about **3.3 orders of magnitude below that
floor**, so **all 25 of 25** interventions fitted at exactly `alpha = 0.05`. A
fit resting on its bound is not estimating the parameter: it returns the closest
curve the optimizer was *allowed* to express. Those fits were estimating an
offset with a fixed shape, not a decay rate.

Found by A1 of the adversarial audit (`AUDIT_ADVERSARIAL.md`): an independent
from-scratch re-implementation with no bounds disagreed with the audit on
exactly one intervention pair in each OLMES metric, which traced back to the
bound.

## What was fixed

1. **Adaptive bounds.** `adaptive_alpha_floor` scales the floor to the data's
   own dynamic range (decay per decade of compute), clamped to `[1e-8, 0.05]`,
   so metrics with BPB-like decay keep the historical floor and only
   slow-decaying metrics get a lower one. `fit_power_law(..., adaptive_bounds=True)`
   is now the default; pass `False` for the historical behaviour.
2. **Loud failure.** `detect_bound_pins` reports any parameter resting on a
   bound; `fit_power_law(..., require_interior=True)` raises `BoundPinError`
   instead of returning a pinned fit; and `asla.analysis.fits.bound_pin_report`
   runs over a whole design. Every audit design now records a pin report, and
   the rendered report prints a visible warning block for any design with pins.
   This class of silent failure cannot reach a report unannounced again.

## What changed in the numbers — and what did not

The corrected re-run (`FIRST_AUDIT.md`, `results/first_audit/`) reproduced
**every reported decision number on all four data-axis designs**, identical to
the prior version. The prior version is preserved as
`FIRST_AUDIT_v1_fixed_alpha_bounds.md` and
`results/first_audit_v1_fixed_alpha_bounds/`.

That is not because the fix did nothing. Measured by `audit/bounds_fix_impact.py`:

| metric | old alpha | new alpha | median projection change | as % of between-intervention spread | Spearman(old, new) | ranking identical |
|---|---|---|---|---|---|---|
| `c4_en_bits_per_token` | [0.1385, 0.1649] (interior) | [0.1385, 0.1649] | 1.3e-07 | 0.0% | 1.000000 | yes |
| `olmes_macro_error` | **all pinned at 0.0500** | [0.0171, 0.0337] | 6.5e-03 | **23.8%** | 1.000000 | yes |
| `olmes_macro_correct_prob_per_char_deficit` | [0.0713, 0.4585] (interior) | [0.0713, 0.4585] | 4.0e-08 | 0.0% | 1.000000 | yes |

On OLMES error the fitted exponents moved by more than a factor of two and the
projected target values moved by **24% of the entire between-intervention
spread** — a large change. The decision numbers did not move because the
distortion was **monotone**: pinning `alpha` above its true value biases every
intervention's projection in the same direction by a similar amount, since these
interventions share a fit range and differ mainly in level, not curvature. Decision
metrics depend only on the ordering of projections, and a monotone distortion
preserves ordering (Spearman exactly 1.000000).

**This is a fact about this dataset, not a general licence.** We got lucky, and
the luck is demonstrable rather than merely assertable.

### Demonstrated boundary condition

`audit/curvature_boundary_demo.py` builds three synthetic suites on the same
ladder, with the same seed noise, and applies the *same* alpha-floor defect. The
families are matched on the two things that would otherwise confound the
comparison: the same target spread (equally much signal to detect) and true
exponents below the 0.05 floor (so the defect binds on both). The only
difference is geometry.

| family | interventions pinned by the defect | pairs reordered per trial | trials with identical ordering | Spearman(defective, corrected) |
|---|---|---|---|---|
| `level` (shared exponent, offsets differ) | 8/8 | **0.017** | 98% | 0.99960 |
| `curvature` (pinned on the ladder, floors and exponents differ) | 1/8 | **1.150** | 33% | 0.96190 |
| `mixed` | 4/8 | 0.567 | 52% | 0.98571 |

On level geometry **all 8 interventions fit at the bound and the decisions are
still identical in 98% of trials** - the DataDecide situation reproduced
synthetically. On curvature geometry the same defect reorders
**69x more pairs**, and the ordering differs in
67% of trials. The caveat is therefore not a hedge: it is a
boundary condition with a measured location.

**C4 is unaffected either way.** Its exponents were always interior
([0.1385, 0.1649]), the adaptive floor never binds, and the independent
unconstrained re-derivation agrees with the audit to 8e-7 in projected value.
The headline metric is identical under both fits, and the headline decomposition
(14 fit-error flips, 2 inherited crossovers, all 12 excess flips attributable to
fit error) is byte-identical across versions.

## What the fix exposed

Freeing `alpha` did not make the OLMES fits well-posed; it moved the pathology
to the floor. With `alpha` free, 25 of 25 OLMES-error fits now put `E` at zero,
and profiling the likelihood shows it is **exactly flat** in `E`
(`SSR(E=0)/SSR(best) = 1.0000`). That is a standalone result about downstream
accuracy metrics, reported in `AUDIT_ADVERSARIAL.md`: three-parameter
scaling-law extrapolation is not identifiable on them.

## The two-sided implication

This episode is the sharpest available illustration of the project's central
methodological argument, and it cuts both ways. Both halves belong in the paper.

**In favour of decision-level evaluation.** The bounded fit was badly wrong in
parameter space - exponents off by a factor of three, projections displaced by
24% of the between-intervention spread - and *exactly right* in decision space.
Fit quality and decision quality are not the same thing, and optimising or
reporting the former does not certify the latter. That is the argument for
measuring what we actually care about, which is what ASLA does.

**Against over-reading a good decision number.** The same fact says decision
accuracy is *insensitive* to real estimator defects. A pipeline can be
substantively broken and still produce the right ordering, which means a good
decision-accuracy number certifies less than it appears to. It certifies the
decision, on this suite, under this geometry - not the estimator, and not
transportability to a suite whose interventions differ in curvature. Concretely:
our own headline decomposition is unchanged by a defect that made every OLMES
exponent meaningless, so nobody should read that stability as evidence the
fitting is sound.

The practical consequence is that decision-level evaluation needs
parameter-level diagnostics alongside it, not instead of it. That is why the
bound-pin report now runs on every design and prints into the report rather than
living in a log.

## Methodological point for the paper

Fit bounds tuned for one metric family silently misfit another. A bound chosen
to stabilise BPB fits converted every accuracy fit into an offset-only model
without any error, warning, or visible symptom in the reported numbers — and in
this case without changing them either, which is worse, because nothing would
ever have prompted a check. Any pipeline that fits scaling laws across metric
families should (a) scale its bounds to the data rather than hardcoding them,
and (b) treat a fit resting on a bound as a failure to report, not a result to
use.
