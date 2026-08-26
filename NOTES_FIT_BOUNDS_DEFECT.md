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

**This is a fact about this dataset, not a general licence.** On a suite where
interventions differ in curvature rather than level, the same defect would
reorder them and the decision numbers would change. The correct reading is that
we got lucky, not that bounds do not matter.

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

## Methodological point for the paper

Fit bounds tuned for one metric family silently misfit another. A bound chosen
to stabilise BPB fits converted every accuracy fit into an offset-only model
without any error, warning, or visible symptom in the reported numbers — and in
this case without changing them either, which is worse, because nothing would
ever have prompted a check. Any pipeline that fits scaling laws across metric
families should (a) scale its bounds to the data rather than hardcoding them,
and (b) treat a fit resting on a bound as a failure to report, not a result to
use.
