# Pre-registration: robustness of the sign across geometries, and what drives the design dependence

Committed before any slope for these tasks is computed. Data: DataDecide C4 bits per token, 750M excluded.
Analysis only; no paper edits.

## The frozen statistic

- **Designs:** fit windows of at least 3 consecutive rungs (consecutive within the geometry's sizes), each
  scored at a target above it.
- **Kernel:** the expected-error charge of projection minus single-scale ranking, against the target's own
  three-seed evidence, in points.
- **Slope:** the fixed-target pooled slope. Slices are (target, rung count) with at least 3 windows, and the
  slope is the pooled within-slice OLS slope of excess on log10 lever arm, in points per decade.
- **Uncertainty:**
  - SE: the leave-one-recipe-out jackknife.
  - Interval: estimate ± 1.96 SE. Every calibrated critical value for fixed-target statistics was below 1.96,
    so the normal value is used.
- **Computation:** a fast path precomputes each window's kernel once. It must reproduce
  `asla.analysis.confound.window_designs` with `asla.analysis.fixed_target` exactly on a check set of
  geometries; that is a test.

## Task 1: is the sign robust across geometries?

### Valid geometry

A subset of DataDecide's 13 sizes whose largest member is a target of 150M, 300M, 530M or 1B with at least 5
sizes below it.

- **Targets used:** every target in the subset with at least 5 subset sizes below it. Each such target gives
  at least three 3-rung windows.
- **Determinacy:** all four targets have at least 50% of pairs determined on the full data, and determinacy
  depends only on the target.
- **Count:** 5,682 geometries.
  - By top target: 1B 3,302, 530M 1,486, 300M 638, 150M 256.
  - By size count: 6 sizes 1,632; 7 sizes 1,680; 8 sizes 1,278; 9 sizes 714; 10 sizes 286; 11 sizes 78;
    12 sizes 13; 13 sizes 1.

### Reported

- the number of geometries;
- the distribution of slopes (quantiles);
- the share positive, the share with an interval excluding zero on the positive side, and the share
  significantly negative;
- all of these broken down by top target, by size count, and by the number of targets.

The geometries overlap heavily, so they are not pooled into a test. This is a description of robustness.

### Predicted bands

- **R1:** share positive between 85% and 100%.
- **R2:** share positive with an interval excluding zero between 60% and 90%.
- **R3:** share significantly negative between 0% and 5%.
- **R4:** the share positive and excluding zero is lowest for geometries topped by 1B and highest for those
  topped by 150M.
- **R5:** the share positive and excluding zero rises with size count.

## Task 2: what drives the design dependence?

### Influence

At each target (150M, 300M, 530M, 1B):

- Take the full ladder up to that target; the slope uses that target only.
- Drop each size below the target in turn and recompute.
- **Influence** = (slope without the size − full slope) / SE of the full slope.
- **Flag** a size if |influence| > 1, or if its removal flips the sign.

### Decomposition

Two designs:

| Design | Sizes | Real slope | Curve-implied |
|---|---|---|---|
| 300M, 8 sizes | 4M, 6M, 14M, 20M, 60M, 90M, 150M, 300M | +21.8 | +11.0 |
| 1B, 6 sizes | 4M, 10M, 60M, 90M, 300M, 1B | +3.8 | +7.3 |

**(i) Individual sizes.**

- Shapley contributions of each size the design removes from the full ladder up to its top target to the
  change in slope from that full ladder to the design.
- The value function is the slope on the design's sizes plus any subset of the removed sizes, with the
  design's targets.

**(ii) Rung count and top-rung position.**

The gap (design slope − curve-implied slope) splits exactly into per-window terms. Each window contributes
(x − x̄)(y − c) / Σ(x − x̄)², where:

- x is the window's lever arm;
- y is the design excess;
- c is the fixed-target curve value at that lever arm (from `results/target_scoring/excess_curve.json`, the
  target's own curve);
- the means and denominator are within slices.

These per-window terms are summed by slice (target and rung count) and by top-rung size.

**Concentration rule (decided now).** The dependence is "concentrated" if the two sizes with the largest
|Shapley contribution| account for at least 60% of the sum of |Shapley contributions|. Otherwise it is
"spread".

**For flagged or concentrated sizes, report:**

- misfit: the between-recipe SD of the recipe-specific residual at that size;
- noise: the median relative seed SD;
- tokens per parameter;
- the gaps in log10 compute to the neighbouring sizes.

### Predictions

- **I1:** the sizes on either side of DataDecide's largest compute gap (20M and 60M, 0.95 decades apart) are
  flagged at one or more targets.
- **I2:** the dependence in both designs is concentrated (under the rule above).

## Outputs

- `results/target_scoring/sign_robustness.json`
- `results/target_scoring/design_dependence.json`
- `asla/analysis/geometry_scan.py`
- `scripts/run_sign_robustness.py`
- `tests/test_geometry_scan.py`
