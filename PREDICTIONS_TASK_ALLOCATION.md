# Pre-registration: optimal allocation (Task 2c/2d)

Written before computing the retrospective evaluation on DataDecide. The
allocation solver and the equivalence result (Task 2a/2b) are already committed at
`9edfe81`; nothing below has been run.

Frozen before computing: splits (`asla/analysis/splits.py`), the pairwise
estimand, the cell-wise seed bootstrap, and the held-out target. The deviation
function used by the bias-aware design is estimated on the fitting range only
(Task 4); where 2c needs one before Task 4 exists, it is fitted on the training
interventions only and that provenance is recorded exactly as `lambda` is in
`NOTES_SHRINKAGE_STRENGTH.md`.

## What is already known, and therefore not being predicted

These are measured and committed; they constrain the predictions but are not
themselves at stake.

- Single-scale ranking at the top rung: **1.33%** pairwise mis-selection on the
  primary DataDecide design (`results/adversarial/a1_independent_rederivation.json`).
- Across 319 specifications in the specification curve, plain projection **never**
  beat single-scale ranking: `max_excess_among_winning = 0.0`, and the 15
  "winning" designs all win by exactly zero (`results/adversarial/a2_dose_response.json`).
- Excess mis-selection tracks the **lever arm**, not the number of fitted budgets:
  Spearman +0.564 (p = 3.2e-28) against log lever arm, versus -0.136 against
  budget count.
- Design-for-estimation and design-for-decision coincide in the well-specified
  linear model (correlation 0.99995; <=0.11% excess variance at `C*`). The SL2
  head-to-head therefore cannot be a contest of argmins, and P4 below states what
  it is instead.

## The hard bar

**Does the optimally-allocated estimator beat single-scale ranking at matched
exploration compute on DataDecide?**

My honest confidence: **no, at roughly 20%.** This is against the framing I was
asked to build, and I am stating it in advance rather than after seeing the
result. The reasoning is that single-scale ranking at the top rung has *zero fit
variance* --- it estimates no curve and extrapolates nothing --- while every
projection estimator pays a variance cost proportional to `h*`, plus a bias cost
from misspecification that no allocation can remove. Optimal allocation shrinks
the first term and can *worsen* the second, because the variance-optimal design
puts most of its runs on cheap, badly-biased rungs. A better ladder does not fix
an estimator whose error is dominated by functional-form bias.

Where the method can still win: at large lever arms, where single-scale is not
available at all because there is no run at or near the target.

## Predictions

### P1 --- Shape of the decision-optimal allocation
The cost-constrained decision-optimal design will be **sparse (at most 2 support
points)** and will **not** place the majority of its cost at the top rung of the
ladder.

- Point prediction: cost share at the top rung **< 0.25**.
- Refuted if: cost share at the top rung >= 0.50 on the primary DataDecide design.
- Rationale: the c-optimal design for a single target direction needs at most two
  points, and cost weighting makes cheap rungs attractive; the exploratory probe
  on a synthetic DataDecide-like ladder gave two support points with **zero**
  cost at the top rung.
- **This prediction is the one I most expect to be qualitatively right and
  practically misleading**, because it is a pure-variance statement. P5 is where
  it gets tested against bias.

### P2 --- Variance reduction at matched compute
The decision-optimal allocation will reduce the target-gap variance factor
relative to the uniform 3-seed ladder at equal FLOPs.

- Point prediction: variance ratio (optimal / uniform) in **[0.25, 0.45]**.
  The synthetic probe gave 0.1728 / 0.5219 = **0.331**; the interval allows for
  the real ladder's different spacing.
- Refuted if: ratio > 0.9 (no meaningful gain) or the optimum fails to beat
  uniform at all.
- UNDERPOWERED if: the bootstrap CI on the ratio spans 1.0.

### P3 --- The hard bar, on decision accuracy
At matched exploration compute on DataDecide, the decision-optimally-allocated
projection estimator will **not** beat single-scale ranking.

- Point prediction: mis-selection **>= 1.33%**, i.e. excess over single-scale
  **>= 0**, consistent with all 319 prior specifications.
- Confirmed (as a negative) if: excess >= 0 with the bootstrap CI excluding a
  win of more than 0.5 percentage points.
- **Refuted if**: excess < 0 with a bootstrap CI excluding 0. That would be the
  first time in this project that any projection estimator beat single-scale, and
  it would need the artifact check applied before being believed --- specifically,
  confirm the comparison is at genuinely matched FLOPs and that the single-scale
  arm is not being charged for runs the projection arm also uses.
- UNDERPOWERED if: the CI on the excess spans both -0.5pp and +0.5pp.

### P4 --- SL2 head-to-head (reframed, because the argmin contest is settled)
Implementing SL2's estimation objective as an allocation rule and scoring it on
**decision accuracy** will produce a result statistically indistinguishable from
the decision-optimal allocation.

- Point prediction: absolute difference in mis-selection **< 0.5 percentage
  points**, and cost shares agreeing within **0.02**.
- Refuted if: the two differ by >= 1 percentage point in mis-selection.
- **Interpretation is fixed in advance so the outcome cannot be spun.** A null
  here is *not* a failure to find a contrast; it is the empirical form of the
  analytic result that the two objectives coincide. The contribution becomes
  "design-for-estimation is already decision-optimal under the linear model, and
  the place it fails is bias, not objective choice" --- which is a sharper and
  more defensible claim than the one in the original framing, and it credits SL2
  rather than strawmanning it.

### P5 --- Bias-aware allocation departs from variance-optimal
Once the measured compute-structured deviation is supplied, the bias-aware
optimum will shift cost **up** the ladder relative to the variance-only optimum.

- Point prediction: cheap-rung cost share strictly decreases; top-rung cost share
  increases by **>= 0.05** at the deviation magnitude estimated from data.
- Refuted if: the bias-aware and variance-only allocations agree within 0.02 in
  cost share at every rung, i.e. the measured deviation is too small to matter.
- UNDERPOWERED if: the estimated deviation's own uncertainty is large enough that
  the shift's sign is not determined (bootstrap CI on the top-rung share change
  spans 0).
- This is the prediction that carries the contribution. If P5 holds and P3 also
  holds (no win on DataDecide), the honest summary is that allocation matters for
  *variance* while the binding constraint is *bias* --- and that is the bridge to
  Task 4.

## The concentration outcome is a result either way

Stated in advance so there is no pressure to force a spread.

- **If the optimum concentrates at the top rung**, that is a theoretical
  explanation of the empirical puzzle this audit produced: single-scale ranking is
  strong *because* the decision-optimal design is nearly degenerate at the top
  rung, so ranking at the top rung is close to the optimal transductive design.
  Single-scale ranking would then be near-optimal by construction rather than by
  luck, which is the answer to DataDecide's open question and a stronger result
  than a small win on mis-selection.
- **If the optimum spreads**, then single-scale ranking discards usable
  information, and the gap between it and the optimal design is the headroom the
  method can claim.

Both are publishable; neither is a fallback. The exploratory probe points toward
"spreads", which makes the first outcome the surprising one and therefore the one
to guard against wishful reading.

## What would make me abandon the allocation contribution

If P2 holds (variance genuinely falls) but P3's excess is *large and positive*
(say >= 5 percentage points), then better allocation actively hurts decisions,
which would mean the variance-optimal design concentrates runs where the model is
most wrong. That is a coherent outcome given our measurements, it would refute the
premise that better design helps selection, and it should be reported as the
headline rather than buried.
