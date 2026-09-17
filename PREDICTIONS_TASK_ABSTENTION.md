# Pre-registration: stopping rule and abstention (Task 3)

Written before computing. The allocation results (Task 2) are committed at
`661d818`; nothing in Task 3 has been run.

Frozen before computing: the pairwise estimand, the cell-wise seed bootstrap,
the leaderboard entries and sigma sources used by
`results/resolution/resolution_study.json`, and the conformal interval
implementation in `asla/analysis/conformal.py`. Any hyperparameter introduced
here (interval type, calibration set) has its provenance recorded as `lambda`
does in `NOTES_SHRINKAGE_STRENGTH.md`.

> **Post-hoc correction (recorded, not silently edited).** The scatter figures in
> section 0 were written as 4.22x / 17.8x. A later audit found that ratio pooled
> seed standard errors as `mean(se)` against a dof-corrected residual scale; a
> ratio of scales requires `sqrt(mean(se^2))`. The corrected values are 2.90x and
> 8.40x, now computed in `scripts/run_theory_v2.py`. Nothing in A1-A5 depends on
> the magnitude --- the predictions turn on whether the error is compute-structured,
> not on how large it is --- and no verdict changes.

## 0. Why this task, and what changed

Tasks 1-2 closed the space of correctable causes for the projection failure:
better estimators (shared-exponent, EB shrinkage, ensemble, checkpoint-augmented),
a better objective (decision- vs estimation-optimal, proved to coincide), and a
better ladder (optimally allocated, matched compute) all fail to beat a baseline
that fits nothing and extrapolates nothing. The remaining constructive question is
not how to select better but **when no selection is possible**.

**The two channels, kept separate from here on.** This distinction was blurred in
earlier framing and is now load-bearing:

| channel | what it does | status |
| --- | --- | --- |
| **bias** | moves the point estimate | **empty on our data** |
| **variance** | widens the interval, leaves the point estimate alone | **intact and large** |

The bias channel is empty because the measured deviation is a *wave, not a trend*:
signed mean residuals oscillate across the ladder (+0.045, -0.040, +0.044, -0.043)
and correlate with `log C` at only **-0.022** (1B) and **+0.007** (60M). An
oscillating deviation is absorbed into the intercept, which is orthogonal to the
extrapolation direction, so it barely biases the projection. Measured
consequence: bias-aware and variance-only allocations are identical to five
decimal places (P5, refuted).

The variance channel is untouched by that finding. Residual scatter is 2.90x the
seed standard error (8.40x variance inflation), and `phi` --- a pure
variance-inflation quantity --- varies 2.7-8.9x across ladders and under-predicts
held-out structure by 12x. **So Task 4 is no longer about correcting the design.
It is about correcting the confidence widths that this task's abstention rule
depends on**, and Task 3 must therefore report its guarantee as conditional on
widths whose calibration Task 4 tests.

## 1. What Réda et al. imply for the narrower claim

Réda, Tirinzoni and Degenne (arXiv:2111.01479, NeurIPS 2021) prove a lower bound
for δ-correct Top-m identification under misspecified linear models and conclude
that *"knowing the scale of the deviation from linearity is necessary to exploit
the structure of the problem."*

**The scope of that necessity matters and was checked.** It is necessary *to
exploit the linear structure*, i.e. to beat the sample complexity obtainable with
no structure at all --- not necessary for δ-correctness itself. The literature
states the consequence directly: without a known bound, "any algorithm cannot
achieve a better sample complexity than that obtainable when no structure is
available."

*Verification note.* The formal ε-deviation model and the theorem numbering could
not be extracted from the PDF with the tools available here; this reading rests on
the abstract's own wording, which is unambiguous about "exploit the structure",
corroborated by secondary statements of the same result. The claims below are
therefore scoped to what that wording supports, and the paper is cited for the
*direction* of the result, not for a quoted theorem number.

**Corollary for us.** Two regimes, and we can say which one we are in:

- *Structure-exploiting.* Project along the fitted power law to `C*` and claim the
  variance reduction the parametric form buys. This requires a known deviation
  scale. Ours is unknown and, worse, its compute-structured component is
  near-zero, so there is little structure to exploit in the extrapolation
  direction even if we knew it.
- *Structure-free.* Use per-arm empirical evidence without trusting the
  parametric extrapolation. δ-correctness remains achievable; the price is the
  unstructured sample complexity.

The honest position is that scaling-law selection sits in the **structure-free**
regime for the extrapolation decision, and that a δ-PAC abstention rule is
therefore still available while a structure-exploiting speed-up is not. That is a
weaker but defensible guarantee, and it is the one Task 3 tries to deliver.

## 2. The deliverable

A rule that, with probability >= 1 - δ, either returns the true best recipe at
`C*` or abstains. Abstention is the formalisation of the resolution-limit result:
**437 of 468 adjacent leaderboard orderings (93.4%) are already unresolvable at
the seed budget actually used**, and 28 are unidentifiable at any seed budget.
A leaderboard that always returns a ranking is running a procedure with no
correctness guarantee.

## 3. Predictions

### A1 --- Naive parametric widths do not achieve nominal coverage
Bootstrap and conformal projection intervals built on fit-range data will
undercover the true target value.

- Point prediction: at nominal 90%, empirical coverage **below 0.85** for the
  bootstrap interval and **below 0.70** for the conformal interval, on a ladder
  extrapolating several-fold beyond its top rung.
- Refuted if: either interval achieves >= 0.88 empirical coverage.
- UNDERPOWERED if: the coverage CI spans the nominal level.
- Rationale, stated as a prior rather than as a result: the module docstring for
  `scripts/run_calibration_study.py` reports ~0.82-0.86 and ~0.55 from a `--fast`
  exploratory run. **Those numbers are not evidence** --- they are a fast-mode
  smoke test, they have never been committed to `results/`, and A1 is scored only
  against a full run written to `results/abstention/`.

### A2 --- Abstention rate falls with budget and rises as δ falls
Sweeping δ over {0.20, 0.10, 0.05, 0.01} and the seed budget over {1, 3, 5, 10,
20}, the abstention rate will be monotone decreasing in budget at fixed δ and
monotone increasing as δ decreases at fixed budget.

- Point prediction: at δ = 0.05 and 3 seeds, abstention **> 0.85** on DataDecide
  adjacent pairs, consistent with the 93.4% unresolved figure.
- Refuted if: either monotonicity is violated beyond bootstrap noise, or
  abstention at δ = 0.05 with 3 seeds is < 0.70.
- This is the headline output and it is expected to confirm; the interest is in
  the *shape* of the curve, specifically the budget at which abstention drops
  below one half.

### A3 --- Most pairs cannot be resolved at any affordable budget
The seed budget required to resolve a majority of adjacent pairs at δ = 0.05 will
exceed what any published suite has used.

- Point prediction: median seeds required **> 20** across DataDecide adjacent
  pairs; the 28 pairs already flagged unidentifiable remain unresolved at every
  budget tested.
- Refuted if: median seeds required <= 10.

### A4 --- A correct rule is not free: it abstains where the baseline guesses right
Among pairs where single-scale ranking happens to be correct, the δ-PAC rule will
abstain on a substantial fraction --- i.e. it gives up accuracy it cannot certify.

- Point prediction: the rule abstains on **> 60%** of pairs that single-scale
  ranking calls correctly at δ = 0.05, 3 seeds.
- Refuted if: < 30%.
- **Interpretation fixed in advance.** A high number here is *not* an argument
  against abstention and must not be reported as one. It quantifies how much of
  the leaderboard's apparent accuracy is uncertified rather than earned. The
  honest framing is that the baseline is right more often than it can prove, and
  that a procedure with a guarantee must say so.

### A5 --- Coverage conditions the guarantee (the bridge to Task 4)
The δ-PAC guarantee holds only if the intervals are calibrated. Using the
empirical coverage from A1 in place of the nominal level, the *effective* δ will
be materially worse than the nominal one.

- Point prediction: effective error rate at nominal δ = 0.05 exceeds **0.10**
  (i.e. at least a two-fold degradation) when intervals are built parametrically.
- Refuted if: effective error rate <= 0.07.
- If confirmed, the guarantee must be stated as conditional and Task 4's width
  correction becomes load-bearing rather than optional. **This is the prediction
  that decides whether Task 3's contribution stands on its own or depends on
  Task 4.**

## 4. Honest confidence

- A1: 85% confirm. Extrapolation is not exchangeable with interpolation and the
  conformal scores measure the wrong error.
- A2: 90% confirm. It largely restates a measured result in a new form; the risk
  is a monotonicity violation from bootstrap noise at small budgets.
- A3: 75% confirm.
- A4: 70% confirm.
- A5: 80% confirm, and this is the one I most want to be wrong, because a
  confirmation means the headline abstention numbers are themselves conditional
  on a correction that has not yet been validated.

## 5. What would make me abandon this task's contribution

If A1 is refuted --- if off-the-shelf intervals are in fact calibrated for
extrapolation --- then the abstention rule is a routine application of existing
machinery and the resolution-limit result already says everything interesting.
The contribution would reduce to the empirical abstention curve, which is worth
reporting but is not a method.

If A2's abstention rate at realistic budgets came out *low* (< 0.30), it would
contradict the committed resolution study, and the discrepancy would have to be
resolved before either number is reported.
