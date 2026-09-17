# Formulation: scaling-law selection as transductive pure exploration

This states the problem the audit has been circling, in the language of
transductive linear bandits, and marks precisely where the model is an
assumption and where our own measurements say it fails.

Implemented in `asla/analysis/transductive.py` and `asla/analysis/allocation.py`;
every claim below that is numerical is checked in `tests/test_transductive.py` or
`tests/test_allocation.py`.

## 1. The problem

**Recipes.** A finite set `R` of candidate training interventions (data recipes,
optimizers). `|R| = 25` for DataDecide.

**Sampled arms.** `A = {(r, C) : r in R, C in ladder}`, the (recipe, compute
budget) pairs we can afford to train. Pulling `(r, C)` costs `C` FLOPs and
returns a noisy evaluation of recipe `r` trained at budget `C`.

**Target set.** `Z = {(r, C*) : r in R}` --- the same recipes at the target
budget `C*`, which is never trained by anyone. This is what makes the problem
transductive in the sense of Fiez, Jain, Jamieson and Ratliff (NeurIPS 2019,
arXiv:1906.08399): the set we sample and the set we must decide about are
disjoint.

**Feature map.** In log space the two-parameter power law `log E = a - alpha log C`
is linear in

    x(C) = (1, -log C)

so recipe `r`'s value at any budget is the linear functional `theta_r' x(C)` with
`theta_r = (a_r, alpha_r)`.

**Noise.** Cell means at `(r, C)` are averages over seeds; we model them as
`theta_r' x(C) + eta` with `eta` zero-mean and independent across cells. Section 4
records exactly how this fails.

**Decision objective.** Identify `argmin_r theta_r' x(C*)`, equivalently recover
the sign of

    (theta_r - theta_s)' x(C*)

for every pair `(r, s)`. The decision depends only on *differences* at the
target; the level of any single projection is a nuisance.

## 2. Target-gap variance under an allocation

An allocation assigns run counts `n_j >= 0` to ladder budgets `C_j`, with
information matrix

    A(n) = sum_j n_j x(C_j) x(C_j)'

Recipes are fitted independently on a shared design, so for a single recipe

    Var(theta_r' x(C*)) = sigma^2 * x(C*)' A(n)^{-1} x(C*)

and, since two recipes' fits are independent given the design,

    Var((theta_r - theta_s)' x(C*)) = 2 sigma^2 * x(C*)' A(n)^{-1} x(C*)

The scalar `h*(n) = x(C*)' A(n)^{-1} x(C*)` is the whole design quantity. Note the
units: `h*` is a *whole-design* variance factor for `sum_j n_j` observations, not
a per-observation one. Getting this wrong costs a factor of `sum n_j`, which on a
uniform ladder is a clean factor of `k` and cancels in any comparison of two
equal-`n` designs --- it is invisible except against an absolute quantity.

### 2a. Recovery of the existing `h*` result (verified, not asserted)

For the uniform allocation (`n_j = 1` for each of `k` distinct budgets),

    h* = 1/k + (u* - ubar)^2 / S_uu,   u = log C, S_uu = sum (u_j - ubar)^2

which is the textbook OLS prediction leverage this project derived independently
and uses throughout. `tests/test_transductive.py` checks the two computations
agree on five ladders, and a separate check confirms agreement with the shipped
`scripts/run_theory_check.py` implementation to **1.3e-12** over 500 random
ladders. The general transductive objective therefore reduces to our existing
result as its uniform-allocation special case, as it must.

## 3. Two departures from classical transductive BAI

**Cost weighting.** Fiez et al. charge one sample per pull; here a pull at budget
`C` costs `C` FLOPs, so the design is chosen on a cost-weighted simplex
`sum_j n_j C_j <= B`. This changes the optimum qualitatively: cheap rungs are
nearly free, so a pure-variance objective will buy very many of them.

*Correction to an earlier framing of ours.* Cost weighting alone is **not** novel
in the scaling-law design setting --- SL2 (arXiv:2604.22753) already weights by
`c(x)` with `6ND` for dense training. The novelty is cost weighting inside a
*decision* objective.

**Misspecification.** Classical guarantees assume the linear model holds. Ours
does not, and Section 4 is the honest accounting.

## 4. Where the model is an assumption, and where it fails

This section exists so nothing downstream can quietly assume a well-specified
model. These are measured, committed numbers, not concerns.

1. **Residual scatter is 4.22x the seed standard error** --- a 17.8x variance
   inflation. The dominant term in projection error is functional-form error,
   which more seeds do not shrink.
2. **The error is compute-structured, so no scalar can absorb it.** A constant
   variance inflation `phi` fails: it varies 2.7-8.9x across ladders, is monotone
   in lever arm and ladder length, and under-predicts held-out structure by 12x.
3. **The floor `E` is unidentified on accuracy metrics** --- the profile
   likelihood is flat in `E`, so the three-parameter form is not identified from
   the ladder alone.
4. **The deviation oscillates with compute rather than trending.** Signed mean
   residuals across the DataDecide ladder run +0.045, -0.040, +0.044, -0.043,
   with correlation to `log C` of only **-0.022**. This matters for what a
   bias-aware design can do: an oscillating deviation is largely absorbed into the
   intercept and biases the projection far less than its magnitude suggests.

Consequence for the theory: the linear model supplies the *design* and the
*skeleton* of a guarantee, but the guarantee's validity is exactly what Task 4
must test rather than inherit.

## 5. An analytic result: design-for-estimation equals design-for-decision

Under the model of Section 1-2, with a design shared across recipes:

    Var(gap at C*) = 2 * Var(level at C*)

The factor 2 does not depend on the allocation `n`. Therefore

    argmin_n Var(gap) = argmin_n Var(level)

and minimising target-region prediction error selects the **same allocation** as
minimising target-gap variance. The framing "design for estimation versus design
for decision" has no content in the well-specified homoscedastic case.

Measured, to confirm the algebra is not hiding a numerical caveat: the two
objectives correlate at **0.99995** over random designs with their ratio varying
by **0.6%**, and an estimation-optimal design whose target region spans three
decades costs only **0.11%** excess variance at `C*`.

**What breaks the equivalence.** The equality above uses (i) one design shared by
all recipes, (ii) homoscedastic noise, and (iii) an unbiased estimator. Relaxing
(iii) is what matters here: under an additive deviation `b(C)` the projection
acquires the exact bias

    bias = x(C*)' A(n)^{-1} X' W b,   W = diag(n)

and the design must then minimise `variance + bias^2`, which is *not* the
estimation objective. Only the signed, common-mode part of `b` contributes ---
using a magnitude summary such as RMS here manufactures a bias that cancellation
would otherwise remove, and on DataDecide that mistake overstates the projection
bias eighteen-fold.

So the real contrast is **bias-aware versus variance-only design**, not decision
versus estimation. That is a narrower claim than we set out to make, it credits
SL2 rather than strawmanning it, and it is the one the measurements support.

## 6. Relation to prior work

Positioning is recorded in full in `NOVELTY_CHECK.md`. In brief: Fiez et al.
supply the decision structure and the design quantity but assume uniform costs
and a well-specified model; SL2 shares the domain and the cost weighting but
optimises fit accuracy for a single law; Réda, Tirinzoni and Degenne
(arXiv:2111.01479) handle misspecified fixed-confidence identification but prove
that *knowing the scale of the deviation is necessary* to exploit the structure
--- and assume that scale is known. Our deviation is neither known nor uniform,
which is what Task 4 addresses.
