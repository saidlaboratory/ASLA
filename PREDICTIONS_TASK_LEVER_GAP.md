# Pre-registration: the lever-arm gap, the DataDecide code search, Monte Carlo precision

Written 2026-09-26, before any computation in this round. No paper prose changes.

## Correction to the premise

The random-candidate worlds of `scripts/run_lever_arm_calibration.py` do **not**
use parametric power-law truth. Each world resamples the real recipes' shrunk
seed-mean curves with replacement and adds a smooth power-law shift
dE + A(e^{dlogA} - 1)C^-alpha. The real curves' misspecification is therefore
retained, so the 0.66-versus-0.87 gap cannot be explained by "parametric truth
has no misspecification". Candidate explanations that remain:

- **(i)** Resampling with replacement creates near-duplicate recipes, whose pairs
  have small true gaps set by the smooth shift. That changes the pair mix.
- **(ii)** The shift itself dilutes the real curves' differential structure.
- **(iii)** The real estimate is inflated by noise.

## 1a: the discriminating check

Fixed-candidate worlds keep the 25 real recipes with their shrunk curves. They
are identical to arm B3 at amplitude 1 below, and number 150. True rho is the
Spearman correlation between log lever arm and each design's true excess,
averaged over worlds. The interval is a bootstrap over worlds. A noiseless
readout is also reported: every ranker run once on the truth itself.

- **P1a.** Fixed-candidate true rho is at least 0.80. The gap is then a property of
  the random-candidate construction (i or ii), not a test artifact and not
  inflation of the real estimate. If it lies between 0.60 and 0.72, near the
  random value, explanation (iii) applies: the real estimate is inflated, and
  that is the finding.

## 1b: controlled misspecification dose

- **Parametric truth.** mu_P(r, C) = E_r + A_r C^-alpha_r, fitted to each recipe's
  shrunk curve over the full ladder (750M excluded).
- **Residuals.** R(r, C) = mu_real - mu_P, decomposed by rank-one SVD as
  R = a m^T + E: a_r are the loadings, m(C) the shared shape, E the remainder.
  The loading CV is reported.
- **Arms.** All fixed-candidate, 150 worlds each, same noise model:
  - A: mu_P (variance only);
  - B1: mu_P + mean(a) m (common mode, constant loadings);
  - B2: mu_P + (a - mean(a)) m + E (recipe-specific part);
  - B3(lambda): mu_P + lambda R, for lambda in {0.5, 1, 2}. B3(0) is A and B3(1)
    is the real shrunk truth.
- **Readouts per arm.** True rho with a bootstrap interval over worlds. The OLS
  slope of true excess on log lever arm, in points per log unit, with a
  bootstrap interval. The mean estimated rho.
- **Attribution, on the slope scale** (slopes are closer to additive than rho):
  A's share of B3(1)'s slope, B1's and B2's increments over A, and the
  interaction remainder. The real effect is the slope of the real data's
  per-design mean kernel on log lever arm; any gap to B3(1) is reported as
  unexplained.

**Predictions.**

- **P1b-1.** |rho(B1) - rho(A)| <= 0.10: common-mode error cancels. Refuted if B1
  raises rho by more than 0.10.
- **P1b-2.** rho(B2) - rho(A) >= 0.10, and B2's slope increment is at least twice B1's.
- **P1b-3.** True rho and slope rise monotonically in lambda.
- **P1b-4.** rho(A) is between 0.3 and 0.8. Variance alone produces a positive
  lever-arm effect, because extrapolation variance grows with leverage.
- **P1b-5.** The loading CV on shrunk-truth residuals is between 0.05 and 0.35
  (0.16 was measured on fit-budget residuals).

## 1c: readouts from the same worlds

- **Ensemble.** The true mis-selection difference, ensemble minus single-scale, at
  the primary design (fit to 300M, target 530M), in A and in B3(1).
  - **P1c-1.** The ensemble is worse in A by at least 2 points: its flexibility fits
    noise. It is also worse in B3(1).
- **Power.** The Hoeffding components (zeta1, zeta2) of the projection-versus-
  single-scale kernel at the primary design, averaged over worlds per arm. The
  candidates needed for 2 points at power 0.8, with the calibrated random-
  candidate critical value of 2.99, are computed for A and for B3(1).
  - **Framing.** The earlier 128 used real-data components, not a variance-only
    population. It is compared with both.
  - **P1c-2.** B3(1) needs between 90 and 200 candidates. A needs fewer, because
    well-specified truth has smaller pair-level variance in projection's errors.

## Task 2: DataDecide's decision-accuracy code

A direct search of the public repositories: the DataDecide release and its
linked GitHub repositories, and the evaluation and scaling-law tooling the
paper cites. Nothing is reconstructed. If the code is found, it is run against
the released predictions.

- **P2.** Not found as a standalone released script; the search locations are
  reported.

## Task 3: Monte Carlo precision of fixed-candidate p-values

Each reported fixed-candidate p has two Monte Carlo sources:

- **(a)** The real-data bootstrap SE, B draws. Its error is measured by batch
  means over independent batches.
- **(b)** The calibrated null distribution, from a finite number of simulated
  statistics. Its error is binomial on the tail fraction.

B is increased until (a)'s error is below the last reported digit. For (b), the
number of simulated statistics that would be needed is reported. Where that is
impractical, the p-value is reported to one significant figure, with the
reason.

- **P3.** Source (a) is fixed by B of about 4000. Source (b) is impractical to
  reduce to the third decimal for p around 0.03 (about 30,000 statistics would
  be needed), so those p-values are reported to one significant figure.
