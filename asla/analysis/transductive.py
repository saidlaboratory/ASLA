"""Transductive experimental design for scaling-law selection.

The selection problem is transductive in the sense of Fiez, Jain, Jamieson and
Ratliff (NeurIPS 2019, arXiv:1906.08399): we sample one set of arms and must
decide about a different set that is never sampled.

* Sampled arms ``A``: pairs ``(recipe, C)`` we can afford to train, ``C <= C_max``.
* Target set ``Z``: the same recipes at the target budget ``C*``, never trained.

In log space the two-parameter power law ``log E = a - alpha * log C`` is linear
in the feature ``x(C) = (1, -log C)``, so the target value of recipe ``r`` is the
linear functional ``theta_r' x(C*)``. Selecting the best recipe means recovering
the sign of ``(theta_r - theta_s)' x(C*)`` for every pair ``(r, s)``.

Two departures from the classical setting drive everything here:

1. **Cost weighting.** In transductive BAI each pull costs one sample. Here a
   pull at budget ``C`` costs ``C`` FLOPs, so a design spends its budget on a
   *cost-weighted* simplex. Fiez et al. do not consider heterogeneous costs; SL2
   (arXiv:2604.22753) does weight by cost but optimises prediction error, not a
   decision. Cost weighting inside a decision objective is the novel combination.

2. **Misspecification.** The classical guarantee assumes the linear model holds.
   Ours does not: residual scatter is 2.90x the seed standard error and the error
   is compute-structured. Nothing in this module assumes otherwise; it computes
   the design quantities implied by the linear model, and
   :mod:`asla.analysis.misspec` is where the deviation is measured and the
   widths are corrected.

Because each recipe is fitted independently, the design matrix is shared across
recipes and the per-pair variance factorises into a common scalar --- the OLS
prediction leverage ``h*`` --- times the per-recipe noise level. That is why the
general transductive objective collapses to the ``h*`` result this project
already derived; :func:`leverage_from_allocation` and the tests in
``tests/test_transductive.py`` verify that recovery numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np

Weights = Union[Sequence[float], np.ndarray]

# Numerical floor for allocation weights treated as "support" of a design.
SUPPORT_TOLERANCE = 1e-9


def feature(budget: float) -> np.ndarray:
    """Return the log-space design feature ``x(C) = (1, -log C)``.

    The sign convention makes the second coordinate of ``theta`` the decay
    exponent ``alpha`` in ``log E = a - alpha log C``, so a *larger* second
    coordinate means a faster-improving recipe.
    """

    value = float(budget)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("budget must be a finite positive number")
    return np.array([1.0, -np.log(value)], dtype=float)


def design_features(budgets: Iterable[float]) -> np.ndarray:
    """Stack :func:`feature` over budgets into a ``(k, 2)`` matrix."""

    values = [float(b) for b in budgets]
    if not values:
        raise ValueError("at least one budget is required")
    return np.vstack([feature(b) for b in values])


def information_matrix(budgets: Sequence[float], weights: Weights) -> np.ndarray:
    """Return ``A(lambda) = sum_j lambda_j x_j x_j'`` for a design over budgets.

    ``weights`` need not be normalised; callers that want the standard
    probability-simplex convention should normalise first. Keeping the raw scale
    lets the cost-constrained solver work in units of "number of runs".
    """

    x = design_features(budgets)
    w = np.asarray(weights, dtype=float)
    if w.shape != (x.shape[0],):
        raise ValueError("weights must have one entry per budget")
    if np.any(w < 0):
        raise ValueError("design weights must be non-negative")
    return x.T @ (w[:, None] * x)


def _solve_quadratic_form(matrix: np.ndarray, vector: np.ndarray) -> float:
    """Return ``vector' matrix^{-1} vector``, or ``inf`` when unidentifiable.

    A design supported on a single budget cannot identify a two-parameter line;
    the information matrix is then singular and the variance is infinite. That is
    the mathematically correct answer and callers depend on it, so we return
    ``inf`` rather than raising.
    """

    try:
        solved = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return float("inf")
    value = float(vector @ solved)
    if not np.isfinite(value) or value < 0:
        return float("inf")
    return value


def target_variance_factor(
    budgets: Sequence[float],
    weights: Weights,
    target: float,
) -> float:
    """Return ``x(C*)' A(lambda)^{-1} x(C*)``.

    Multiplying by the per-observation noise variance gives the variance of the
    projected target value for one recipe. With ``weights`` counting runs per
    budget this is the variance of the projection from that many runs.
    """

    return _solve_quadratic_form(information_matrix(budgets, weights), feature(target))


def pair_variance_factor(
    budgets: Sequence[float],
    weights: Weights,
    target: float,
) -> float:
    """Return the variance factor for a *difference* of two recipe projections.

    Recipes are fitted independently on the same design, so their projection
    errors are independent given the design and the gap variance is twice the
    single-recipe factor. This is the quantity the decision depends on: the
    decision is the sign of a difference, never the level of one projection.
    """

    return 2.0 * target_variance_factor(budgets, weights, target)


def leverage_from_allocation(
    budgets: Sequence[float],
    weights: Weights,
    target: float,
) -> float:
    """Return the OLS prediction leverage ``h*`` implied by an allocation.

    **Units.** ``weights`` are *run counts*, not a probability measure, and the
    returned value is the variance factor for the whole design: the projection
    variance is ``sigma^2 * h*``, with no further division by ``n``. This
    matches the classical textbook form ``1/k + (u* - ubar)^2 / S_uu``, which is
    likewise a whole-dataset quantity for ``k`` observations.

    Getting this wrong is easy and was caught here by the recovery test: dividing
    the weights by their total instead yields the per-observation leverage, which
    is a factor of ``sum(weights)`` too large. The two conventions differ by
    exactly the number of runs, so on a uniform ladder the error is a clean
    factor of ``k`` and is invisible in any ratio of two designs at equal ``n``.
    """

    values = [float(w) for w in weights]
    if sum(values) <= 0:
        raise ValueError("design weights must have positive total mass")
    return target_variance_factor(budgets, values, target)


def classical_leverage(budgets: Sequence[float], target: float) -> dict[str, float]:
    """Return ``h* = 1/k + (u* - ubar)^2 / S_uu`` and its intermediates.

    This is the pre-existing project formula, reproduced here verbatim in its own
    algebra so that the recovery check compares two genuinely independent
    computations rather than one function calling the other.
    """

    u = np.log(np.asarray([float(b) for b in budgets], dtype=float))
    k = int(u.size)
    if k == 0:
        raise ValueError("at least one budget is required")
    ubar = float(u.mean())
    s_uu = float(np.sum((u - ubar) ** 2))
    u_star = float(np.log(float(target)))
    if s_uu <= 0:
        return {
            "k": float(k),
            "ubar": ubar,
            "S_uu": s_uu,
            "u_star": u_star,
            "one_over_k": 1.0 / k,
            "offset_term": float("inf"),
            "leverage": float("inf"),
        }
    offset = (u_star - ubar) ** 2 / s_uu
    return {
        "k": float(k),
        "ubar": ubar,
        "S_uu": s_uu,
        "u_star": u_star,
        "one_over_k": 1.0 / k,
        "offset_term": offset,
        "leverage": 1.0 / k + offset,
    }


def lever_arm(budgets: Sequence[float], target: float) -> float:
    """Return ``L = C* / C_max``, the extrapolation lever arm."""

    values = [float(b) for b in budgets]
    if not values:
        raise ValueError("at least one budget is required")
    return float(target) / max(values)


@dataclass(frozen=True)
class Allocation:
    """A cost-constrained design over budgets.

    ``runs`` is the (possibly fractional) number of runs at each budget; solvers
    return the relaxed solution and rounding is a separate, explicit step.
    """

    budgets: tuple[float, ...]
    runs: tuple[float, ...]
    target: float

    def __post_init__(self) -> None:
        if len(self.budgets) != len(self.runs):
            raise ValueError("budgets and runs must have equal length")
        if not self.budgets:
            raise ValueError("an allocation needs at least one budget")
        if any(r < 0 for r in self.runs):
            raise ValueError("run counts must be non-negative")

    @property
    def cost(self) -> float:
        """Total FLOP cost: ``sum_j runs_j * C_j``."""

        return float(sum(r * b for r, b in zip(self.runs, self.budgets)))

    @property
    def support(self) -> tuple[float, ...]:
        """Budgets carrying non-negligible weight."""

        return tuple(b for b, r in zip(self.budgets, self.runs) if r > SUPPORT_TOLERANCE)

    @property
    def total_runs(self) -> float:
        return float(sum(self.runs))

    def pair_variance(self, noise_variance: float = 1.0) -> float:
        """Variance of a projected target *gap* under this allocation."""

        return noise_variance * pair_variance_factor(self.budgets, self.runs, self.target)

    def concentration(self) -> float:
        """Fraction of total cost spent at the largest supported budget.

        The qualitative question Task 2 asks --- does the optimum concentrate at
        the top rung or spread? --- is read off this number.
        """

        if self.cost <= 0:
            return float("nan")
        top = max(self.budgets)
        spent_top = sum(r * b for r, b in zip(self.runs, self.budgets) if np.isclose(b, top))
        return float(spent_top / self.cost)

    def as_dict(self) -> dict[str, object]:
        return {
            "budgets": list(self.budgets),
            "runs": list(self.runs),
            "target": self.target,
            "cost": self.cost,
            "total_runs": self.total_runs,
            "support": list(self.support),
            "concentration_at_top": self.concentration(),
        }


def uniform_allocation(budgets: Sequence[float], target: float, runs_per_budget: float = 1.0) -> Allocation:
    """The ladder used in practice: equal run counts at every budget."""

    values = tuple(sorted(float(b) for b in budgets))
    return Allocation(values, tuple(float(runs_per_budget) for _ in values), float(target))


def equal_cost_allocation(budgets: Sequence[float], target: float, total_cost: float) -> Allocation:
    """Spend the same number of FLOPs at each budget.

    A natural second baseline: unlike the uniform ladder it is cost-balanced, so
    it buys many cheap runs and few expensive ones without optimising anything.
    """

    values = tuple(sorted(float(b) for b in budgets))
    share = float(total_cost) / len(values)
    return Allocation(values, tuple(share / b for b in values), float(target))
