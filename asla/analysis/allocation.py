"""Cost-constrained allocation: which budgets to train at, and how many runs.

Given a FLOP budget ``B`` and a candidate ladder of compute budgets, choose run
counts ``n_j`` at each budget ``C_j`` subject to ``sum_j n_j C_j <= B``, so as to
minimise the variance of the projected target *gap*.

The objective is the decision quantity ``x(C*)' A(n)^{-1} x(C*)``, not the fit
error over the ladder. Those differ: a design can estimate the curve well
everywhere and still be poor at resolving the sign of a difference at ``C*``.
:func:`solve_estimation_optimal` implements the competing objective (SL2,
arXiv:2604.22753, minimises target-region prediction error; here reduced to its
design core) so the contrast can be scored as a number rather than asserted.

**Convexity.** ``A(n)`` is linear in ``n`` and ``v' A^{-1} v`` is convex in ``A``
on the positive-definite cone, so the objective is convex in ``n`` and the
feasible set is a simplex scaled by costs. The cost constraint binds at the
optimum (more runs never hurt), so we optimise on the boundary.

**Why the optimum can be sparse.** With a two-parameter model the information
matrix is 2x2, and by Caratheodory an optimal design needs at most 2 support
points (3 in general for a symmetric 2x2, but the c-optimal solution for a single
target direction needs at most 2). This is why the "concentrate or spread"
question has a clean answer rather than a diffuse one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from asla.analysis.transductive import (
    Allocation,
    Weights,
    design_features,
    feature,
    information_matrix,
)

# Below this weight a support point is treated as unused when reporting.
PRUNE_WEIGHT = 1e-6
# Ridge added to the information matrix only to keep the interior solver finite;
# it is never used in reported variances, which recompute without it.
SOLVER_RIDGE = 1e-12


@dataclass(frozen=True)
class AllocationResult:
    """An allocation plus the provenance needed to reproduce it."""

    allocation: Allocation
    objective: str
    objective_value: float
    budget_flops: float
    solver: str
    converged: bool
    n_starts: int

    def as_dict(self) -> dict[str, object]:
        out = dict(self.allocation.as_dict())
        out.update(
            {
                "objective": self.objective,
                "objective_value": self.objective_value,
                "budget_flops": self.budget_flops,
                "solver": self.solver,
                "converged": self.converged,
                "n_starts": self.n_starts,
            }
        )
        return out


def _decision_objective(runs: np.ndarray, budgets: Sequence[float], target: float) -> float:
    """Variance factor of the target gap (up to the factor 2 common to all pairs)."""

    matrix = information_matrix(budgets, runs) + SOLVER_RIDGE * np.eye(2)
    x_star = feature(target)
    try:
        return float(x_star @ np.linalg.solve(matrix, x_star))
    except np.linalg.LinAlgError:
        return float("inf")


def _estimation_objective(runs: np.ndarray, budgets: Sequence[float], target: float) -> float:
    """Average prediction variance over a target *region*: the SL2-style criterion.

    SL2 (arXiv:2604.22753) minimises ``MSPE_tar``, the mean squared prediction
    error averaged over a finite set ``X_tar`` of expensive target configurations
    (verified against the paper: ``MSPE_tar = (1/|X_tar|) E_theta ||F(theta) -
    fbar||^2``, with no pilot-region term). Its design core, with fitted-parameter
    uncertainty as the only stochastic term, is the average of
    ``x(C)' A^{-1} x(C)`` over that region; here the decade centred on ``C*``.

    **This objective is near-identical to the decision objective, and that is a
    result, not a defect of this implementation.** Under a shared homoscedastic
    design the gap variance is exactly twice the level variance, so the two have
    the same argmin; even a three-decade target region costs a negligible excess
    variance at ``C*`` (measured in ``results/target_scoring/fit_structure.json``).
    Design-for-estimation and design-for-decision provably coincide in the
    well-specified linear model (proof: the paper's appendix). The contrast that *does* matter
    is :func:`_bias_aware_objective`.
    """

    matrix = information_matrix(budgets, runs) + SOLVER_RIDGE * np.eye(2)
    region = np.exp(np.linspace(np.log(target / np.sqrt(10.0)), np.log(target * np.sqrt(10.0)), 9))
    features = design_features(region)
    try:
        solved = np.linalg.solve(matrix, features.T)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.mean(np.sum(features.T * solved, axis=0)))


def projection_bias(
    runs: Weights,
    budgets: Sequence[float],
    target: float,
    deviation: Sequence[float],
) -> float:
    """Bias of the target projection induced by a per-budget model deviation.

    If the truth is ``x(C)'theta + b(C)`` rather than ``x(C)'theta``, weighted
    least squares carries the deviation into the projection as

        bias = x(C*)' A(n)^{-1} X' W b

    with ``W = diag(runs)``. This is exact, not an approximation: it is the
    deterministic part of the estimator under an additive deviation.

    The design consequence is the interesting one. Variance alone is minimised by
    piling runs onto the cheapest budgets, because they are cheap; but under a
    deviation that grows as compute falls --- which is what we measured --- those
    are exactly the biased points. A bias-aware design must trade them off.
    """

    matrix = information_matrix(budgets, runs) + SOLVER_RIDGE * np.eye(2)
    x_design = design_features(budgets)
    weights = np.asarray(runs, dtype=float)
    dev = np.asarray(deviation, dtype=float)
    if dev.shape != weights.shape:
        raise ValueError("deviation must have one entry per budget")
    try:
        return float(feature(target) @ np.linalg.solve(matrix, x_design.T @ (weights * dev)))
    except np.linalg.LinAlgError:
        return float("inf")


def _bias_aware_objective(
    runs: np.ndarray,
    budgets: Sequence[float],
    target: float,
    deviation: Sequence[float],
) -> float:
    """Mean squared error of the target projection: variance plus squared bias.

    This is where designing for a decision under *measured* misspecification
    departs from designing for fit under an assumed-correct model.
    """

    variance = _decision_objective(runs, budgets, target)
    if not np.isfinite(variance):
        return float("inf")
    bias = projection_bias(runs, budgets, target, deviation)
    if not np.isfinite(bias):
        return float("inf")
    return variance + bias**2


OBJECTIVES: dict[str, Callable[[np.ndarray, Sequence[float], float], float]] = {
    "decision": _decision_objective,
    "estimation": _estimation_objective,
}


def _solve(
    budgets: Sequence[float],
    target: float,
    budget_flops: float,
    objective: str,
    *,
    n_starts: int = 12,
    seed: int = 0,
    min_runs: float = 0.0,
    max_runs: float | None = None,
    deviation: Sequence[float] | None = None,
) -> AllocationResult:
    """Minimise the chosen objective subject to the FLOP constraint.

    Multi-start SLSQP. The problem is convex, so any local optimum is global;
    multiple starts guard against the solver stalling on the boundary rather
    than against local minima, and we keep the best. Run counts span orders of
    magnitude across a ladder, so this can stop short of the optimum: on the
    DataDecide ladder at 1B it sits about 2% above the exhaustive small-support
    optimum computed in ``scripts/run_fit_structure_check.py``.
    """

    values = tuple(sorted(float(b) for b in budgets))
    k = len(values)
    if k < 2:
        raise ValueError("a two-parameter design needs at least two distinct budgets")
    if budget_flops <= 0:
        raise ValueError("FLOP budget must be positive")
    costs = np.asarray(values, dtype=float)
    if float(costs.min()) * 2.0 > budget_flops:
        raise ValueError("FLOP budget cannot afford two runs at the cheapest budget")

    func: Callable[[np.ndarray, Sequence[float], float], float]
    if objective == "bias_aware":
        if deviation is None:
            raise ValueError("the bias_aware objective requires a per-budget deviation")
        dev = [float(d) for d in deviation]
        if len(dev) != k:
            raise ValueError("deviation must have one entry per budget")

        def func(runs: np.ndarray, b: Sequence[float], t: float) -> float:
            return _bias_aware_objective(runs, b, t, dev)
    else:
        func = OBJECTIVES[objective]
    constraints = [{"type": "ineq", "fun": lambda n: budget_flops - float(costs @ n)}]
    ceiling = float("inf") if max_runs is None else float(max_runs)
    bounds = [(min_runs, min(budget_flops / c, ceiling)) for c in costs]
    if max_runs is not None and float(costs.min()) * 2.0 > budget_flops:
        raise ValueError("FLOP budget cannot afford two runs at the cheapest budget")
    rng = np.random.default_rng(seed)

    best: tuple[float, np.ndarray, bool] | None = None
    for start in range(n_starts):
        # Annotated because the three branches produce arrays whose shape types
        # differ under numpy's shape-generic stubs: np.ones(k) is inferred as the
        # precise 1-D tuple[int] while the other two are tuple[int, ...]. Without
        # this, mypy binds the variable to the narrow type from the first branch
        # and rejects the others.
        weights: np.ndarray
        if start == 0:
            weights = np.ones(k)
        elif start == 1:
            weights = 1.0 / costs
        else:
            weights = rng.dirichlet(np.ones(k))
        x0 = weights / float(costs @ weights) * budget_flops
        x0 = np.clip(x0, min_runs, None)
        res = minimize(
            func,
            x0,
            args=(values, target),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not np.all(np.isfinite(res.x)):
            continue
        value = func(np.asarray(res.x, dtype=float), values, target)
        if not np.isfinite(value):
            continue
        if best is None or value < best[0]:
            best = (value, np.asarray(res.x, dtype=float), bool(res.success))

    if best is None:
        raise RuntimeError("allocation solver failed from every start")

    value, runs, converged = best
    runs = np.where(runs < PRUNE_WEIGHT, 0.0, runs)
    return AllocationResult(
        allocation=Allocation(values, tuple(float(r) for r in runs), float(target)),
        objective=objective,
        objective_value=float(value),
        budget_flops=float(budget_flops),
        solver="SLSQP",
        converged=converged,
        n_starts=n_starts,
    )


def solve_decision_optimal(
    budgets: Sequence[float],
    target: float,
    budget_flops: float,
    *,
    n_starts: int = 12,
    seed: int = 0,
    min_runs: float = 0.0,
    max_runs: float | None = None,
) -> AllocationResult:
    """Allocation minimising target-gap variance: design for the decision."""

    return _solve(
        budgets,
        target,
        budget_flops,
        "decision",
        n_starts=n_starts,
        seed=seed,
        min_runs=min_runs,
        max_runs=max_runs,
    )


def solve_estimation_optimal(
    budgets: Sequence[float],
    target: float,
    budget_flops: float,
    *,
    n_starts: int = 12,
    seed: int = 0,
    min_runs: float = 0.0,
    max_runs: float | None = None,
) -> AllocationResult:
    """Allocation minimising target-region prediction variance: the SL2 criterion."""

    return _solve(
        budgets,
        target,
        budget_flops,
        "estimation",
        n_starts=n_starts,
        seed=seed,
        min_runs=min_runs,
        max_runs=max_runs,
    )


def solve_bias_aware(
    budgets: Sequence[float],
    target: float,
    budget_flops: float,
    deviation: Sequence[float],
    *,
    n_starts: int = 12,
    seed: int = 0,
    min_runs: float = 0.0,
    max_runs: float | None = None,
) -> AllocationResult:
    """Allocation minimising projection MSE under a per-budget model deviation."""

    return _solve(
        budgets,
        target,
        budget_flops,
        "bias_aware",
        n_starts=n_starts,
        seed=seed,
        min_runs=min_runs,
        max_runs=max_runs,
        deviation=deviation,
    )


def scale_to_budget(allocation: Allocation, budget_flops: float) -> Allocation:
    """Rescale run counts so the design spends exactly ``budget_flops``.

    Used to put a hand-built ladder (which has its own natural cost) on equal
    footing with a solved allocation for a matched-compute comparison.
    """

    cost = allocation.cost
    if cost <= 0:
        raise ValueError("cannot rescale an allocation with zero cost")
    factor = float(budget_flops) / cost
    return Allocation(
        allocation.budgets,
        tuple(r * factor for r in allocation.runs),
        allocation.target,
    )


def round_allocation(allocation: Allocation, budget_flops: float, max_runs: float | None = None) -> Allocation:
    """Round fractional runs to integers without exceeding the FLOP budget.

    Floors every entry, then spends the remainder greedily on whichever budget
    gives the largest marginal variance reduction per FLOP. A design needs two
    distinct supported budgets to identify the line, so if flooring leaves fewer
    than two, the cheapest affordable second point is seeded first.
    """

    budgets = np.asarray(allocation.budgets, dtype=float)
    runs = np.floor(np.asarray(allocation.runs, dtype=float))
    spent = float(budgets @ runs)

    supported = int(np.sum(runs > 0))
    if supported < 2:
        order = np.argsort(budgets)
        for idx in order:
            if runs[idx] > 0:
                continue
            if spent + budgets[idx] <= budget_flops:
                runs[idx] += 1.0
                spent += float(budgets[idx])
                supported += 1
            if supported >= 2:
                break

    while True:
        best_idx = -1
        best_gain = 0.0
        current = _decision_objective(runs, allocation.budgets, allocation.target)
        for idx in range(len(budgets)):
            if spent + budgets[idx] > budget_flops:
                continue
            if max_runs is not None and runs[idx] + 1.0 > max_runs:
                continue
            trial = runs.copy()
            trial[idx] += 1.0
            gain = current - _decision_objective(trial, allocation.budgets, allocation.target)
            gain_per_flop = gain / float(budgets[idx])
            if gain_per_flop > best_gain:
                best_gain = gain_per_flop
                best_idx = idx
        if best_idx < 0:
            break
        runs[best_idx] += 1.0
        spent += float(budgets[best_idx])

    return Allocation(allocation.budgets, tuple(float(r) for r in runs), allocation.target)
