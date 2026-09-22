"""Tests for cost-constrained allocation.

Two results are pinned here as properties rather than left to a script:

* :func:`test_decision_and_estimation_objectives_coincide` --- in the
  well-specified homoscedastic model, designing for the decision and designing
  for target-region fit give the same allocation. This is a negative result for
  the "design for estimation vs design for decision" framing and it is asserted,
  not merely reported, so that any future change that appears to create a gap
  must confront it.
* :func:`test_bias_shifts_allocation_away_from_cheap_rungs` --- once a
  compute-structured deviation is present, the optimum moves up the ladder. That
  is where the real contrast lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from asla.analysis.allocation import (
    projection_bias,
    round_allocation,
    scale_to_budget,
    solve_bias_aware,
    solve_decision_optimal,
    solve_estimation_optimal,
)
from asla.analysis.transductive import (
    Allocation,
    target_variance_factor,
    uniform_allocation,
)

LADDER = (4e6, 1.6e7, 6e7, 2.4e8, 1.0e9)
TARGET = 1.0e10
BUDGET = float(sum(LADDER) * 3)


def test_solver_respects_flop_budget() -> None:
    result = solve_decision_optimal(LADDER, TARGET, BUDGET)
    assert result.allocation.cost <= BUDGET * (1 + 1e-6)
    assert result.converged


def test_solver_beats_uniform_ladder_at_matched_cost() -> None:
    """The optimum must be at least as good as the ladder it is compared against."""

    result = solve_decision_optimal(LADDER, TARGET, BUDGET)
    uniform = scale_to_budget(uniform_allocation(LADDER, TARGET, 3.0), BUDGET)
    uniform_variance = target_variance_factor(LADDER, uniform.runs, TARGET)
    assert result.objective_value <= uniform_variance


def test_optimal_design_is_sparse() -> None:
    """A two-parameter c-optimal design needs at most two support points."""

    result = solve_decision_optimal(LADDER, TARGET, BUDGET)
    assert len(result.allocation.support) <= 2


def test_decision_and_estimation_objectives_coincide() -> None:
    """Design-for-decision equals design-for-estimation in the linear model.

    Under a shared homoscedastic design the target-gap variance is exactly twice
    the target-level variance, so the two objectives have identical argmins. The
    allocations are required to agree closely in cost share; the residual
    difference comes only from averaging over a region rather than a point.
    """

    decision = solve_decision_optimal(LADDER, TARGET, BUDGET)
    estimation = solve_estimation_optimal(LADDER, TARGET, BUDGET)
    dec_share = np.array([r * b for r, b in zip(decision.allocation.runs, LADDER)]) / decision.allocation.cost
    est_share = np.array([r * b for r, b in zip(estimation.allocation.runs, LADDER)]) / estimation.allocation.cost
    assert np.max(np.abs(dec_share - est_share)) < 0.02

    # And the estimation-optimal design is near-optimal for the decision too.
    decision_variance_of_estimation_design = target_variance_factor(LADDER, estimation.allocation.runs, TARGET)
    assert decision_variance_of_estimation_design <= decision.objective_value * 1.01


def test_projection_bias_is_zero_without_deviation() -> None:
    runs = [10.0] * len(LADDER)
    assert projection_bias(runs, LADDER, TARGET, [0.0] * len(LADDER)) == pytest.approx(0.0)


def test_projection_bias_passes_through_exact_linear_deviation() -> None:
    """A deviation that is itself linear in the feature is absorbed into theta.

    If ``b(C) = c0 + c1 * (-log C)`` the deviation is indistinguishable from a
    shift of the parameter, so the projection bias must equal that same linear
    functional evaluated at the target --- not zero, and not something else.
    """

    runs = [10.0] * len(LADDER)
    c0, c1 = 0.3, 0.05
    deviation = [c0 + c1 * (-np.log(b)) for b in LADDER]
    expected = c0 + c1 * (-np.log(TARGET))
    assert projection_bias(runs, LADDER, TARGET, deviation) == pytest.approx(expected, rel=1e-8)


def test_projection_bias_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        projection_bias([1.0] * len(LADDER), LADDER, TARGET, [0.0])


def test_bias_shifts_allocation_away_from_cheap_rungs() -> None:
    """Compute-structured deviation moves the optimum up the ladder.

    The deviation grows as compute falls, which is the structure measured in this
    project. A variance-only design piles runs on the cheapest rung; a bias-aware
    design must spend more of its budget higher up.
    """

    deviation = [0.05 * (b / max(LADDER)) ** -0.5 for b in LADDER]
    variance_only = solve_decision_optimal(LADDER, TARGET, BUDGET)
    bias_aware = solve_bias_aware(LADDER, TARGET, BUDGET, deviation)

    def cheap_share(result: object) -> float:
        alloc = result.allocation  # type: ignore[attr-defined]
        return float(alloc.runs[0] * LADDER[0] / alloc.cost)

    assert cheap_share(bias_aware) < cheap_share(variance_only)


def test_bias_aware_requires_deviation() -> None:
    with pytest.raises(ValueError):
        solve_bias_aware(LADDER, TARGET, BUDGET, [0.1])


def test_solver_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        solve_decision_optimal((1e6,), TARGET, BUDGET)
    with pytest.raises(ValueError):
        solve_decision_optimal(LADDER, TARGET, 0.0)
    with pytest.raises(ValueError):
        solve_decision_optimal(LADDER, TARGET, 1.0)  # cannot afford two cheap runs


def test_scale_to_budget_preserves_shape() -> None:
    alloc = uniform_allocation(LADDER, TARGET, 3.0)
    scaled = scale_to_budget(alloc, BUDGET)
    assert scaled.cost == pytest.approx(BUDGET)
    ratios = [s / a for s, a in zip(scaled.runs, alloc.runs)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_scale_to_budget_rejects_zero_cost() -> None:
    with pytest.raises(ValueError):
        scale_to_budget(Allocation(LADDER, (0.0,) * len(LADDER), TARGET), BUDGET)


def test_round_allocation_stays_within_budget_and_identifiable() -> None:
    result = solve_decision_optimal(LADDER, TARGET, BUDGET)
    rounded = round_allocation(result.allocation, BUDGET)
    assert rounded.cost <= BUDGET
    assert all(float(r).is_integer() for r in rounded.runs)
    # Must keep at least two supported budgets or the line is unidentifiable.
    assert len(rounded.support) >= 2
    assert np.isfinite(target_variance_factor(LADDER, rounded.runs, TARGET))


def test_round_allocation_recovers_identifiability_from_single_point() -> None:
    """Flooring can strand a design on one budget; rounding must repair it."""

    stranded = Allocation(LADDER, (1.5, 0.2, 0.0, 0.0, 0.0), TARGET)
    rounded = round_allocation(stranded, BUDGET)
    assert len(rounded.support) >= 2
    assert np.isfinite(target_variance_factor(LADDER, rounded.runs, TARGET))
