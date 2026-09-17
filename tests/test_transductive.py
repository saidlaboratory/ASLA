"""Tests for the transductive design algebra.

The central test is :func:`test_uniform_allocation_recovers_classical_leverage`:
the transductive variance factor under a uniform allocation must equal the
project's pre-existing ``h* = 1/k + (u*-ubar)^2/S_uu``. If it did not, one of the
two derivations would be wrong and nothing downstream could be trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from asla.analysis.transductive import (
    Allocation,
    classical_leverage,
    design_features,
    equal_cost_allocation,
    feature,
    information_matrix,
    lever_arm,
    leverage_from_allocation,
    pair_variance_factor,
    target_variance_factor,
    uniform_allocation,
)

# A DataDecide-like ladder: 4M to 1B in roughly half-decade steps, target above it.
LADDER = (4e6, 1.6e7, 6e7, 2.4e8, 1.0e9)
TARGET = 1.0e10


def test_feature_sign_convention() -> None:
    x = feature(np.e)
    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(-1.0)


def test_feature_rejects_nonpositive() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            feature(bad)


def test_design_features_shape() -> None:
    assert design_features(LADDER).shape == (len(LADDER), 2)


@pytest.mark.parametrize(
    "budgets,target",
    [
        (LADDER, TARGET),
        ((1e6, 1e7, 1e8), 1e9),
        ((3e5, 9e5, 2.7e6, 8.1e6), 1e8),
        (LADDER, 1.0e9 * 1.0001),  # target barely above the top rung
        ((1e6, 1e12), 1e13),  # extreme two-point ladder
    ],
)
def test_uniform_allocation_recovers_classical_leverage(budgets, target) -> None:
    """The h* recovery theorem, checked numerically on several ladders.

    ``classical_leverage`` re-implements the textbook formula in its own algebra;
    ``leverage_from_allocation`` goes through the transductive information
    matrix. They must agree to floating-point precision.
    """

    transductive = leverage_from_allocation(budgets, [1.0] * len(budgets), target)
    classical = classical_leverage(budgets, target)["leverage"]
    assert transductive == pytest.approx(classical, rel=1e-10)


def test_leverage_matches_hand_rolled_ols_variance() -> None:
    """Pin the *units* against an independent OLS prediction-variance computation.

    This is the test that caught a normalisation bug: dividing weights by their
    total gives the per-observation leverage, a factor of ``sum(weights)`` too
    large. Because that error is a clean constant on a uniform ladder, it cancels
    in any ratio of two equal-``n`` designs and would have gone unnoticed in
    every comparison downstream. Comparing against an absolute quantity, rather
    than another design, is what makes it visible.
    """

    budgets = (1e6, 1e7, 1e8, 1e9)
    target = 1e10
    u = np.log(np.asarray(budgets))
    x = np.vstack([np.ones(u.size), u]).T  # plain a + b*u parametrisation
    x_star = np.array([1.0, np.log(target)])
    ols_variance = float(x_star @ np.linalg.solve(x.T @ x, x_star))
    assert leverage_from_allocation(budgets, [1.0] * len(budgets), target) == pytest.approx(ols_variance, rel=1e-10)


def test_more_runs_reduce_variance_inversely() -> None:
    """Doubling every run count must halve the projection variance."""

    one = leverage_from_allocation(LADDER, [1.0] * len(LADDER), TARGET)
    two = leverage_from_allocation(LADDER, [2.0] * len(LADDER), TARGET)
    assert two == pytest.approx(one / 2.0, rel=1e-12)


def test_recovery_holds_for_nonuniform_replication() -> None:
    """Replicating a budget r times must match a ladder listing it r times.

    This is the bridge between "weights are run counts" and "weights are a
    measure": duplicating a design point is the same as doubling its weight.
    """

    budgets = (1e6, 1e7, 1e8)
    weights = [3.0, 1.0, 1.0]
    expanded = (1e6, 1e6, 1e6, 1e7, 1e8)
    a = leverage_from_allocation(budgets, weights, 1e9)
    b = leverage_from_allocation(expanded, [1.0] * len(expanded), 1e9)
    assert a == pytest.approx(b, rel=1e-10)


def test_single_budget_design_is_unidentifiable() -> None:
    """One budget cannot identify a two-parameter line: variance is infinite."""

    assert np.isinf(target_variance_factor((1e6,), (1.0,), 1e9))
    assert np.isinf(classical_leverage((1e6,), 1e9)["leverage"])


def test_pair_variance_is_twice_single() -> None:
    single = target_variance_factor(LADDER, [1.0] * len(LADDER), TARGET)
    assert pair_variance_factor(LADDER, [1.0] * len(LADDER), TARGET) == pytest.approx(2.0 * single)


def test_leverage_grows_with_lever_arm() -> None:
    """Extrapolating further from the same ladder must cost more variance."""

    near = leverage_from_allocation(LADDER, [1.0] * len(LADDER), 2.0e9)
    far = leverage_from_allocation(LADDER, [1.0] * len(LADDER), 1.0e11)
    assert far > near


def test_information_matrix_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        information_matrix(LADDER, [1.0] * (len(LADDER) - 1))
    with pytest.raises(ValueError):
        information_matrix(LADDER, [-1.0] + [1.0] * (len(LADDER) - 1))


def test_lever_arm() -> None:
    assert lever_arm(LADDER, TARGET) == pytest.approx(TARGET / max(LADDER))


def test_allocation_cost_and_concentration() -> None:
    alloc = Allocation((1e6, 1e9), (10.0, 1.0), 1e10)
    assert alloc.cost == pytest.approx(10 * 1e6 + 1e9)
    # Nearly all cost sits at the top rung despite ten times as many cheap runs.
    assert alloc.concentration() == pytest.approx(1e9 / (10 * 1e6 + 1e9))
    assert alloc.support == (1e6, 1e9)
    assert alloc.total_runs == pytest.approx(11.0)


def test_allocation_validation() -> None:
    with pytest.raises(ValueError):
        Allocation((1e6,), (1.0, 2.0), 1e9)
    with pytest.raises(ValueError):
        Allocation((), (), 1e9)
    with pytest.raises(ValueError):
        Allocation((1e6,), (-1.0,), 1e9)


def test_uniform_allocation_helper() -> None:
    alloc = uniform_allocation(LADDER, TARGET, runs_per_budget=3.0)
    assert alloc.runs == (3.0,) * len(LADDER)
    assert alloc.budgets == tuple(sorted(LADDER))


def test_equal_cost_allocation_spends_equally() -> None:
    alloc = equal_cost_allocation(LADDER, TARGET, total_cost=1e12)
    spends = [r * b for r, b in zip(alloc.runs, alloc.budgets)]
    assert all(s == pytest.approx(spends[0]) for s in spends)
    assert alloc.cost == pytest.approx(1e12)


def test_zero_weight_allocation_rejected() -> None:
    with pytest.raises(ValueError):
        leverage_from_allocation(LADDER, [0.0] * len(LADDER), TARGET)
