"""Tests for the fit-structure check and its design-theorem companion figures."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from asla.analysis.transductive import pair_variance_factor, target_variance_factor

check = importlib.import_module("scripts.run_fit_structure_check")

LADDER = (1e17, 3e17, 1e18, 3e18, 1e19)
TARGET = 1e20
FLOPS = float(sum(LADDER) * 3)


def test_optimal_design_needs_no_more_than_three_support_points() -> None:
    runs = check._optimal(check._point_objective, LADDER, TARGET, FLOPS)
    assert np.count_nonzero(runs > 1e-9) <= 3
    assert float(np.asarray(LADDER) @ runs) == pytest.approx(FLOPS, rel=1e-6)


def test_optimal_search_beats_every_uniform_and_two_point_design() -> None:
    """The exhaustive search must not lose to any design it could have found."""

    best = check._point_objective(check._optimal(check._point_objective, LADDER, TARGET, FLOPS), LADDER, TARGET)
    rng = np.random.default_rng(0)
    for _ in range(200):
        shares = rng.dirichlet(np.ones(len(LADDER)))
        runs = shares * FLOPS / np.asarray(LADDER)
        assert best <= check._point_objective(runs, LADDER, TARGET) * (1 + 1e-9)


def test_region_optimum_wins_on_its_own_objective() -> None:
    """The defect this guards: a solver whose region optimum lost to the point optimum."""

    def region(runs: np.ndarray, b: tuple[float, ...], t: float) -> float:
        return check._region_objective(runs, b, t, 3.0)

    point_runs = check._optimal(check._point_objective, LADDER, TARGET, FLOPS)
    region_runs = check._optimal(region, LADDER, TARGET, FLOPS)
    assert region(region_runs, LADDER, TARGET) <= region(point_runs, LADDER, TARGET) * (1 + 1e-9)


def test_pair_variance_is_twice_level_for_arbitrary_designs() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        runs = rng.dirichlet(np.ones(len(LADDER))) * 10
        assert pair_variance_factor(LADDER, runs, TARGET) == pytest.approx(2 * target_variance_factor(LADDER, runs, TARGET))


def test_committed_fit_structure_findings_all_stand_without_target_labels() -> None:
    from asla.provenance import Source

    out = Source.load("target_scoring/fit_structure.json")
    findings = out.get("findings")
    assert findings, "no findings recorded"
    for name, row in findings.items():
        assert row["uses_target_labels"] is False, name
        assert row["stands"] is True, name
    assert out.number("design_theorem.identity_check_pair_over_level.min") == pytest.approx(2.0)
    for decades in ("1_decades", "3_decades"):
        assert out.number(f"design_theorem.region_optimal_designs.{decades}.region_gain_over_point_optimal_design") <= 0
