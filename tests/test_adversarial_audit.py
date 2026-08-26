"""Tests for the adversarial-audit modules under audit/."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))

import dependence_and_data_checks as dep  # noqa: E402
import detector_power as dp  # noqa: E402
import independent_rederivation as ir  # noqa: E402


def test_independent_power_law_fit_recovers_known_parameters():
    compute = np.array([1e15, 2e15, 4e15, 8e15, 1.6e16, 3.2e16], dtype=float)
    truth = (0.5, 3.0, 0.12)
    values = truth[0] + truth[1] * compute ** (-truth[2])
    e, a, alpha = ir.fit_power_law_independent(compute, values)
    predicted = e + a * compute ** (-alpha)
    assert np.allclose(predicted, values, rtol=1e-6)
    assert alpha == pytest.approx(truth[2], rel=1e-3)


def test_pairwise_mis_selection_matches_hand_count():
    truth = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert ir.pairwise_mis_selection({"a": 1.0, "b": 2.0, "c": 3.0}, truth) == 0.0
    assert ir.pairwise_mis_selection({"a": 3.0, "b": 2.0, "c": 1.0}, truth) == 1.0
    # only the (b, c) pair is inverted -> 1 of 3
    assert ir.pairwise_mis_selection({"a": 1.0, "b": 3.0, "c": 2.0}, truth) == pytest.approx(1 / 3)


def test_flips_ignore_exact_ties():
    truth = {"a": 1.0, "b": 1.0, "c": 2.0}
    assert ir.flips({"a": 2.0, "b": 1.0, "c": 3.0}, truth) == []


def test_welch_p_matches_scipy_and_handles_single_seed():
    from scipy.stats import ttest_ind

    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 7), rng.normal(0.6, 2.0, 5)
    assert ir.welch_p(x, y) == pytest.approx(float(ttest_ind(x, y, equal_var=False).pvalue))
    assert ir.welch_p(np.array([1.0]), y) is None


def test_benjamini_hochberg_step_up_and_none_handling():
    # BH step-up with m=4, q=0.05: sorted p = [.001, .02, .04, .9] against
    # thresholds k/m*q = [.0125, .025, .0375, .05]. The largest k with
    # p_(k) <= k/m*q is k=2, so the two smallest p-values are rejected.
    assert ir.benjamini_hochberg([0.001, 0.9, 0.02, 0.04], q=0.05) == [True, False, True, False]
    assert ir.benjamini_hochberg([0.9, 0.8], q=0.05) == [False, False]
    assert ir.benjamini_hochberg([None, 0.001], q=0.05) == [False, True]
    assert ir.benjamini_hochberg([], q=0.05) == []


def test_benjamini_yekutieli_is_more_conservative_than_bh():
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04]
    bh = ir.benjamini_hochberg(p_values, q=0.05)
    by = dep.benjamini_yekutieli(p_values, q=0.05)
    assert sum(by) <= sum(bh)
    assert by[0] is True or by[0] == bh[0]


def test_make_pair_builds_a_true_crossover_and_a_true_non_crossover():
    budgets = np.array([1e16, 1e17, 1e18, 1e19], dtype=float)
    target = 7e20
    for crossover in (True, False):
        a, b = dp.make_pair(target_gap=0.01, crossover=crossover, budgets=budgets, target=target, ladder_gap=0.02)
        assert a is not None
        a_ladder = dp.curve(budgets, **a)[-1]
        b_ladder = dp.curve(budgets, **b)[-1]
        a_target = float(dp.curve(np.asarray([target]), **a)[0])
        b_target = float(dp.curve(np.asarray([target]), **b)[0])
        assert a_target < b_target  # 'a' always wins at the target
        assert bool(a_ladder > b_ladder) is crossover  # order on the ladder flips only when crossing


def test_simulate_once_recovers_a_noiseless_crossover_exactly():
    budgets = np.array([1e16, 1e17, 1e18, 1e19], dtype=float)
    target = 7e20
    a, b = dp.make_pair(0.01, True, budgets, target, 0.02)
    out = dp.simulate_once(a, b, budgets, target, 0.0, 0.0, 3, np.random.default_rng(0))
    assert out["observed_flip"] is True
    noncross_a, noncross_b = dp.make_pair(0.01, False, budgets, target, 0.02)
    out2 = dp.simulate_once(noncross_a, noncross_b, budgets, target, 0.0, 0.0, 3, np.random.default_rng(0))
    assert out2["observed_flip"] is False


def test_curvature_families_are_matched_on_signal_and_differ_only_in_geometry():
    import curvature_boundary_demo as cb

    level = cb.build_truth("level")
    curvature = cb.build_truth("curvature")
    # both have real, comparable target spread, so neither arm is at chance
    assert cb._target_spread(level) > 0.02
    assert cb._target_spread(curvature) > 0.02
    # level curves share one exponent; curvature curves do not
    assert len({round(alpha, 9) for _, _, alpha in level.values()}) == 1
    assert len({round(alpha, 9) for _, _, alpha in curvature.values()}) == len(curvature)
    # every level exponent sits below the 0.05 defect floor so the defect binds
    assert all(alpha < 0.05 for _, _, alpha in level.values())


def test_defect_is_monotone_on_level_geometry_and_reorders_on_curvature():
    import curvature_boundary_demo as cb

    level = cb.run_family("level", n_trials=8, seed=11)
    curvature = cb.run_family("curvature", n_trials=8, seed=11)
    # the defect binds on the level family (that is the point: pinned yet harmless)
    assert level["n_pinned_under_defect"] >= 1
    # decisions are essentially untouched on level geometry ...
    assert level["mean_pairs_reordered"] < 0.5
    assert level["mean_spearman_defective_vs_corrected"] > 0.99
    # ... and materially disturbed on curvature geometry
    assert curvature["mean_pairs_reordered"] > level["mean_pairs_reordered"]
    assert curvature["fraction_trials_ordering_identical"] < 1.0


def test_curvature_truths_reject_a_floor_above_the_ladder_pin():
    import curvature_boundary_demo as cb

    with pytest.raises(ValueError, match="exceeds the ladder pin"):
        cb._curvature_truths(4, floor_lo=0.0, floor_hi=5.0)
