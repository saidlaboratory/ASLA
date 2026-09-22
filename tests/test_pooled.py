"""Tests for the partial-pooling projection estimators."""

import numpy as np
import pandas as pd
import pytest

from asla.analysis.pooled import (
    empirical_bayes_strength,
    exponent_variance_components,
    fit_per_intervention,
    fit_shared_exponent,
    fit_shrunk,
    make_shrinkage_ranker,
    pooled_ranker_suite,
    project,
    shared_exponent_ranker,
)


def _grid(alphas: dict[str, float], noise: float = 0.0, seed: int = 0, n_seeds: int = 3) -> pd.DataFrame:
    x = np.array([1e15, 1e16, 1e17, 1e18, 1e19], dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for i, (name, alpha) in enumerate(sorted(alphas.items())):
        for compute in x:
            value = 0.5 + 0.02 * i + 3.0 * compute ** (-alpha)
            for s in range(n_seeds):
                rows.append(
                    {
                        "intervention": name,
                        "intervention_class": "data",
                        "compute": float(compute),
                        "seed": s,
                        "bpb": value + rng.normal(0.0, noise),
                    }
                )
    return pd.DataFrame(rows)


BUDGETS = (1e15, 1e16, 1e17, 1e18, 1e19)


def test_shared_exponent_recovers_a_common_truth():
    df = _grid({n: 0.15 for n in "abcd"})
    params = fit_shared_exponent(df, BUDGETS)
    alphas = {name: par[2] for name, par in params.items()}
    assert len(set(np.round(list(alphas.values()), 9))) == 1  # one exponent for all
    assert alphas["a"] == pytest.approx(0.15, rel=1e-3)
    # per-intervention floors are preserved, and the ordering is right
    ranked = project(params, 1e21)
    assert list(ranked.index) == ["a", "b", "c", "d"]


def test_shrinkage_endpoints_reproduce_plain_and_complete_pooling():
    df = _grid({"a": 0.12, "b": 0.15, "c": 0.18})
    plain = fit_per_intervention(df, BUDGETS)
    at_zero = fit_shrunk(df, BUDGETS, strength=0.0)
    at_one = fit_shrunk(df, BUDGETS, strength=1.0)
    for name in plain:
        assert at_zero.params[name][2] == pytest.approx(plain[name][2], rel=1e-6)
    pooled = at_one.pooled_exponent
    assert all(par[2] == pytest.approx(pooled, rel=1e-9) for par in at_one.params.values())
    assert at_one.pooled_exponent == pytest.approx(np.mean([plain[n][2] for n in plain]), rel=1e-6)


def test_shrinkage_interpolates_monotonically():
    df = _grid({"a": 0.12, "b": 0.15, "c": 0.18})
    plain = fit_per_intervention(df, BUDGETS)["a"][2]
    pooled = fit_shrunk(df, BUDGETS, strength=1.0).pooled_exponent
    previous = plain
    for strength in (0.25, 0.5, 0.75, 1.0):
        alpha = fit_shrunk(df, BUDGETS, strength=strength).params["a"][2]
        assert abs(alpha - pooled) <= abs(previous - pooled) + 1e-12
        previous = alpha


def test_empirical_bayes_strength_endpoints_and_direction():
    assert empirical_bayes_strength(between_variance=0.0, within_variance=1.0) == 1.0
    assert empirical_bayes_strength(between_variance=1.0, within_variance=0.0) == 0.0
    assert empirical_bayes_strength(1.0, 1.0) == pytest.approx(0.5)
    assert empirical_bayes_strength(0.0, 0.0) == 1.0  # degenerate: nothing to distinguish, pool


def test_variance_components_shrink_more_when_exponents_are_really_equal():
    identical = _grid({n: 0.15 for n in "abcd"}, noise=0.01, seed=1)
    different = _grid({"a": 0.08, "b": 0.14, "c": 0.20, "d": 0.26}, noise=0.01, seed=1)
    b_same, w_same, _ = exponent_variance_components(identical, BUDGETS, n_boot=12, rng=np.random.default_rng(0))
    b_diff, w_diff, _ = exponent_variance_components(different, BUDGETS, n_boot=12, rng=np.random.default_rng(0))
    assert b_same < b_diff  # genuinely different exponents show more between-variance
    assert empirical_bayes_strength(b_same, w_same) > empirical_bayes_strength(b_diff, w_diff)


def test_shrinkage_reduces_projection_variance_on_noisy_equal_exponent_data():
    """The mechanism under test: pooling should stabilise the projected target value."""

    truth_alpha = 0.15
    plain_spread, pooled_spread = [], []
    for seed in range(12):
        df = _grid({n: truth_alpha for n in "abcd"}, noise=0.02, seed=seed)
        plain_spread.append(fit_per_intervention(df, BUDGETS)["a"][2])
        pooled_spread.append(fit_shrunk(df, BUDGETS, strength=1.0).params["a"][2])
    assert np.std(pooled_spread) < np.std(plain_spread)


def test_rankers_expose_the_expected_interface():
    df = _grid({"a": 0.12, "b": 0.15, "c": 0.18})
    shared = shared_exponent_ranker(df, BUDGETS, 1e21)
    assert list(shared.index) == sorted(shared.index, key=lambda n: shared[n])
    ranker = make_shrinkage_ranker(0.5)
    assert ranker.__name__ == "shrinkage_ranker_0.5"
    ranked = ranker(df, BUDGETS, 1e21)
    assert set(ranked.index) == {"a", "b", "c"}
    suite = pooled_ranker_suite((0.0, 0.5, 1.0))
    assert {"shrinkage_0", "shrinkage_0.5", "shrinkage_1", "shrinkage_eb", "shared_exponent"} <= set(suite)


def test_rankers_refuse_held_out_test_rows():
    df = _grid({"a": 0.12, "b": 0.15})
    df.attrs["split_role"] = "test"
    with pytest.raises(AssertionError, match="held-out test rows"):
        shared_exponent_ranker(df, BUDGETS, 1e21)


def test_fixed_exponent_solve_keeps_floor_and_amplitude_nonnegative():
    from asla.analysis.pooled import _fit_with_fixed_exponent

    x = np.array([1.0, 2.0, 4.0, 8.0])
    increasing = np.array([1.0, 1.2, 1.4, 1.6])  # rising values force A <= 0 at a positive exponent
    e, a, alpha = _fit_with_fixed_exponent(x, increasing, 0.15)
    assert a >= 0.0 and e >= 0.0 and alpha == 0.15


def test_shrinkage_strength_can_be_restricted_to_training_interventions():
    """The empirical-Bayes strength must not be estimated from evaluation interventions."""

    df = _grid({"a": 0.10, "b": 0.14, "c": 0.18, "d": 0.22}, noise=0.02, seed=3)
    train = ["a", "b"]
    leaky = fit_shrunk(df, BUDGETS, n_boot=10, rng=np.random.default_rng(0))
    split = fit_shrunk(df, BUDGETS, n_boot=10, rng=np.random.default_rng(0), strength_df=df[df["intervention"].isin(train)])
    assert leaky.strength_estimated_from == "same_rows_as_evaluation"
    assert leaky.strength_source == ("a", "b", "c", "d")
    assert split.strength_estimated_from == "separate_strength_rows"
    assert split.strength_source == ("a", "b")
    # both still fit and project every intervention
    assert set(split.params) == {"a", "b", "c", "d"}


def test_shrinkage_ranker_restricts_strength_rows_but_ranks_everything():
    df = _grid({"a": 0.10, "b": 0.14, "c": 0.18, "d": 0.22}, noise=0.02, seed=4)
    ranker = make_shrinkage_ranker(None, n_boot=8, seed=0, strength_interventions=["a", "b"])
    ranked = ranker(df, BUDGETS, 1e21)
    assert set(ranked.index) == {"a", "b", "c", "d"}


def test_fixed_strength_rankers_estimate_nothing_and_are_unaffected_by_the_split():
    df = _grid({"a": 0.10, "b": 0.14, "c": 0.18}, noise=0.02, seed=5)
    fixed = fit_shrunk(df, BUDGETS, strength=0.5, n_boot=4, rng=np.random.default_rng(0))
    assert fixed.strength_estimated_from == "fixed_constant"
    assert fixed.shrinkage == 0.5
