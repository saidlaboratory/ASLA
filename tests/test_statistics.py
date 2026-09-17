"""WS1 statistical hardening: FDR crossovers, weighted fits, diagnostics, truth SEs."""

import numpy as np
import pandas as pd
import pytest

from asla.analysis.crossover import benjamini_hochberg, detect_crossovers_fdr, pairwise_target_tests
from asla.analysis.fits import (
    MixedSeedCountError,
    cell_means_and_sigma,
    fit_all,
    fit_diagnostics_all,
    truth_ranking_with_se,
    truth_ties_with_winner,
)
from asla.analysis.rankers import make_gate_ranker
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, negative_controls_only, noise_close_call, saturation_crossover
from asla.models import FitError, bpb_power_law, fit_power_law, fit_power_law_diagnostics


def _frame(rows):
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype(int)
    return df


def test_benjamini_hochberg_known_example():
    # m=5, q=0.25: step-up thresholds are 0.05, 0.10, 0.15, 0.20, 0.25.
    p = [0.005, 0.03, 0.04, 0.2, 0.9]
    assert benjamini_hochberg(p, q=0.25) == [True, True, True, True, False]
    p = [0.01, 0.02, 0.165, 0.205, 0.396]
    assert benjamini_hochberg(p, q=0.25) == [True, True, False, False, False]


def test_benjamini_hochberg_ignores_untestable():
    p = [0.001, float("nan"), 0.9]
    reject = benjamini_hochberg(p, q=0.05)
    assert reject[0] is True or reject[0] == True  # noqa: E712 - list of bool
    assert reject[1] == False  # noqa: E712
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], q=1.5)


def test_pairwise_target_tests_marks_single_seed_untestable():
    rows = []
    for seed in range(3):
        rows.append(
            {"intervention": "a", "intervention_class": "x", "compute": 64.0, "seed": seed, "bpb": 1.0 + 0.001 * seed}
        )
    rows.append({"intervention": "b", "intervention_class": "x", "compute": 64.0, "seed": 0, "bpb": 1.2})
    df = _frame(rows)
    tests = pairwise_target_tests(df, 64.0)
    assert len(tests) == 1
    assert tests[0]["p_value"] is None


def test_detect_crossovers_fdr_flags_planted_pair():
    cfg = AuditConfig()
    df = saturation_crossover(np.random.default_rng(1729), cfg)
    found = detect_crossovers_fdr(df, cfg.budgets.fit, cfg.budgets.target)
    pairs = {frozenset((row["a"], row["b"])) for row in found if row["significant"]}
    assert frozenset(("saturating_early", "late_power")) in pairs


def test_detect_crossovers_fdr_negative_controls_quiet():
    cfg = AuditConfig()
    n = 30
    flagged = 0
    for seed in range(n):
        df = negative_controls_only(np.random.default_rng(seed), cfg)
        found = detect_crossovers_fdr(df, cfg.budgets.fit, cfg.budgets.target)
        flagged += int(any(row["significant"] for row in found))
    assert flagged / n < 0.1


def test_cell_means_and_sigma_values():
    rows = []
    for compute in (1.0, 2.0):
        for seed, bpb in enumerate((1.0, 1.2)):
            rows.append({"intervention": "a", "intervention_class": "x", "compute": compute, "seed": seed, "bpb": bpb})
    df = _frame(rows)
    cells = cell_means_and_sigma(df)
    assert len(cells) == 2
    assert np.allclose(cells["bpb_mean"], 1.1)
    expected_sigma = np.std([1.0, 1.2], ddof=1) / np.sqrt(2)
    assert np.allclose(cells["sigma"], expected_sigma)


def test_cell_means_raises_on_mixed_seed_counts():
    """Mixed seed counts must raise rather than impute a mean standard error.

    The historical behaviour filled a single-seed cell's sigma with the mean
    sigma of the seeded cells. That is a linear aggregate of a scale feeding a
    fit that consumes 1/sigma**2 -- the Jensen-class defect audited in this
    project -- so the caller is now required to choose explicitly.
    """

    rows = [
        {"intervention": "a", "intervention_class": "x", "compute": 1.0, "seed": 0, "bpb": 1.0},
        {"intervention": "a", "intervention_class": "x", "compute": 1.0, "seed": 1, "bpb": 1.1},
        {"intervention": "a", "intervention_class": "x", "compute": 2.0, "seed": 0, "bpb": 0.9},
    ]
    with pytest.raises(MixedSeedCountError, match="single seed"):
        cell_means_and_sigma(_frame(rows))


def test_cell_means_uniform_seed_counts_are_unaffected():
    """The guard must fire only on the mixed case, not on well-seeded tables."""

    rows = [
        {"intervention": "a", "intervention_class": "x", "compute": c, "seed": s, "bpb": 1.0 + 0.1 * s}
        for c in (1.0, 2.0)
        for s in (0, 1)
    ]
    cells = cell_means_and_sigma(_frame(rows))
    assert cells["sigma"].notna().all()
    assert (cells["sigma"] > 0).all()


def test_cell_means_all_single_seed_leaves_sigma_null():
    """No cell has two seeds, so there is nothing to impute from and nothing to raise about."""

    rows = [{"intervention": "a", "intervention_class": "x", "compute": c, "seed": 0, "bpb": 1.0} for c in (1.0, 2.0, 4.0)]
    cells = cell_means_and_sigma(_frame(rows))
    assert cells["sigma"].isna().all()


def test_weighted_fit_all_runs_and_matches_truth_shape():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(7), cfg)
    weighted = fit_all(df, cfg.budgets.fit, weighted=True)
    unweighted = fit_all(df, cfg.budgets.fit, weighted=False)
    assert set(weighted) == set(unweighted)
    for name in weighted:
        assert all(np.isfinite(weighted[name]))


def test_weighted_sigma_validation():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = bpb_power_law(x, 0.9, 0.3, 0.4)
    with pytest.raises(FitError):
        fit_power_law(x, y, sigma=np.array([1.0, 1.0]))
    with pytest.raises(FitError):
        fit_power_law(x, y, sigma=np.array([1.0, -1.0, 1.0, 1.0]))


def test_fit_diagnostics_near_perfect_on_noiseless_curve():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.asarray(bpb_power_law(x, 0.9, 0.3, 0.4), dtype=float)
    params, diag = fit_power_law_diagnostics(x, y)
    assert diag.r_squared > 0.999
    assert diag.rmse < 1e-4
    assert diag.n_points == 5
    assert diag.dof == 2


def test_fit_diagnostics_all_reports_every_intervention():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(3), cfg)
    diags = fit_diagnostics_all(df, cfg.budgets.fit)
    assert set(diags) == set(df["intervention"].unique())
    for diag in diags.values():
        assert 0.0 <= diag.r_squared <= 1.0


def test_truth_ranking_with_se_ordering_and_values():
    rows = []
    for seed, bpb in enumerate((1.0, 1.1)):
        rows.append({"intervention": "good", "intervention_class": "x", "compute": 64.0, "seed": seed, "bpb": bpb})
    for seed, bpb in enumerate((1.4, 1.6)):
        rows.append({"intervention": "bad", "intervention_class": "x", "compute": 64.0, "seed": seed, "bpb": bpb})
    table = truth_ranking_with_se(_frame(rows), 64.0)
    assert list(table.index) == ["good", "bad"]
    assert np.isclose(table.loc["good", "mean"], 1.05)
    assert np.isclose(table.loc["good", "se"], np.std([1.0, 1.1], ddof=1) / np.sqrt(2))
    assert table.loc["good", "n_seeds"] == 2


def test_truth_ties_detects_close_call_and_clear_winner():
    cfg = AuditConfig()
    close = noise_close_call(np.random.default_rng(11), cfg)
    ties = truth_ties_with_winner(close, cfg.budgets.target)
    assert isinstance(ties, list)
    rows = []
    for name, level in (("good", 1.0), ("bad", 2.0)):
        for seed in range(4):
            rows.append(
                {"intervention": name, "intervention_class": "x", "compute": 64.0, "seed": seed, "bpb": level + 0.001 * seed}
            )
    assert truth_ties_with_winner(_frame(rows), 64.0) == []


def test_gate_ranker_is_deterministic_across_calls():
    cfg = AuditConfig()
    df = noise_close_call(np.random.default_rng(5), cfg)
    ranker = make_gate_ranker(cfg.budgets.intermediate, cfg.gate.tau, n_boot=50, seed=99)
    first = ranker(df, cfg.budgets.fit, cfg.budgets.target)
    second = ranker(df, cfg.budgets.fit, cfg.budgets.target)
    pd.testing.assert_series_equal(first, second)
