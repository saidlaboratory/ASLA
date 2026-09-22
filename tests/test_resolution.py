"""Tests for the leaderboard resolution-limit diagnostic."""

import pytest

from asla.analysis.resolution import (
    minimum_detectable_gap,
    resolution_report,
    seeds_required,
    sensitivity,
)


def test_minimum_detectable_gap_shrinks_with_seeds_and_grows_with_noise():
    base = minimum_detectable_gap(n_seeds=4, sigma=1.0)
    assert minimum_detectable_gap(n_seeds=16, sigma=1.0) < base
    assert minimum_detectable_gap(n_seeds=4, sigma=2.0) == pytest.approx(2 * base)
    # a multiplicity correction makes the bar stricter
    assert minimum_detectable_gap(n_seeds=4, sigma=1.0, n_comparisons=10) > base
    with pytest.raises(ValueError, match="at least 2 seeds"):
        minimum_detectable_gap(n_seeds=1, sigma=1.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        minimum_detectable_gap(n_seeds=4, sigma=0.0)


def test_seeds_required_is_the_inverse_of_minimum_detectable_gap():
    sigma = 0.01
    for n in (4, 9, 25):
        gap = minimum_detectable_gap(n_seeds=n, sigma=sigma)
        # the gap detectable at n seeds needs about n seeds to detect
        assert seeds_required(gap, sigma) in range(n - 1, n + 2)
    assert seeds_required(gap=0.0, sigma=1.0) is None
    # an absurdly small gap is reported as infeasible rather than a huge number
    assert seeds_required(gap=1e-9, sigma=1.0, max_seeds=1000) is None


def test_resolution_report_reproduces_the_motivating_optimizer_case():
    """1.2B/8xC: NAdamW vs SOAP differ by 6e-6 nats against sigma 1.4e-3."""

    entries = {"Muon": 2.748363, "NAdamW": 2.748523, "SOAP": 2.748529, "AdamW": 2.752277}
    report = resolution_report(entries, sigma=0.0014312, seed_budget=3)
    assert report["ranking"] == ["Muon", "NAdamW", "SOAP", "AdamW"]
    pairs = {f"{p['better']}|{p['worse']}": p for p in report["adjacent_pairs"]}
    tie = pairs["NAdamW|SOAP"]
    assert tie["noise_to_gap"] > 100
    assert tie["identifiable_at_any_budget"] is False
    assert tie["seeds_required"] is None
    # the widest gap is identifiable in principle, just not at 3 seeds
    assert pairs["SOAP|AdamW"]["identifiable_at_any_budget"] is True
    assert report["n_unidentifiable_at_any_budget"] == 1
    assert report["top_k_fully_resolved"] is False


def test_report_respects_metric_direction_and_multiplicity():
    entries = {"a": 0.90, "b": 0.80, "c": 0.70}
    lower = resolution_report(entries, sigma=0.01, seed_budget=5)
    higher = resolution_report(entries, sigma=0.01, seed_budget=5, lower_is_better=False)
    assert lower["ranking"] == ["c", "b", "a"]
    assert higher["ranking"] == ["a", "b", "c"]
    corrected = resolution_report(entries, sigma=0.05, seed_budget=5)
    uncorrected = resolution_report(entries, sigma=0.05, seed_budget=5, multiplicity="none")
    assert corrected["minimum_detectable_gap"] > uncorrected["minimum_detectable_gap"]
    with pytest.raises(ValueError, match="unknown multiplicity"):
        resolution_report(entries, sigma=0.01, multiplicity="holm")


def test_wide_gaps_are_resolvable_and_narrow_ones_are_not():
    wide = resolution_report({"a": 1.0, "b": 5.0}, sigma=0.01, seed_budget=3)
    assert wide["n_unresolved_at_budget"] == 0
    assert wide["top_k_fully_resolved"] is True
    narrow = resolution_report({"a": 1.0, "b": 1.0001}, sigma=0.01, seed_budget=3)
    assert narrow["n_unresolved_at_budget"] == 1


def test_sensitivity_spans_noise_and_budget_and_is_monotone():
    entries = {"a": 1.0, "b": 1.02, "c": 1.10}
    rows = sensitivity(entries, sigma=0.01, multipliers=(0.5, 1.0, 2.0), seed_budgets=(3, 10))
    assert len(rows) == 6
    # more seeds never make the detectable gap larger at fixed noise
    for multiplier in (0.5, 1.0, 2.0):
        at_multiplier = sorted([r for r in rows if r["sigma_multiplier"] == multiplier], key=lambda r: r["seed_budget"])
        gaps = [r["minimum_detectable_gap"] for r in at_multiplier]
        assert gaps == sorted(gaps, reverse=True)
    # more noise never makes it smaller at fixed budget
    at_budget = sorted([r for r in rows if r["seed_budget"] == 3], key=lambda r: r["sigma_multiplier"])
    assert [r["minimum_detectable_gap"] for r in at_budget] == sorted([r["minimum_detectable_gap"] for r in at_budget])


def test_report_requires_at_least_two_entries():
    with pytest.raises(ValueError, match="at least two entries"):
        resolution_report({"only": 1.0}, sigma=0.01)
