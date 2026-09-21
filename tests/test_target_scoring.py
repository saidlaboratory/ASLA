"""Tests for scoring against a target whose ordering is itself uncertain."""

from __future__ import annotations

import pytest

from asla.analysis.target_scoring import (
    benjamini_hochberg_threshold,
    decomposition,
    score_ranker,
    target_evidence,
    welch_pair,
)


def test_welch_df_is_not_four_when_variances_differ() -> None:
    """4 df is the maximum at n=3, attained only under equal variances.

    The draft asserted 4 df for all three-seed comparisons. That is the
    best case, and it is anti-conservative everywhere else.
    """

    _, _, df_equal, _ = welch_pair(1.0, 0.01, 3, 2.0, 0.01, 3)
    assert df_equal == pytest.approx(4.0, abs=1e-9)

    _, _, df_unequal, _ = welch_pair(1.0, 0.001, 3, 2.0, 0.05, 3)
    assert df_unequal < 2.5


def test_welch_pair_requires_two_observations() -> None:
    with pytest.raises(ValueError):
        welch_pair(1.0, 0.01, 1, 2.0, 0.01, 3)


def test_benjamini_hochberg_threshold_matches_step_up() -> None:
    assert benjamini_hochberg_threshold([0.001, 0.02, 0.5], 0.05) == pytest.approx(0.02)
    assert benjamini_hochberg_threshold([0.4, 0.6], 0.05) == 0.0
    assert benjamini_hochberg_threshold([], 0.05) == 0.0


def _three_candidates(gap: float, sd: float = 0.001):
    means = {"a": 1.0, "b": 1.0 + gap, "c": 1.0 + 2 * gap}
    sds = {k: sd for k in means}
    counts = {k: 3 for k in means}
    return means, sds, counts


def test_wide_gaps_are_determined_and_narrow_ones_are_not() -> None:
    wide = target_evidence(*_three_candidates(0.5), multiplicity="bonferroni")
    assert all(e.determined for e in wide)

    narrow = target_evidence(*_three_candidates(1e-6), multiplicity="bonferroni")
    assert not any(e.determined for e in narrow)


def test_bonferroni_is_stricter_than_bh_which_is_stricter_than_none() -> None:
    means, sds, counts = _three_candidates(0.004)
    counts_by_rule = {
        rule: sum(1 for e in target_evidence(means, sds, counts, multiplicity=rule) if e.determined)
        for rule in ("bonferroni", "bh", "none")
    }
    assert counts_by_rule["bonferroni"] <= counts_by_rule["bh"] <= counts_by_rule["none"]


def test_unknown_multiplicity_rejected() -> None:
    means, sds, counts = _three_candidates(0.1)
    with pytest.raises(ValueError):
        target_evidence(means, sds, counts, multiplicity="holm")


def test_expected_error_charges_less_than_observed_on_undetermined_pairs() -> None:
    """A ranker must not be fully charged for missing a coin flip."""

    means, sds, counts = _three_candidates(1e-6)
    evidence = target_evidence(means, sds, counts, multiplicity="none")
    # Predict the opposite of the observed order everywhere.
    predictions = {(e.left, e.right): -e.gap for e in evidence}
    scored = score_ranker(predictions, evidence)
    assert scored["observed_rate"] == pytest.approx(1.0)
    # Near-coin-flip pairs cost about half in expectation, not one.
    assert 0.4 < scored["expected_rate"] < 0.6


def test_decomposition_counts_the_paired_discordant_cells() -> None:
    """introduced and corrected are exactly the discordant cells."""

    means, sds, counts = _three_candidates(0.5)
    evidence = target_evidence(means, sds, counts, multiplicity="none")
    truth = {(e.left, e.right): e.gap for e in evidence}
    keys = sorted(truth)

    left = dict(truth)
    right = dict(truth)
    left[keys[0]] = -truth[keys[0]]  # projection wrong, single right -> introduced
    right[keys[1]] = -truth[keys[1]]  # projection right, single wrong -> corrected
    left[keys[2]] = -truth[keys[2]]
    right[keys[2]] = -truth[keys[2]]  # both wrong -> shared

    split = decomposition(left, right, evidence)
    assert split["introduced"] == 1
    assert split["corrected"] == 1
    assert split["shared"] == 1
    assert split["net_introduced"] == 0
    assert split["both_correct"] == 0


def test_shipped_reproduction_check_is_recorded() -> None:
    """The committed audit counts must be checked, not assumed."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "target_scoring" / "target_scoring.json"
    if not path.exists():  # pragma: no cover - results not always materialised
        pytest.skip("target scoring results not present")
    check = json.loads(path.read_text())["reproduction_check"]
    assert "reproduces" in check
    # Two independent implementations must agree with each other.
    assert (
        check["recomputed_with_audit_detectors"]["projection_flips"]
        == check["recomputed_in_this_script"]["projection_flips"]
    )
