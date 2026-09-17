"""Tests for the delta-PAC abstention rule.

The important ones check the *guarantee*, not just the plumbing:
:func:`test_empirical_error_rate_respects_delta` simulates the null and confirms
the family-wise error rate is held, and
:func:`test_effective_error_rate_exposes_undercoverage` checks that a nominal
delta cannot be quoted as earned when the intervals undercover.
"""

from __future__ import annotations

import numpy as np
import pytest

from asla.analysis.abstention import (
    Decision,
    abstention_rate,
    bonferroni_delta,
    certify_leaderboard,
    certify_pair,
    effective_error_rate,
    glr_threshold,
    seeds_to_resolve,
)


def test_bonferroni_splits_delta() -> None:
    assert bonferroni_delta(0.05, 10) == pytest.approx(0.005)
    with pytest.raises(ValueError):
        bonferroni_delta(0.0, 1)
    with pytest.raises(ValueError):
        bonferroni_delta(0.05, 0)


def test_glr_threshold_grows_with_comparisons_and_confidence() -> None:
    assert glr_threshold(0.05, 1) == pytest.approx(1.959963985, rel=1e-6)
    assert glr_threshold(0.05, 10) > glr_threshold(0.05, 1)
    assert glr_threshold(0.01, 1) > glr_threshold(0.05, 1)


def test_certify_pair_abstains_on_small_gap() -> None:
    verdict = certify_pair("a", "b", 1.000, 1.001, 0.01, 0.05)
    assert verdict.decision is Decision.ABSTAIN
    assert verdict.abstained


def test_certify_pair_certifies_clear_gap() -> None:
    """Lower is better, so the smaller mean wins."""

    verdict = certify_pair("a", "b", 1.00, 2.00, 0.01, 0.05)
    assert verdict.decision is Decision.LEFT
    verdict = certify_pair("a", "b", 2.00, 1.00, 0.01, 0.05)
    assert verdict.decision is Decision.RIGHT


def test_certify_pair_respects_direction_flag() -> None:
    verdict = certify_pair("a", "b", 2.00, 1.00, 0.01, 0.05, lower_is_better=False)
    assert verdict.decision is Decision.LEFT


def test_certify_pair_zero_standard_error() -> None:
    """Zero noise certifies any real gap, and abstains on an exact tie."""

    assert certify_pair("a", "b", 1.0, 2.0, 0.0, 0.05).decision is Decision.LEFT
    assert certify_pair("a", "b", 1.0, 1.0, 0.0, 0.05).decision is Decision.ABSTAIN


def test_certify_pair_rejects_bad_standard_error() -> None:
    with pytest.raises(ValueError):
        certify_pair("a", "b", 1.0, 2.0, -1.0, 0.05)
    with pytest.raises(ValueError):
        certify_pair("a", "b", 1.0, 2.0, float("nan"), 0.05)


def test_t_threshold_far_exceeds_normal_at_small_seed_counts() -> None:
    """At 3 seeds the Gaussian quantile is anti-conservative by 3.6x.

    This is the difference between a guarantee that holds and one that does not,
    at exactly the seed count leaderboards actually use.
    """

    normal = glr_threshold(0.05, 300)
    student = glr_threshold(0.05, 300, n_seeds=3)
    assert student / normal > 3.5
    # The approximation only becomes adequate at large seed counts.
    assert glr_threshold(0.05, 300, n_seeds=50) / normal < 1.1
    # Monotone in seeds: more data, lower threshold.
    thresholds = [glr_threshold(0.05, 300, n_seeds=n) for n in (3, 5, 10, 20, 50)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_glr_threshold_rejects_too_few_seeds() -> None:
    with pytest.raises(ValueError):
        glr_threshold(0.05, 1, n_seeds=1)


def test_empirical_error_rate_with_estimated_sigma() -> None:
    """The guarantee where it is hardest: sigma estimated from 3 seeds.

    The Gaussian threshold fails badly here. This test draws real seed samples,
    estimates the standard error from them exactly as the study does, and checks
    the family-wise error rate is still held. It is the test that would have
    caught the normal-vs-t error, which the known-sigma simulation below cannot.
    """

    rng = np.random.default_rng(4242)
    delta = 0.05
    n_entries = 5
    n_seeds = 3
    sigma = 0.1
    trials = 2000
    n_comparisons = n_entries * (n_entries - 1) // 2

    gaussian_errors = 0
    student_errors = 0
    for _ in range(trials):
        draws = {f"e{i}": rng.normal(0.0, sigma, size=n_seeds) for i in range(n_entries)}
        means = {name: float(v.mean()) for name, v in draws.items()}
        errors = {name: float(v.std(ddof=1)) / np.sqrt(n_seeds) for name, v in draws.items()}
        gaussian = certify_leaderboard(means, errors, delta, adjacent_only=False)
        student = certify_leaderboard(means, errors, delta, adjacent_only=False, n_seeds=n_seeds)
        if any(not v.abstained for v in gaussian):
            gaussian_errors += 1
        if any(not v.abstained for v in student):
            student_errors += 1

    gaussian_rate = gaussian_errors / trials
    student_rate = student_errors / trials
    assert n_comparisons == 10
    # The t-based rule holds the guarantee.
    assert student_rate <= delta, f"t-based family-wise error {student_rate:.4f} exceeds {delta}"
    # The Gaussian rule does not, which is why n_seeds must be passed.
    assert gaussian_rate > student_rate


def test_empirical_error_rate_respects_delta() -> None:
    """The guarantee under known sigma, where the Gaussian threshold is valid.

    With all entries truly equal, every certified ordering is an error. The
    family-wise error rate must stay at or below delta. Note this simulation
    supplies the TRUE standard error, so it cannot detect a wrong reference
    distribution; see the estimated-sigma test above for that.
    """

    rng = np.random.default_rng(20260917)
    delta = 0.05
    n_entries = 5
    sigma = 0.1
    n_seeds = 3
    standard_error = sigma / np.sqrt(n_seeds)
    trials = 4000
    families_with_error = 0

    for _ in range(trials):
        means = {f"e{i}": float(rng.normal(0.0, standard_error)) for i in range(n_entries)}
        errors = {name: standard_error for name in means}
        verdicts = certify_leaderboard(means, errors, delta, adjacent_only=False)
        if any(not v.abstained for v in verdicts):
            families_with_error += 1

    observed = families_with_error / trials
    # Bonferroni is conservative, so the rate should sit at or below delta.
    assert observed <= delta + 0.01, f"family-wise error {observed:.4f} exceeds delta {delta}"


def test_power_recovers_true_ordering_when_gaps_are_large() -> None:
    """The rule must not abstain on everything: with clear gaps it certifies."""

    rng = np.random.default_rng(7)
    truth = {f"e{i}": float(i) for i in range(4)}
    errors = {name: 0.01 for name in truth}
    observed = {name: value + float(rng.normal(0, 0.01)) for name, value in truth.items()}
    verdicts = certify_leaderboard(observed, errors, 0.05)
    assert abstention_rate(verdicts) == pytest.approx(0.0)
    assert all(v.decision is Decision.LEFT for v in verdicts)


def test_abstention_rate_increases_as_delta_falls() -> None:
    entries = {"a": 1.000, "b": 1.030, "c": 1.065}
    errors = {name: 0.012 for name in entries}
    rates = [abstention_rate(certify_leaderboard(entries, errors, d)) for d in (0.20, 0.05, 0.001)]
    assert rates[0] <= rates[1] <= rates[2]


def test_abstention_rate_empty() -> None:
    assert np.isnan(abstention_rate([]))


def test_certify_leaderboard_requires_two_entries() -> None:
    with pytest.raises(ValueError):
        certify_leaderboard({"a": 1.0}, {"a": 0.1}, 0.05)


def test_certify_leaderboard_adjacent_vs_all_pairs() -> None:
    entries = {"a": 1.0, "b": 2.0, "c": 3.0}
    errors = {name: 0.01 for name in entries}
    assert len(certify_leaderboard(entries, errors, 0.05, adjacent_only=True)) == 2
    assert len(certify_leaderboard(entries, errors, 0.05, adjacent_only=False)) == 3
    # Multiplicity counts all pairs either way: the ordering is the claim.
    adjacent = certify_leaderboard(entries, errors, 0.05, adjacent_only=True)
    assert all(v.n_comparisons == 3 for v in adjacent)


def test_seeds_to_resolve_scales_inverse_square_of_gap() -> None:
    """Halving the gap quadruples the seeds required, in the Gaussian limit.

    Under the t correction the ratio is smaller (about 2.8 here), because the
    degrees of freedom grow with the answer: a design needing more seeds also
    gets a smaller quantile. The pure inverse-square law is the ``use_t=False``
    case, and that is what is asserted exactly.
    """

    wide = seeds_to_resolve(0.10, 0.05, 0.05, use_t=False)
    narrow = seeds_to_resolve(0.05, 0.05, 0.05, use_t=False)
    assert wide is not None and narrow is not None
    assert narrow / wide == pytest.approx(4.0, rel=0.15)

    wide_t = seeds_to_resolve(0.10, 0.05, 0.05)
    narrow_t = seeds_to_resolve(0.05, 0.05, 0.05)
    assert wide_t is not None and narrow_t is not None
    assert 2.0 < narrow_t / wide_t < 4.0


def test_seeds_to_resolve_t_demands_more_than_normal() -> None:
    """The t correction costs seeds, most of all when few are available."""

    with_t = seeds_to_resolve(0.02, 0.05, 0.05, n_comparisons=300, use_t=True)
    with_z = seeds_to_resolve(0.02, 0.05, 0.05, n_comparisons=300, use_t=False)
    assert with_t is not None and with_z is not None
    assert with_t > with_z


def test_seeds_to_resolve_never_returns_one() -> None:
    """A single seed gives no variance estimate; two is the floor."""

    result = seeds_to_resolve(1e6, 0.05, 0.05)
    assert result is not None and result >= 2


def test_seeds_to_resolve_returns_none_for_unresolvable() -> None:
    """A gap buried in noise gets None, not a silently clipped cap."""

    assert seeds_to_resolve(1e-9, 1.0, 0.05) is None
    assert seeds_to_resolve(0.0, 1.0, 0.05) is None
    assert seeds_to_resolve(-1.0, 1.0, 0.05) is None


def test_seeds_to_resolve_grows_with_multiplicity() -> None:
    single = seeds_to_resolve(0.1, 0.05, 0.05, n_comparisons=1)
    many = seeds_to_resolve(0.1, 0.05, 0.05, n_comparisons=300)
    assert single is not None and many is not None
    assert many > single


def test_effective_error_rate_exposes_undercoverage() -> None:
    """Undercovering intervals inflate the real error rate above nominal."""

    # Perfect calibration leaves delta unchanged.
    assert effective_error_rate(0.05, 0.90, 0.90) == pytest.approx(0.05)
    # Coverage of 0.55 against nominal 0.90 means 4.5x the intended miss rate.
    assert effective_error_rate(0.05, 0.55, 0.90) == pytest.approx(0.225, rel=1e-6)
    # Over-coverage reduces the effective error rate.
    assert effective_error_rate(0.05, 0.95, 0.90) < 0.05
    # Never exceeds 1.
    assert effective_error_rate(0.5, 0.0, 0.9) == pytest.approx(1.0)


def test_effective_error_rate_validates_inputs() -> None:
    with pytest.raises(ValueError):
        effective_error_rate(0.0, 0.9, 0.9)
    with pytest.raises(ValueError):
        effective_error_rate(0.05, 1.5, 0.9)
    with pytest.raises(ValueError):
        effective_error_rate(0.05, 0.9, 1.0)
