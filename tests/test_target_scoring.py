"""Tests for scoring against a target whose ordering is itself uncertain."""

from __future__ import annotations

import pytest

from asla.analysis.target_scoring import (
    benjamini_hochberg_threshold,
    candidate_bootstrap,
    candidates_for_power,
    decomposition,
    expected_charges,
    jackknife_standard_error,
    score_ranker,
    target_evidence,
    u_statistic_components,
    u_statistic_test,
    u_statistic_variance,
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


def test_bootstrap_p_value_never_exceeds_one() -> None:
    """Ties at zero must not push the p-value above 1."""

    import numpy as np

    from asla.analysis.target_scoring import bootstrap_two_sided_p

    assert bootstrap_two_sided_p([0.0] * 90 + [0.1] * 5 + [-0.1] * 5) == 1.0
    assert bootstrap_two_sided_p([1.0] * 100) == 0.0
    assert 0.0 < bootstrap_two_sided_p([1.0] * 90 + [-1.0] * 10) < 1.0
    assert np.isnan(bootstrap_two_sided_p([]))


# --- Candidate-level inference on pair-mean statistics ---


def _additive_kernel(n: int, sd_u: float, sd_e: float, seed: int) -> dict[tuple[str, str], float]:
    """h(a, b) = u_a + u_b + e_ab: zeta1 = sd_u^2 and zeta2 = 2 sd_u^2 + sd_e^2 exactly."""

    from itertools import combinations

    import numpy as np

    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sd_u, size=n)
    names = [f"c{i:02d}" for i in range(n)]
    return {(names[i], names[j]): u[i] + u[j] + rng.normal(0.0, sd_e) for i, j in combinations(range(n), 2)}


def test_expected_charges_sum_to_score_ranker_expected_errors() -> None:
    evidence = target_evidence(
        {"a": 1.0, "b": 1.1, "c": 1.3}, {"a": 0.1, "b": 0.1, "c": 0.1}, {"a": 3, "b": 3, "c": 3}
    )
    predictions = {("a", "b"): -1.0, ("a", "c"): 1.0, ("b", "c"): -1.0}
    charges = expected_charges(predictions, evidence)
    assert sum(charges.values()) == pytest.approx(score_ranker(predictions, evidence)["expected_errors"])


def test_candidate_bootstrap_centres_on_the_full_sample_statistic() -> None:
    import numpy as np

    kernel = _additive_kernel(25, 1.0, 1.0, seed=0)
    draws = candidate_bootstrap(kernel, np.random.default_rng(1), 2000)
    assert np.mean(draws) == pytest.approx(np.mean(list(kernel.values())), abs=0.1)


def test_u_statistic_components_recover_known_zetas_on_average() -> None:
    import numpy as np

    estimates = [u_statistic_components(_additive_kernel(40, 1.0, 2.0, seed=s)) for s in range(200)]
    assert np.mean([e["zeta1"] for e in estimates]) == pytest.approx(1.0, rel=0.15)
    assert np.mean([e["zeta2"] for e in estimates]) == pytest.approx(6.0, rel=0.1)


def test_bootstrap_jackknife_and_formula_agree_on_the_standard_error() -> None:
    """The defect this guards: a unique-set bootstrap overstated the spread."""

    import numpy as np

    kernel = _additive_kernel(25, 1.0, 1.0, seed=3)
    components = u_statistic_components(kernel)
    formula = np.sqrt(u_statistic_variance(components["zeta1"], components["zeta2"], 25))
    boot = float(np.std(candidate_bootstrap(kernel, np.random.default_rng(4), 3000)))
    jack = jackknife_standard_error(kernel)
    assert boot == pytest.approx(formula, rel=0.2)
    assert jack == pytest.approx(formula, rel=0.2)


def test_u_statistic_variance_matches_simulation() -> None:
    import numpy as np

    values = [np.mean(list(_additive_kernel(25, 1.0, 1.0, seed=s).values())) for s in range(1500)]
    assert np.var(values) == pytest.approx(u_statistic_variance(1.0, 3.0, 25), rel=0.15)


def test_candidates_for_power_grows_as_the_effect_shrinks() -> None:
    small = candidates_for_power(1.0, 3.0, delta=0.2)
    large = candidates_for_power(1.0, 3.0, delta=0.5)
    assert small is not None and large is not None and small > large
    assert candidates_for_power(0.0, 0.0, delta=0.1) == 3


def test_weighted_bootstrap_is_conservative_when_pair_noise_dominates() -> None:
    """Why the U-statistic test is the headline: at zeta2/zeta1 near this project's
    value the weighted bootstrap overstates the standard error, the formula does not."""

    import numpy as np

    truth = np.sqrt(u_statistic_variance(1.0, 38.0, 25))
    boot, formula = [], []
    for seed in range(25):
        kernel = _additive_kernel(25, 1.0, 6.0, seed=seed)
        boot.append(np.std(candidate_bootstrap(kernel, np.random.default_rng(seed), 600)))
        formula.append(u_statistic_test(kernel)["standard_error"])
    assert np.mean(boot) / truth > 1.15
    assert np.mean(formula) / truth == pytest.approx(1.0, abs=0.1)


def test_u_statistic_test_reports_zero_difference_as_p_one() -> None:
    kernel = {("a", "b"): 0.0, ("a", "c"): 0.0, ("a", "d"): 0.0, ("b", "c"): 0.0, ("b", "d"): 0.0, ("c", "d"): 0.0}
    result = u_statistic_test(kernel)
    assert result["p_two_sided"] == 1.0 and not result["excludes_zero"]
