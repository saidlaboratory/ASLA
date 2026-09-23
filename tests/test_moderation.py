"""Tests for empirical-Bayes variance moderation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from asla.analysis.moderation import (
    fit_prior,
    homogeneity,
    moderate,
    moderated_evidence,
    posterior_df,
    trigamma_inverse,
)
from asla.analysis.target_scoring import target_evidence


@pytest.mark.parametrize("y", [0.05, 0.7, 3.0, 40.0])
def test_trigamma_inverse_inverts_trigamma(y: float) -> None:
    assert trigamma_inverse(float(special.polygamma(1, y))) == pytest.approx(y, rel=1e-6)


def _hierarchical(n: int, d0: float, s0: float, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sigma2 = s0 * d0 / rng.chisquare(d0, size=n)
    s2 = sigma2 * rng.chisquare(d, size=n) / d
    return sigma2, s2


def test_fit_prior_recovers_known_hyperparameters_on_average() -> None:
    rng = np.random.default_rng(0)
    fits = [fit_prior(_hierarchical(400, 8.0, 2.0, 2, rng)[1], 2) for _ in range(60)]
    assert np.median([f.df_prior for f in fits]) == pytest.approx(8.0, rel=0.3)
    assert np.median([f.var_prior for f in fits]) == pytest.approx(2.0, rel=0.15)


def test_moderation_beats_raw_variances_in_log_error() -> None:
    rng = np.random.default_rng(1)
    sigma2, s2 = _hierarchical(25, 6.0, 1.0, 2, rng)
    prior = fit_prior(s2, 2)
    moderated = moderate(s2, prior)
    assert np.mean(np.log(moderated / sigma2) ** 2) < np.mean(np.log(s2 / sigma2) ** 2)


def test_homogeneous_variances_give_infinite_prior_df() -> None:
    rng = np.random.default_rng(2)
    s2 = rng.chisquare(2, size=2000) / 2  # every cell has sigma^2 = 1
    prior = fit_prior(s2, 2)
    assert prior.df_prior > 50 or not np.isfinite(prior.df_prior)
    if not np.isfinite(prior.df_prior):
        assert np.allclose(moderate(s2, prior), s2.mean())
        assert posterior_df(prior) == float("inf")


def test_moderated_evidence_adds_degrees_of_freedom() -> None:
    means = {"a": 1.0, "b": 1.2, "c": 1.5, "d": 1.9}
    s2 = {"a": 0.01, "b": 0.02, "c": 0.015, "d": 0.012}
    counts = dict.fromkeys(means, 3)
    prior = fit_prior(list(s2.values()), 2)
    moderated = moderated_evidence(means, s2, counts, prior)
    raw = target_evidence(means, {k: np.sqrt(v) for k, v in s2.items()}, counts)
    assert all(m.welch_df >= r.welch_df for m, r in zip(moderated, raw))
    assert all(0.5 <= m.probability_observed_order_correct <= 1 for m in moderated)


def test_homogeneity_reports_both_tests() -> None:
    rng = np.random.default_rng(3)
    groups = [rng.normal(0, 1, 3) for _ in range(20)]
    out = homogeneity(groups)
    assert 0 <= out["bartlett_p"] <= 1 and 0 <= out["brown_forsythe_p"] <= 1
