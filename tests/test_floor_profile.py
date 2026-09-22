"""Tests for the floor profile-likelihood interval."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

floor_profile = importlib.import_module("scripts.run_floor_profile")

COMPUTE = np.geomspace(1e17, 1e20, 11)


def _curve(floor: float, amplitude: float, alpha: float, noise: float, seed: int) -> np.ndarray:
    u = COMPUTE / np.exp(np.mean(np.log(COMPUTE)))
    rng = np.random.default_rng(seed)
    return floor + amplitude * u ** (-alpha) + rng.normal(0.0, noise, size=u.size)


def test_profile_ssr_is_no_worse_than_the_log_space_fit() -> None:
    """The fix: the profile minimises the response-scale residual."""

    y = _curve(2.0, 1.0, 0.3, 0.01, seed=0)
    u = COMPUTE / np.exp(np.mean(np.log(COMPUTE)))
    floor = 1.5
    slope, intercept = np.polyfit(np.log(u), np.log(y - floor), 1)
    log_space = float(np.sum((floor + np.exp(intercept) * u**slope - y) ** 2))
    profiled, _ = floor_profile.profile_ssr(u, y, floor)
    assert profiled <= log_space + 1e-12


def test_profile_ssr_rejects_floor_above_an_observation() -> None:
    y = _curve(2.0, 1.0, 0.3, 0.0, seed=0)
    ssr, params = floor_profile.profile_ssr(COMPUTE / 1e18, y, float(np.min(y)) + 0.1)
    assert ssr == float("inf")
    assert np.all(np.isnan(params))


def test_identified_floor_interval_covers_truth_and_excludes_zero() -> None:
    y = _curve(2.0, 1.0, 0.5, 0.002, seed=1)
    out = floor_profile.profile_interval(COMPUTE, y, sigma_mean=0.002)
    low, high = out["interval"]
    assert low <= 2.0 <= high
    assert not out["interval_includes_zero"]
    assert out["known_sigma"]["misspecification_ratio"] == pytest.approx(1.0, rel=1.0)


def test_curve_without_curvature_leaves_the_floor_unidentified() -> None:
    """A pure power law with zero floor and heavy noise must not pin E away from 0."""

    y = _curve(0.0, 0.5, 0.05, 0.01, seed=2)
    out = floor_profile.profile_interval(COMPUTE, y)
    assert out["interval_includes_zero"]
    assert "known_sigma" not in out
