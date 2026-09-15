"""Tests for the H-norm / H-curv discriminating test."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hnorm_hcurv_test import residuals_in_space, synthetic_control  # noqa: E402


def test_synthetic_control_separates_amplitude_from_curvature():
    """The decisive factorial: amplitude sharing governs cancellation, curvature does not."""

    control = synthetic_control(n_interventions=20, seed=0)
    grid = control["grid"]
    shared = [r for r in grid if r["misspecification_amplitude"] == "shared"]
    per_intervention = [r for r in grid if r["misspecification_amplitude"] == "per_intervention"]
    # curvature spread spans 0 -> 0.6 in both arms
    assert {r["true_alpha_cv"] for r in shared} == {0.0, 0.3, 0.6}
    # shared amplitude: cancellation stays near zero regardless of curvature
    assert all(r["cancellation"] < 0.05 for r in shared)
    assert max(r["cancellation"] for r in shared) - min(r["cancellation"] for r in shared) < 0.1
    # per-intervention amplitude: cancellation is destroyed regardless of curvature
    assert all(r["cancellation"] > 0.8 for r in per_intervention)
    # and the residuals are real misspecification, not numerical noise
    assert all(r["mean_abs_relative_residual"] > 1e-4 for r in grid)
    assert control["summary"]["shared"]["varies_with_alpha_cv"] is False


def test_control_residuals_are_not_numerical_noise():
    """Guards the bug that invalidated the first control attempt: exact power laws
    produce residuals at 1e-16, so PCA of them is meaningless."""

    control = synthetic_control(n_interventions=10, seed=1)
    assert all(r["mean_abs_relative_residual"] > 1e-6 for r in control["grid"])


def test_residuals_in_space_supports_three_spaces_and_differs_between_them():
    # The data must be MISSPECIFIED for residuals to exist at all: an exact power
    # law leaves residuals at ~1e-16 in every space and the comparison is vacuous.
    # This is the same trap that invalidated the first synthetic control attempt.
    x = np.geomspace(1e15, 1e19, 8)
    log_x = np.log(x)
    wiggle = 0.02 * np.sin(2 * np.pi * (log_x - log_x.min()) / (log_x.max() - log_x.min()))
    values = {f"r{i}": (0.5 + 0.02 * i + 3.0 * x ** (-0.15)) * (1 + wiggle) for i in range(5)}
    cells = pd.DataFrame(values, index=x).T
    spaces = {s: residuals_in_space(cells, s) for s in ("absolute", "relative", "log")}
    for frame in spaces.values():
        assert frame.shape == (5, 8)
    assert np.abs(spaces["relative"].to_numpy()).mean() > 1e-4  # real misspecification
    # relative residuals are the absolute ones divided by the fitted value, so they differ in scale
    assert not np.allclose(spaces["absolute"].to_numpy(), spaces["relative"].to_numpy())


def test_a_shared_amplitude_cancels_and_a_varying_one_does_not():
    from scripts.run_common_mode_gate import cancellation_ratio

    x = np.geomspace(1e15, 1e19, 10)
    log_x = np.log(x)
    wiggle = 0.02 * np.sin(2 * np.pi * (log_x - log_x.min()) / (log_x.max() - log_x.min()))
    rng = np.random.default_rng(0)
    shared = pd.DataFrame(
        {f"r{i}": (0.55 + 0.01 * i + 3.0 * x ** (-0.15)) * (1 + wiggle) for i in range(15)}, index=x
    ).T
    varying = pd.DataFrame(
        {f"r{i}": (0.55 + 0.01 * i + 3.0 * x ** (-0.15)) * (1 + rng.normal(0, 1) * wiggle) for i in range(15)},
        index=x,
    ).T
    assert cancellation_ratio(residuals_in_space(shared, "relative"))["mean_ratio"] < 0.05
    assert cancellation_ratio(residuals_in_space(varying, "relative"))["mean_ratio"] > 0.5
