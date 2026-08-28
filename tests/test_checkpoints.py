"""Tests for checkpoint-augmented, correlation-aware fitting."""

import numpy as np
import pandas as pd
import pytest

from asla.analysis.checkpoints import (
    _per_scale_sigma,
    autocorrelation_summary,
    effective_sample_size,
    estimate_run_autocorrelation,
    fit_checkpoint_augmented,
    lag1_autocorrelation,
    make_checkpoint_ranker,
)
from asla.models import fit_power_law


def test_lag1_autocorrelation_recovers_a_known_ar1_process():
    rng = np.random.default_rng(0)
    white = rng.normal(size=500)
    assert lag1_autocorrelation(white) < 0.1
    series = np.zeros(500)
    for i in range(1, 500):
        series[i] = 0.6 * series[i - 1] + rng.normal()
    assert 0.45 < lag1_autocorrelation(series) < 0.75
    # negative autocorrelation is clipped: we decline to claim more information than rows
    alternating = np.array([1.0, -1.0] * 50)
    assert lag1_autocorrelation(alternating) == 0.0
    assert lag1_autocorrelation(np.array([1.0, 2.0])) == 0.0


def test_effective_sample_size_matches_the_ar1_formula():
    assert effective_sample_size(41, 0.242) == pytest.approx(41 * (1 - 0.242) / (1 + 0.242))
    assert effective_sample_size(10, 0.0) == 10.0
    assert effective_sample_size(10, 0.99) >= 1.0  # floored, never zero
    assert effective_sample_size(0, 0.5) == 0.0


def test_uniform_sigma_is_a_no_op_so_the_correction_must_vary_across_scales():
    """The trap this module guards against: a per-run constant weight cancels."""

    x = np.array([1e15, 1e16, 1e17, 1e18, 1e19], dtype=float)
    y = 0.5 + 3.0 * x ** (-0.15)
    baseline = np.asarray(fit_power_law(x, y))
    # A constant sigma cancels in the normal equations, so it changes the fitted
    # parameters only to optimizer tolerance (~1e-4 relative), not statistically.
    for constant in (2.0, 17.0):
        scaled = np.asarray(fit_power_law(x, y, sigma=np.full(len(x), constant)))
        assert np.allclose(scaled, baseline, rtol=1e-3)
    # A sigma that varies across points does change the fit materially - but only
    # when the data do not already determine the curve exactly, which is the
    # realistic case. This is why the correlation correction must differ between
    # scales rather than scaling a whole run uniformly.
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0.0, 0.01, size=len(y))
    noisy_baseline = np.asarray(fit_power_law(x, noisy))
    noisy_varied = np.asarray(fit_power_law(x, noisy, sigma=np.array([1.0, 1.0, 1.0, 10.0, 10.0])))
    assert np.max(np.abs((noisy_varied - noisy_baseline) / noisy_baseline)) > 1e-2


def test_per_scale_sigma_penalises_checkpoint_dense_scales():
    # two scales: one with many densely-spaced checkpoints, one with few
    dense = np.geomspace(1e15, 2e15, 30)
    sparse = np.geomspace(1e18, 2e18, 4)
    compute = np.concatenate([dense, sparse])
    values = 0.5 + 3.0 * compute ** (-0.15)
    labels = np.array(["small"] * len(dense) + ["big"] * len(sparse))
    sigma = _per_scale_sigma(compute, values, labels)
    assert sigma is not None
    # the dense scale gets the larger sigma (less weight per point)
    assert sigma[: len(dense)].mean() > sigma[len(dense) :].mean()
    assert _per_scale_sigma(compute, values, None) is None


def _grid(n_checkpoints_per_scale: dict[str, int], alpha: float = 0.15, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for offset, name in enumerate(("a", "b", "c")):
        for scale, count in n_checkpoints_per_scale.items():
            base = {"4M": 1e15, "150M": 1e17, "300M": 1e18}[scale]
            for compute in np.geomspace(base, base * 3, count):
                value = 0.5 + 0.02 * offset + 3.0 * compute ** (-alpha)
                for s in range(2):
                    rows.append(
                        {
                            "intervention": name,
                            "intervention_class": "data",
                            "compute": float(compute),
                            "seed": s,
                            "bpb": value + rng.normal(0.0, noise),
                            "scale_label": scale,
                        }
                    )
    return pd.DataFrame(rows)


def test_checkpoint_fit_recovers_a_known_exponent_and_reports_diagnostics():
    df = _grid({"4M": 6, "150M": 10, "300M": 12})
    budgets = tuple(sorted(df["compute"].unique()))
    params, diagnostics = fit_checkpoint_augmented(df, budgets, weighting="ar1")
    assert set(params) == {"a", "b", "c"}
    assert params["a"][2] == pytest.approx(0.15, rel=1e-2)
    summary = autocorrelation_summary(diagnostics)
    assert summary["n_runs"] == 3 and 0.0 <= summary["mean_rho"] <= 1.0
    assert summary["mean_effective_points"] <= summary["mean_n_points"]


def test_naive_and_corrected_fits_differ_when_checkpoint_density_is_uneven():
    df = _grid({"4M": 4, "150M": 30, "300M": 30}, noise=0.01, seed=3)
    budgets = tuple(sorted(df["compute"].unique()))
    naive, _ = fit_checkpoint_augmented(df, budgets, weighting="naive")
    corrected, _ = fit_checkpoint_augmented(df, budgets, weighting="ar1")
    assert any(not np.allclose(naive[n], corrected[n], rtol=1e-6) for n in naive)


def test_ranker_uses_every_checkpoint_below_the_target_and_never_the_target():
    df = _grid({"4M": 5, "150M": 8, "300M": 8})
    target = float(df["compute"].max())
    ranker = make_checkpoint_ranker("ar1")
    ranked = ranker(df, (float(df["compute"].min()),), target)
    assert list(ranked.index) == ["a", "b", "c"]
    assert ranker.__name__ == "checkpoint_ranker_ar1"


def test_ranker_refuses_held_out_rows_and_unknown_weighting():
    df = _grid({"4M": 5, "150M": 8, "300M": 8})
    budgets = tuple(sorted(df["compute"].unique()))
    with pytest.raises(ValueError, match="unknown weighting"):
        fit_checkpoint_augmented(df, budgets, weighting="bogus")  # type: ignore[arg-type]
    df.attrs["split_role"] = "test"
    with pytest.raises(AssertionError, match="held-out test rows"):
        fit_checkpoint_augmented(df, budgets)


def test_estimate_run_autocorrelation_reports_deflation():
    x = np.geomspace(1e15, 1e19, 40)
    y = 0.5 + 3.0 * x ** (-0.15)
    out = estimate_run_autocorrelation(x, y, "a")
    assert out.n_points == 40
    assert 0.0 <= out.rho < 1.0
    assert out.effective_sample_size <= out.n_points
    assert out.weight == pytest.approx(out.effective_sample_size / out.n_points)
