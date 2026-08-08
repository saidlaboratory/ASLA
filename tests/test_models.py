import numpy as np
import pytest

from asla.models import FitError, bootstrap_projection, bpb_power_law, fit_power_law


def test_fit_power_law_recovers_clean_parameters():
    params = (0.82, 0.55, 0.37)
    compute = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    bpb = bpb_power_law(compute, *params)
    fitted = fit_power_law(compute, bpb)
    assert np.allclose(fitted, params, rtol=0.03, atol=0.015)


def test_bootstrap_projection_interval_contains_truth_often():
    rng = np.random.default_rng(123)
    params = (0.86, 0.42, 0.40)
    target = 64.0
    contains = 0
    trials = 25
    compute_base = np.repeat(np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 5)
    truth = float(bpb_power_law(target, *params))
    for _ in range(trials):
        bpb = bpb_power_law(compute_base, *params) + rng.normal(0.0, 0.004, size=len(compute_base))
        point, std, lo, hi = bootstrap_projection(compute_base, bpb, target, 80, rng)
        assert std >= 0.0
        assert lo <= point <= hi or lo <= truth <= hi
        contains += int(lo <= truth <= hi)
    assert contains / trials >= 0.65


def test_fit_power_law_is_unit_invariant_at_flop_scale():
    alpha = 0.5
    params = (0.90, 0.33 * (1e18) ** alpha, alpha)
    compute = np.array([1e18, 2e18, 4e18, 8e18])
    bpb = bpb_power_law(compute, *params)
    fitted = fit_power_law(compute, bpb)
    target = 6.4e19
    truth = float(bpb_power_law(target, *params))
    projected = float(bpb_power_law(target, *fitted))
    assert abs(projected - truth) < 1e-3


def test_fit_power_law_projection_matches_across_unit_conventions():
    params_rel = (0.86, 0.42, 0.40)
    compute_rel = np.array([1.0, 2.0, 4.0, 8.0])
    bpb = bpb_power_law(compute_rel, *params_rel)
    unit = 1e18
    fitted_rel = fit_power_law(compute_rel, bpb)
    fitted_raw = fit_power_law(compute_rel * unit, bpb)
    target_rel = 64.0
    proj_rel = float(bpb_power_law(target_rel, *fitted_rel))
    proj_raw = float(bpb_power_law(target_rel * unit, *fitted_raw))
    assert abs(proj_rel - proj_raw) < 1e-4


def test_fit_power_law_rejects_nonfinite_inputs():
    compute = np.array([1.0, 2.0, 4.0])
    bpb = np.array([1.2, np.nan, 1.0])
    with pytest.raises(FitError, match="finite"):
        fit_power_law(compute, bpb)


def test_bootstrap_projection_rejects_nonpositive_bootstrap_count():
    compute = np.array([1.0, 2.0, 4.0])
    bpb = np.array([1.2, 1.1, 1.0])
    with pytest.raises(FitError, match="n_boot"):
        bootstrap_projection(compute, bpb, 8.0, 0, np.random.default_rng(1))


def test_projection_bootstrap_preserves_every_compute_cell(monkeypatch):
    seen = []

    def fake_fit(compute, bpb, sigma=None):
        seen.append(set(np.asarray(compute, dtype=float)))
        return (0.8, 0.4, 0.5)

    monkeypatch.setattr("asla.models.fit_power_law", fake_fit)
    compute = np.repeat(np.array([1.0, 2.0, 4.0]), [2, 3, 4])
    bpb = np.linspace(1.4, 1.0, len(compute))
    bootstrap_projection(compute, bpb, 8.0, 5, np.random.default_rng(2))
    assert len(seen) == 6  # point fit plus five bootstrap fits
    assert all(scales == {1.0, 2.0, 4.0} for scales in seen)


def test_projection_bootstrap_reports_insufficient_fit_success(monkeypatch):
    calls = 0

    def fail_after_point(compute, bpb, sigma=None):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FitError("adversarial bootstrap failure")
        return (0.8, 0.4, 0.5)

    monkeypatch.setattr("asla.models.fit_power_law", fail_after_point)
    compute = np.repeat(np.array([1.0, 2.0, 4.0]), 2)
    bpb = np.linspace(1.4, 1.0, len(compute))
    with pytest.raises(FitError, match="too few stratified bootstrap resamples"):
        bootstrap_projection(compute, bpb, 8.0, 5, np.random.default_rng(2))
