"""WS2 ensemble projection: new curve families, pseudo-BMA weights, reliability."""

import numpy as np
import pytest

from asla.analysis.ensemble import (
    ensemble_projection,
    ensemble_rank,
    ensemble_report,
    pseudo_bma_weights,
)
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, saturation_crossover
from asla.models import (
    FitError,
    bpb_damped_power_law,
    bpb_power_law,
    bpb_saturating,
    fit_damped_power_law,
    fit_saturating,
)


def test_fit_saturating_recovers_noiseless_params():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.asarray(bpb_saturating(x, 0.95, 0.10, 2.0), dtype=float)
    floor, drop, c_half = fit_saturating(x, y)
    assert np.isclose(floor, 0.95, atol=5e-3)
    assert np.isclose(drop, 0.10, atol=5e-3)
    assert np.isclose(c_half, 2.0, rtol=0.15)


def test_fit_saturating_unit_invariant():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.asarray(bpb_saturating(x, 0.95, 0.10, 2.0), dtype=float)
    unit = 1e18
    rel = fit_saturating(x, y)
    raw = fit_saturating(x * unit, y)
    pred_rel = bpb_saturating(64.0, *rel)
    pred_raw = bpb_saturating(64.0 * unit, *raw)
    assert np.isclose(pred_rel, pred_raw, rtol=1e-3)


def test_fit_damped_power_law_recovers_plain_power_law_predictions():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.asarray(bpb_power_law(x, 0.9, 0.3, 0.4), dtype=float)
    params = fit_damped_power_law(x, y)
    pred = bpb_damped_power_law(x, *params)
    assert np.allclose(pred, y, atol=2e-3)


def test_fit_damped_power_law_requires_four_budgets():
    x = np.array([1.0, 2.0, 4.0])
    y = np.asarray(bpb_power_law(x, 0.9, 0.3, 0.4), dtype=float)
    with pytest.raises(FitError):
        fit_damped_power_law(x, y)


def test_pseudo_bma_weights_properties():
    w = pseudo_bma_weights(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(w, 1 / 3)
    w = pseudo_bma_weights(np.array([0.001, 0.1]))
    assert w[0] > w[1]
    assert np.isclose(w.sum(), 1.0)
    scaled = pseudo_bma_weights(np.array([0.001, 0.1]) * 1e6)
    assert np.allclose(w, scaled)
    with pytest.raises(ValueError):
        pseudo_bma_weights(np.array([-1.0, 0.5]))


def test_ensemble_flags_short_ladder_and_resolves_long_ladder_on_saturating_truth():
    # On a short ladder the families are indistinguishable inside the fitting
    # range — the ensemble must express that as disagreement, not certainty.
    rng = np.random.default_rng(0)
    short = np.repeat(np.array([1.0, 2.0, 4.0, 8.0]), 8)
    y_short = np.asarray(bpb_saturating(short, 0.95, 0.10, 2.0), dtype=float) + rng.normal(0, 0.002, len(short))
    proj_short = ensemble_projection(short, y_short, 64.0)
    assert proj_short.disagreement > 0.002  # larger than the seed noise scale

    # With budgets deep enough to reveal curvature, the saturating family
    # should dominate and the ensemble point should approach the truth.
    long = np.repeat(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]), 8)
    y_long = np.asarray(bpb_saturating(long, 0.95, 0.10, 2.0), dtype=float) + rng.normal(0, 0.002, len(long))
    proj_long = ensemble_projection(long, y_long, 64.0)
    weights = {fam.name: fam.weight for fam in proj_long.families}
    assert weights["saturating"] == max(weights.values())
    truth = float(bpb_saturating(64.0, 0.95, 0.10, 2.0))
    assert abs(proj_long.point - truth) < abs(proj_short.point - truth)
    assert proj_long.disagreement < proj_short.disagreement


def test_ensemble_agrees_with_power_law_on_power_law_truth():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(1729), cfg)
    rows = df[df["intervention"] == "late_scaler"]
    mask = rows["compute"].astype(float).apply(lambda c: bool(np.any(np.isclose(c, np.asarray(cfg.budgets.fit)))))
    rows = rows[mask]
    proj = ensemble_projection(
        rows["compute"].to_numpy(dtype=float),
        rows["bpb"].to_numpy(dtype=float),
        cfg.budgets.target,
    )
    power = next(fam for fam in proj.families if fam.name == "power_law")
    assert np.isfinite(proj.point)
    assert power.weight > 0.1


def test_ensemble_requires_four_distinct_budgets():
    x = np.array([1.0, 2.0, 4.0])
    y = np.asarray(bpb_power_law(x, 0.9, 0.3, 0.4), dtype=float)
    with pytest.raises(FitError):
        ensemble_projection(x, y, 64.0)


def test_ensemble_rank_covers_all_interventions_and_is_deterministic():
    cfg = AuditConfig()
    df = saturation_crossover(np.random.default_rng(3), cfg)
    first = ensemble_rank(df, cfg.budgets.fit, cfg.budgets.target)
    second = ensemble_rank(df, cfg.budgets.fit, cfg.budgets.target)
    assert set(first.index) == set(df["intervention"].unique())
    assert (first.values == second.values).all()
    assert list(first.values) == sorted(first.values)


def test_ensemble_report_reliability_flags_saturation():
    cfg = AuditConfig()
    df = saturation_crossover(np.random.default_rng(1729), cfg)
    from asla.analysis.audit import seed_noise_report

    band = seed_noise_report(df, cfg.budgets.target)["noise_band"]
    report = ensemble_report(df, cfg.budgets.fit, cfg.budgets.target, noise_band=band)
    assert set(report) == set(df["intervention"].unique())
    rho_saturating = report["saturating_early"]["reliability"]
    assert rho_saturating is not None and rho_saturating > 1.0
    none_report = ensemble_report(df, cfg.budgets.fit, cfg.budgets.target, noise_band=None)
    assert all(entry["reliability"] is None for entry in none_report.values())
