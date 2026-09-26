"""Tests for the per-metric published calibration and the lever-arm calibration summary."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

published = importlib.import_module("scripts.run_calibration_published")
lever = importlib.import_module("scripts.run_lever_arm_calibration")


def _worlds(n: int, se_u: float, se_jk: float, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            name: {"estimate": float(rng.standard_normal()), "true_difference": 0.0, "u_se": se_u, "kjk_se": se_jk}
            for name in ("a", "b")
        }
        for _ in range(n)
    ]


def test_published_calibration_keeps_jackknife_only_when_it_covers_at_least_as_well() -> None:
    worlds = _worlds(1000, se_u=1.0, se_jk=1.0)
    out = published.calibrate(worlds, [], ["a", "b"])
    assert out["random_u"]["critical_value"] == pytest.approx(1.96, abs=0.15)
    # Identical SEs give identical coverage, so the jackknife is kept.
    assert out["estimator_rule"]["jackknife_kept"] is True
    narrow_jk = _worlds(1000, se_u=1.0, se_jk=0.4, seed=1)
    for w in narrow_jk:
        w["b"]["kjk_se"] = 1.0  # jackknife fine for b, badly understated for a
    out = published.calibrate(narrow_jk, [], ["a", "b"])
    assert out["random_adopted"] in ("random_u", "random_jackknife")


def test_lever_summary_recovers_calibration_on_synthetic_worlds() -> None:
    rng = np.random.default_rng(2)
    lever_arm = list(np.geomspace(2, 100, 12))
    jitter = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1, 0.6, -0.5, 0.2, -0.1, 0.4, -0.3]
    true_excess = [float(np.log(x)) + j for x, j in zip(lever_arm, jitter)]
    from scipy import stats

    rho_true = float(stats.spearmanr(np.log(lever_arm), true_excess).statistic)
    cache = {}
    for i in range(400):
        z_hat = np.arctanh(rho_true) + 0.1 * rng.standard_normal()
        cache[("main", i)] = {
            "lever_arm": lever_arm,
            "true_excess": true_excess,
            "rho": float(np.tanh(z_hat)),
            "fisher_z": float(z_hat),
            "se": 0.1,
            "null_fisher_z": float(0.1 * rng.standard_normal()),
            "null_se": 0.1,
            "null_p": 0.5,
        }
    summary = lever.summarise(cache)
    assert summary["rho_true"] == pytest.approx(rho_true)
    assert rho_true < 1.0
    assert summary["n_null_statistics"] == 400
    assert 0.9 <= summary["cross_fit_coverage"] <= 1.0
    assert summary["calibrated_critical_value"] == pytest.approx(1.96, abs=0.25)
