"""Tests for the leave-one-out noise bracket and the leaderboard-resolution helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

nb = importlib.import_module("scripts.run_noise_bracket")
lr = importlib.import_module("scripts.run_leaderboard_resolution")


def test_loo_is_exact_on_a_polynomial_and_needs_enough_points() -> None:
    x = np.linspace(0, 1, 6)
    y = 3.0 + 0.2 * x - 0.1 * x**2
    assert np.allclose(nb.loo(x, y, "quadratic"), 0.0, atol=1e-10)
    assert nb.loo(x[:3], y[:3], "quadratic") is None  # two training points cannot identify three parameters
    assert nb.loo(x[:4], y[:4], "quadratic") is not None
    assert nb.loo(np.array([0, 0, 1, 2, 3, 4.0]), y, "spline") is None  # tied x


def test_loo_recovers_noise_scale_for_a_correct_model() -> None:
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(400):
        x = np.linspace(0, 1, 12)
        y = (3.0 + 0.2 * x) * (1 + 0.002 * rng.standard_normal(12))
        errors += nb.loo(x, y, "linear")
    rms = float(np.sqrt(np.mean(np.square(errors))))
    assert 0.002 < rms < 0.0025  # leave-one-out inflates a correct model's noise only slightly (leverage)


def test_residual_df_and_coordinates() -> None:
    x = np.linspace(0, 1, 6)
    ss, df = nb.residual(x, 1 + x, "linear")
    assert df == 4 and ss == pytest.approx(0.0, abs=1e-20)
    assert nb.residual(x, 1 + x, "spline") is None
    assert np.allclose(nb.coordinate("beta2", np.array([0.9, 0.99])), np.log([0.1, 0.01]))
    assert np.allclose(nb.coordinate("learning_rate", np.array([1e-3, 2e-3])), np.log([1e-3, 2e-3]))
    assert np.allclose(nb.coordinate("weight_decay", np.array([0.0, 0.1])), [0.0, 0.1])


def test_bound_uses_best_flexibility_and_corrects_ratios_below_one() -> None:
    summary = {"a": {"loo_rms": 0.004, "loo_n": 30}, "b": {"loo_rms": 0.002, "loo_n": 30}}
    b = nb.bound_for(summary, {"a": 3.0, "b": 4.0})
    assert b["flexibility"] == "b" and b["upper_95"] > 0.002 and b["calibrated_estimate"] == pytest.approx(0.0005)
    low = nb.bound_for(summary, {"b": 0.5})
    assert low["upper_95"] == pytest.approx(2 * b["upper_95"])


def test_marin_reorder_and_fisher() -> None:
    cells = {("a", "f"): {"y": 1.0}, ("b", "f"): {"y": 1.1}, ("a", "t"): {"y": 0.95}, ("b", "t"): {"y": 0.90}}
    r = lr.marin_reorder(cells, [("a", "b")], "f", "t", 1e-4)
    assert r == {"pairs": 1, "determined": 1, "reordered": 1, "rate": 1.0}
    assert lr.marin_reorder(cells, [("a", "b")], "f", "t", 1.0)["determined"] == 0
    assert lr.fisher({"determined": 0, "reordered": 0}, {"determined": 10, "reordered": 0}) is None
    assert lr.fisher({"determined": 50, "reordered": 20}, {"determined": 300, "reordered": 1}) < 1e-10
