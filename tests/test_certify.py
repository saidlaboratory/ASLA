"""Tests for the Stage-2 certifying selector's helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

cf = importlib.import_module("scripts.run_certify")


def test_certify_needs_every_form_and_enough_margin() -> None:
    fc = {"a": np.array([1.0, 2.0, 2.01]), "b": np.array([1.0, 2.0, 2.01])}
    var = {"a": np.full(3, 0.01), "b": np.full(3, 0.01)}
    c = cf.certify(fc, var)
    assert c[0, 1] and c[0, 2] and not c[1, 2] and not c[1, 0]  # 1 vs 2 is within noise
    fc_dis = {"a": fc["a"], "b": np.array([2.5, 2.0, 2.01])}  # form b disagrees on (0, 1)
    assert not cf.certify(fc_dis, var)[0, 1]
    assert cf.possibly_best(c).tolist() == [True, False, False]


def test_kappa_recovers_a_known_misfit_and_ignores_common_mode() -> None:
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(400):
        L = np.full(25, 0.8)
        r = 0.01 * L * rng.standard_normal(25) + 0.05  # recipe-specific misfit 1% per decade, plus a common mode
        preds.append({"r": r, "u": np.zeros(25), "L": L})
    assert cf.calibrate_kappa(preds) == pytest.approx(0.01, rel=0.05)
    assert cf.calibrate_kappa([{"r": np.full(5, 0.3), "u": np.zeros(5), "L": np.ones(5)}]) == 0.0


def test_conformal_quantile_and_too_few_scores() -> None:
    preds = [{"r": np.linspace(0.01, 0.2, 40), "L": np.ones(40)}]
    q = cf.conformal_q(preds)
    assert q is not None and 0.18 < q <= 0.2
    assert cf.conformal_q([{"r": np.array([0.1]), "L": np.ones(1)}]) is None


def test_evaluate_charges_reversal_probability() -> None:
    rng = np.random.default_rng(1)
    m, se = np.array([1.0, 1.0, 5.0]), np.array([0.1, 0.1, 0.1])
    cert = np.zeros((3, 3), dtype=bool)
    cert[0, 2] = True  # clearly right
    cert[0, 1] = True  # a coin flip under the truth's uncertainty
    out = cf.evaluate(cert, m, se, rng)
    assert out["certified"] == 2 and out["fcr"] == pytest.approx(0.25, abs=1e-6)
    assert out["possibly_best_size"] == 1 and out["contains_observed_best"]
