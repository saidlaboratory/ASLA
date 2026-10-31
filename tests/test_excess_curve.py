"""The excess curve: the saturating fit recovers planted parameters, and coincidence holds for identical curves."""

from __future__ import annotations

import numpy as np

from asla.analysis import excess_curve as ec
from asla.analysis.confound import WindowDesign


def test_fit_recovers_planted_initial_slope_and_scale():
    lever = np.linspace(0.2, 4.0, 40)
    y = ec.saturating(lever, 12.0, 1.5)
    s, tau = ec.fit(lever, y)
    assert np.isclose(s, 12.0, rtol=1e-3) and np.isclose(tau, 1.5, rtol=1e-3)


def test_linear_data_puts_tau_at_its_bound():
    lever = np.linspace(0.2, 4.0, 40)
    s, tau = ec.fit(lever, 8.0 * lever)
    assert tau > 19.0 and 7.0 < s < 9.0


def _designs(targets, levers, recipes):
    out = []
    for t in targets:
        for L in levers:
            kern = {(a, b): 0.0 for i, a in enumerate(recipes) for b in recipes[i + 1:]}
            out.append(WindowDesign(start=0, top=0, target=t, lever=L, log_top=0.0, rungs=3, kernel=kern))
    return out


def test_identical_curves_do_not_reject_and_gaps_are_zero():
    recipes = [f"r{i}" for i in range(8)]
    ds = _designs([1, 2], [0.3, 0.8, 1.2, 1.8, 2.5], recipes)
    rng = np.random.default_rng(0)
    for d in ds:  # the same lever-arm curve at both targets, plus recipe-level noise shared across targets
        for p in d.kernel:
            d.kernel[p] = 6.0 * d.lever + rng.normal(0, 0.5) + 0.0
    jk = ec.jackknife(ds, [1, 2])
    res = ec.coincidence(jk, [1, 2])
    assert res["binned"]["df"] >= 1
    assert res["binned_max_gap"] < 1.0
