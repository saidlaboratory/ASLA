"""Tests for the tolerance-aware sign-change count and the degenerate-fit rule."""

from __future__ import annotations

import importlib

import numpy as np

st = importlib.import_module("scripts.run_sign_ties")


def test_tolerance_drops_small_gaps_and_counts_across_them() -> None:
    gaps = np.array([
        [0.01, -0.01, 0.01, -0.01],  # three clear changes
        [0.01, 1e-12, -0.01, -0.02],  # one change across a numerical tie
        [0.01, -0.0005, 0.01, 0.02],  # the dip is below tau: no change
        [1e-12, -1e-12, 1e-12, -1e-12],  # all ties: no change
    ])  # fmt: skip
    assert st.changes_with_tolerance(gaps, 1e-9).tolist() == [3, 1, 2, 0]
    assert st.changes_with_tolerance(gaps, 0.001).tolist() == [3, 1, 0, 0]
    assert st.changes_with_tolerance(gaps[None].repeat(2, 0), 0.001).shape == (2, 4)


def test_linear_gap_in_common_log_x_changes_sign_at_most_once() -> None:
    rng = np.random.default_rng(0)
    lx = np.log([1e18, 3e18, 1e19, 3e19])
    slopes, intercepts = rng.normal(-0.05, 0.02, 30), rng.normal(0.0, 0.05, 30)
    values = np.exp(intercepts[:, None] + slopes[:, None] * lx[None])
    assert st.changes_with_tolerance(st.pair_gaps(values), 1e-9).max() <= 1


def test_pair_gaps_order_and_sign() -> None:
    v = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    g = st.pair_gaps(v)
    assert g.shape == (3, 2) and np.allclose(g[0], [np.log(0.5), np.log(2.0)])


def test_degenerate_rule() -> None:
    assert st.degenerate(None, 1.0) == "no fit returned"
    assert st.degenerate({"slope": -0.2, "asymptote": 0.999}, 1.0) == "floor at its bound"
    assert st.degenerate({"slope": -2.0, "asymptote": 0.5}, 1.0) == "exponent at its upper bound"
    assert st.degenerate({"slope": -1e-6, "asymptote": 0.5}, 1.0) == "exponent at its lower bound"
    assert st.degenerate({"slope": -0.2, "asymptote": 0.8}, 1.0) is None
