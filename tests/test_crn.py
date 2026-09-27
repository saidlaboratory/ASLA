"""Tests for the CRN estimators: seed-index intraclass correlation, matched-gap variances, rho_D."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

cm = importlib.import_module("scripts.run_crn_compute_multipliers")
pp = importlib.import_module("scripts.run_crn_polypythias")


def _table(seed_effect: float, noise: float, n_recipes: int = 40, n_seeds: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recipe = rng.normal(0, 1, n_recipes)[:, None]
    shared = rng.normal(0, seed_effect, n_seeds)[None, :]
    return pd.DataFrame(recipe + shared + rng.normal(0, noise, (n_recipes, n_seeds)))


def test_icc_recovers_the_shared_seed_share() -> None:
    table = _table(seed_effect=1.0, noise=1.0)  # true r = 0.5
    out = cm.seed_icc(table)
    assert abs(out["r"] - 0.5) < 0.15
    assert out["r_ci"][0] < 0.5 < out["r_ci"][1]
    assert abs(out["reduction_factor"] - 1 / (1 - out["r"])) < 1e-12


def test_icc_is_near_zero_without_a_shared_seed_effect() -> None:
    out = cm.seed_icc(_table(seed_effect=0.0, noise=1.0))
    assert abs(out["r"]) < 0.1


def test_matched_gaps_vary_less_when_seeds_are_shared() -> None:
    shared = cm.pair_variances(_table(seed_effect=2.0, noise=0.5, n_recipes=6, n_seeds=8))
    independent = cm.pair_variances(_table(seed_effect=0.0, noise=0.5, n_recipes=6, n_seeds=8, seed=1))
    assert shared["pooled_matched_over_mismatched"] < 0.5
    assert 0.5 < independent["pooled_matched_over_mismatched"] < 1.6


def test_rho_d_and_its_interval() -> None:
    out = pp.rho_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert np.isclose(out["rho_d"], 0.5)
    assert out["rho_d_ci"][0] < 0.1 and out["rho_d_ci"][1] > 0.9  # F(2, 2) with three runs per arm
    assert pp.rho_d([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["rho_d"] == 0.0


def test_gate_rule() -> None:
    gate = importlib.import_module("scripts.run_crn_gate")
    assert gate.decide(0.3, 0.05) == "recommend"
    assert gate.decide(-0.03, 0.08) == "do not recommend"
    assert gate.decide(None, 0.25) == "recommend"
    assert gate.decide(0.1, 0.4) == "not decided by the rule"
