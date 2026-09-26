"""Tests for the lever-arm dose arms, the DataDecide code check helpers, and Monte Carlo precision."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest

arms = importlib.import_module("scripts.run_lever_arm_arms")
precision = importlib.import_module("scripts.run_mc_precision")
code_check = importlib.import_module("scripts.run_datadecide_code_check")


def test_decimals_supported_matches_the_last_digit_rule() -> None:
    assert precision.decimals_supported(0.004) == 2  # 0.03 is safe, 0.030 is not
    assert precision.decimals_supported(0.0004) == 3
    assert precision.decimals_supported(0.02) == 1


def test_arm_residuals_decompose_the_full_residual() -> None:
    ctx = arms._context()
    residual = ctx["arm_residual"]
    # Common mode plus recipe-specific equals the full residual (B1 + B2 = B3 at amplitude 1).
    assert np.allclose(residual["B1"] + residual["B2"], residual["B3_1"])
    assert np.allclose(residual["A"], 0.0)
    assert np.allclose(residual["B3_2"], 2 * residual["B3_1"])
    # B1 has constant loadings: every row is the same curve.
    assert np.allclose(residual["B1"], residual["B1"][0])


def test_arm_truth_at_full_amplitude_is_the_shrunk_real_truth() -> None:
    ctx = arms._context()
    truth = arms.arm_truth("B3_1")
    merged = truth.final.merge(ctx["base"].final, on=["intervention", "scale_label", "seed", "step"], suffixes=("", "_base"))
    assert np.allclose(merged["mu"], merged["mu_base"])


def test_summarise_arm_recovers_a_planted_monotone_effect() -> None:
    lever = list(np.geomspace(2, 200, 10))
    rows = [
        {
            "lever_arm": lever,
            "true_excess": [float(np.log(x)) for x in lever],
            "rho_hat": 0.9,
            "ensemble_true_excess": 5.0,
            "projection_true_excess": 1.0,
            "zeta1": 1.0,
            "zeta2": 30.0,
        }
        for _ in range(5)
    ]
    out = arms.summarise_arm(rows, np.random.default_rng(0))
    assert out["true_rho"] == pytest.approx(1.0)
    assert out["true_slope_pp_per_log"] == pytest.approx(1.0)
    assert out["candidates_for_2pp_power_0_8"] is not None


def test_their_preprocessing_keeps_default_seed_and_named_sizes() -> None:
    macro = pd.DataFrame(
        {
            "params": ["1B", "1B", "4M"],
            "data": ["Dolma1.7", "Dolma1.7", "Dolma1.7"],
            "task": ["arc_easy"] * 3,
            "step": [10, 10, 10],
            "seed": ["default", "large aux 2", "default"],
            "chinchilla": ["5xC"] * 3,
            "metrics": [json.dumps({"primary_metric": 0.5})] * 3,
        }
    )
    out = code_check.their_preprocessing(macro, {"dolma17": "Dolma1.7"})
    assert len(out) == 1 and out.iloc[0]["group"] == "dolma17" and out.iloc[0]["size"] == "1B"


def test_floor_is_never_reported_tighter_than_it_is() -> None:
    assert precision._round_up(1 / 1350) == "0.0008"  # 0.00074 rounds up, never down to 0.0007
    assert precision._round_up(0.02) == "0.02"
    assert float(precision._round_up(1 / 1080)) >= 1 / 1080
