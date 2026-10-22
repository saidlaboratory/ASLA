"""Tests for the break diagnosis and the certified-leaderboard helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

bd = importlib.import_module("scripts.run_break_diagnosis")
cl = importlib.import_module("scripts.run_certified_leaderboard")


def test_settings_reads_nested_optimizer_fields_and_tokens_per_param() -> None:
    info = {"total_tokens": 2000, "model_size": 100, "run_completion_timestamp": "t",
            "train_config": {"num_train_steps": 10, "train_batch_size": 128, "train_seq_len": 4096, "data_seed": None,
                             "optimizer_config": {"lr": 0.01, "warmup": 0, "lr_schedule": "linear", "decay": 0.8,
                                                  "weight_decay": 0.1, "min_lr_ratio": 0}},
            "resources": {"tpu_type": "v5p-32"}, "num_devices": 32}  # fmt: skip
    s = bd.settings(info)
    assert s["tokens_per_param"] == 20 and s["lr"] == 0.01 and s["warmup"] == 0 and s["decay"] == 0.8
    assert s["batch"] == 128 and s["hardware"] == "v5p-32" and s["schedule"] == "linear"


def test_change_and_flags() -> None:
    assert bd.change(128, 256) == 2.0 and bd.change("a", "a") == "same" and bd.change("a", "b") == "a -> b"
    table = {s: {"300m": {f: 1 for f in bd.FIELDS}, "520m": {f: 1 for f in bd.FIELDS}} for s in "ab"}
    table["a"]["520m"]["batch"] = 2
    f = bd.flags(table, {"all": ["a", "b"]})["all"]
    assert f["batch"]["flagged"] and not f["lr"]["flagged"]


def test_submission_step_is_detected_only_when_present() -> None:
    x = {"130m": 1e18, "300m": 1e19, "520m": 5e19, "1_2b": 2e20}
    sig = {r: 0.002 for r in x}

    def cells(step: float) -> dict:
        return {
            ("s", r): {"x": v, "y": 3.0 * v**-0.05 * (np.exp(step) if r in ("520m", "1_2b") else 1.0)} for r, v in x.items()
        }

    flat = bd.shape(cells(0.0), ["s"], [], sig)["submission_step"]["s"]
    stepped = bd.shape(cells(0.03), ["s"], [], sig)["submission_step"]["s"]
    assert abs(flat["z"]) < 1e-6 and stepped["d"] == pytest.approx(0.03, abs=1e-9) and stepped["z"] > 1.96


def test_fo_reorder_counts_determined_flips() -> None:
    optima = {"a": {"300m": {"optimum": 3.00}, "520m": {"optimum": 2.80}},
              "b": {"300m": {"optimum": 3.10}, "520m": {"optimum": 2.70}}}  # fmt: skip
    r = bd.fo_reorder(optima, "300m", "520m", {"300m": 1e-4, "520m": 1e-4})
    assert r["determined"] == 1 and r["reordered"] == 1 and r["genuine"] == 1 and r["flipped"] == [["a", "b"]]


def test_certify_needs_noise_and_form_agreement() -> None:
    primary = np.array([1.0, 1.1, 1.2])
    rng = np.random.default_rng(0)
    draws = primary[None] + 0.01 * rng.standard_normal((400, 3))
    assert cl.certify(draws, primary, primary) == [(0, 1), (0, 2), (1, 2)]
    assert (0, 1) not in cl.certify(draws, primary, np.array([1.2, 1.1, 1.3]))  # forms disagree on (0, 1)
    assert cl.certify(primary[None] + 0.2 * rng.standard_normal((400, 3)), primary, primary) == []


def test_realized_rate_and_sigma_star_interpolation() -> None:
    assert cl.realized([(0, 1), (1, 2)], np.array([1.0, 0.9, 1.5])) == {"certified": 2, "wrong": 1, "rate": 0.5}
    rows = {"0.001": {"rate": 0.2}, "0.002": {"rate": 0.1}, "0.004": {"rate": 0.0}}
    s = cl.sigma_star(rows)
    assert s["bracket"] == [0.002, 0.004] and s["sigma_star"] == pytest.approx(np.sqrt(0.002 * 0.004))
    assert cl.sigma_star({"0.001": {"rate": 0.5}})["sigma_star"] is None


def test_loglog_forecast_is_exact_on_a_power_law() -> None:
    x = np.array([[1e18, 1e19, 1e20]])
    assert cl.loglog_forecast(x, 2.0 * x**-0.1, np.array([1e22]))[0] == pytest.approx(2.0 * 1e22**-0.1)
