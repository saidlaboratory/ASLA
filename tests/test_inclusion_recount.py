"""Tests for the inclusion-recount helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

ir = importlib.import_module("scripts.run_inclusion_recount")
RUNGS = ("130m", "300m", "520m", "1_2b")


def _table(batches: dict[str, list[int]]) -> dict:
    base = {k: "x" for k in ir.PROTOCOL}
    return {s: {r: dict(base, batch=b[i]) for i, r in enumerate(RUNGS)} for s, b in batches.items()}


def test_change_vector_matches_only_identical_per_size_changes() -> None:
    t = _table({"a": [128, 128, 256, 256], "b": [64, 64, 128, 128], "c": [128, 128, 128, 256]})
    assert ir.change_vector(t, "a", "300m", "520m") == ir.change_vector(t, "b", "300m", "520m")  # both x2
    assert ir.change_vector(t, "a", "300m", "520m") != ir.change_vector(t, "c", "300m", "520m")
    assert ir.change_vector(t, "a", "520m", "1_2b") != ir.change_vector(t, "c", "520m", "1_2b")


def test_step_sensitivity_is_zero_on_an_exact_power_law() -> None:
    x = {"130m": 2e18, "300m": 1e19, "520m": 5e19, "1_2b": 2e20}
    cells = {("s", r): {"x": v, "y": 3.0 * v**-0.05} for r, v in x.items()}
    table = _table({"s": [128, 128, 256, 256]})
    out = ir.sensitivity(cells, table, ["s"], {"s": 0.8}, {r: 0.002 for r in RUNGS})["by_submission"]["s"]
    assert out["batch_change_rung"] == "520m"
    assert out["B_minus_B0"] == pytest.approx(0.0, abs=1e-12) and out["C_minus_B0"] == pytest.approx(0.0, abs=1e-12)
    assert out["B0_loglog"] == pytest.approx(3.0 * 1e22**-0.05)
    assert not out["cannot_be_separated_from_step"]


def test_step_at_last_rung_is_not_identified_without_two_post_points() -> None:
    x = {"130m": 2e18, "300m": 1e19, "520m": 5e19, "1_2b": 2e20}
    cells = {("s", r): {"x": v, "y": 3.0 * v**-0.05 * (1.01 if r == "1_2b" else 1.0)} for r, v in x.items()}
    out = ir.sensitivity(cells, _table({"s": [128, 128, 128, 256]}), ["s"], {"s": 0.8}, {r: 0.002 for r in RUNGS})
    row = out["by_submission"]["s"]
    assert row["batch_change_rung"] == "1_2b" and row["C_post_only"].startswith("not identified")
    assert row["B_step"] == pytest.approx(3.0 * 1e22**-0.05 * 1.01)


def test_jackknife_needs_enough_candidates() -> None:
    assert ir.jackknife({("a", "b"): 100.0}) is None
    se = ir.jackknife({("a", "b"): 100.0, ("a", "c"): 0.0, ("b", "c"): 0.0, ("c", "d"): 0.0})
    assert se is not None and np.isfinite(se)
