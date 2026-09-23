"""Tests for the simulation-calibrated critical values and p-values."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

calibration = importlib.import_module("scripts.run_calibration")


def test_calibrated_p_is_monotone_and_floored() -> None:
    grid = list(np.quantile(np.abs(np.random.default_rng(0).standard_normal(5000)), np.linspace(0, 1, 2001)))
    p_small, p_large = calibration.calibrated_p(0.5, grid, 5000), calibration.calibrated_p(2.5, grid, 5000)
    assert p_small > p_large
    assert calibration.calibrated_p(1.96, grid, 5000) == pytest.approx(0.05, abs=0.01)
    assert calibration.calibrated_p(50.0, grid, 5000) == pytest.approx(1 / 5000)


def _rows(se_scale: float, n: int = 200, seed: int = 1) -> tuple[list[dict], dict]:
    """Worlds where estimate - truth ~ N(0, 1) and the reported SE is se_scale."""

    rng = np.random.default_rng(seed)
    names = ["a", "b"]
    rows = []
    for kind in ("fixed", "random", "random_jk"):
        for i in range(n):
            rows.append({"kind": kind, "index": i, "comparisons": {}})
    by = {(r["kind"], r["index"]): r for r in rows}
    for i in range(n):
        for name in names:
            deviation_f, deviation_r = rng.standard_normal(2)
            by[("fixed", i)]["comparisons"][name] = {"estimate": deviation_f, "boot_se": se_scale}
            by[("random", i)]["comparisons"][name] = {"estimate": deviation_r, "u_se": se_scale, "jk_se": se_scale}
            by[("random_jk", i)]["comparisons"][name] = {"estimate": deviation_r, "kjk_se": se_scale, "u_se": se_scale}
    summary = {"by_comparator": {n: {"theta_fixed": 0.0, "theta_random": 0.0} for n in names}}
    return rows, summary


def test_correct_standard_errors_give_a_critical_value_near_1_96() -> None:
    rows, summary = _rows(1.0, n=1500)
    out = calibration.calibrated_procedures(rows, summary)
    assert out["fixed_bootstrap"]["critical_value"] == pytest.approx(1.96, abs=0.12)
    assert all(0.92 <= v <= 0.98 for v in out["fixed_bootstrap"]["cross_fit_coverage"].values())


def test_understated_standard_errors_are_compensated_by_the_critical_value() -> None:
    rows, summary = _rows(0.5, n=1500)
    out = calibration.calibrated_procedures(rows, summary)
    assert out["random_u"]["critical_value"] == pytest.approx(2 * 1.96, abs=0.25)


def test_mismatched_regenerated_worlds_are_refused() -> None:
    rows, summary = _rows(1.0, n=10)
    for row in rows:
        if row["kind"] == "random_jk":
            row["comparisons"]["a"]["estimate"] += 1.0
    with pytest.raises(ValueError, match="do not reproduce"):
        calibration.calibrated_procedures(rows, summary)
