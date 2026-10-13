"""Tests for the aligned-compute Marin helpers and the leaderboard fix patch."""

from __future__ import annotations

import importlib
import shutil
import subprocess

import numpy as np
import pytest

al = importlib.import_module("scripts.run_marin_aligned")
rig = importlib.import_module("scripts.run_marin_rigour")


def _cells(xs: dict[str, list[float]], f) -> dict:
    return {(s, b): {"x": x, "y": float(f(s, x)), "train_flops": 0.0} for s, v in xs.items() for b, x in zip(rig.BUDGETS, v)}


def test_alignment_moves_along_the_fit_and_keeps_residuals() -> None:
    xs = {"a": [1e18, 4e18, 2e19, 8e19], "b": [1.2e18, 4e18, 2.4e19, 9.6e19]}

    def law(s: str, x: float) -> float:
        return (3.0 if s == "a" else 3.1) * x**-0.05

    cells = _cells(xs, law)
    cells[("a", "520m")]["y"] *= 1.01  # a residual the alignment must keep
    target = al.common_x(["a", "b"], cells)
    for fit in (al.loglog_fit, al.three_fit):
        aligned = al.aligned_cells(["a", "b"], cells, target, fit, rig.BUDGETS)
        assert aligned[("b", "1_2b")]["x"] == target["1_2b"]
        assert aligned[("b", "300m")]["y"] == pytest.approx(law("b", target["300m"]), rel=1e-6)
    aligned = al.aligned_cells(["a", "b"], cells, target, al.loglog_fit, rig.BUDGETS)
    f = al.loglog_fit(np.array(xs["a"]), np.array([cells[("a", b)]["y"] for b in rig.BUDGETS]))
    assert aligned[("a", "520m")]["y"] / f(target["520m"]) == pytest.approx(cells[("a", "520m")]["y"] / f(2e19))


def test_common_x_is_clipped_into_every_observed_range() -> None:
    xs = {"a": [1e18, 4e18, 2e19, 8e19], "b": [1.2e18, 4e18, 2.4e19, 9.6e19], "c": [1.3e18, 4e18, 2.5e19, 1e20]}
    cells = _cells(xs, lambda s, x: x**-0.05)
    clipped, raw = al.common_x(list(xs), cells), al.common_x(list(xs), cells, clip=False)
    assert raw["1_2b"] == 9.6e19 and clipped["1_2b"] == 8e19
    assert raw["130m"] == 1.2e18 and clipped["130m"] == 1.3e18
    assert clipped["300m"] == raw["300m"]


def test_masked_pairs_are_never_determined() -> None:
    values = {"a": 1.0, "b": 1.1, "c": 1.2}
    se = al.masked_se(values, 1e-4, {("a", "b")})
    assert np.isfinite(se[("a", "b")]) and se[("a", "c")] == np.inf and se[("b", "c")] == np.inf
    cells = {(s, b): {"x": 1.0, "y": v + (0.15 if (s, b) == ("a", "1_2b") else 0.0)} for s, v in values.items()
             for b in ("520m", "1_2b")}  # fmt: skip
    r = al.reorder_rates(list(values), cells, {("a", "b"), ("a", "c")}, {"cv": 1e-4})["cv"]
    assert r["determined"] == 2 and r["reordered"] == 1


def test_class_survival_needs_both_size_and_significance() -> None:
    dd = {"determined": 296, "reordered": 3, "rate": 3 / 296}
    assert al.fisher_vs_dd({"determined": 190, "reordered": 41, "rate": 41 / 190}, dd)["survives"]
    small = al.fisher_vs_dd({"determined": 41, "reordered": 2, "rate": 2 / 41}, dd)
    assert not small["survives"] and small["difference_pp"] < 5
    assert not al.fisher_vs_dd({"determined": 0, "reordered": 0, "rate": None}, dd)["survives"]


@pytest.mark.skipif(not (rig.NODE_DIR / "regression.ts").exists() or shutil.which("patch") is None,
                    reason="their regression.ts is not cached")  # fmt: skip
def test_fix_patch_applies_to_their_regression_ts(tmp_path) -> None:
    target = tmp_path / "regression.ts"
    shutil.copy(rig.NODE_DIR / "regression.ts", target)
    subprocess.run(["patch", "-s", str(target), str(al.PATCH)], check=True)
    text = target.read_text()
    assert "errorTolerance: 1e-15" in text and "p.x / X_SCALE" in text and "Math.pow(X_SCALE, slope)" in text
