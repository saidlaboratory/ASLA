"""Tests for the regime map, the ensemble verdict rule, and the DataDecide tie and target helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

regime = importlib.import_module("scripts.run_regime_map")
sensitivity = importlib.import_module("scripts.run_datadecide_sensitivity")
arms = importlib.import_module("scripts.run_lever_arm_arms")


def test_map_cells_reproduce_the_existing_arms() -> None:
    ctx = regime.context("c4")
    old = arms._context()
    base = ctx["parametric"].to_numpy()
    assert np.allclose(base, old["parametric"].to_numpy(), atol=1e-6)
    # lambda = 1 on the map is arm B3 x 1; lambda = 0 is arm B1.
    assert np.allclose(regime.truth_table(ctx, "map", 1.0).to_numpy() - base, old["arm_residual"]["B3_1"], atol=1e-6)
    assert np.allclose(regime.truth_table(ctx, "map", 0.0).to_numpy() - base, old["arm_residual"]["B1"], atol=1e-6)
    assert np.allclose(regime.truth_table(ctx, "ens", 2.0).to_numpy() - base, old["arm_residual"]["B3_2"], atol=1e-6)


def test_common_random_numbers_scale_the_noise_exactly() -> None:
    ctx = regime.context("c4")
    mu = regime.truth_table(ctx, "map", 1.0)
    one = regime.make_truth(ctx, mu, 1.0)
    half = regime.make_truth(ctx, mu, 0.5)
    final_one, _ = regime.simulate(one, np.random.default_rng(3))
    final_half, _ = regime.simulate(half, np.random.default_rng(3))
    assert np.allclose(final_half["bpb"] - final_half["mu"], 0.5 * (final_one["bpb"] - final_one["mu"]))


def test_noiseless_arm_a_projection_is_exact() -> None:
    ctx = regime.context("c4")
    rows = regime.noiseless_designs(ctx, regime.truth_table(ctx, "A", 0.0))
    assert len(rows) == 55
    assert all(r["projection_scored"] > 0 and r["single_scored"] > 0 for r in rows)


def test_crossings_interpolate_sign_changes() -> None:
    assert regime.crossings([0, 1, 2], [-1.0, 1.0, 3.0]) == [0.5]
    assert regime.crossings([0, 1, 2], [1.0, 2.0, 3.0]) == []
    assert regime.crossings([0, 1, 2], [-1.0, 1.0, -1.0]) == [0.5, 1.5]


def test_candidates_needed_falls_with_smaller_variance() -> None:
    assert regime.candidates_needed(1.0, 50.0) < regime.candidates_needed(4.0, 200.0)
    assert regime.candidates_needed(0.0, 0.0) == 4


def _weights(top: str) -> dict[str, dict[str, float]]:
    return {f"r{i}": {f: (1.0 if f == top else 0.0) for f in regime.FAMILIES} for i in range(5)}


def test_ensemble_verdict_separates_switching_from_smooth() -> None:
    switch = regime.ensemble_verdict(
        [1.0, 1.0, 9.0, 1.0],
        [0.1] * 4,
        [_weights("power_law"), _weights("power_law"), _weights("saturating"), _weights("power_law")],
    )
    assert switch["verdict"] == "family_switching"
    smooth = regime.ensemble_verdict([1.0, 2.0, 3.0, 2.5], [0.1] * 4, [_weights("power_law")] * 4)
    assert smooth["verdict"] == "smooth_peak"
    jump_same_family = regime.ensemble_verdict([1.0, 1.0, 9.0, 1.0], [0.1] * 4, [_weights("power_law")] * 4)
    assert jump_same_family["verdict"] == "indeterminate"


def test_grid_extends_by_doubling_until_bracketed() -> None:
    s, lam = regime.grids({"x_noise": 1.0, "y_recipe_specific": 1.0}, {"x_noise": 4.9, "y_recipe_specific": 1.5})
    assert s == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert lam == list(regime.BASE_GRID)


def test_tie_blocks_and_reorder() -> None:
    values = [0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4]
    blocks = sensitivity.tie_blocks(values)
    assert blocks == [(1, 3), (4, 7)]
    names = ["a", "b", "c", "d", "e", "f", "g"]
    assert sensitivity.reorder(names, blocks, [(1, 0), (2, 1, 0)]) == ["a", "c", "b", "d", "g", "f", "e"]
    assert sensitivity.tie_blocks([1.0, 2.0]) == []


def test_stable_target_orders_ties_by_input_order() -> None:
    frame = pd.DataFrame(
        {
            "task": ["t"] * 3,
            "size": ["1B"] * 3,
            "model": ["m1", "m2", "m3"],
            "group": ["x", "y", "z"],
            "step": [10, 10, 10],
            "mix": ["x", "y", "z"],
            "acc": [0.5, 0.4, 0.5],
        }
    )
    import sys
    import types

    stub = types.ModuleType("utils.dataloader")
    stub.get_slice = lambda df, task=None: df[df["task"] == task].reset_index(drop=True)  # type: ignore[attr-defined]
    saved = sys.modules.get("utils.dataloader")
    sys.modules["utils.dataloader"] = stub
    try:
        out = sensitivity.stable_target(frame, "1B", "t", "acc")
    finally:
        if saved is None:
            del sys.modules["utils.dataloader"]
        else:
            sys.modules["utils.dataloader"] = saved
    assert list(out["group"]) == ["y", "x", "z"]
