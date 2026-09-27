"""Tests for the transfer machinery: hybrids, coordinate transforms, physicality, ladders."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

tw = importlib.import_module("scripts.transfer_worlds")


def test_median_gap_is_the_median_absolute_pairwise_difference() -> None:
    assert tw.median_gap(pd.Series([0.0, 1.0, 3.0])) == 2.0


def test_hybrid_without_swaps_is_the_base_metric() -> None:
    c4 = tw.context("c4")
    truth, sigma = tw.hybrid("c4", {})
    assert np.allclose(truth.to_numpy(), c4["mu"].to_numpy())
    assert np.allclose(sigma.to_numpy(), c4["sigma_table"].to_numpy())


def test_all_swapped_hybrid_is_the_other_metric_in_base_units() -> None:
    c4, olmes = tw.context("c4"), tw.context("olmes")
    every = dict.fromkeys(("i", "ii", "iii", "iv", "v"), "olmes")
    truth, sigma = tw.hybrid("c4", every)
    expected = olmes["mu"].loc[c4["recipes"], c4["scales"]] * c4["g"] / olmes["g"]
    assert np.allclose(truth.to_numpy(), expected.to_numpy())
    pi_o, a_o = tw.noise_profile(olmes)
    pi_c, a_c = tw.noise_profile(c4)
    assert np.allclose(
        (sigma / c4["g"]).pow(2).mean(axis=0).pow(0.5).to_numpy() / a_o, pi_o[c4["scales"]].to_numpy(), rtol=1e-6
    )


def test_residual_split_recombines_to_the_full_swap() -> None:
    split = tw.hybrid("c4", {"iv_common": "olmes", "iv_specific": "olmes"})[0]
    whole = tw.hybrid("c4", {"iv": "olmes"})[0]
    assert np.allclose(split.to_numpy(), whole.to_numpy())


def test_transform_is_identity_at_own_coordinates_and_hits_the_target() -> None:
    c4 = tw.context("c4")
    design = tw.PRIMARY
    mu, sigma, applied = tw.transform_to(
        "c4", tw.own_coordinates("c4", design), design, c4["scales"], frozenset({"ii", "iii"})
    )
    assert np.allclose(mu.to_numpy(), c4["mu"].to_numpy())
    assert np.allclose(sigma.to_numpy(), c4["sigma_table"].to_numpy())
    target = tw.own_coordinates("olmes", design)
    _, _, moved = tw.transform_to("c4", target, design, c4["scales"], frozenset({"iii"}))
    for key in ("q1", "q2", "q3", "q5_level", "q6_decay"):
        assert np.isclose(moved["achieved"][key], target[key], rtol=1e-6), key


def test_physical_flags_a_rise_beyond_one_seed_sd() -> None:
    mu = pd.DataFrame([[3.0, 2.0, 2.5]], columns=["a", "b", "c"])
    small = pd.DataFrame([[1.0, 0.1, 0.1]], columns=["a", "b", "c"])
    large = pd.DataFrame([[1.0, 1.0, 1.0]], columns=["a", "b", "c"])
    assert not tw.physical(mu, small, "c4")["tolerant_monotone"]
    assert tw.physical(mu, large, "c4")["tolerant_monotone"]
    assert not tw.physical(mu, large, "c4")["strict_monotone"]
    assert not tw.physical(mu, large, "olmes")["in_range"]


def test_ladder_designs_pick_the_extreme_lever_arms() -> None:
    compute = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 100.0, "300M": 150.0, "530M": 400.0})
    designs = tw.ladder_designs(list(compute.index), compute)
    assert designs["long"] == ("c", "530M")
    assert designs["short"] == ("d", "300M")  # L = 1.5, the smallest
    assert designs["primary"] == ("300M", "530M")
