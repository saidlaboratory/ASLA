"""Tests for the misspecification-pattern helpers and the DataDecide paper check."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

pattern = importlib.import_module("scripts.run_pattern")
posthoc = importlib.import_module("scripts.run_pattern_posthoc")
paper = importlib.import_module("scripts.run_datadecide_paper_check")
tw = importlib.import_module("scripts.transfer_worlds")


def test_crossover_rate_counts_reversed_pairs() -> None:
    a = pd.Series({"x": 1.0, "y": 2.0, "z": 3.0})
    b = pd.Series({"x": 1.0, "y": 3.0, "z": 2.0})
    assert pattern.crossover_rate(a, a) == 0.0
    assert np.isclose(pattern.crossover_rate(a, b), 1 / 3)


def test_in_sample_residual_vanishes_on_an_exact_power_law() -> None:
    compute = pd.Series({"a": 1e15, "b": 1e16, "c": 1e17, "d": 1e18})
    rows = {r: [e + 2.0 * float(c) ** -0.2 for c in compute] for r, e in (("r1", 0.5), ("r2", 0.7), ("r3", 0.9))}
    table = pd.DataFrame(rows, index=list(compute.index)).T
    residual = pattern.in_sample_residual(table, compute, ("c", "d"))
    assert np.allclose(residual.to_numpy(), 0.0, atol=1e-4)


def test_permuted_truth_holds_amplitude_and_spans_alignment() -> None:
    c4 = tw.context("c4")
    spec = c4["specific_table"].to_numpy()
    for c in (-1.0, 0.0, 1.0):
        truth, info = pattern.permuted_truth(c, ("300M", "530M"))
        permuted = (truth - c4["parametric"]).to_numpy() - c4["common"]
        assert np.allclose(np.sort(permuted, axis=0), np.sort(spec, axis=0))  # same rows, reassigned
        assert np.isclose(info["specific_rms"], info["measured_specific_rms"])
    assert pattern.permuted_truth(1.0, ("300M", "530M"))[1]["achieved_spearman_with_y0"] > 0.99
    assert pattern.permuted_truth(-1.0, ("300M", "530M"))[1]["achieved_spearman_with_y0"] < -0.99


def test_posthoc_quantities() -> None:
    q = posthoc.quantities(np.array([1.0, -1.0, 0.0]), np.array([1.0, -1.0, 0.0]), 2.0)
    assert np.isclose(q["persistence"], 1.0)
    assert q["transient_size_in_gaps"] == 0.0


def test_quote_returns_context_or_none() -> None:
    text = "alpha beta gamma delta"
    assert "gamma" in paper.quote(text, "gamma", width=3)
    assert paper.quote(text, "epsilon") is None
