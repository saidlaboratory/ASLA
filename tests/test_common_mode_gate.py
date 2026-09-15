"""Tests for the Task 0 common-mode gate."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_common_mode_gate import (  # noqa: E402
    cancellation_ratio,
    pairwise_residual_correlation,
    shared_component_fraction,
    shared_component_shape,
)


def _frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=[f"r{i}" for i in range(matrix.shape[0])], columns=[1.0, 2.0, 4.0, 8.0, 16.0])


def test_cancellation_ratio_is_one_for_independent_residuals():
    rng = np.random.default_rng(0)
    independent = _frame(rng.normal(0, 1, size=(30, 5)))
    ratio = cancellation_ratio(independent)["mean_ratio"]
    assert 0.85 < ratio < 1.15  # ~1 under independence


def test_cancellation_ratio_is_near_zero_for_a_pure_common_offset():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 1, size=(1, 5))
    residuals = _frame(np.repeat(shared, 30, axis=0) + rng.normal(0, 0.01, size=(30, 5)))
    assert cancellation_ratio(residuals)["mean_ratio"] < 0.05


def test_shared_component_shape_separates_offset_from_scaled_shape():
    rng = np.random.default_rng(2)
    shape = rng.normal(0, 1, size=(1, 5))
    # offset: every intervention carries the same shared vector
    offset = _frame(np.repeat(shape, 25, axis=0) + rng.normal(0, 0.02, size=(25, 5)))
    # scaled: the same shape but with per-intervention amplitudes spanning zero
    amplitudes = rng.normal(0, 1, size=(25, 1))
    scaled = _frame(amplitudes * shape + rng.normal(0, 0.02, size=(25, 5)))
    offset_shape = shared_component_shape(offset)
    scaled_shape = shared_component_shape(scaled)
    # both have a dominant first component ...
    assert offset_shape["pc1_variance_fraction"] > 0.9
    assert scaled_shape["pc1_variance_fraction"] > 0.9
    # ... but only the offset has near-constant, same-sign loadings
    assert offset_shape["loading_cv"] < 0.5 and offset_shape["loadings_same_sign"]
    assert scaled_shape["loading_cv"] >= 0.5
    assert "cancels" in offset_shape["interpretation"]
    assert "does not cancel" in scaled_shape["interpretation"]
    # and the decisive consequence: only the offset case cancels
    assert cancellation_ratio(offset)["mean_ratio"] < 0.1
    assert cancellation_ratio(scaled)["mean_ratio"] > 0.5


def test_a_large_pc1_alone_does_not_imply_cancellation():
    """The central trap: PC1 fraction is not sufficient evidence for the premise."""

    rng = np.random.default_rng(3)
    shape = rng.normal(0, 1, size=(1, 5))
    amplitudes = rng.normal(0, 1, size=(25, 1))
    scaled = _frame(amplitudes * shape + rng.normal(0, 0.02, size=(25, 5)))
    assert shared_component_fraction(scaled)["pc1_fraction"] > 0.9
    assert cancellation_ratio(scaled)["mean_ratio"] > 0.5  # high PC1, no cancellation


def test_residual_correlation_reflects_shared_structure():
    rng = np.random.default_rng(4)
    shared = rng.normal(0, 1, size=(1, 5))
    correlated = _frame(np.repeat(shared, 20, axis=0) + rng.normal(0, 0.05, size=(20, 5)))
    independent = _frame(rng.normal(0, 1, size=(20, 5)))
    assert pairwise_residual_correlation(correlated)["mean"] > 0.9
    assert abs(pairwise_residual_correlation(independent)["mean"]) < 0.3


def test_shared_component_fraction_handles_degenerate_input():
    zeros = _frame(np.zeros((5, 5)))
    out = shared_component_fraction(zeros)
    assert np.isnan(out["pc1_fraction"]) and np.isnan(out["one_way_fraction"])
