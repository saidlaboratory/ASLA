"""Tests for the candidate-level re-test of published comparisons and the correction curve."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from asla.analysis.target_scoring import u_statistic_variance
from asla.provenance import Source

published = importlib.import_module("scripts.run_published_comparisons")


def test_agreement_scores_ties_as_half_and_higher_as_better() -> None:
    target = pd.Series({"a": 3.0, "b": 2.0, "c": 2.0})
    values = pd.Series({"a": 1.0, "b": 0.5, "c": 0.9})
    agreement = published._agreement(values, target)
    assert agreement[("a", "b")] == 1.0
    assert agreement[("b", "c")] == 0.5  # tied target
    values_wrong = pd.Series({"a": 0.0, "b": 0.5, "c": 0.9})
    assert published._agreement(values_wrong, target)[("a", "b")] == 0.0


@pytest.mark.parametrize("n", [5, 25, 100])
@pytest.mark.parametrize("ratio", [0.0, 0.035, 0.2])
def test_ratio_curve_equals_candidate_over_pair_standard_error(n: int, ratio: float) -> None:
    """R(n) is the closed form of sqrt(candidate variance / pair-resampled variance)."""

    zeta2 = 1.0
    zeta1 = ratio * zeta2
    pair_variance = zeta2 / (n * (n - 1) / 2)
    expected = np.sqrt(u_statistic_variance(zeta1, zeta2, n) / pair_variance)
    assert published.ratio_curve(ratio)[str(n)] == pytest.approx(expected)


def test_curve_range_clips_negative_zeta1_estimates() -> None:
    out = published._curve_range([-0.02, 0.01, 0.05, 0.1])
    assert out["n_negative_zeta1_estimates"] == 1
    for curve in out["curve_by_percentile"].values():
        assert all(np.isfinite(v) and v >= 1.0 for v in curve.values())


def test_mapping_refuses_an_inexact_match() -> None:
    fits = pd.DataFrame(
        {
            "task": [published.MACRO],
            "metric": ["primary_metric"],
            "setup": [published.SL_VARIANTS[0]],
            "mix": ["x"],
            "stacked_y": [0.5],
        }
    )
    with pytest.raises(ValueError, match="no unique exact target match"):
        published._mapping(fits, pd.Series({"A": 0.51, "B": 0.49}))


def test_committed_result_reports_the_failed_gate_rather_than_hiding_it() -> None:
    """D1 did not reproduce the released decision_acc; the JSON must say so."""

    out = Source.load("external/published_comparisons.json")
    assert out.get("predictions.P1_gate.confirmed") is False
    assert "exploratory" in out.get("d1_gate_note")
    assert out.get("predictions.P4_signal_and_noise_bpb.status") == "not reconstructed"
