"""The design-power simulator: truths follow DataDecide's curves, and the summary's framings behave."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(
    not (REPO / "data" / "datadecide_runs.parquet").exists(), reason="raw DataDecide table not harvested"
)


@needs_data
def test_truth_draws_decrease_with_compute_and_vary_by_recipe():
    from asla.analysis.design_power import draw_truth

    x = np.linspace(16.0, 20.5, 6)
    truth = draw_truth(8, x, np.random.default_rng(0))
    assert len(truth) == 8
    assert all(np.all(np.diff(v) < 0) for v in truth.values())
    assert np.std([v[-1] for v in truth.values()]) > 0


def test_summary_counts_worlds_without_a_test_as_failures():
    from asla.analysis.design_power import summarise

    worlds = [{"estimate": 10.0, "se": 1.0, "true": 10.0}] * 50 + [{"estimate": None}] * 50
    out = summarise(worlds, betas=(11.7,))
    assert out["worlds_with_test"] == 50
    assert np.isclose(out["fixed_candidate"]["power_at_11.7"], 0.5)
    assert out["fixed_candidate"]["false_positive_rate_positive_at_0"] == 0.0
