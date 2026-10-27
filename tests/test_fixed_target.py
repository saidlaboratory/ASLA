"""Fixed-target slices: counts as pre-registered, and a planted slope is recovered."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asla.analysis.confound import window_designs
from asla.analysis.fixed_target import admitted_targets, fixed_target_slices, statistics, target_varied_slices

REPO = Path(__file__).resolve().parents[1]
METRIC_FILE = REPO / "data" / "datadecide_runs.parquet"
needs_data = pytest.mark.skipif(not METRIC_FILE.exists(), reason="raw DataDecide table not harvested")


@pytest.fixture(scope="module")
def setup():
    frame = pd.read_parquet(METRIC_FILE)
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    designs = window_designs(frame)
    labels = list(frame.groupby("scale_label")["compute"].first().sort_values().index)
    return designs, labels


@needs_data
def test_admitted_targets_and_counts_match_preregistration(setup):
    designs, labels = setup
    targets = admitted_targets(designs)
    assert [labels[t] for t in targets] == ["150M", "300M", "530M", "1B"]
    ft = fixed_target_slices(designs, targets)
    assert sum(len(gs) for gs in ft.values()) == 26
    assert sum(len(g) for gs in ft.values() for g in gs) == 152
    for t, gs in ft.items():
        for g in gs:
            assert len({designs[i].target for i in g}) == 1 and len({designs[i].rungs for i in g}) == 1
    tv = target_varied_slices(designs)
    assert len(tv) == 36 and sum(map(len, tv)) == 192


@needs_data
def test_planted_fixed_target_slope_is_recovered(setup):
    designs, labels = setup
    targets = admitted_targets(designs)
    ft, tv = fixed_target_slices(designs, targets), target_varied_slices(designs)
    lever = np.array([d.lever for d in designs])
    offsets = np.array([d.target * 10.0 + d.rungs for d in designs])  # absorbed by slice fixed effects
    planted = 4.0 * lever + offsets
    window_offsets = np.array([d.start * 10.0 + d.top for d in designs])  # absorbed in the target-varied slices
    planted_tv = 4.0 * lever + window_offsets
    stats_ = statistics(designs, planted, planted_tv, ft, tv, labels)
    assert np.isclose(stats_["fixed_target_pooled"], 4.0)
    assert all(np.isclose(stats_[f"fixed_target_{labels[t]}"], 4.0) for t in targets)
    assert np.isclose(stats_["target_varied_determined"], 4.0)


@needs_data
def test_determined_pairs_are_kernel_pairs(setup):
    designs, _ = setup
    for d in designs[:20]:
        assert d.determined_pairs <= set(d.kernel)
