"""The confound analysis generalises the lever-arm designs to fit windows; check it reduces to them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asla.analysis.confound import enumerate_designs, jackknife, slices, statistics, window_designs

REPO = Path(__file__).resolve().parents[1]
METRIC_FILE = REPO / "data" / "datadecide_runs.parquet"

needs_data = pytest.mark.skipif(not METRIC_FILE.exists(), reason="raw DataDecide table not harvested")


def test_design_count_matches_preregistration():
    assert len(enumerate_designs(13)) == 220
    assert sum(1 for s, _, _ in enumerate_designs(13) if s == 0) == 55


@pytest.fixture(scope="module")
def real_designs():
    frame = pd.read_parquet(METRIC_FILE)
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    return frame, window_designs(frame)


@needs_data
def test_windows_from_the_smallest_rung_reproduce_the_lever_arm_designs(real_designs):
    from asla.analysis.lever_arm import designs

    frame, windows = real_designs
    nested = {(d.max_fit_label, d.target_label): d for d in designs(frame)}
    order = list(frame.groupby("scale_label")["compute"].first().sort_values().index)
    matched = 0
    for w in windows:
        if w.start != 0:
            continue
        reference = nested[(order[w.top], order[w.target])]
        assert np.isclose(w.lever, np.log10(reference.lever_arm))
        assert reference.kernel.keys() == w.kernel.keys()
        assert np.allclose([w.kernel[p] for p in reference.kernel], list(reference.kernel.values()))
        matched += 1
    assert matched == 55


@needs_data
def test_slice_counts_match_preregistration(real_designs):
    _, windows = real_designs
    groups = slices(windows)
    assert {k: len(v) for k, v in groups.items()} == {"lever": 36, "top": 29, "rungs": 36}
    # The pre-registration counted 100 designs in slice (ii) from log compute rounded to three decimals; on exact
    # values the 8M-to-20M design (lever arm 0.6994) falls in the 0.6 bin, not the 0.8 one. The rule is applied as
    # written and the count recorded as a deviation in confound.json.
    assert {k: sum(map(len, v)) for k, v in groups.items()} == {"lever": 192, "top": 101, "rungs": 192}


@needs_data
def test_statistics_recover_planted_coefficients(real_designs):
    _, windows = real_designs
    groups = slices(windows)
    lever = np.array([d.lever for d in windows])
    top = np.array([d.log_top for d in windows])
    rungs = np.array([d.rungs for d in windows], dtype=float)
    planted = 1.0 + 3.0 * lever - 2.0 * top + 0.5 * rungs
    stats_ = statistics(windows, planted, groups)
    assert np.isclose(stats_["joint_lever"], 3.0)
    assert np.isclose(stats_["joint_top"], -2.0)
    assert np.isclose(stats_["joint_rungs"], 0.5)
    assert np.isclose(stats_["slice_lever"], 3.0)
    assert np.isclose(stats_["slice_rungs"], 0.5)
    # Slice (ii) holds rung count fixed and absorbs residual lever-arm variation with a covariate.
    assert np.isclose(stats_["slice_top"], -2.0)


@needs_data
def test_jackknife_reports_every_statistic(real_designs):
    _, windows = real_designs
    out = jackknife(windows, slices(windows))
    assert set(out) == {"joint_lever", "joint_top", "joint_rungs", "slice_lever", "slice_top", "slice_rungs"}
    assert all(v["se"] > 0 for v in out.values())
