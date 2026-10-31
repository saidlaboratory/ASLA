"""The geometry scanner reproduces the frozen window-design and fixed-target code on real geometries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (REPO / "data" / "datadecide_runs.parquet").exists(), reason="raw DataDecide table not harvested"
)


def frozen_slope(frame, labels_subset, target_labels):
    from asla.analysis.confound import window_designs
    from asla.analysis.fixed_target import fixed_target_slices, jackknife, target_varied_slices

    sub = frame[frame["scale_label"].isin(labels_subset)].reset_index(drop=True)
    ds = window_designs(sub)
    order = list(sub.groupby("scale_label")["compute"].first().sort_values().index)
    targets = [order.index(t) for t in target_labels]
    ft = {t: g for t, g in fixed_target_slices(ds, targets).items() if g}
    return jackknife(ds, ft, target_varied_slices(ds), order)["fixed_target_pooled"]


@pytest.mark.parametrize(
    "subset,targets",
    [
        (["4M", "6M", "14M", "20M", "60M", "90M", "150M", "300M"], ["150M", "300M"]),
        (["4M", "10M", "60M", "90M", "300M", "1B"], ["1B"]),
        (["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "1B"], ["150M", "300M", "530M", "1B"]),
    ],
)
def test_scanner_matches_frozen_code(subset, targets):
    import importlib

    from asla.analysis.geometry_scan import Scanner

    frame = importlib.import_module("scripts.run_confound")._load_frame()
    sc = Scanner(frame)
    idx = [sc.labels.index(s) for s in subset]
    res = sc.slope(idx, [sc.labels.index(t) for t in targets])
    ref = frozen_slope(frame, subset, targets)
    assert np.isclose(res["estimate"], ref["estimate"], rtol=1e-9, atol=1e-9)
    assert np.isclose(res["se"], ref["se"], rtol=1e-9, atol=1e-9)
