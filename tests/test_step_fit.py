"""The step-term fit reduces to the frozen power law, and subset slopes reduce to the scanner's."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asla.analysis.step_fit import fit_power_law_step
from asla.models import bpb_power_law, fit_power_law

REPO = Path(__file__).resolve().parents[1]


def test_no_boundary_matches_frozen_fit():
    c = np.logspace(16, 19, 6)
    y = bpb_power_law(c, 0.8, 30.0, 0.12) + np.array([0.003, -0.002, 0.001, 0.0, -0.001, 0.002])
    e, a, alpha, d = fit_power_law_step(c, y, [])
    ref = fit_power_law(c, y)
    assert d == ()
    assert np.allclose([e, a, alpha], ref, rtol=1e-6)


def test_step_is_recovered():
    c = np.logspace(16, 19, 7)
    y = bpb_power_law(c, 0.8, 30.0, 0.12) + np.where(c >= c[3], -0.02, 0.0)
    e, a, alpha, d = fit_power_law_step(c, y, [c[3]])
    assert d[0] == pytest.approx(-0.02, abs=2e-3)


@pytest.mark.skipif(not (REPO / "data" / "datadecide_runs.parquet").exists(), reason="raw DataDecide table not harvested")
def test_subset_stat_on_all_designs_matches_scanner():
    import importlib

    from asla.analysis.geometry_scan import Scanner
    from asla.analysis.step_fit import subset_stat

    sc = Scanner(importlib.import_module("scripts.run_confound")._load_frame())
    sizes, targets = list(range(11)), [9, 10]
    ref = sc.slope(sizes, targets)
    got = subset_stat(sc, sc.designs(sizes, targets))
    assert np.isclose(got["estimate"], ref["estimate"]) and np.isclose(got["se"], ref["se"])
