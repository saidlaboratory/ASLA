"""Budget-indexed fixed-target designs: the frozen statistic recovers a planted slope; evidence is sound."""

from __future__ import annotations

import numpy as np

from asla.analysis.budget_ladder import designs, known_variance_evidence
from asla.analysis.fixed_target import fixed_target_slices, statistics, target_varied_slices


def _world(n_budgets: int = 9):
    x = 16.0 + 0.3 * np.arange(n_budgets)
    loss = {f"r{i}": 2.0 + 30.0 * (10.0**x) ** -0.12 + 0.01 * i for i in range(5)}
    sds = {t: {r: 0.001 for r in loss} for t in range(n_budgets)}
    return loss, x, sds


def test_designs_cover_every_window_below_each_target():
    loss, x, sds = _world()
    ds = designs(loss, x, sds, targets=[8])
    # windows of at least 3 consecutive rungs among the 8 rungs below the target
    expected = sum(f - 1 for f in range(2, 8))
    assert len(ds) == expected
    assert all(np.isclose(d.lever, x[8] - x[d.top]) for d in ds)


def test_frozen_statistic_recovers_planted_slope():
    loss, x, sds = _world()
    ds = designs(loss, x, sds, targets=[6, 7, 8])
    ft = fixed_target_slices(ds, [6, 7, 8])
    tv = target_varied_slices(ds)
    lever = np.array([d.lever for d in ds])
    planted = 7.0 * lever + np.array([d.target * 3.0 + d.rungs for d in ds])
    labels = [str(i) for i in range(20)]
    assert np.isclose(statistics(ds, planted, planted, ft, tv, labels)["fixed_target_pooled"], 7.0)


def test_known_variance_evidence_determination():
    ev = {(e.left, e.right): e for e in known_variance_evidence({"a": 1.0, "b": 1.1, "c": 1.0001}, {"a": 0.001, "b": 0.001, "c": 0.001})}
    assert ev[("a", "b")].determined and not ev[("a", "c")].determined
