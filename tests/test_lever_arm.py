"""Tests for the lever-arm test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asla.analysis.lever_arm import designs, loro_test

LADDER = {f"s{k}": 1e17 * 3**k for k in range(8)}


def _frame(n: int = 10, noise: float = 0.002, seed: int = 0, curvature: float = 0.0, truth_seed: int = 0) -> pd.DataFrame:
    """Truth fixed by ``truth_seed``; ``seed`` draws only the noise."""

    alphas = 0.2 + 0.01 * np.random.default_rng(truth_seed).standard_normal(n)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n):
        level, alpha = 2.0 + 0.05 * r, alphas[r]
        for label, compute in LADDER.items():
            x = np.log(compute / 1e17)
            mu = level + 3.0 * (compute / 1e17) ** -alpha + curvature * (r % 3 - 1) * x**2 / 100
            for s in range(3):
                rows.append(
                    {
                        "intervention": f"r{r}",
                        "scale_label": label,
                        "compute": compute,
                        "seed": s,
                        "bpb": mu + noise * rng.standard_normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_designs_enumerate_every_fit_target_pair_with_three_budgets() -> None:
    out = designs(_frame())
    # targets s3..s7, fits s2..target-1: 1 + 2 + 3 + 4 + 5
    assert len(out) == 15
    assert all(d.lever_arm > 1 for d in out)


def test_loro_statistics_are_well_formed() -> None:
    test = loro_test(designs(_frame(curvature=1.0)))
    assert -1 <= test["rho"] <= 1 and test["se_fisher_z"] > 0
    assert len(test["leave_one_out_rho"]) == 10
    assert test["ci"][0] <= test["rho"] <= test["ci"][1]


def test_replicate_comparator_is_null_by_construction() -> None:
    rhos = []
    for seed in range(20):
        # Same truth, independent noise: the replicate's true excess is zero at every design.
        frame, replicate = _frame(seed=seed, noise=0.02), _frame(seed=100 + seed, noise=0.02)
        out = designs(frame, replicate=replicate, with_projection=False)
        rhos.append(loro_test(out, replicate=True)["rho"])
        assert all(d.kernel == {} for d in out)
    assert abs(float(np.mean(rhos))) < 0.3


def test_true_excess_requires_truth_and_projection() -> None:
    frame = _frame()
    truth = {label: g.groupby("intervention")["bpb"].mean() for label, g in frame.groupby("scale_label")}
    with_truth = designs(frame, truth_targets=truth)
    assert all(d.true_excess is not None for d in with_truth)
    assert all(d.true_excess is None for d in designs(frame, truth_targets=truth, replicate=frame, with_projection=False))
