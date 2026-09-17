"""Tests for the allocation study's helpers.

These cover the parts that could silently produce a wrong number: the stable
seed (which ``--resume`` depends on), the deviation estimator's provenance, and
the guard that refuses to feed a two-point design to a three-parameter fit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_allocation_study", Path(__file__).resolve().parents[1] / "scripts" / "run_allocation_study.py"
)
assert _SPEC is not None and _SPEC.loader is not None
study = importlib.util.module_from_spec(_SPEC)
sys.modules["run_allocation_study"] = study
_SPEC.loader.exec_module(study)


def _synthetic_runs(n_interventions: int = 4, seeds: int = 3) -> pd.DataFrame:
    """A small well-specified table: exact power laws plus tiny seed noise."""

    budgets = [1e16, 3e16, 1e17, 3e17, 1e18]
    rng = np.random.default_rng(0)
    rows = []
    for index in range(n_interventions):
        floor = 2.0 + 0.01 * index
        amplitude = 50.0 + index
        alpha = 0.18 + 0.002 * index
        for budget in budgets:
            mean = floor + amplitude * budget**-alpha
            for seed in range(seeds):
                rows.append(
                    {
                        "intervention": f"recipe_{index}",
                        "intervention_class": "data",
                        "compute": budget,
                        "seed": seed,
                        "bpb": mean + rng.normal(0, 1e-4),
                        "metric_name": "bpb",
                        "params_n": 1e7,
                        "tokens_d": 1e9,
                        "tokens_per_param": 100.0,
                        "scale_label": f"S{budget:.0e}",
                        "step": 1000,
                        "seed_label": str(seed),
                        "tuning_quality": "ok",
                    }
                )
    return pd.DataFrame(rows)


def test_stable_seed_is_deterministic_across_calls() -> None:
    """--resume depends on this; Python's hash() would not be stable."""

    first = study._stable_seed(123, "frac_0.1", "decision_optimal")
    second = study._stable_seed(123, "frac_0.1", "decision_optimal")
    assert first == second
    assert first != study._stable_seed(123, "frac_0.3", "decision_optimal")
    assert first != study._stable_seed(124, "frac_0.1", "decision_optimal")
    assert 0 <= first < 2**32


def test_estimate_deviation_records_provenance() -> None:
    df = _synthetic_runs()
    budgets = tuple(sorted(df["compute"].unique()))
    names = sorted(df["intervention"].unique())[:2]
    info = study.estimate_deviation(df, budgets, names)
    assert info["estimated_from"] == "train_interventions_only"
    assert info["interventions_used"] == names
    assert len(info["deviation"]) == len(budgets)
    assert all(value >= 0 for value in info["deviation"])


def test_estimate_deviation_is_small_for_well_specified_data() -> None:
    """Exact power laws must give a near-zero deviation, not a spurious one."""

    df = _synthetic_runs()
    budgets = tuple(sorted(df["compute"].unique()))
    info = study.estimate_deviation(df, budgets, sorted(df["intervention"].unique()))
    assert max(info["deviation"]) < 1e-2


def test_evaluate_allocation_refuses_two_point_design() -> None:
    """A two-point design cannot identify the three-parameter estimator."""

    df = _synthetic_runs()
    budgets = tuple(sorted(df["compute"].unique()))
    runs = (3.0, 0.0, 3.0, 0.0, 0.0)
    result = study.evaluate_allocation(
        df, budgets, runs, 1e19, sorted(df["intervention"].unique()), np.random.default_rng(0), 4
    )
    assert result["mis_selection"] is None
    assert "unidentifiable" in result["error"]
    assert result["n_support"] == 2


def test_evaluate_allocation_reports_single_support_separately() -> None:
    df = _synthetic_runs()
    budgets = tuple(sorted(df["compute"].unique()))
    result = study.evaluate_allocation(
        df,
        budgets,
        (3.0, 0.0, 0.0, 0.0, 0.0),
        1e19,
        sorted(df["intervention"].unique()),
        np.random.default_rng(0),
        4,
    )
    assert result["mis_selection"] is None
    assert "fewer than two" in result["error"]


def test_evaluate_single_scale_perfect_when_ordering_preserved() -> None:
    """Monotone, well-separated recipes must give zero mis-selection."""

    df = _synthetic_runs()
    names = sorted(df["intervention"].unique())
    result = study.evaluate_single_scale(df, 3e17, 1e18, names, np.random.default_rng(0), 20, 3)
    assert result["mis_selection"] == pytest.approx(0.0)
    assert result["cost_flops"] == pytest.approx(3e17 * 3 * len(names))


def test_load_datadecide_excludes_off_trajectory_scale(tmp_path: Path) -> None:
    df = _synthetic_runs()
    df.loc[df.index[:3], "scale_label"] = "750M"
    path = tmp_path / "runs.parquet"
    df.to_parquet(path)
    loaded = study.load_datadecide(path)
    assert "750M" not in set(loaded["scale_label"])
