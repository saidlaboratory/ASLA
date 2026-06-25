"""Paper-grade audit routines with cell-wise bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets, truth_ranking
from asla.analysis.metrics import decision_metrics
from asla.analysis.rankers import Ranker
from asla.config import GateConfig
from asla.data.schema import validate


def cell_seed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return seed counts for each ``(intervention, compute)`` cell."""

    validate(df)
    return (
        df.groupby(["intervention", "compute"], sort=True)["seed"]
        .nunique()
        .reset_index(name="seed_count")
        .sort_values(["intervention", "compute"], kind="mergesort")
        .reset_index(drop=True)
    )


def under_seeded_cells(df: pd.DataFrame, min_seeds: int = 2) -> list[dict[str, Any]]:
    """Return cells with fewer than ``min_seeds`` distinct seeds."""

    counts = cell_seed_counts(df)
    rows = counts[counts["seed_count"] < min_seeds]
    return [
        {
            "intervention": str(row.intervention),
            "compute": float(row.compute),
            "seed_count": int(row.seed_count),
        }
        for row in rows.itertuples(index=False)
    ]


def seed_noise_report(df: pd.DataFrame, target: float, k: float = GateConfig().noise_band_k) -> dict[str, Any]:
    """Estimate target-budget seed noise and report under-seeded cells.

    Variance is pooled across target cells with at least two seeds. If no target
    cell has two seeds, the noise band is infinite so significant crossovers are
    not fabricated from unestimated noise.
    """

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    variances: list[float] = []
    counts: list[int] = []
    target_under_seeded: list[dict[str, Any]] = []
    for (intervention, compute), group in target_df.groupby(["intervention", "compute"], sort=True):
        n = int(group["seed"].nunique())
        if n < 2:
            target_under_seeded.append(
                {"intervention": str(intervention), "compute": float(compute), "seed_count": n}
            )
            continue
        variances.append(float(group["bpb"].var(ddof=1)))
        counts.append(n)
    if not variances:
        return {
            "noise_band": float("inf"),
            "pooled_variance": None,
            "adequately_seeded_cells": 0,
            "under_seeded_cells": under_seeded_cells(df),
            "target_under_seeded_cells": target_under_seeded,
        }
    pooled_var = float(np.mean(variances))
    mean_n = float(np.mean(counts))
    return {
        "noise_band": float(k * np.sqrt(pooled_var / mean_n)),
        "pooled_variance": pooled_var,
        "adequately_seeded_cells": len(variances),
        "under_seeded_cells": under_seeded_cells(df),
        "target_under_seeded_cells": target_under_seeded,
    }


def resample_runs_by_cell(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap runs by resampling seeds within each intervention-budget cell."""

    validate(df)
    pieces: list[pd.DataFrame] = []
    for (_, _), group in df.groupby(["intervention", "compute"], sort=True, dropna=False):
        seeds = np.asarray(sorted(group["seed"].unique()), dtype=int)
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        for replicate_seed, seed in enumerate(sampled):
            rows = group[group["seed"] == seed].copy()
            rows["seed"] = int(replicate_seed)
            pieces.append(rows)
    if not pieces:
        return df.iloc[0:0].copy()
    out = pd.concat(pieces, ignore_index=True)
    out.attrs = dict(df.attrs)
    validate(out)
    return out


def _with_extra_metrics(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    out["mis_selection_rate"] = 1.0 - float(metrics["top1_acc"])
    out["mean_regret"] = float(metrics["regret"])
    return out


def _summarize_distribution(point: float, values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def audit_with_ci(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    rankers: Mapping[str, Ranker],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Audit rankers and bootstrap decision metrics by resampling seeds within cells."""

    validate(df)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    fit_budgets = normalize_budgets(budgets, target=target)
    truth = truth_ranking(df, target)
    point_metrics: dict[str, dict[str, float]] = {}
    boot_metrics: dict[str, dict[str, list[float]]] = {}
    for name, ranker in rankers.items():
        projected = ranker(df, fit_budgets, target)
        metrics = _with_extra_metrics(decision_metrics(projected, truth, k=min(3, len(projected))))
        point_metrics[name] = metrics
        boot_metrics[name] = {metric_name: [] for metric_name in metrics}

    for _ in range(n_boot):
        boot_df = resample_runs_by_cell(df, rng)
        boot_truth = truth_ranking(boot_df, target)
        for name, ranker in rankers.items():
            projected = ranker(boot_df, fit_budgets, target)
            metrics = _with_extra_metrics(decision_metrics(projected, boot_truth, k=min(3, len(projected))))
            for metric_name, value in metrics.items():
                boot_metrics[name][metric_name].append(float(value))

    ranker_results: dict[str, Any] = {}
    for name, metrics in point_metrics.items():
        ranker_results[name] = {
            metric_name: _summarize_distribution(point, boot_metrics[name][metric_name])
            for metric_name, point in metrics.items()
        }

    noise = seed_noise_report(df, target)
    return {
        "rankers": ranker_results,
        "noise": noise,
        "under_seeded_cells": noise["under_seeded_cells"],
    }

