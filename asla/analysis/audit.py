"""Paper-grade audit routines with cell-wise bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.fits import (
    fit_diagnostics_all,
    normalize_budgets,
    truth_ranking,
    truth_ranking_with_se,
    truth_ties_with_winner,
)
from asla.analysis.metrics import decision_metrics
from asla.analysis.rankers import Ranker
from asla.config import GateConfig
from asla.data.schema import validate
from asla.models import FitError


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

    Variance is pooled across target-budget cells with at least two seeds.
    When no target cell has two seeds, pooling falls back to all adequately
    seeded cells and ``noise_band_source`` reports the fallback. If no cell at
    all has two seeds, the noise band is unestimated and significant
    crossovers are not fabricated from unestimated noise.
    """

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    all_variances: list[float] = []
    all_counts: list[int] = []
    target_variances: list[float] = []
    target_counts: list[int] = []
    all_under_seeded: list[dict[str, Any]] = []
    target_under_seeded: list[dict[str, Any]] = []
    for (intervention, compute), group in df.groupby(["intervention", "compute"], sort=True):
        n = int(group["seed"].nunique())
        is_target = bool(np.isclose(float(compute), float(target)))
        if n < 2:
            cell = {"intervention": str(intervention), "compute": float(compute), "seed_count": n}
            all_under_seeded.append(cell)
            if is_target:
                target_under_seeded.append(cell)
            continue
        all_variances.append(float(group["bpb"].var(ddof=1)))
        all_counts.append(n)
        if is_target:
            target_variances.append(all_variances[-1])
            target_counts.append(n)
    if target_variances:
        variances, counts, source = target_variances, target_counts, "target_cells"
    elif all_variances:
        variances, counts, source = all_variances, all_counts, "all_cells"
    else:
        return {
            "noise_band": None,
            "noise_band_estimated": False,
            "noise_band_source": None,
            "pooled_variance": None,
            "adequately_seeded_cells": 0,
            "under_seeded_cells": all_under_seeded,
            "target_under_seeded_cells": target_under_seeded,
        }
    pooled_var = float(np.mean(variances))
    mean_n = float(np.mean(counts))
    return {
        "noise_band": float(k * np.sqrt(pooled_var / mean_n)),
        "noise_band_estimated": True,
        "noise_band_source": source,
        "pooled_variance": pooled_var,
        "adequately_seeded_cells": len(variances),
        "under_seeded_cells": all_under_seeded,
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


def _summarize_distribution(point: float, values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"point": float(point), "lo": None, "hi": None}
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
    if not rankers:
        raise ValueError("at least one ranker is required")
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

    truth_table = truth_ranking_with_se(df, target)
    truth_report = {
        name: {
            "mean": float(row["mean"]),
            "se": None if not np.isfinite(row["se"]) else float(row["se"]),
            "n_seeds": int(row["n_seeds"]),
        }
        for name, row in truth_table.iterrows()
    }
    try:
        diagnostics = {name: asdict(diag) for name, diag in fit_diagnostics_all(df, fit_budgets).items()}
    except FitError:
        diagnostics = {}

    return {
        "rankers": ranker_results,
        "noise": noise,
        "under_seeded_cells": noise["under_seeded_cells"],
        "truth": truth_report,
        "truth_ties": truth_ties_with_winner(df, target),
        "fit_diagnostics": diagnostics,
    }
