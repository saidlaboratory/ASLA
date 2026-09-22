"""Paper-grade audit routines with cell-wise bootstrap confidence intervals."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Literal

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
from asla.models import FitError, FitForm

Estimand = Literal["single_design_seed_sensitivity", "pairwise_decisions"]
ESTIMANDS: tuple[Estimand, ...] = ("single_design_seed_sensitivity", "pairwise_decisions")


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

    Within-cell sample variances are pooled with ``n - 1`` degrees-of-freedom
    weights. The representative target-mean SEM uses the actual seed count of
    every target cell, including one-seed cells whose variance must be borrowed.
    Variance is pooled across target-budget cells with at least two seeds.
    When no target cell has two seeds, pooling falls back to all adequately
    seeded cells and ``noise_band_source`` reports the fallback. If no cell at
    all has two seeds, the noise band is unestimated and significant
    crossovers are not fabricated from unestimated noise.
    """

    validate(df)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a finite positive number")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("noise-band multiplier k must be a finite positive number")
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    all_variances: list[float] = []
    all_counts: list[int] = []
    target_variances: list[float] = []
    target_counts: list[int] = []
    target_mean_counts: list[int] = []
    all_under_seeded: list[dict[str, Any]] = []
    target_under_seeded: list[dict[str, Any]] = []
    for (intervention, compute), group in df.groupby(["intervention", "compute"], sort=True):
        n = int(group["seed"].nunique())
        is_target = bool(np.isclose(float(compute), float(target)))
        if is_target:
            target_mean_counts.append(n)
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
            "noise_band_k": float(k),
            "representative_mean_sem": None,
            "representative_gap_sem": None,
            "noise_band_estimated": False,
            "noise_band_source": None,
            "noise_band_seed_count_source": "target_cells",
            "noise_variance_model": "homoscedastic pooled within-cell variance",
            "pooled_variance": None,
            "pooled_degrees_of_freedom": 0,
            "effective_seed_count": None,
            "adequately_seeded_cells": 0,
            "under_seeded_cells": all_under_seeded,
            "target_under_seeded_cells": target_under_seeded,
        }
    degrees_of_freedom = np.asarray(counts, dtype=float) - 1.0
    pooled_var = float(np.average(np.asarray(variances, dtype=float), weights=degrees_of_freedom))
    mean_inverse_n = float(np.mean(1.0 / np.asarray(target_mean_counts, dtype=float)))
    effective_n = float(1.0 / mean_inverse_n)
    representative_mean_sem = float(np.sqrt(pooled_var * mean_inverse_n))
    representative_gap_sem = float(np.sqrt(2.0) * representative_mean_sem)
    return {
        "noise_band": float(k * representative_mean_sem),
        "noise_band_k": float(k),
        "representative_mean_sem": representative_mean_sem,
        "representative_gap_sem": representative_gap_sem,
        "noise_band_estimated": True,
        "noise_band_source": source,
        "noise_band_seed_count_source": "target_cells",
        "noise_variance_model": "homoscedastic pooled within-cell variance",
        "pooled_variance": pooled_var,
        "pooled_degrees_of_freedom": int(np.sum(degrees_of_freedom)),
        "effective_seed_count": effective_n,
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


def _winner_only_metrics(ranking: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Return only metrics licensed by a top-1-only decision policy."""

    if ranking.empty:
        raise ValueError("winner-only ranker returned no intervention")
    winner = str(ranking.index[0])
    truth_names = set(truth.index.astype(str))
    if winner not in truth_names:
        raise ValueError(f"winner-only ranker selected unknown intervention {winner!r}")
    truth = truth.rename(index=str)
    true_winner = str(truth.index[0])
    regret = float(truth.loc[winner] - truth.min())
    top1 = float(winner == true_winner)
    return {"top1_acc": top1, "regret": regret, "mis_selection_rate": 1.0 - top1, "mean_regret": regret}


def _evaluate_ranker(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    ranker: Ranker,
    estimand: Estimand,
) -> tuple[dict[str, float], int]:
    """Evaluate one ranker for the selected replication unit."""

    winner_only = getattr(ranker, "asla_ranking_scope", None) == "top1_only"

    def evaluate(table: pd.DataFrame, k: int) -> dict[str, float]:
        ranking = ranker(table, budgets, target)
        truth = truth_ranking(table, target)
        if winner_only:
            return _winner_only_metrics(ranking, truth)
        return _with_extra_metrics(decision_metrics(ranking, truth, k=k))

    if estimand == "single_design_seed_sensitivity":
        return evaluate(df, min(3, int(df["intervention"].nunique()))), 1
    if estimand != "pairwise_decisions":
        raise ValueError(f"unknown estimand {estimand!r}; choose one of {ESTIMANDS}")
    interventions = sorted(df["intervention"].astype(str).unique())
    pairs = list(itertools.combinations(interventions, 2))
    if not pairs:
        raise ValueError("pairwise_decisions requires at least two interventions")
    values: dict[str, list[float]] = {}
    for a, b in pairs:
        pair_df = df[df["intervention"].astype(str).isin([a, b])]
        for name, value in evaluate(pair_df, 1).items():
            values.setdefault(name, []).append(float(value))
    return {name: float(np.mean(metric_values)) for name, metric_values in values.items()}, len(pairs)


def _estimand_metadata(estimand: Estimand, replication_count: int) -> dict[str, object]:
    """Describe the selected quantity without selecting a paper headline."""

    common = {
        "name": estimand,
        "replication_count": int(replication_count),
        "bootstrap_unit": "seeds resampled within observed intervention-compute cells",
        "assumptions": [
            "seed rows are exchangeable within each intervention-compute cell",
            "cells are resampled independently; paired-seed dependence across cells is not preserved",
            "the intervention set and budget design are fixed",
        ],
    }
    if estimand == "single_design_seed_sensitivity":
        return {
            **common,
            "replication_unit": "one complete-candidate-set leaderboard decision",
            "point_interpretation": "binary wrong-selection indicator for the complete intervention set",
            "uncertainty_scope": "within-cell seed-bootstrap sensitivity of one fixed decision",
        }
    return {
        **common,
        "replication_unit": "unordered intervention pair",
        "point_interpretation": "mean decision metric over every unordered pair in this fixed runs table",
        "uncertainty_scope": "within-cell seed-bootstrap sensitivity for a fixed finite set of dependent pairs",
    }


def _summarize_distribution(point: float, values: list[float]) -> dict[str, float | int | str | None]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {
            "point": float(point),
            "lo": None,
            "hi": None,
            "confidence_level": 0.95,
            "method": "percentile bootstrap",
            "n_boot": 0,
        }
    bounds = np.asarray(np.percentile(arr, [2.5, 97.5]), dtype=float)
    return {
        "point": float(point),
        "lo": float(bounds[0]),
        "hi": float(bounds[1]),
        "confidence_level": 0.95,
        "method": "percentile bootstrap",
        "n_boot": int(len(arr)),
    }


def audit_with_ci(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    rankers: Mapping[str, Ranker],
    n_boot: int,
    rng: np.random.Generator,
    *,
    estimand: Estimand,
    fit_form: FitForm = "compute_power_law",
    weighted: bool = False,
) -> dict[str, Any]:
    """Audit rankers for an explicit estimand using a cell-wise seed bootstrap."""

    validate(df)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not rankers:
        raise ValueError("at least one ranker is required")
    if estimand not in ESTIMANDS:
        raise ValueError(f"unknown estimand {estimand!r}; choose one of {ESTIMANDS}")
    fit_budgets = normalize_budgets(budgets, target=target)
    point_metrics: dict[str, dict[str, float]] = {}
    boot_metrics: dict[str, dict[str, list[float]]] = {}
    bootstrap_failures = {name: 0 for name in rankers}
    metric_availability: dict[str, dict[str, object]] = {}
    replication_count: int | None = None
    for name, ranker in rankers.items():
        metrics, ranker_replication_count = _evaluate_ranker(df, fit_budgets, target, ranker, estimand)
        if replication_count is None:
            replication_count = ranker_replication_count
        elif replication_count != ranker_replication_count:
            raise ValueError("rankers disagreed on estimand replication count")
        point_metrics[name] = metrics
        boot_metrics[name] = {metric_name: [] for metric_name in metrics}
        all_ranking_metrics = {
            "top1_acc",
            "regret",
            "mis_selection_rate",
            "mean_regret",
            "topk_recall",
            "pairwise_acc",
            "kendall_tau",
            "spearman",
        }
        reported = sorted(metrics)
        omitted = sorted(all_ranking_metrics - set(reported))
        metric_availability[name] = {
            "ranking_scope": getattr(ranker, "asla_ranking_scope", "full_field"),
            "reported": reported,
            "omitted": {metric: "requires a coherent full-field ranking" for metric in omitted},
        }

    for _ in range(n_boot):
        boot_df = resample_runs_by_cell(df, rng)
        for name, ranker in rankers.items():
            try:
                metrics, _ = _evaluate_ranker(boot_df, fit_budgets, target, ranker, estimand)
            except FitError:
                bootstrap_failures[name] += 1
                continue
            for metric_name, value in metrics.items():
                boot_metrics[name][metric_name].append(float(value))

    ranker_results: dict[str, Any] = {}
    bootstrap_diagnostics: dict[str, dict[str, float | int]] = {}
    min_successes = min(n_boot, max(10, int(np.ceil(0.25 * n_boot))))
    for name, metrics in point_metrics.items():
        successes = n_boot - bootstrap_failures[name]
        if successes < min_successes:
            raise FitError(
                f"too few audit bootstrap resamples succeeded for ranker {name!r}: "
                f"{successes}/{n_boot} (minimum {min_successes})"
            )
        ranker_results[name] = {
            metric_name: _summarize_distribution(point, boot_metrics[name][metric_name])
            for metric_name, point in metrics.items()
        }
        bootstrap_diagnostics[name] = {
            "requested": int(n_boot),
            "successful": int(successes),
            "failed": int(bootstrap_failures[name]),
            "success_fraction": float(successes / n_boot),
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
    diagnostics_availability: dict[str, object]
    try:
        diagnostics = {
            name: asdict(diag)
            for name, diag in fit_diagnostics_all(df, fit_budgets, fit_form=fit_form, weighted=weighted).items()
        }
        diagnostics_availability = {"included": True}
    except FitError as exc:
        diagnostics = {}
        diagnostics_availability = {"included": False, "reason": str(exc)}

    return {
        "rankers": ranker_results,
        "noise": noise,
        "under_seeded_cells": noise["under_seeded_cells"],
        "truth": truth_report,
        "truth_ties": truth_ties_with_winner(df, target),
        "fit_diagnostics": diagnostics,
        "analysis_availability": {"fit_diagnostics": diagnostics_availability},
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "ranker_metric_availability": metric_availability,
        "estimand": _estimand_metadata(estimand, int(replication_count or 0)),
    }
