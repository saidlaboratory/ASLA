"""Per-intervention scaling-law fits and rankings."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from asla.analysis import splits
from asla.data.schema import validate
from asla.models import bpb_power_law, bootstrap_projection, fit_power_law


def normalize_budgets(budgets: Iterable[float], *, target: float | None = None) -> tuple[float, ...]:
    """Return sorted distinct positive budgets and reject target leakage when provided."""

    values = tuple(sorted({float(value) for value in budgets}))
    if not values:
        raise ValueError("at least one fitting budget is required")
    if any(value <= 0 or not np.isfinite(value) for value in values):
        raise ValueError("fitting budgets must be finite positive numbers")
    if target is not None and any(np.isclose(value, float(target)) for value in values):
        raise ValueError(f"target budget {target} cannot be included in fitting budgets")
    return values


def _budget_mask(series: pd.Series, budgets: Iterable[float]) -> pd.Series:
    values = np.asarray(normalize_budgets(budgets), dtype=float)
    return series.astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))


def fit_all(df: pd.DataFrame, budgets: Iterable[float]) -> Dict[str, Tuple[float, float, float]]:
    """Fit a power law for every intervention using only ``budgets`` rows."""

    validate(df)
    splits.assert_not_test(df)
    fit_budgets = normalize_budgets(budgets)
    fit_df = df[_budget_mask(df["compute"], fit_budgets)]
    if fit_df.empty:
        raise ValueError(f"no rows found for fitting budgets {fit_budgets}")
    params: Dict[str, Tuple[float, float, float]] = {}
    for intervention, group in fit_df.groupby("intervention", sort=True):
        params[str(intervention)] = fit_power_law(
            group["compute"].to_numpy(dtype=float),
            group["bpb"].to_numpy(dtype=float),
        )
    return params


def project_ranking(df: pd.DataFrame, budgets: Iterable[float], target: float) -> pd.Series:
    """Return projected BPB at ``target`` sorted ascending by intervention."""

    fit_budgets = normalize_budgets(budgets, target=target)
    params = fit_all(df, fit_budgets)
    values = {name: float(bpb_power_law(target, *par)) for name, par in params.items()}
    return pd.Series(values).sort_values(kind="mergesort")


def truth_ranking(df: pd.DataFrame, target: float) -> pd.Series:
    """Return measured mean BPB at ``target`` sorted ascending by intervention."""

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    values = target_df.groupby("intervention", sort=True)["bpb"].mean()
    return values.sort_values(kind="mergesort")


def projection_with_uncertainty(
    df_iv: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Fit and bootstrap one intervention's target projection."""

    validate(df_iv)
    fit_budgets = normalize_budgets(budgets, target=target)
    fit_df = df_iv[_budget_mask(df_iv["compute"], fit_budgets)]
    if fit_df.empty:
        raise ValueError(f"no rows found for fitting budgets {fit_budgets}")
    return bootstrap_projection(
        fit_df["compute"].to_numpy(dtype=float),
        fit_df["bpb"].to_numpy(dtype=float),
        target,
        n_boot,
        rng,
    )
