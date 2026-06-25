"""Per-intervention scaling-law fits and rankings."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from asla.analysis import splits
from asla.data.schema import validate
from asla.models import FitForm, bpb_chinchilla, bpb_power_law, bootstrap_projection, fit_chinchilla, fit_power_law


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


def _require_chinchilla_columns(df: pd.DataFrame) -> None:
    """Raise clearly if Chinchilla inputs are absent."""

    missing = [col for col in ("params_n", "tokens_d") if col not in df.columns]
    if missing:
        raise ValueError(f"fit_form='chinchilla' requires columns {missing}")


def fit_all(
    df: pd.DataFrame,
    budgets: Iterable[float],
    fit_form: FitForm = "compute_power_law",
) -> Dict[str, tuple[float, ...]]:
    """Fit a scaling law for every intervention using only ``budgets`` rows."""

    validate(df)
    splits.assert_not_test(df)
    if fit_form == "chinchilla":
        _require_chinchilla_columns(df)
    elif fit_form != "compute_power_law":
        raise ValueError(f"unknown fit_form: {fit_form}")
    fit_budgets = normalize_budgets(budgets)
    fit_df = df[_budget_mask(df["compute"], fit_budgets)]
    if fit_df.empty:
        raise ValueError(f"no rows found for fitting budgets {fit_budgets}")
    all_interventions = set(df["intervention"].astype(str).unique())
    fit_interventions = set(fit_df["intervention"].astype(str).unique())
    missing = sorted(all_interventions - fit_interventions)
    if missing:
        raise ValueError(f"no fitting rows found for interventions: {missing}")
    params: Dict[str, tuple[float, ...]] = {}
    for intervention, group in fit_df.groupby("intervention", sort=True):
        if fit_form == "compute_power_law":
            params[str(intervention)] = fit_power_law(
                group["compute"].to_numpy(dtype=float),
                group["bpb"].to_numpy(dtype=float),
            )
        else:
            params[str(intervention)] = fit_chinchilla(
                group["params_n"].to_numpy(dtype=float),
                group["tokens_d"].to_numpy(dtype=float),
                group["bpb"].to_numpy(dtype=float),
            )
    return params


def project_ranking(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    fit_form: FitForm = "compute_power_law",
) -> pd.Series:
    """Return projected BPB at ``target`` sorted ascending by intervention."""

    fit_budgets = normalize_budgets(budgets, target=target)
    params = fit_all(df, fit_budgets, fit_form=fit_form)
    if fit_form == "compute_power_law":
        values = {name: float(bpb_power_law(target, *par)) for name, par in params.items()}
    elif fit_form == "chinchilla":
        _require_chinchilla_columns(df)
        target_df = df[np.isclose(df["compute"].astype(float), float(target))]
        if target_df.empty:
            raise ValueError(f"no target rows found at budget {target} for Chinchilla projection inputs")
        values = {}
        for name, par in params.items():
            group = target_df[target_df["intervention"].astype(str) == name]
            if group.empty:
                raise ValueError(f"no target Chinchilla inputs found for intervention {name!r}")
            target_inputs = group[["params_n", "tokens_d"]].to_numpy(dtype=float)
            if (not np.isfinite(target_inputs).all()) or (target_inputs <= 0).any():
                raise ValueError(f"target Chinchilla inputs for intervention {name!r} must be finite positive values")
            n_target = float(group["params_n"].mean())
            d_target = float(group["tokens_d"].mean())
            values[name] = float(bpb_chinchilla(n_target, d_target, *par))
    else:
        raise ValueError(f"unknown fit_form: {fit_form}")
    return pd.Series(values).sort_values(kind="mergesort")


def truth_ranking(df: pd.DataFrame, target: float) -> pd.Series:
    """Return measured mean BPB at ``target`` sorted ascending by intervention."""

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    all_interventions = set(df["intervention"].astype(str).unique())
    target_interventions = set(target_df["intervention"].astype(str).unique())
    missing = sorted(all_interventions - target_interventions)
    if missing:
        raise ValueError(f"no target rows found for interventions: {missing}")
    values = target_df.groupby("intervention", sort=True)["bpb"].mean()
    return values.sort_values(kind="mergesort")


def projection_with_uncertainty(
    df_iv: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    n_boot: int,
    rng: np.random.Generator,
    fit_form: FitForm = "compute_power_law",
) -> tuple[float, float, float, float]:
    """Fit and bootstrap one intervention's target projection."""

    if fit_form != "compute_power_law":
        raise NotImplementedError("projection bootstrap uncertainty is currently implemented for compute_power_law only")
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
