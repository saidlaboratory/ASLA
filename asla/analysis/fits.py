"""Per-intervention scaling-law fits and rankings."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from asla.analysis import splits
from asla.data.schema import validate
from asla.models import (
    FitDiagnostics,
    FitForm,
    bootstrap_projection,
    bpb_chinchilla,
    bpb_power_law,
    fit_chinchilla,
    fit_diagnostics,
    fit_power_law,
)


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


def resolve_fit_budgets(
    df: pd.DataFrame,
    target: float,
    requested: Iterable[float] | None = None,
    intermediate_budget: float | None = None,
) -> tuple[float, ...]:
    """Resolve fitting budgets while keeping a reserved exploration budget held out."""

    validate(df)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a finite positive number")
    computes = tuple(sorted(float(value) for value in df["compute"].unique()))
    if not any(np.isclose(value, target) for value in computes):
        raise ValueError(f"target budget {target:g} is not present in the runs table")
    if intermediate_budget is not None:
        if not np.isfinite(intermediate_budget) or intermediate_budget <= 0:
            raise ValueError("intermediate budget must be a finite positive number")
        if not any(np.isclose(value, intermediate_budget) for value in computes):
            raise ValueError(f"intermediate budget {intermediate_budget:g} is not present in the runs table")
        if intermediate_budget >= target or np.isclose(intermediate_budget, target):
            raise ValueError("intermediate budget must be strictly below the target budget")

    requested_values = None if requested is None else [float(value) for value in requested]
    if requested_values is not None and intermediate_budget is not None and any(
        np.isclose(value, intermediate_budget) for value in requested_values
    ):
        raise ValueError("fitting budgets must not include the reserved intermediate budget")
    candidates = requested_values if requested_values is not None else [value for value in computes if value < target]
    if intermediate_budget is not None:
        candidates = [value for value in candidates if not np.isclose(value, intermediate_budget)]
    missing = [value for value in candidates if not any(np.isclose(value, observed) for observed in computes)]
    if missing:
        raise ValueError(f"requested fitting budgets are absent from the runs table: {missing}")
    budgets = normalize_budgets(candidates, target=target)
    if len(budgets) < 3:
        raise ValueError(f"need at least 3 distinct fitting budgets; found {len(budgets)}")
    if intermediate_budget is not None and any(budget >= intermediate_budget for budget in budgets):
        raise ValueError("all fitting budgets must be strictly below the reserved intermediate budget")
    return budgets


def _budget_mask(series: pd.Series, budgets: Iterable[float]) -> pd.Series:
    values = np.asarray(normalize_budgets(budgets), dtype=float)
    return series.astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))


def _require_chinchilla_columns(df: pd.DataFrame) -> None:
    """Raise clearly if Chinchilla inputs are absent."""

    missing = [col for col in ("params_n", "tokens_d") if col not in df.columns]
    if missing:
        raise ValueError(f"fit_form='chinchilla' requires columns {missing}")


def cell_means_and_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level rows into per-cell means with seed-aware standard errors.

    Each ``(intervention, compute)`` cell yields the mean BPB, the number of
    distinct seeds, and ``sigma = sd / sqrt(n_seeds)``. Cells with one seed
    cannot estimate a standard error; their sigma is imputed as the mean sigma
    of the adequately seeded cells so weighted fits neither drop them nor let
    them dominate. When no cell has two seeds, all sigmas are null and callers
    should fall back to unweighted fitting.
    """

    validate(df)
    records: list[dict[str, object]] = []
    for (intervention, compute), group in df.groupby(["intervention", "compute"], sort=True):
        n = int(group["seed"].nunique())
        sd = float(group["bpb"].std(ddof=1)) if n >= 2 else float("nan")
        record: dict[str, object] = {
            "intervention": str(intervention),
            "compute": float(compute),
            "bpb_mean": float(group["bpb"].mean()),
            "n_seeds": n,
            "sigma": sd / np.sqrt(n) if np.isfinite(sd) else float("nan"),
        }
        for col in ("params_n", "tokens_d"):
            if col in group.columns:
                record[col] = float(group[col].mean())
        records.append(record)
    cells = pd.DataFrame.from_records(records)
    estimated = cells["sigma"].dropna()
    if not estimated.empty:
        floor = float(estimated[estimated > 0].min()) if (estimated > 0).any() else 1e-12
        fill = float(max(estimated.mean(), floor))
        cells["sigma"] = cells["sigma"].fillna(fill).clip(lower=floor)
    return cells


def fit_all(
    df: pd.DataFrame,
    budgets: Iterable[float],
    fit_form: FitForm = "compute_power_law",
    weighted: bool = False,
) -> Dict[str, tuple[float, ...]]:
    """Fit a scaling law for every intervention using only ``budgets`` rows.

    With ``weighted=True`` each intervention is fit on per-cell seed means
    weighted by their standard errors (``sd/sqrt(n_seeds)``); when no cell has
    two seeds the fit silently falls back to the unweighted seed-level fit.
    """

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

    cells: pd.DataFrame | None = None
    if weighted:
        cells = cell_means_and_sigma(fit_df)
        if cells["sigma"].isna().all():
            cells = None

    params: Dict[str, tuple[float, ...]] = {}
    for intervention, group in fit_df.groupby("intervention", sort=True):
        if cells is not None:
            cell_group = cells[cells["intervention"] == str(intervention)]
            x = cell_group["compute"].to_numpy(dtype=float)
            y = cell_group["bpb_mean"].to_numpy(dtype=float)
            sigma = cell_group["sigma"].to_numpy(dtype=float)
            if fit_form == "compute_power_law":
                params[str(intervention)] = fit_power_law(x, y, sigma=sigma)
            else:
                params[str(intervention)] = fit_chinchilla(
                    cell_group["params_n"].to_numpy(dtype=float),
                    cell_group["tokens_d"].to_numpy(dtype=float),
                    y,
                    sigma=sigma,
                )
        elif fit_form == "compute_power_law":
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


def fit_diagnostics_all(
    df: pd.DataFrame,
    budgets: Iterable[float],
    fit_form: FitForm = "compute_power_law",
    weighted: bool = False,
) -> Dict[str, FitDiagnostics]:
    """Return residual diagnostics of each intervention's fit on the fitting budgets."""

    fit_budgets = normalize_budgets(budgets)
    params = fit_all(df, fit_budgets, fit_form=fit_form, weighted=weighted)
    fit_df = df[_budget_mask(df["compute"], fit_budgets)]
    diagnostics: Dict[str, FitDiagnostics] = {}
    for intervention, group in fit_df.groupby("intervention", sort=True):
        par = params[str(intervention)]
        y = group["bpb"].to_numpy(dtype=float)
        if fit_form == "compute_power_law":
            y_pred = np.asarray(bpb_power_law(group["compute"].to_numpy(dtype=float), *par), dtype=float)
            n_params = 3
        else:
            y_pred = np.asarray(
                bpb_chinchilla(
                    group["params_n"].to_numpy(dtype=float),
                    group["tokens_d"].to_numpy(dtype=float),
                    *par,
                ),
                dtype=float,
            )
            n_params = 5
        diagnostics[str(intervention)] = fit_diagnostics(y, y_pred, n_params=n_params)
    return diagnostics


def project_ranking(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    fit_form: FitForm = "compute_power_law",
    weighted: bool = False,
) -> pd.Series:
    """Return projected BPB at ``target`` sorted ascending by intervention."""

    fit_budgets = normalize_budgets(budgets, target=target)
    params = fit_all(df, fit_budgets, fit_form=fit_form, weighted=weighted)
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
            target_designs = group[["params_n", "tokens_d"]].drop_duplicates()
            if len(target_designs) != 1:
                raise ValueError(
                    f"target Chinchilla inputs for intervention {name!r} must be identical across seeds; "
                    f"found {len(target_designs)} designs"
                )
            n_target = float(target_designs["params_n"].iloc[0])
            d_target = float(target_designs["tokens_d"].iloc[0])
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


def truth_ranking_with_se(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Return measured target means with seed standard errors, sorted by mean.

    Columns: ``mean``, ``se`` (``sd/sqrt(n_seeds)``; null for single-seed
    cells), and ``n_seeds``. The index is the intervention name.
    """

    ranking = truth_ranking(df, target)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    rows: dict[str, dict[str, float]] = {}
    for intervention, group in target_df.groupby("intervention", sort=True):
        n = int(group["seed"].nunique())
        sd = float(group["bpb"].std(ddof=1)) if n >= 2 else float("nan")
        rows[str(intervention)] = {
            "mean": float(group["bpb"].mean()),
            "se": sd / np.sqrt(n) if np.isfinite(sd) else float("nan"),
            "n_seeds": n,
        }
    out = pd.DataFrame.from_dict(rows, orient="index").loc[ranking.index]
    out["n_seeds"] = out["n_seeds"].astype(int)
    return out


def truth_ties_with_winner(df: pd.DataFrame, target: float, z: float = 2.0) -> list[str]:
    """Return interventions statistically tied with the measured target winner.

    An intervention ties the winner when its mean-gap to the winner is within
    ``z`` combined standard errors. Interventions whose SE is unestimated
    (single seed) are conservatively reported as tied, because the data cannot
    rule the tie out. The winner itself is excluded from the returned list.
    """

    table = truth_ranking_with_se(df, target)
    if len(table) < 2:
        return []
    winner = table.index[0]
    winner_se = float(table.loc[winner, "se"])
    ties: list[str] = []
    for name in table.index[1:]:
        gap = float(table.loc[name, "mean"] - table.loc[winner, "mean"])
        se = float(table.loc[name, "se"])
        if not np.isfinite(winner_se) or not np.isfinite(se):
            ties.append(str(name))
            continue
        combined = float(np.sqrt(winner_se**2 + se**2))
        if gap <= z * combined:
            ties.append(str(name))
    return ties


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
