"""Crossover detection and fitted crossover-budget estimates."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from asla.analysis.fits import fit_all, project_ranking, truth_ranking
from asla.config import GateConfig
from asla.data.schema import validate
from asla.models import FitError, bpb_power_law, fit_power_law


def seed_noise_band(df: pd.DataFrame, target: float, k: float = GateConfig().noise_band_k) -> float:
    """Return ``k`` times the pooled standard error of target-budget means."""

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    variances: list[float] = []
    counts: list[int] = []
    for _, group in target_df.groupby("intervention", sort=True):
        if len(group) > 1:
            variances.append(float(group["bpb"].var(ddof=1)))
            counts.append(len(group))
    if not variances:
        return 0.0
    pooled_var = float(np.mean(variances))
    mean_n = float(np.mean(counts))
    return float(k * np.sqrt(pooled_var / mean_n))


def detect_crossovers(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
) -> List[Tuple[str, str, float]]:
    """Detect significant pairs whose projected and true target orders disagree."""

    projected = project_ranking(df, budgets, target)
    truth = truth_ranking(df, target)
    band = seed_noise_band(df, target)
    names = sorted(set(projected.index.astype(str)) & set(truth.index.astype(str)))
    found: list[tuple[str, str, float]] = []
    for a, b in itertools.combinations(names, 2):
        projected_gap = float(projected.loc[a] - projected.loc[b])
        true_gap = float(truth.loc[a] - truth.loc[b])
        if abs(true_gap) <= band:
            continue
        if np.sign(projected_gap) != np.sign(true_gap):
            found.append((a, b, true_gap))
    return found


def crossover_budget(params_a: tuple[float, float, float], params_b: tuple[float, float, float]) -> float | None:
    """Return the fitted curve intersection budget, or ``None`` when absent."""

    grid = np.logspace(-3, 6, 4000)
    diff = np.asarray(bpb_power_law(grid, *params_a) - bpb_power_law(grid, *params_b), dtype=float)
    exact = np.where(np.isclose(diff, 0.0, atol=1e-10))[0]
    if len(exact):
        return float(grid[int(exact[0])])
    signs = np.sign(diff)
    idxs = np.where(signs[:-1] * signs[1:] < 0)[0]
    if len(idxs) == 0:
        return None
    i = int(idxs[0])
    x1, x2 = np.log(grid[i]), np.log(grid[i + 1])
    y1, y2 = diff[i], diff[i + 1]
    if y2 == y1:
        return float(grid[i])
    root_log = x1 - y1 * (x2 - x1) / (y2 - y1)
    return float(np.exp(root_log))


def crossover_budget_ci(
    df: pd.DataFrame,
    a: str,
    b: str,
    budgets: Iterable[float],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float] | None:
    """Bootstrap the fitted crossover budget for two interventions."""

    validate(df)
    budget_values = np.asarray(tuple(budgets), dtype=float)
    roots: list[float] = []
    for _ in range(n_boot):
        params: dict[str, tuple[float, float, float]] = {}
        for name in (a, b):
            group = df[(df["intervention"] == name) & df["compute"].apply(lambda c: np.any(np.isclose(c, budget_values)))]
            if group.empty:
                continue
            idx = rng.integers(0, len(group), size=len(group))
            sample = group.iloc[idx]
            try:
                params[name] = fit_power_law(sample["compute"].to_numpy(float), sample["bpb"].to_numpy(float))
            except FitError:
                continue
        if a in params and b in params:
            root = crossover_budget(params[a], params[b])
            if root is not None:
                roots.append(root)
    if not roots:
        return None
    arr = np.asarray(roots, dtype=float)
    lo, med, hi = np.percentile(arr, [5.0, 50.0, 95.0])
    return float(med), float(lo), float(hi)


def fitted_crossover_for_pair(
    df: pd.DataFrame,
    a: str,
    b: str,
    budgets: Iterable[float],
) -> float | None:
    """Fit two interventions and return their fitted crossover budget if any."""

    params = fit_all(df[df["intervention"].isin([a, b])], budgets)
    return crossover_budget(params[a], params[b])

