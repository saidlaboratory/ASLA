"""Crossover detection and fitted crossover-budget estimates."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from asla.analysis.audit import seed_noise_report
from asla.analysis.fits import fit_all, project_ranking, truth_ranking
from asla.config import GateConfig
from asla.data.schema import validate
from asla.models import FitError, FitForm, bpb_power_law, fit_power_law


def seed_noise_band(df: pd.DataFrame, target: float, k: float = GateConfig().noise_band_k) -> float:
    """Return ``k`` times the pooled standard error of target-budget means."""

    band = seed_noise_report(df, target, k=k)["noise_band"]
    return float("inf") if band is None else float(band)


def detect_crossovers(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    fit_form: FitForm = "compute_power_law",
) -> List[Tuple[str, str, float]]:
    """Detect significant pairs whose projected and true target orders disagree."""

    projected = project_ranking(df, budgets, target, fit_form=fit_form)
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


def naive_crossover_budget(params_a: tuple[float, float, float], params_b: tuple[float, float, float]) -> float | None:
    """Return the intersection of two fitted power-law curves, or ``None`` when absent."""

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


def crossover_budget(params_a: tuple[float, float, float], params_b: tuple[float, float, float]) -> float | None:
    """Backward-compatible alias for :func:`naive_crossover_budget`."""

    return naive_crossover_budget(params_a, params_b)


def mechanism_crossover_budget(*args: object, **kwargs: object) -> float:
    """Stub for the future muP-derived mechanism crossover-budget predictor.

    This later-phase predictor is intentionally not implemented here and must
    not fabricate a number.
    """

    raise NotImplementedError("mechanism_crossover_budget is a future muP-based predictor and is not implemented")


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
            root = naive_crossover_budget(params[a], params[b])
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
    return naive_crossover_budget(params[a], params[b])
