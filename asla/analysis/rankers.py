"""Ranker interfaces for comparing decision rules."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets, project_ranking
from asla.analysis.gate import gate_pick
from asla.models import FitForm

Ranker = Callable[[pd.DataFrame, tuple[float, ...], float], pd.Series]


def projection_ranker(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    fit_form: FitForm = "compute_power_law",
) -> pd.Series:
    """Rank interventions by scaling-law projection at the target budget."""

    return project_ranking(df, budgets, target, fit_form=fit_form)


def make_projection_ranker(fit_form: FitForm = "compute_power_law") -> Ranker:
    """Return a projection ranker bound to a fit form."""

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        return projection_ranker(df, budgets, target, fit_form=fit_form)

    _ranker.__name__ = f"projection_ranker_{fit_form}"
    return _ranker


def single_scale_ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
    """Rank by mean BPB at the largest fitting budget."""

    fit_budgets = normalize_budgets(budgets, target=target)
    largest = float(max(fit_budgets))
    rows = df[np.isclose(df["compute"].astype(float), largest)]
    if rows.empty:
        raise ValueError(f"no rows found at largest fitting budget {largest}")
    return rows.groupby("intervention", sort=True)["bpb"].mean().sort_values(kind="mergesort")


def make_gate_ranker(
    intermediate_budget: float,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Ranker:
    """Return a ranker whose winner is selected by the uncertainty gate."""

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        pick = gate_pick(df, budgets, target, intermediate_budget, tau, n_boot, rng)
        projection = project_ranking(df, budgets, target)
        if pick not in projection.index:
            raise ValueError(f"gate picked unknown intervention {pick!r}")
        ordered = projection.copy()
        ordered.loc[pick] = float(ordered.min()) - 1e-12
        return ordered.sort_values(kind="mergesort")

    _ranker.__name__ = "gate_ranker"
    return _ranker
