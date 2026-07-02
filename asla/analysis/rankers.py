"""Ranker interfaces for comparing decision rules."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from asla.analysis.ensemble import ensemble_rank
from asla.analysis.fits import normalize_budgets, project_ranking
from asla.analysis.gate import gate_pick
from asla.models import FitForm

Ranker = Callable[[pd.DataFrame, tuple[float, ...], float], pd.Series]


def ensemble_ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
    """Rank interventions by misspecification-aware ensemble projection."""

    return ensemble_rank(df, budgets, target)


def projection_ranker(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    fit_form: FitForm = "compute_power_law",
    weighted: bool = False,
) -> pd.Series:
    """Rank interventions by scaling-law projection at the target budget."""

    return project_ranking(df, budgets, target, fit_form=fit_form, weighted=weighted)


def make_projection_ranker(fit_form: FitForm = "compute_power_law", weighted: bool = False) -> Ranker:
    """Return a projection ranker bound to a fit form and weighting choice."""

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        return projection_ranker(df, budgets, target, fit_form=fit_form, weighted=weighted)

    _ranker.__name__ = f"projection_ranker_{fit_form}" + ("_weighted" if weighted else "")
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
    seed: int,
) -> Ranker:
    """Return a ranker whose winner is selected by the uncertainty gate.

    A fresh generator is derived from ``seed`` on every call so the ranker is
    deterministic regardless of how many times or in what order it runs —
    sharing one mutating generator across rankers and bootstrap replicates
    would make results depend on call order.
    """

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        rng = np.random.default_rng(int(seed))
        pick = gate_pick(df, budgets, target, intermediate_budget, tau, n_boot, rng)
        projection = project_ranking(df, budgets, target)
        if pick not in projection.index:
            raise ValueError(f"gate picked unknown intervention {pick!r}")
        ordered = projection.copy()
        ordered.loc[pick] = float(ordered.min()) - 1e-12
        return ordered.sort_values(kind="mergesort")

    _ranker.__name__ = "gate_ranker"
    return _ranker
