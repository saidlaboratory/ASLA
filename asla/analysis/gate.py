"""Projection baselines and the uncertainty gate."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets, project_ranking, projection_with_uncertainty, truth_ranking
from asla.analysis.metrics import decision_metrics
from asla.config import AuditConfig
from asla.data.schema import validate
from asla.data.synthetic import true_ranking as synthetic_true_ranking


def plain_projection_pick(df: pd.DataFrame, budgets: Iterable[float], target: float) -> str:
    """Pick the intervention with the lowest projected target BPB."""

    ranking = project_ranking(df, budgets, target)
    return str(ranking.index[0])


def largest_single_run_pick(df: pd.DataFrame, budgets: Iterable[float], target: float) -> str:
    """Pick by mean BPB at the largest fitting budget."""

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    largest = float(max(fit_budgets))
    rows = df[np.isclose(df["compute"].astype(float), largest)]
    if rows.empty:
        raise ValueError(f"no rows found at largest fitting budget {largest}")
    means = rows.groupby("intervention", sort=True)["bpb"].mean().sort_values(kind="mergesort")
    return str(means.index[0])


def gate_pick_detailed(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    intermediate_budget: float,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Apply the uncertainty gate and return the pick with its decision path.

    The result maps ``pick`` to the selected intervention, ``escalated`` to
    whether the intermediate budget was consulted, and ``top_pair`` to the two
    interventions the gate compared (``None`` with a single intervention).
    """

    validate(df)
    ranking = project_ranking(df, budgets, target)
    if len(ranking) < 2:
        return {"pick": str(ranking.index[0]), "escalated": False, "top_pair": None}
    top1, top2 = str(ranking.index[0]), str(ranking.index[1])
    p1 = projection_with_uncertainty(df[df["intervention"] == top1], budgets, target, n_boot, rng)
    p2 = projection_with_uncertainty(df[df["intervention"] == top2], budgets, target, n_boot, rng)
    denom = float(np.sqrt(p1[1] ** 2 + p2[1] ** 2))
    gap = float(ranking.iloc[1] - ranking.iloc[0])
    g = np.inf if denom == 0.0 and gap > 0 else gap / denom if denom > 0 else 0.0
    if g >= tau:
        return {"pick": top1, "escalated": False, "top_pair": (top1, top2)}
    extended = tuple(sorted(set((*tuple(float(b) for b in budgets), float(intermediate_budget)))))
    extended_df = df[df["intervention"].isin([top1, top2])]
    at_intermediate = extended_df[np.isclose(extended_df["compute"].astype(float), float(intermediate_budget))]
    missing = sorted({top1, top2} - set(at_intermediate["intervention"].astype(str)))
    if missing:
        raise ValueError(
            f"gate escalation needs runs at intermediate budget {intermediate_budget} "
            f"for interventions {missing}; none were found"
        )
    pick = plain_projection_pick(extended_df, extended, target)
    return {"pick": pick, "escalated": True, "top_pair": (top1, top2)}


def gate_pick(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    intermediate_budget: float,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> str:
    """Apply the projection uncertainty gate and return the selected intervention."""

    return str(gate_pick_detailed(df, budgets, target, intermediate_budget, tau, n_boot, rng)["pick"])


def _regret_for_pick(pick: str, truth: pd.Series) -> float:
    return float(truth.loc[pick] - truth.min())


def evaluation_truth(df: pd.DataFrame, target: float) -> pd.Series:
    """Use noiseless synthetic truth when available, otherwise measured target means."""

    try:
        return synthetic_true_ranking(df, target)
    except ValueError:
        return truth_ranking(df, target)


_evaluation_truth = evaluation_truth


def monte_carlo(
    scenario_fn: Callable[[np.random.Generator, AuditConfig | None], pd.DataFrame],
    config: AuditConfig | None = None,
    n_trials: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate plain projection, largest-budget, and gate decisions over trials."""

    cfg = config or AuditConfig()
    trials = cfg.counts.n_trials if n_trials is None else n_trials
    if trials <= 0:
        raise ValueError("n_trials must be positive")
    root_rng = rng or np.random.default_rng(cfg.seeds.seed)
    regret = {"plain": [], "largest": [], "gate": []}
    wrong = {"plain": [], "largest": [], "gate": []}
    for _ in range(trials):
        data_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        gate_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        df = scenario_fn(data_rng, cfg)
        truth = _evaluation_truth(df, cfg.budgets.target)
        true_best = str(truth.index[0])
        plain = plain_projection_pick(df, cfg.budgets.fit, cfg.budgets.target)
        largest = largest_single_run_pick(df, cfg.budgets.fit, cfg.budgets.target)
        gate = gate_pick(
            df,
            cfg.budgets.fit,
            cfg.budgets.target,
            cfg.budgets.intermediate,
            cfg.gate.tau,
            cfg.gate.n_boot,
            gate_rng,
        )
        for name, pick in (("plain", plain), ("largest", largest), ("gate", gate)):
            regret[name].append(_regret_for_pick(pick, truth))
            wrong[name].append(float(pick != true_best))

    out: dict[str, dict[str, float]] = {}
    for name in ("plain", "largest", "gate"):
        out[name] = {
            "mean_regret": float(np.mean(regret[name])),
            "wrong_pick_rate": float(np.mean(wrong[name])),
        }
    return out


def decision_report(df: pd.DataFrame, budgets: Iterable[float], target: float, k: int = 2) -> dict[str, float]:
    """Return decision metrics for projected versus measured target rankings."""

    return decision_metrics(project_ranking(df, budgets, target), truth_ranking(df, target), k=k)
