"""Leave-one-budget-out conformal intervals for scaling-law projections.

The bootstrap gate's intervals inherit the parametric family's optimism: when
every resample fits the same wrong curve, the band can be arbitrarily narrow
around a wrong projection. The conformal alternative scores how badly the
fitted family *actually* predicts held-out budgets — refit without each
ladder budget, record the absolute error against that budget's seed mean —
and wraps the target projection in the empirical quantile of those scores.

Honesty note: exact conformal coverage guarantees need exchangeable scores;
extrapolating beyond the ladder is not exchangeable with interpolating inside
it, so the guarantee here is heuristic. That is precisely why the benchmark
includes an empirical calibration study comparing nominal and observed
coverage for both interval sources (`scripts/run_calibration_study.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets, project_ranking
from asla.data.schema import validate
from asla.models import FitError, bpb_power_law, fit_power_law


def conformal_projection_interval(
    compute: np.ndarray,
    bpb: np.ndarray,
    target: float,
    alpha: float = 0.1,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for a power-law projection at ``target``.

    Scores are leave-one-budget-out: for each distinct budget, refit on the
    remaining budgets and record the absolute error against the held-out
    budget's mean BPB. The interval is the point projection plus/minus the
    conformal quantile ``ceil((R+1)(1-alpha))/R`` of the scores. At least
    four distinct budgets are required so each refit keeps three.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    x = np.asarray(compute, dtype=float)
    y = np.asarray(bpb, dtype=float)
    distinct = np.unique(x)
    if len(distinct) < 4:
        raise FitError(f"conformal interval requires at least 4 distinct compute budgets, got {len(distinct)}")
    params = fit_power_law(x, y)
    point = float(bpb_power_law(float(target), *params))
    scores: list[float] = []
    for budget in distinct:
        held = np.isclose(x, budget)
        try:
            loo_params = fit_power_law(x[~held], y[~held])
        except FitError:
            continue
        pred = float(bpb_power_law(float(budget), *loo_params))
        scores.append(abs(pred - float(np.mean(y[held]))))
    if len(scores) < 3:
        raise FitError(f"too few leave-one-budget-out fits succeeded: {len(scores)}")
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - alpha)))
    q = float(ordered[min(rank, len(ordered)) - 1])
    return point, point - q, point + q


def conformal_gate_pick(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    intermediate_budget: float,
    alpha: float = 0.1,
) -> str:
    """Gate the top-2 comparison on conformal-interval overlap.

    Same escalation contract as the bootstrap gate: when the top two
    projections' intervals overlap, refit both on the ladder extended by the
    intermediate budget and take that projection's winner; the escalation
    fails loudly when intermediate runs are missing. Deterministic — no RNG.
    """

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    ranking = project_ranking(df, fit_budgets, target)
    if len(ranking) < 2:
        return str(ranking.index[0])
    top1, top2 = str(ranking.index[0]), str(ranking.index[1])
    budget_values = np.asarray(fit_budgets, dtype=float)
    intervals: dict[str, tuple[float, float, float]] = {}
    for name in (top1, top2):
        group = df[df["intervention"].astype(str) == name]
        mask = group["compute"].astype(float).apply(lambda c: bool(np.any(np.isclose(c, budget_values))))
        rows = group[mask]
        intervals[name] = conformal_projection_interval(
            rows["compute"].to_numpy(dtype=float),
            rows["bpb"].to_numpy(dtype=float),
            target,
            alpha=alpha,
        )
    separated = intervals[top1][2] < intervals[top2][1]
    if separated:
        return top1
    extended = tuple(sorted(set((*fit_budgets, float(intermediate_budget)))))
    extended_df = df[df["intervention"].isin([top1, top2])]
    at_intermediate = extended_df[np.isclose(extended_df["compute"].astype(float), float(intermediate_budget))]
    missing = sorted({top1, top2} - set(at_intermediate["intervention"].astype(str)))
    if missing:
        raise ValueError(
            f"gate escalation needs runs at intermediate budget {intermediate_budget} "
            f"for interventions {missing}; none were found"
        )
    extended_ranking = project_ranking(extended_df, extended, target)
    return str(extended_ranking.index[0])
