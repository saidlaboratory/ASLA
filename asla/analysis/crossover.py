"""Crossover detection and fitted crossover-budget estimates."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, List, Tuple, cast

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import ttest_ind

from asla.analysis.audit import resample_runs_by_cell, seed_noise_report
from asla.analysis.fits import fit_all, normalize_budgets, project_ranking, truth_ranking
from asla.config import GateConfig
from asla.data.schema import validate
from asla.models import FitError, FitForm, bpb_power_law


def seed_noise_band(df: pd.DataFrame, target: float, k: float = GateConfig().noise_band_k) -> float:
    """Return ``k`` times the pooled representative target-mean standard error."""

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


def benjamini_hochberg(p_values: Iterable[float], q: float = 0.05) -> list[bool]:
    """Return per-hypothesis rejections under Benjamini–Hochberg FDR control.

    Non-finite p-values are never rejected and do not count toward the number
    of tested hypotheses.
    """

    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    p = np.asarray(list(p_values), dtype=float)
    reject = np.zeros(len(p), dtype=bool)
    tested = np.where(np.isfinite(p))[0]
    m = len(tested)
    if m == 0:
        return reject.tolist()
    order = tested[np.argsort(p[tested], kind="mergesort")]
    below = np.where(p[order] <= (np.arange(1, m + 1) / m) * q)[0]
    if len(below):
        reject[order[: int(below[-1]) + 1]] = True
    return reject.tolist()


def pairwise_target_tests(df: pd.DataFrame, target: float) -> list[dict[str, Any]]:
    """Welch-test every intervention pair's seed-level BPB at the target budget.

    Pairs where either side has fewer than two seeds get ``p_value=None`` —
    the difference is untestable, not significant.
    """

    validate(df)
    target_df = df[np.isclose(df["compute"].astype(float), float(target))]
    if target_df.empty:
        raise ValueError(f"no rows found at target budget {target}")
    groups = {
        str(name): group["bpb"].to_numpy(dtype=float)
        for name, group in target_df.groupby("intervention", sort=True)
    }
    seeds = {
        str(name): int(group["seed"].nunique())
        for name, group in target_df.groupby("intervention", sort=True)
    }
    results: list[dict[str, Any]] = []
    for a, b in itertools.combinations(sorted(groups), 2):
        gap = float(np.mean(groups[a]) - np.mean(groups[b]))
        if seeds[a] < 2 or seeds[b] < 2:
            p_value: float | None = None
        else:
            p_value = float(ttest_ind(groups[a], groups[b], equal_var=False).pvalue)
        results.append({"a": a, "b": b, "true_gap": gap, "p_value": p_value})
    return results


def detect_order_flips_fdr(
    df: pd.DataFrame,
    predicted: pd.Series,
    target: float,
    q: float = 0.05,
) -> list[dict[str, Any]]:
    """Report pairs whose predicted order disagrees with the measured target order.

    ``predicted`` is any lower-is-better score per intervention (a scaling-law
    projection, a single-scale mean, ...). ``significant`` is true only when the
    measured target gap survives Benjamini-Hochberg at level ``q`` across all
    disagreeing pairs; untestable pairs (single-seed cells) carry
    ``p_value=None`` and are always reported as not significant.
    """

    truth = truth_ranking(df, target)
    names = set(predicted.index.astype(str)) & set(truth.index.astype(str))
    predicted = predicted.rename(index=str)
    tests = pairwise_target_tests(df, target)
    disagreeing = []
    for row in tests:
        a, b = row["a"], row["b"]
        if a not in names or b not in names:
            continue
        predicted_gap = float(predicted.loc[a] - predicted.loc[b])
        if predicted_gap == 0.0 or row["true_gap"] == 0.0:
            continue
        if np.sign(predicted_gap) != np.sign(row["true_gap"]):
            disagreeing.append({**row, "predicted_gap": predicted_gap})
    rejects = benjamini_hochberg(
        [row["p_value"] if row["p_value"] is not None else float("nan") for row in disagreeing],
        q=q,
    )
    for row, significant in zip(disagreeing, rejects):
        row["significant"] = bool(significant)
    return disagreeing


def detect_crossovers_fdr(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    q: float = 0.05,
    fit_form: FitForm = "compute_power_law",
) -> list[dict[str, Any]]:
    """Detect projection-versus-truth order disagreements with per-pair Welch tests under BH FDR control.

    See :func:`detect_order_flips_fdr`; the predicted score here is the
    scaling-law projection at the target.
    """

    projected = project_ranking(df, budgets, target, fit_form=fit_form)
    return detect_order_flips_fdr(df, projected, target, q=q)


def detect_single_scale_flips_fdr(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    q: float = 0.05,
) -> list[dict[str, Any]]:
    """Order flips between the seed-mean ranking at the largest fitting budget and the target.

    This is the crossover notion of single-scale ranking (DataDecide's
    baseline): a pair whose winner at the largest small budget is not the
    measured winner at the target, tested against seed noise at the target.
    """

    fit_budgets = normalize_budgets(budgets, target=target)
    largest = float(max(fit_budgets))
    rows = df[np.isclose(df["compute"].astype(float), largest)]
    if rows.empty:
        raise ValueError(f"no rows found at largest fitting budget {largest}")
    predicted = rows.groupby("intervention", sort=True)["bpb"].mean()
    return detect_order_flips_fdr(df, predicted, target, q=q)


def naive_crossover_budget(
    params_a: tuple[float, float, float],
    params_b: tuple[float, float, float],
    search_range: tuple[float, float] = (1e-9, 1e30),
) -> float | None:
    """Return the intersection of two fitted power-law curves, or ``None`` when absent.

    ``search_range`` spans compute in raw units; the default covers both O(1)
    relative units and raw FLOP counts. The difference of two three-parameter
    power laws has at most two sign changes, so a dense log grid is reliable.
    """

    lo, hi = float(search_range[0]), float(search_range[1])
    if not (0 < lo < hi) or not np.isfinite(hi):
        raise ValueError(f"search_range must be finite positive bounds with lo < hi, got {search_range}")
    n_decades = np.log10(hi) - np.log10(lo)
    grid = np.logspace(np.log10(lo), np.log10(hi), max(4000, int(300 * n_decades)))
    diff = np.asarray(bpb_power_law(grid, *params_a) - bpb_power_law(grid, *params_b), dtype=float)
    if not np.isfinite(diff).all():
        raise ValueError("fitted curves produced non-finite values in the crossover search range")
    nonzero = np.flatnonzero(diff != 0.0)
    if len(nonzero) < 2:
        return None
    bracket = next(
        (
            (int(left), int(right))
            for left, right in zip(nonzero[:-1], nonzero[1:])
            if np.signbit(diff[left]) != np.signbit(diff[right])
        ),
        None,
    )
    if bracket is None:
        return None
    left_idx, right_idx = bracket
    x1, x2 = np.log(grid[left_idx]), np.log(grid[right_idx])

    def _difference_at_log_compute(log_compute: float) -> float:
        compute = float(np.exp(log_compute))
        return float(bpb_power_law(compute, *params_a) - bpb_power_law(compute, *params_b))

    root_log = brentq(_difference_at_log_compute, x1, x2)
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
) -> dict[str, float | int | str | None]:
    """Bootstrap a fitted crossover while preserving every compute cell.

    The crossing fraction uses every requested draw as its denominator. The
    interval is conditional on a crossing being found and is omitted when too
    few roots exist; fit failures and non-crossing draws remain visible.
    """

    validate(df)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    budget_values = np.asarray(tuple(budgets), dtype=float)
    pair_df = df[
        df["intervention"].astype(str).isin([a, b])
        & df["compute"].astype(float).apply(lambda value: bool(np.any(np.isclose(value, budget_values))))
    ]
    missing = sorted({a, b} - set(pair_df["intervention"].astype(str)))
    if missing:
        raise ValueError(f"no fitting rows found for crossover interventions: {missing}")
    roots: list[float] = []
    fit_successes = 0
    for _ in range(n_boot):
        sample = resample_runs_by_cell(pair_df, rng)
        try:
            params = fit_all(sample, budget_values)
        except (FitError, ValueError):
            continue
        fit_successes += 1
        params_a = cast(tuple[float, float, float], params[a])
        params_b = cast(tuple[float, float, float], params[b])
        root = naive_crossover_budget(params_a, params_b)
        if root is not None:
            roots.append(root)
    min_roots = min(n_boot, max(10, int(np.ceil(0.25 * n_boot))))
    report: dict[str, float | int | str | None] = {
        "median": None,
        "lo": None,
        "hi": None,
        "confidence_level": 0.90,
        "interval_method": "percentile bootstrap conditional on a crossing being found",
        "minimum_roots_for_interval": int(min_roots),
        "crossing_found_fraction": float(len(roots) / n_boot),
        "crossings_found": int(len(roots)),
        "non_crossing_draws": int(fit_successes - len(roots)),
        "fit_success_fraction": float(fit_successes / n_boot),
        "fit_successes": int(fit_successes),
        "fit_failures": int(n_boot - fit_successes),
        "n_boot": int(n_boot),
    }
    if len(roots) >= min_roots:
        arr = np.asarray(roots, dtype=float)
        bounds = np.asarray(np.percentile(arr, [5.0, 50.0, 95.0]), dtype=float)
        report.update({"median": float(bounds[1]), "lo": float(bounds[0]), "hi": float(bounds[2])})
    return report


def fitted_crossover_for_pair(
    df: pd.DataFrame,
    a: str,
    b: str,
    budgets: Iterable[float],
) -> float | None:
    """Fit two interventions and return their fitted crossover budget if any."""

    params = fit_all(df[df["intervention"].isin([a, b])], budgets)
    params_a = cast(tuple[float, float, float], params[a])
    params_b = cast(tuple[float, float, float], params[b])
    return naive_crossover_budget(params_a, params_b)
