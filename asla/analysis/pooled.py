"""Partial-pooling projection estimators: shared exponent and hierarchical shrinkage.

Motivation. The Phase 1 audit found that the excess mis-selection of
scaling-law projection over single-scale ranking is fit/extrapolation
*variance*, not crossover bias, and that adding model flexibility (a
three-family ensemble) makes selection worse. The symmetric prediction is that
*removing* flexibility by pooling information across interventions should
reduce projection variance and improve decisions. These estimators test it.

Practitioners already use the least-flexible member of this family as a
reporting device: the Olmo Hybrid work fits a shared exponent across
architectures because unconstrained per-architecture fits have confidence
intervals too wide to compare. Here it is evaluated as a *decision rule*.

Estimators, in decreasing flexibility:

* ``fit_per_intervention`` - the existing plain fit, no pooling (reference).
* ``fit_shrunk`` with ``strength=lam`` - each intervention's exponent is pulled
  a fraction ``lam`` of the way toward the pooled exponent, then the remaining
  parameters are refitted with the exponent held fixed. ``lam=0`` recovers the
  plain fit; ``lam=1`` is complete pooling on the exponent.
* ``fit_shared_exponent`` - one exponent for all interventions, fitted jointly;
  each intervention keeps its own floor and coefficient. Equivalent to
  ``fit_shrunk(strength=1.0)`` but fitted jointly rather than by shrink-then-refit.

The empirical-Bayes shrinkage strength is estimated from the data as the noise
share of the observed exponent spread,

    lam = within_variance / (within_variance + between_variance),

where ``between`` is the sample variance of the per-intervention exponents and
``within`` is the mean sampling variance of a single exponent, estimated by a
cell-wise seed bootstrap. When the observed spread is entirely explained by
sampling noise, ``lam -> 1`` (pool completely); when the exponents are
precisely estimated and genuinely different, ``lam -> 0`` (do not pool). This
is the James-Stein logic applied to the exponent only; floors and coefficients
are never pooled, because they carry the intervention's actual quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from asla.analysis import splits
from asla.analysis.fits import normalize_budgets
from asla.data.schema import validate
from asla.models import FitError, bpb_power_law


@dataclass(frozen=True)
class PooledFit:
    """A pooled projection fit: per-intervention parameters plus the pooling used."""

    params: dict[str, tuple[float, float, float]]
    shrinkage: float
    pooled_exponent: float
    per_intervention_exponent: dict[str, float]
    between_variance: float
    within_variance: float


def _cell_means(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-intervention (compute, mean value) arrays over the fitting budgets."""

    values = np.asarray(budgets, dtype=float)
    mask = df["compute"].astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))
    fit_df = df[mask]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, group in fit_df.groupby("intervention", sort=True):
        cells = group.groupby("compute", sort=True)["bpb"].mean()
        out[str(name)] = (cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))
    return out


def fit_per_intervention(df: pd.DataFrame, budgets: Iterable[float]) -> dict[str, tuple[float, float, float]]:
    """Plain per-intervention power-law fit on cell means (the no-pooling reference)."""

    from asla.models import fit_power_law

    fit_budgets = normalize_budgets(budgets)
    out: dict[str, tuple[float, float, float]] = {}
    for name, (x, y) in _cell_means(df, fit_budgets).items():
        out[name] = fit_power_law(x, y)
    return out


def _fit_with_fixed_exponent(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Fit ``E + A C^-alpha`` with ``alpha`` held fixed; linear in (E, A) so solved by least squares."""

    basis = np.vstack([np.ones_like(x), x ** (-alpha)]).T
    solution, *_ = np.linalg.lstsq(basis, y, rcond=None)
    e, a = float(solution[0]), float(solution[1])
    # Keep the curve physically meaningful: a non-negative floor and a
    # non-negative amplitude. Clamping to the boundary and re-solving the free
    # parameter is what a bounded optimizer would do at the same point.
    if a < 0.0:
        a = 0.0
        e = float(np.mean(y))
    if e < 0.0:
        e = 0.0
        denominator = float(np.sum((x ** (-alpha)) ** 2))
        a = float(np.sum(y * x ** (-alpha)) / denominator) if denominator > 0 else 0.0
    return (e, a, float(alpha))


def fit_shared_exponent(df: pd.DataFrame, budgets: Iterable[float]) -> dict[str, tuple[float, float, float]]:
    """Fit one exponent shared by every intervention, with per-intervention floor and coefficient.

    The shared exponent is found by a one-dimensional search that minimises the
    total squared residual after solving each intervention's linear (E, A)
    exactly at that exponent. This is the profile likelihood in ``alpha``, and
    it is far more stable than optimising 2K+1 parameters jointly.
    """

    fit_budgets = normalize_budgets(budgets)
    cells = _cell_means(df, fit_budgets)
    if not cells:
        raise FitError("shared-exponent fit requires at least one intervention")
    scale = float(np.min(np.concatenate([x for x, _ in cells.values()])))
    scaled = {name: (x / scale, y) for name, (x, y) in cells.items()}

    def total_residual(alpha: float) -> float:
        total = 0.0
        for x, y in scaled.values():
            e, a, _ = _fit_with_fixed_exponent(x, y, alpha)
            total += float(np.sum((e + a * x ** (-alpha) - y) ** 2))
        return total

    grid = np.geomspace(1e-4, 2.0, 240)
    losses = np.asarray([total_residual(a) for a in grid], dtype=float)
    best = int(np.argmin(losses))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, len(grid) - 1)]
    # golden-section refine inside the bracketing interval
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if total_residual(c) < total_residual(d):
            hi = d
        else:
            lo = c
    alpha = float((lo + hi) / 2.0)
    out: dict[str, tuple[float, float, float]] = {}
    for name, (x, y) in scaled.items():
        e, a_scaled, _ = _fit_with_fixed_exponent(x, y, alpha)
        out[name] = (e, a_scaled * scale**alpha, alpha)
    return out


def exponent_variance_components(
    df: pd.DataFrame,
    budgets: Iterable[float],
    n_boot: int = 40,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Return (between_variance, within_variance, per-intervention exponents).

    ``between`` is the sample variance of the fitted exponents across
    interventions. ``within`` is the mean sampling variance of one
    intervention's exponent, estimated by resampling seeds inside every
    ``(intervention, compute)`` cell. Both are needed for the empirical-Bayes
    shrinkage strength.
    """

    fit_budgets = normalize_budgets(budgets)
    generator = np.random.default_rng(0) if rng is None else rng
    params = fit_per_intervention(df, fit_budgets)
    exponents = {name: float(par[2]) for name, par in params.items()}
    between = float(np.var(np.asarray(list(exponents.values()), dtype=float), ddof=1)) if len(exponents) > 1 else 0.0

    values = np.asarray(fit_budgets, dtype=float)
    mask = df["compute"].astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))
    fit_df = df[mask]
    within_estimates: list[float] = []
    for name, group in fit_df.groupby("intervention", sort=True):
        draws: list[float] = []
        for _ in range(n_boot):
            pieces = []
            for _, cell in group.groupby("compute", sort=True):
                index = generator.integers(0, len(cell), len(cell))
                pieces.append(cell.iloc[index])
            resampled = pd.concat(pieces, ignore_index=True)
            resampled["seed"] = np.arange(len(resampled))
            try:
                cells = resampled.groupby("compute", sort=True)["bpb"].mean()
                from asla.models import fit_power_law

                draws.append(float(fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))[2]))
            except (FitError, ValueError, np.linalg.LinAlgError):
                continue
        if len(draws) > 1:
            within_estimates.append(float(np.var(np.asarray(draws, dtype=float), ddof=1)))
    within = float(np.mean(within_estimates)) if within_estimates else 0.0
    return between, within, exponents


def empirical_bayes_strength(between_variance: float, within_variance: float) -> float:
    """Shrinkage strength ``within / (within + between)``, clipped to [0, 1].

    All spread explained by sampling noise gives 1 (pool completely); precisely
    estimated and genuinely different exponents give 0 (do not pool).
    """

    total = within_variance + between_variance
    if total <= 0:
        return 1.0
    return float(min(1.0, max(0.0, within_variance / total)))


def fit_shrunk(
    df: pd.DataFrame,
    budgets: Iterable[float],
    strength: float | None = None,
    n_boot: int = 40,
    rng: np.random.Generator | None = None,
) -> PooledFit:
    """Shrink each exponent toward the pooled exponent, then refit floor and coefficient.

    ``strength=None`` estimates the strength from the data by empirical Bayes.
    ``strength=0`` reproduces the plain fit; ``strength=1`` fixes every exponent
    at the pooled value.
    """

    validate(df)
    splits.assert_not_test(df)
    fit_budgets = normalize_budgets(budgets)
    between, within, exponents = exponent_variance_components(df, fit_budgets, n_boot=n_boot, rng=rng)
    lam = empirical_bayes_strength(between, within) if strength is None else float(strength)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"shrinkage strength must lie in [0, 1], got {lam}")
    pooled_exponent = float(np.mean(np.asarray(list(exponents.values()), dtype=float)))
    cells = _cell_means(df, fit_budgets)
    scale = float(np.min(np.concatenate([x for x, _ in cells.values()])))
    params: dict[str, tuple[float, float, float]] = {}
    for name, (x, y) in cells.items():
        alpha = (1.0 - lam) * exponents[name] + lam * pooled_exponent
        e, a_scaled, _ = _fit_with_fixed_exponent(x / scale, y, alpha)
        params[name] = (e, a_scaled * scale**alpha, float(alpha))
    return PooledFit(
        params=params,
        shrinkage=lam,
        pooled_exponent=pooled_exponent,
        per_intervention_exponent=exponents,
        between_variance=between,
        within_variance=within,
    )


def project(params: Mapping[str, tuple[float, float, float]], target: float) -> pd.Series:
    """Project fitted parameters to the target budget, sorted ascending (lower is better)."""

    values = {name: float(bpb_power_law(target, *par)) for name, par in params.items()}
    return pd.Series(values).sort_values(kind="mergesort")


def shared_exponent_ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
    """Ranker: complete pooling on the exponent."""

    validate(df)
    splits.assert_not_test(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    return project(fit_shared_exponent(df, fit_budgets), target)


def make_shrinkage_ranker(strength: float | None = None, n_boot: int = 40, seed: int = 0):
    """Return a ranker using hierarchical shrinkage at a fixed or data-estimated strength."""

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        fit_budgets = normalize_budgets(budgets, target=target)
        fit = fit_shrunk(df, fit_budgets, strength=strength, n_boot=n_boot, rng=np.random.default_rng(seed))
        return project(fit.params, target)

    label = "eb" if strength is None else f"{strength:g}"
    _ranker.__name__ = f"shrinkage_ranker_{label}"
    return _ranker


def pooled_ranker_suite(strengths: Iterable[float], n_boot: int = 40, seed: int = 0) -> dict[str, object]:
    """Rankers spanning no pooling to complete pooling, for the flexibility sweep."""

    suite: dict[str, object] = {}
    for strength in strengths:
        suite[f"shrinkage_{strength:g}"] = make_shrinkage_ranker(strength, n_boot=n_boot, seed=seed)
    suite["shrinkage_eb"] = make_shrinkage_ranker(None, n_boot=n_boot, seed=seed)
    suite["shared_exponent"] = shared_exponent_ranker
    return suite


__all__ = [
    "PooledFit",
    "empirical_bayes_strength",
    "exponent_variance_components",
    "fit_per_intervention",
    "fit_shared_exponent",
    "fit_shrunk",
    "make_shrinkage_ranker",
    "pooled_ranker_suite",
    "project",
    "shared_exponent_ranker",
]
