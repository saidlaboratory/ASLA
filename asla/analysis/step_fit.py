"""Window-subset fixed-target slopes and a step-term projection, for the config-step analysis
(PREDICTIONS_TASK_CONFIG_STEPS.md).

``subset_stat`` restricts the frozen pooled fixed-target slope to any set of (window, target) designs, with
slices (target, rung count) of at least three designs recomputed inside the set, and returns the estimate,
its leave-one-recipe-out replicates and the jackknife SE, so contrasts and ratios between subsets share
replicates. ``fit_power_law_step`` is fit_power_law's model, bounds and start plus one level shift per regime
boundary inside the window; ``step_kernel`` is the frozen kernel with projections from that fit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

from asla.analysis.geometry_scan import Scanner
from asla.analysis.target_scoring import expected_charges
from asla.models import FitError, adaptive_alpha_floor

DELTA_BOUND = 0.5


def fit_power_law_step(compute: np.ndarray, bpb: np.ndarray, boundaries: Sequence[float]) -> tuple[float, float, float, tuple[float, ...]]:
    """(E, A, alpha, deltas) for E + A C^-alpha + sum_b delta_b 1[C >= C_b]; A in raw compute units."""

    x = np.asarray(compute, dtype=float)
    y = np.asarray(bpb, dtype=float)
    k = len(boundaries)
    if len(np.unique(x)) < 3 + k + 1:
        raise FitError("step fit needs at least one residual degree of freedom")
    x_ref = float(np.min(x))
    xs = x / x_ref
    ind = np.array([[1.0 if c >= b - 1e-9 * b else 0.0 for b in boundaries] for c in x]).reshape(len(x), k)
    y_min = float(np.min(y))
    e0 = max(0.0, min(y_min * 0.9, y_min - 1e-6))
    a0 = min(5.0, max(0.01, float(np.max(y) - e0)))
    floor = adaptive_alpha_floor(xs, y)

    def model(i, e, a, alpha, *d):
        idx = np.asarray(i, dtype=int)
        return e + a * xs[idx] ** (-alpha) + ind[idx] @ np.asarray(d, dtype=float)

    p0 = (e0, a0, max(0.3, floor * 2.0), *([0.0] * k))
    lower = [0.0, 0.0, floor, *([-DELTA_BOUND] * k)]
    upper = [y_min, 5.0, 2.0, *([DELTA_BOUND] * k)]
    try:
        p, _ = curve_fit(model, np.arange(len(x)), y, p0=p0, bounds=(lower, upper), maxfev=50000)
    except Exception as exc:  # pragma: no cover
        raise FitError(f"step fit failed: {exc}") from exc
    e, a_s, alpha = (float(v) for v in p[:3])
    return e, a_s * x_ref**alpha, alpha, tuple(float(v) for v in p[3:])


def step_kernel(sc: Scanner, window: tuple[int, ...], t: int, boundaries: Sequence[float]) -> np.ndarray:
    """Frozen kernel at target t, with each recipe projected by the step fit over the window."""

    budgets = sc.compute[list(window)]
    proj = {}
    for r in sc.recipes:
        try:
            e, a, alpha, d = fit_power_law_step(budgets, sc.means[r][list(window)], boundaries)
            proj[r] = float(e + a * sc.compute[t] ** (-alpha) + sum(d))
        except (FitError, RuntimeError, ValueError):
            proj[r] = float("nan")
    single = {r: float(sc.means[r][window[-1]]) for r in sc.recipes}
    pg = {(a, b): proj[a] - proj[b] for a, b in sc.pairs}
    sg = {(a, b): single[a] - single[b] for a, b in sc.pairs}
    ev = sc.evidence(t)
    left, right = expected_charges(pg, ev), expected_charges(sg, ev)
    k = np.full(len(sc.pairs), np.nan)
    for p in left:
        if p in right and p in sc.pair_index:
            k[sc.pair_index[p]] = 100 * (left[p] - right[p])
    return k


def subset_stat(sc: Scanner, ds: list[tuple[tuple[int, ...], int]], kernels: np.ndarray | None = None) -> dict:
    """Pooled fixed-target slope over the designs ds, with LORO replicates (NaN estimate when no slice)."""

    groups = sc.slices(ds)
    nan = {"estimate": float("nan"), "se": float("nan"), "loo": np.full(len(sc.recipes), np.nan), "n_designs": 0, "n_slices": 0}
    if not groups:
        return nan
    x = sc.lever(ds)
    kern = kernels if kernels is not None else np.array([sc.kernel(w, t) for w, t in ds])
    valid = np.isfinite(kern)
    est = Scanner.pooled_slope(np.where(valid, kern, 0.0).sum(axis=1) / valid.sum(axis=1), x, groups)
    loo = np.empty(len(sc.recipes))
    for r in range(len(sc.recipes)):
        keep = valid & ~sc.member[r][None, :]
        loo[r] = Scanner.pooled_slope(np.where(keep, kern, 0.0).sum(axis=1) / keep.sum(axis=1), x, groups)
    return {"estimate": est, "se": jk_se(loo), "loo": loo, "n_designs": len({i for g in groups for i in g}), "n_slices": len(groups)}


def jk_se(loo: np.ndarray) -> float:
    v = np.asarray(loo, dtype=float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


__all__ = ["fit_power_law_step", "jk_se", "step_kernel", "subset_stat"]
