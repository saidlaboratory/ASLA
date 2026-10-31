"""Projection's excess over single-scale ranking as a function of lever arm, at fixed targets (PREDICTIONS_TASK_CURVE.md).

For the designs at each fixed target (every fit window below it, any rung count), the design excess is the
mean expected-error kernel in points. Two descriptions per target:

* binned: the mean excess of the designs whose lever arm falls in each pre-specified bin;
* saturating: least-squares fit of excess = s tau (1 - exp(-L / tau)), L in decades, over the target's designs;
  s is the initial slope (points per decade as L -> 0) and tau the saturation scale in decades (tau at its
  upper bound means no saturation is detectable over the lever arms covered: the curve is linear there).

A common curve fits the same form to all targets' designs together. Inference is the leave-one-recipe-out
jackknife over every statistic. The primary coincidence test compares the binned curves: in every bin that two
or more targets cover, each target's bin mean against the mean across those targets, as a Wald test with the
jackknife covariance. A secondary Wald test compares the initial slopes s across targets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import optimize, stats

from asla.analysis.confound import WindowDesign

BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5)
BOUNDS = ([-100.0, 0.1], [100.0, 20.0])
TOLERANCE = 3.0  # points: a target's curve must stay this close to the common curve over its lever range


def saturating(lever: np.ndarray, slope: float, tau: float) -> np.ndarray:
    return slope * tau * (1.0 - np.exp(-np.asarray(lever, dtype=float) / tau))


def fit(lever: np.ndarray, excess: np.ndarray) -> tuple[float, float]:
    lever, excess = np.asarray(lever, dtype=float), np.asarray(excess, dtype=float)
    ok = np.isfinite(excess)
    slope0 = float(np.polyfit(lever[ok], excess[ok], 1)[0]) if ok.sum() >= 2 else 1.0
    p0 = (float(np.clip(slope0, -99.0, 99.0)), 2.0)
    try:
        (sl, tau), _ = optimize.curve_fit(saturating, lever[ok], excess[ok], p0=p0, bounds=BOUNDS, maxfev=20000)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(sl), float(tau)


def statistics(all_designs: Sequence[WindowDesign], excess: np.ndarray, targets: Sequence[int]) -> dict[str, float]:
    lever = np.array([d.lever for d in all_designs])
    tgt = np.array([d.target for d in all_designs])
    out: dict[str, float] = {}
    keep = np.isin(tgt, list(targets))
    out["common_s"], out["common_tau"] = fit(lever[keep], excess[keep])
    for t in targets:
        m = tgt == t
        out[f"s_{t}"], out[f"tau_{t}"] = fit(lever[m], excess[m])
        for j in range(len(BINS) - 1):
            b = m & (lever >= BINS[j]) & (lever < BINS[j + 1])
            out[f"bin_{t}_{j}"] = float(np.mean(excess[b])) if b.any() else float("nan")
    return out


def excess_vector(all_designs: Sequence[WindowDesign], drop: str | None = None) -> np.ndarray:
    return np.array(
        [np.mean([v for p, v in d.kernel.items() if drop is None or drop not in p]) for d in all_designs]
    )


def jackknife(all_designs: Sequence[WindowDesign], targets: Sequence[int]) -> dict:
    full = statistics(all_designs, excess_vector(all_designs), targets)
    names = sorted({n for d in all_designs for pair in d.kernel for n in pair})
    reps = [statistics(all_designs, excess_vector(all_designs, n), targets) for n in names]
    n = len(names)
    se = {}
    for key in full:
        v = np.array([r[key] for r in reps], dtype=float)
        v = v[np.isfinite(v)]
        se[key] = float(np.sqrt((len(v) - 1) / len(v) * np.sum((v - v.mean()) ** 2))) if len(v) > 1 else float("nan")
    keys = [k for k in full if (k.startswith("bin_") or k.startswith("s_")) and np.isfinite(full[k])]
    theta = np.array([full[k] for k in keys])
    rep_mat = np.array([[r[k] for k in keys] for r in reps], dtype=float)
    centred = rep_mat - np.nanmean(rep_mat, axis=0)
    centred = np.nan_to_num(centred)
    cov = (n - 1) / n * centred.T @ centred
    return {"estimate": full, "se": se, "theta_keys": keys, "theta": theta.tolist(), "theta_cov": cov.tolist()}


def coincidence(jk: dict, targets: Sequence[int], inflation: float = 1.0) -> dict:
    """Binned (primary) and initial-slope (secondary) Wald tests that the targets' curves coincide."""

    keys, theta, cov = jk["theta_keys"], np.array(jk["theta"]), np.array(jk["theta_cov"]) * inflation

    def wald(rows: list[np.ndarray]) -> dict:
        if not rows:
            return {"wald": float("nan"), "df": 0, "p": float("nan")}
        c = np.array(rows)
        d = c @ theta
        v = c @ cov @ c.T
        w = float(d @ np.linalg.pinv(v) @ d)
        df = int(np.linalg.matrix_rank(v))
        return {"wald": w, "df": df, "p": float(stats.chi2.sf(w, df)) if df else float("nan")}

    binned_rows, gaps = [], []
    for j in range(len(BINS) - 1):
        present = [t for t in targets if f"bin_{t}_{j}" in keys]
        if len(present) < 2:
            continue
        idx = [keys.index(f"bin_{t}_{j}") for t in present]
        mean_bin = np.mean(theta[idx])
        for t in present[1:]:  # k - 1 independent contrasts per bin: each target against the bin mean
            row = np.zeros(len(keys))
            for i2 in idx:
                row[i2] -= 1.0 / len(idx)
            row[keys.index(f"bin_{t}_{j}")] += 1.0
            binned_rows.append(row)
        gaps += [abs(theta[i2] - mean_bin) for i2 in idx]
    slope_rows = []
    s_present = [t for t in targets if f"s_{t}" in keys]
    for t in s_present[1:]:
        row = np.zeros(len(keys))
        row[keys.index(f"s_{t}")] = 1.0
        row[keys.index(f"s_{s_present[0]}")] = -1.0
        slope_rows.append(row)
    return {
        "binned": wald(binned_rows),
        "binned_max_gap": float(max(gaps)) if gaps else float("nan"),
        "initial_slope": wald(slope_rows),
    }


def max_departure(all_designs: Sequence[WindowDesign], estimate: dict, targets: Sequence[int]) -> dict[str, float]:
    """Largest gap between each target's fitted curve and the common curve over that target's lever range."""

    out = {}
    lever = np.array([d.lever for d in all_designs])
    tgt = np.array([d.target for d in all_designs])
    for t in targets:
        grid = np.linspace(lever[tgt == t].min(), lever[tgt == t].max(), 200)
        own = saturating(grid, estimate[f"s_{t}"], estimate[f"tau_{t}"])
        common = saturating(grid, estimate["common_s"], estimate["common_tau"])
        out[str(t)] = float(np.max(np.abs(own - common)))
    return out


__all__ = [
    "BINS",
    "TOLERANCE",
    "coincidence",
    "excess_vector",
    "fit",
    "jackknife",
    "max_departure",
    "saturating",
    "statistics",
]
