"""Profile likelihood in the floor E, fixing the grid-and-log-space shortcut.

Pre-registered in PREDICTIONS_TASK_FIT_STRUCTURE.md (F1-F4). For each metric and
intervention on the full-ladder design (fit 4M..300M, target 1B):

* the profile SSR(E) minimises the response-scale residual over (A, alpha) at
  each fixed floor, rather than taking the log-space regression's SSR;
* the primary 95% interval is the F-based profile interval of Bates & Watts,
  which uses the residual variance and so stays honest under misspecification;
* the secondary interval treats the seed variance of a cell mean as known;
* the conditioning of J^T J at the joint optimum is reported in (E, log A, alpha).

Writes ``results/adversarial/floor_profile.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "audit"))

import independent_rederivation as ir  # noqa: E402

from asla.provenance import Source  # noqa: E402

N_PARAMS = 3
LEVEL = 0.95
GRID = 400
SEEDS_PER_CELL = 3
METRICS = ("c4_en_bits_per_token", "olmes_macro_error", "olmes_macro_correct_prob_per_char_deficit")


def _residuals(params: np.ndarray, u: np.ndarray, y: np.ndarray, floor: float) -> np.ndarray:
    log_a, alpha = params
    return floor + np.exp(log_a) * u ** (-alpha) - y


def profile_ssr(u: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, np.ndarray]:
    """Minimum response-scale SSR over (log A, alpha) at a fixed floor.

    ``u`` is compute divided by its geometric mean, which keeps ``A`` of order
    one. Returns ``(inf, nan)`` when the floor is at or above an observation.
    """

    gap = y - floor
    if np.any(gap <= 0):
        return float("inf"), np.full(2, np.nan)
    slope, intercept = np.polyfit(np.log(u), np.log(gap), 1)
    start = np.array([intercept, -slope])
    fit = optimize.least_squares(_residuals, start, args=(u, y, floor), method="lm")
    return float(np.sum(fit.fun**2)), fit.x


def _joint_condition_number(u: np.ndarray, floor: float, params: np.ndarray) -> float:
    log_a, alpha = params
    power = u ** (-alpha)
    jac = np.column_stack((np.ones_like(u), np.exp(log_a) * power, -np.exp(log_a) * power * np.log(u)))
    return float(np.linalg.cond(jac.T @ jac))


def profile_interval(
    compute: np.ndarray, y: np.ndarray, sigma_mean: float | None = None, grid: int = GRID
) -> dict[str, Any]:
    """F-based (and optionally known-sigma) profile intervals for the floor."""

    u = np.asarray(compute, dtype=float) / float(np.exp(np.mean(np.log(compute))))
    y = np.asarray(y, dtype=float)
    n = len(y)
    upper_limit = float(np.min(y))
    floors = np.linspace(0.0, upper_limit, grid, endpoint=False)
    ssr = np.array([profile_ssr(u, y, float(f))[0] for f in floors])
    finite = np.isfinite(ssr)
    best_index = int(np.argmin(np.where(finite, ssr, np.inf)))
    # Refine the optimum between the neighbouring grid points.
    lo = floors[max(best_index - 1, 0)]
    hi = floors[min(best_index + 1, grid - 1)]
    refined = optimize.minimize_scalar(lambda f: profile_ssr(u, y, f)[0], bounds=(lo, hi), method="bounded")
    best_floor = float(refined.x) if refined.fun < ssr[best_index] else float(floors[best_index])
    ssr_min, params = profile_ssr(u, y, best_floor)

    def interval(statistic: Any, critical: float) -> tuple[float, float, bool]:
        def excess(f: float) -> float:
            return float(statistic(profile_ssr(u, y, f)[0]) - critical)

        values = np.array([statistic(s) - critical if np.isfinite(s) else np.inf for s in ssr])
        inside = values <= 0
        left_open = bool(inside[0])
        lower = 0.0
        if not left_open:
            below = [i for i in range(best_index) if not inside[i]]
            i = below[-1] if below else 0
            lower = float(optimize.brentq(excess, floors[i], max(best_floor, floors[i + 1])))
        above = [i for i in range(best_index + 1, grid) if not inside[i]]
        if above:
            j = above[0]
            upper = float(optimize.brentq(excess, min(best_floor, floors[j - 1]), floors[j]))
        else:
            upper = upper_limit
        return lower, upper, left_open

    dispersion = ssr_min / (n - N_PARAMS)
    f_crit = float(stats.f.ppf(LEVEL, 1, n - N_PARAMS))
    low, high, includes_zero = interval(lambda s: (s - ssr_min) / dispersion, f_crit)
    out: dict[str, Any] = {
        "n_points": n,
        "best_floor": best_floor,
        "best_floor_relative_to_ymin": best_floor / upper_limit,
        "ssr_min": ssr_min,
        "ssr_at_zero_floor": float(ssr[0]),
        "residual_dispersion": dispersion,
        "f_critical": f_crit,
        "interval": [low, high],
        "interval_width": high - low,
        "interval_includes_zero": includes_zero,
        "interval_open_at_ymin": high >= upper_limit,
        "interval_width_relative_to_ymin": (high - low) / upper_limit,
        "condition_number": _joint_condition_number(u, best_floor, params),
    }
    if sigma_mean is not None and sigma_mean > 0:
        chi_crit = float(stats.chi2.ppf(LEVEL, 1))
        k_low, k_high, k_zero = interval(lambda s: (s - ssr_min) / sigma_mean**2, chi_crit)
        out["known_sigma"] = {
            "sigma_mean": sigma_mean,
            "interval": [k_low, k_high],
            "interval_includes_zero": k_zero,
            # Above 1, the fit misses by more than seed noise and this interval
            # is too narrow; it is reported, not used for the verdicts.
            "misspecification_ratio": dispersion / sigma_mean**2,
        }
    return out


def run(metric: str) -> dict[str, Any]:
    df, budgets, _ = ir.design_frame(metric)
    fit = df[df["compute"].isin(budgets)]
    cell_var = fit.groupby(["intervention", "compute"])["bpb"].var(ddof=1)
    sigma_mean = float(np.sqrt(cell_var.mean() / SEEDS_PER_CELL))
    rows = []
    for name, group in fit.groupby("intervention", sort=True):
        means = group.groupby("compute", sort=True)["bpb"].mean()
        row = profile_interval(means.index.to_numpy(dtype=float), means.to_numpy(dtype=float), sigma_mean)
        rows.append({"intervention": str(name), **row})
    frame = pd.DataFrame(rows)
    return {
        "metric": metric,
        "n_interventions": len(rows),
        "sigma_mean": sigma_mean,
        "n_interval_excludes_zero": int((~frame["interval_includes_zero"]).sum()),
        "n_known_sigma_excludes_zero": int(sum(not r["known_sigma"]["interval_includes_zero"] for r in rows)),
        "median_interval_width": float(frame["interval_width"].median()),
        "median_interval_width_relative_to_ymin": float(frame["interval_width_relative_to_ymin"].median()),
        "n_interval_open_at_ymin": int(frame["interval_open_at_ymin"].sum()),
        "median_condition_number": float(frame["condition_number"].median()),
        "median_misspecification_ratio": float(np.median([r["known_sigma"]["misspecification_ratio"] for r in rows])),
        "per_intervention": rows,
    }


def predictions(out: dict[str, Any]) -> dict[str, Any]:
    committed = Source.load("adversarial/floor_identifiability.json")
    c4, err = out["c4_en_bits_per_token"], out["olmes_macro_error"]
    old_width = committed.number("c4_en_bits_per_token.median_admissible_floor_width")
    ratio = err["median_condition_number"] / c4["median_condition_number"]
    return {
        "F1_c4_floor_identified": {
            "measured": c4["n_interval_excludes_zero"],
            "threshold": "at least 23 of 25 exclude zero",
            "confirmed": c4["n_interval_excludes_zero"] >= 23,
        },
        "F2_accuracy_floor_unidentified": {
            "measured": err["n_interventions"] - err["n_interval_excludes_zero"],
            "threshold": "at least 20 of 25 include zero",
            "confirmed": err["n_interventions"] - err["n_interval_excludes_zero"] >= 20,
        },
        "F3_conditioning": {
            "measured_ratio": ratio,
            "threshold": "olmes_macro_error median at least 100x C4 median",
            "confirmed": ratio >= 100,
        },
        "F4_fix_tightens_c4": {
            "measured_width": c4["median_interval_width"],
            "committed_admissible_width": old_width,
            "confirmed": c4["median_interval_width"] < old_width,
        },
    }


def main() -> None:
    out: dict[str, Any] = {metric: run(metric) for metric in METRICS}
    out["predictions"] = predictions(out)
    out["method"] = {
        "level": LEVEL,
        "grid": GRID,
        "primary": "F-based profile interval, residual dispersion SSR_min/(n-3)",
        "secondary": "known-sigma chi-square interval, sigma of a three-seed cell mean",
        "conditioning": "cond(J^T J) at the joint optimum in (E, log A, alpha), compute over its geometric mean",
    }
    path = REPO / "results" / "adversarial" / "floor_profile.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for metric in METRICS:
        row = out[metric]
        print(
            f"{metric}: excl0 {row['n_interval_excludes_zero']}/{row['n_interventions']} "
            f"(known-sigma {row['n_known_sigma_excludes_zero']}), width {row['median_interval_width']:.4f}, "
            f"cond {row['median_condition_number']:.3g}, misspec {row['median_misspecification_ratio']:.2f}"
        )
    for key, value in out["predictions"].items():
        print(key, value["confirmed"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
