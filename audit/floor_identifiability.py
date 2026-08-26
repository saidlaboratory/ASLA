"""Is the irreducible-loss floor identifiable on accuracy-type metrics?

With the adaptive alpha bound in place, the OLMES-error fits stop pinning
``alpha`` and start pinning ``E`` at zero instead: 25 of 25 interventions fit a
floor of ~0, i.e. the fitted curve claims downstream error decays to zero at
infinite compute. That is not a bound artifact to be silenced - it says the
three-parameter power law cannot identify a floor from this metric's data.

This module quantifies the claim three ways:

1. **Profile likelihood in E.** Sweep the floor across its feasible range,
   refit the remaining parameters at each value, and report how much the
   residual actually changes. A flat profile means the data do not constrain
   the floor.
2. **Floor identifiability ratio.** Compare the residual at the best floor with
   the residual at E = 0. If they differ by less than the seed noise, the floor
   is unidentifiable in the only sense that matters for extrapolation.
3. **Consequence for projection.** Two curves that agree on the fit range but
   differ in floor diverge at the target. Report the spread of target
   projections across the floors that the fit range cannot distinguish - the
   extrapolation ambiguity attributable to the unidentified floor alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _cells(df: pd.DataFrame, budgets: list[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fit = df[df["compute"].isin(budgets)]
    out = {}
    for name, group in fit.groupby("intervention", sort=True):
        means = group.groupby("compute", sort=True)["bpb"].mean()
        out[str(name)] = (means.index.to_numpy(dtype=float), means.to_numpy(dtype=float))
    return out


def _best_fit_at_floor(x: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, float, float]:
    """Given a fixed floor, find the (A, alpha) minimising squared residual; return (A, alpha, ssr)."""

    residual = y - floor
    if np.any(residual <= 0):
        return (np.nan, np.nan, float("inf"))
    # log(y - E) = log A - alpha log C  is linear, so solve in closed form
    log_c = np.log(x)
    log_r = np.log(residual)
    slope, intercept = np.polyfit(log_c, log_r, 1)
    alpha = float(-slope)
    amplitude = float(np.exp(intercept))
    predicted = floor + amplitude * x ** (-alpha)
    return (amplitude, alpha, float(np.sum((predicted - y) ** 2)))


def profile_floor(df: pd.DataFrame, budgets: list[float], target: float, n_grid: int = 60) -> list[dict[str, Any]]:
    """Profile likelihood in the floor for every intervention, with target projections."""

    rows: list[dict[str, Any]] = []
    seed_sd = float(
        np.sqrt(
            (df[df["compute"].isin(budgets)].groupby(["intervention", "compute"])["bpb"].std(ddof=1) ** 2).mean()
        )
    )
    for name, (x, y) in _cells(df, budgets).items():
        y_min = float(np.min(y))
        floors = np.linspace(0.0, y_min * 0.999, n_grid)
        entries = []
        for floor in floors:
            amplitude, alpha, ssr = _best_fit_at_floor(x, y, float(floor))
            if not np.isfinite(ssr):
                continue
            entries.append(
                {
                    "floor": float(floor),
                    "alpha": alpha,
                    "ssr": ssr,
                    "projection": float(floor + amplitude * target ** (-alpha)),
                }
            )
        if not entries:
            continue
        best = min(entries, key=lambda e: e["ssr"])
        n_points = len(x)
        # floors whose fit is within seed noise of the best fit: indistinguishable
        threshold = best["ssr"] + n_points * seed_sd**2
        admissible = [e for e in entries if e["ssr"] <= threshold]
        projections = [e["projection"] for e in admissible]
        rows.append(
            {
                "intervention": name,
                "best_floor": best["floor"],
                "best_floor_relative_to_ymin": best["floor"] / y_min,
                "ssr_at_best": best["ssr"],
                "ssr_at_zero_floor": entries[0]["ssr"],
                "ssr_ratio_zero_over_best": entries[0]["ssr"] / best["ssr"] if best["ssr"] > 0 else float("inf"),
                "n_admissible_floors": len(admissible),
                "admissible_floor_range": [min(e["floor"] for e in admissible), max(e["floor"] for e in admissible)],
                "projection_at_best": best["projection"],
                "projection_spread_over_admissible": float(max(projections) - min(projections)) if projections else 0.0,
                "y_min_on_fit_range": y_min,
            }
        )
    return rows


def run(metric: str) -> dict[str, Any]:
    df, budgets, target = ir.design_frame(metric)
    rows = profile_floor(df, budgets, target)
    frame = pd.DataFrame(rows)
    target_rows = df[np.isclose(df["compute"], target)]
    target_sd = float(np.sqrt((target_rows.groupby("intervention")["bpb"].std(ddof=1) ** 2).mean()))
    target_spread = float(target_rows.groupby("intervention")["bpb"].mean().std(ddof=1))
    return {
        "metric": metric,
        "n_interventions": int(len(frame)),
        "median_best_floor_relative_to_ymin": float(frame["best_floor_relative_to_ymin"].median()),
        "n_with_floor_at_zero": int((frame["best_floor"] <= 1e-9).sum()),
        "median_ssr_ratio_zero_over_best": float(frame["ssr_ratio_zero_over_best"].median()),
        "median_admissible_floor_width": float(
            (frame["admissible_floor_range"].apply(lambda r: r[1] - r[0])).median()
        ),
        "median_projection_spread_over_admissible_floors": float(frame["projection_spread_over_admissible"].median()),
        "target_seed_sd": target_sd,
        "target_between_intervention_sd": target_spread,
        "projection_ambiguity_over_seed_noise": float(
            frame["projection_spread_over_admissible"].median() / target_sd
        ) if target_sd > 0 else None,
        "projection_ambiguity_over_between_spread": float(
            frame["projection_spread_over_admissible"].median() / target_spread
        ) if target_spread > 0 else None,
        "per_intervention": rows,
    }


def main() -> int:
    out = {metric: run(metric) for metric in ir.TABLES}
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "floor_identifiability.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    for metric, row in out.items():
        print(f"=== {metric}")
        print(f"    best floor / y_min (median):        {row['median_best_floor_relative_to_ymin']:.4f}")
        print(f"    interventions with floor == 0:      {row['n_with_floor_at_zero']}/{row['n_interventions']}")
        print(f"    SSR(zero floor)/SSR(best) median:   {row['median_ssr_ratio_zero_over_best']:.4f}")
        print(f"    admissible floor width (median):    {row['median_admissible_floor_width']:.4f}")
        print(f"    target projection spread over those floors: {row['median_projection_spread_over_admissible_floors']:.4f}")
        print(f"    that spread / target seed sd:       {row['projection_ambiguity_over_seed_noise']:.1f}x")
        print(f"    that spread / between-recipe sd:    {row['projection_ambiguity_over_between_spread']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
