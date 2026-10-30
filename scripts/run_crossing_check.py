"""Is projection's excess over single-scale ranking ever negative at short lever arms, at a fixed target?

Uses the fixed-target designs of PREDICTIONS_TASK_FIXED_TARGET.md (C4 bits per token; targets whose
determined share is at least one half; fit windows of at least three consecutive rungs) and the same
expected-error kernel. Two descriptions, neither pre-registered:

* per design: the excess at each design, with a leave-one-recipe-out jackknife interval at the normal
  critical value; the designs at each target's shortest lever arm are listed, with whether any interval lies
  wholly below zero;
* per target: a linear fit of excess on log10 lever arm across that target's designs (rung count as a
  fixed effect), and the lever arm L* where the fitted line crosses zero, with a jackknife interval. A
  crossing inside the observed lever-arm range is reported as one; outside it, as an extrapolation.

Writes results/target_scoring/crossing_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.confound import window_designs  # noqa: E402
from asla.analysis.fixed_target import admitted_targets, excess_all  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "crossing_check.json"
Z = 1.959963984540054
UNIT = "points of mis-selected pairs; lever arm in decades (log10)"


def _fit(lever: np.ndarray, rungs: np.ndarray, y: np.ndarray) -> tuple[float, dict[int, float]]:
    """Slope on lever arm with a separate intercept per rung count; returns slope and intercepts."""

    levels = sorted(set(rungs.tolist()))
    x = np.column_stack([lever] + [(rungs == k).astype(float) for k in levels])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(beta[0]), {k: float(b) for k, b in zip(levels, beta[1:])}


def _crossing(lever: np.ndarray, rungs: np.ndarray, y: np.ndarray, k: int) -> float:
    slope, intercepts = _fit(lever, rungs, y)
    return -intercepts[k] / slope if slope != 0 else float("nan")


def _jackknife(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    return float(np.sqrt((n - 1) / n * np.sum((arr - arr.mean()) ** 2)))


def main() -> None:
    confound = __import__("scripts.run_confound", fromlist=["_load_frame"])
    frame = confound._load_frame()
    labels = list(frame.groupby("scale_label")["compute"].first().sort_values().index)
    designs = window_designs(frame)
    targets = admitted_targets(designs)
    names = sorted({n for d in designs for pair in d.kernel for n in pair})
    full = excess_all(designs)
    loo = {drop: excess_all(designs, drop) for drop in names}
    out_targets = {}
    any_below = False
    for t in targets:
        idx = [i for i, d in enumerate(designs) if d.target == t]
        lever = np.array([designs[i].lever for i in idx])
        rungs = np.array([designs[i].rungs for i in idx])
        rows = []
        for i in idx:
            se = _jackknife([loo[n][i] for n in names])
            lo, hi = full[i] - Z * se, full[i] + Z * se
            any_below |= hi < 0
            rows.append(
                {
                    "window": f"{labels[designs[i].start]}-{labels[designs[i].top]}",
                    "rungs": designs[i].rungs,
                    "lever": designs[i].lever,
                    "excess": float(full[i]),
                    "interval": [float(lo), float(hi)],
                }
            )
        rows.sort(key=lambda r: (r["lever"], r["rungs"]))
        shortest = min(r["lever"] for r in rows)
        # Crossing for the smallest rung count, the one present at the shortest lever arm.
        k = min(r["rungs"] for r in rows if np.isclose(r["lever"], shortest))
        slope, intercepts = _fit(lever, rungs, full[idx])
        cross = _crossing(lever, rungs, full[idx], k)
        cross_loo = [_crossing(lever, rungs, loo[n][idx], k) for n in names]
        se_cross = _jackknife(cross_loo)
        out_targets[labels[t]] = {
            "n_designs": len(idx),
            "shortest_lever": shortest,
            "longest_lever": float(lever.max()),
            "designs_at_shortest_lever": [r for r in rows if np.isclose(r["lever"], shortest)],
            "min_excess": min(r["excess"] for r in rows),
            "any_interval_below_zero": any(r["interval"][1] < 0 for r in rows),
            "any_point_below_zero": any(r["excess"] < 0 for r in rows),
            "fit_slope_per_decade": slope,
            "fit_rung_count_for_crossing": k,
            "fitted_excess_at_shortest_lever": slope * shortest + intercepts[k],
            "crossing_lever": cross,
            "crossing_interval": [cross - Z * se_cross, cross + Z * se_cross],
            "crossing_inside_observed_range": bool(shortest <= cross <= float(lever.max())),
            "designs": rows,
        }
    record = {
        "status": "descriptive check on existing data; not pre-registered",
        "metric": "c4_en_bits_per_token",
        "unit": UNIT,
        "targets": out_targets,
        "any_design_interval_below_zero": bool(any_below),
        "any_design_point_below_zero": bool(any(v["any_point_below_zero"] for v in out_targets.values())),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for lab, v in out_targets.items():
        print(
            lab,
            f"shortest L {v['shortest_lever']:.2f}",
            [(r["window"], round(r["excess"], 2), [round(x, 2) for x in r["interval"]]) for r in v["designs_at_shortest_lever"]],
            f"min {v['min_excess']:.2f}",
            f"L* {v['crossing_lever']:.2f} [{v['crossing_interval'][0]:.2f}, {v['crossing_interval'][1]:.2f}]",
            "inside" if v["crossing_inside_observed_range"] else "outside",
        )
    print("any design interval below zero:", any_below, "any point below zero:", record["any_design_point_below_zero"])


if __name__ == "__main__":
    main()
