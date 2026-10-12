"""Seed-noise profile across scale for every metric the paper analyses (PREDICTIONS_TASK_MARIN.md, small items).

Profile: RMS over candidates of the per-run seed SD at each scale. Measure: the
log-range log(max / min) across scales, the pre-registered quantity, plus the
OLS slope of log SD on log compute per decade (scale spans differ between suites,
so the slope is the comparable context). DataDecide metrics use the rungs of the
calibration design (4M ... 300M and the 530M target), the rung set whose profile
the Task 5 "flattened" condition halved; there fixed-candidate coverage was
0.86. Flags: "flatter than C4" (log-range below C4's) and "as flat as the tested
flattened profile" (at most half of C4's).

Writes ``results/target_scoring/noise_profiles.json``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "target_scoring" / "noise_profiles.json"
RUNGS = ("4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M")


def summarise(profile: pd.Series, compute: pd.Series) -> dict[str, Any]:
    log_sd = np.log(profile.to_numpy(float))
    slope = float(np.polyfit(np.log10(compute.loc[profile.index].to_numpy(float)), log_sd, 1)[0])
    return {
        "profile": {str(k): float(v) for k, v in profile.items()},
        "log_range": float(log_sd.max() - log_sd.min()),
        "log_sd_slope_per_decade": slope,
    }


def datadecide(metric: str) -> dict[str, Any]:
    lam = importlib.import_module("scripts.run_lever_arm_by_metric")
    tw = importlib.import_module("scripts.transfer_worlds")
    if metric == "c4":
        ctx = tw.context("c4")
        sigma, compute = ctx["base"].sigma, ctx["compute"]
    else:
        ctx = lam.context(metric)
        sigma = ctx["base"].sigma
        compute = ctx["frame"].groupby("scale_label")["compute"].first()
    table = pd.Series(sigma)
    profile = table.groupby(level=1).apply(lambda v: float(np.sqrt(np.mean(np.square(v))))).loc[list(RUNGS)]
    return summarise(profile, compute)


def compute_multipliers(chart: str) -> dict[str, Any]:
    cmr = importlib.import_module("scripts.run_compute_multipliers_replication")
    budgets, _, sd = cmr.load_axis(chart)
    profile = pd.Series({b: float(np.sqrt(np.mean(sd[b].to_numpy() ** 2))) for b in budgets})
    return summarise(profile, pd.Series({b: b for b in budgets}))


def main() -> None:
    metrics: dict[str, Any] = {name: datadecide(name) for name in ("c4", "olmes", "cp", "margin", "norm_cp")}
    metrics["compute_multipliers_recipe_axis_olmes10"] = compute_multipliers("djBI7")
    metrics["compute_multipliers_corpus_axis_olmes10"] = compute_multipliers("lm6Jz")
    c4_range = metrics["c4"]["log_range"]
    for name, m in metrics.items():
        m["flatter_than_c4"] = bool(m["log_range"] < c4_range) if name != "c4" else None
        m["as_flat_as_tested_flattened_profile"] = bool(m["log_range"] <= 0.5 * c4_range) if name != "c4" else None
    out = {
        "measure": "log(max/min) of RMS seed SD across scales (pre-registered); slope per decade as context",
        "c4_log_range": c4_range,
        "tested_flattened_log_range": 0.5 * c4_range,
        "metrics": metrics,
        "not_estimable": {
            "signal_and_noise": "one run per cell; noise borrowed from C4",
            "marin_speedrun": "one run per cell; no replicates",
        },
        "flagged_at_risk": sorted(n for n, m in metrics.items() if m["as_flat_as_tested_flattened_profile"]),
        "flatter_than_c4": sorted(n for n, m in metrics.items() if m["flatter_than_c4"]),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for n, m in metrics.items():
        print(
            f"{n:42s} log-range {m['log_range']:.2f} slope/decade {m['log_sd_slope_per_decade']:+.2f} "
            f"flatter {m['flatter_than_c4']} as-flat-as-tested {m['as_flat_as_tested_flattened_profile']}"
        )


if __name__ == "__main__":
    main()
