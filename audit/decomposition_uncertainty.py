"""A3 follow-up: seed-bootstrap uncertainty on the flip decomposition counts.

A3's simulation shows the flip classifier is noisy in both directions at n=3
seeds: a true crossover is seen as a flip 60-99% of the time, and a
close-but-non-crossing pair is falsely seen as flipped 32-50% of the time when
its ladder gap is within a seed sd. The headline "14 fit error vs 2 inherited
crossover" is therefore a point estimate from one noisy realisation.

This resamples seeds within every (intervention, compute) cell and recomputes
the whole decomposition, giving intervals on the counts and, most importantly,
on the *conclusion*: the fraction of resamples in which the excess of
projection over single-scale ranking is still entirely fit error.

The bounded audit fit is used (matching the headline); the independent fit is
too slow to bootstrap 25 interventions x 11 budgets x many replicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np
import pandas as pd

from asla.analysis.fits import project_ranking

REPO = Path(__file__).resolve().parents[1]


def resample_cells(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby(["intervention", "compute"], sort=False):
        seeds = group["seed"].to_numpy()
        drawn = rng.choice(len(seeds), size=len(seeds), replace=True)
        rows = group.iloc[drawn].copy()
        rows["seed"] = np.arange(len(rows))
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def decompose(df: pd.DataFrame, budgets: list[float], target: float) -> dict[str, int]:
    truth = ir.truth_scores(df, target)
    proj = {str(k): float(v) for k, v in project_ranking(df, tuple(budgets), target).items()}
    single = ir.single_scale_scores(df, budgets)
    proj_flips = ir.flips(proj, truth)
    single_flips = set(ir.flips(single, truth))
    inherited = [p for p in proj_flips if p in single_flips]
    fit_error = [p for p in proj_flips if p not in single_flips]
    repaired = [p for p in single_flips if p not in set(proj_flips)]
    excess = len(proj_flips) - len(single_flips)
    return {
        "projection_flips": len(proj_flips),
        "single_scale_flips": len(single_flips),
        "inherited": len(inherited),
        "fit_error": len(fit_error),
        "repaired": len(repaired),
        "excess": excess,
        "excess_all_fit_error": int(excess <= 0 or (len(fit_error) - len(repaired)) >= excess),
        "fit_error_exceeds_inherited": int(len(fit_error) > len(inherited)),
        "projection_mis_selection": ir.pairwise_mis_selection(proj, truth),
        "single_scale_mis_selection": ir.pairwise_mis_selection(single, truth),
        "projection_worse": int(ir.pairwise_mis_selection(proj, truth) > ir.pairwise_mis_selection(single, truth)),
    }


def run(metric: str, n_boot: int = 300, seed: int = 0) -> dict[str, Any]:
    df, budgets, target = ir.design_frame(metric)
    point = decompose(df, budgets, target)
    rng = np.random.default_rng(seed)
    draws: list[dict[str, int]] = []
    for _ in range(n_boot):
        try:
            draws.append(decompose(resample_cells(df, rng), budgets, target))
        except Exception:  # noqa: BLE001 - a failed fit in a resample is recorded by omission
            continue
    frame = pd.DataFrame(draws)
    out: dict[str, Any] = {"metric": metric, "n_boot": len(frame), "point": point, "intervals": {}}
    for column in ("projection_flips", "single_scale_flips", "inherited", "fit_error", "repaired", "excess"):
        values = frame[column].to_numpy(dtype=float)
        out["intervals"][column] = {
            "point": point[column],
            "mean": float(values.mean()),
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
        }
    out["conclusion_stability"] = {
        "excess_all_fit_error_rate": float(frame["excess_all_fit_error"].mean()),
        "fit_error_exceeds_inherited_rate": float(frame["fit_error_exceeds_inherited"].mean()),
        "projection_worse_rate": float(frame["projection_worse"].mean()),
    }
    return out


def main() -> int:
    results = [run(metric) for metric in ir.TABLES]
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "a3b_decomposition_bootstrap.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    for row in results:
        print(f"=== {row['metric']} (n_boot={row['n_boot']})")
        for name, interval in row["intervals"].items():
            print(f"    {name:20s} point={interval['point']:3d}  95% [{interval['lo']:.0f}, {interval['hi']:.0f}]")
        print(f"    conclusion stability: {json.dumps(row['conclusion_stability'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
