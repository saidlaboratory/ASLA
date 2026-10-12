"""Task 2 of PREDICTIONS_TASK_REVIEW_RISKS.md: single-scale vs projection vs shared-exponent on compute-multipliers.

Only OLMES10 accuracy has the full five-budget ladder (1e17 ... 1e19, +/- 1 sd
bands): chart ``djBI7`` (7 recipes on FineWeb-Edu) and ``lm6Jz`` (7 corpora under
OLMo 2). Rankers work on 1 - accuracy. Designs have at least three consecutive
fitted budgets and a later held-out target. Uncertainty: cell means resampled
as mean + (sd / sqrt 3) t_2, 4,000 draws, assuming normal seed noise,
independence across cells, and sd read from the chart bands; there is no
per-seed data. With 7 candidates, candidate-level inference is meaningless, so
all of this is descriptive with seed-level uncertainty.

Writes ``results/external/compute_multipliers_replication.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "data" / "raw" / "compute_multipliers" / "datawrapper"
OUT = REPO / "results" / "external" / "compute_multipliers_replication.json"
AXES = {"recipe": "djBI7", "corpus": "lm6Jz"}
CHARTS = {
    "djBI7": "OLMES10 vs compute, 7 recipes on FineWeb-Edu, +/- 1 sd (full ladder)",
    "lm6Jz": "OLMES10 vs compute, 7 corpora under OLMo 2, +/- 1 sd (full ladder)",
    "TrYq2": "OLMES10 vs compute, 2019/2025 corner 2x2, +/- 1 sd (full ladder, 4 cells)",
    "Cnl63": "recipe x corpus residual grid at one budget (not a ladder)",
    "KNa3I": "recipe x corpus grid at one budget (not a ladder)",
    "CzRR5": "recipe compute-multiplier table at 1e19 (not a ladder)",
    "k03qL": "corpus compute-multiplier table at 1e19 (not a ladder)",
}
CHANCE_ACCURACY = 0.348
N_DRAWS = 4000
SEED = 20261012


def load_axis(chart: str) -> tuple[list[float], pd.DataFrame, pd.DataFrame]:
    """(budgets, mean accuracy [candidate x budget], sd [candidate x budget])."""

    raw = pd.read_csv(RAW / f"{chart}.csv")
    budgets = [float(c) for c in raw["compute"]]
    names = [c for c in raw.columns[1:] if not c.endswith(("−1 sd", "+1 sd"))]
    mean = pd.DataFrame({n: raw[n].to_numpy(float) for n in names}, index=budgets).T
    sd = pd.DataFrame({n: (raw[f"{n} +1 sd"] - raw[f"{n} −1 sd"]).to_numpy(float) / 2 for n in names}, index=budgets).T
    return budgets, mean, sd


def designs(budgets: list[float]) -> list[tuple[list[float], float]]:
    out = []
    for start in range(len(budgets)):
        for end in range(start + 2, len(budgets) - 1):
            for target in range(end + 1, len(budgets)):
                out.append((budgets[start : end + 1], budgets[target]))
    return out


def _table(error: pd.DataFrame, fit: list[float]) -> pd.DataFrame:
    rows = [
        {"intervention": str(n), "intervention_class": "recipe", "compute": b, "seed": 0, "bpb": float(error.loc[n, b])}
        for n in error.index
        for b in fit
    ]
    return pd.DataFrame(rows)


def predictions(error: pd.DataFrame, fit: list[float], target: float) -> dict[str, pd.Series]:
    from asla.analysis.pooled import shared_exponent_ranker
    from asla.analysis.rankers import projection_ranker, single_scale_ranker

    table = _table(error, fit)
    out = {"single_scale": single_scale_ranker(table, tuple(fit), target)}
    for name, ranker in (("projection", projection_ranker), ("shared_exponent", shared_exponent_ranker)):
        try:
            out[name] = ranker(table, tuple(fit), target)
        except Exception:  # noqa: BLE001 - a failed fit is reported, not fatal
            out[name] = pd.Series({n: float("nan") for n in error.index})
    return out


def _gaps(values: pd.Series) -> dict[tuple[str, str], float]:
    return {(a, b): float(values[a] - values[b]) for a, b in combinations(sorted(values.index), 2)}


def mis(pred: pd.Series, truth: pd.Series) -> float:
    p, t = _gaps(pred), _gaps(truth)
    scored = [(p[k], t[k]) for k in t if t[k] != 0 and np.isfinite(p[k]) and p[k] != 0]
    return float(np.mean([np.sign(a) != np.sign(b) for a, b in scored])) if scored else float("nan")


def axis_result(chart: str, rng: np.random.Generator) -> dict[str, Any]:
    from asla.analysis.moderation import calibrated_target_evidence
    from asla.analysis.target_scoring import decomposition

    budgets, acc, sd = load_axis(chart)
    error = 1 - acc
    top = budgets[-1]
    spread = {
        "target_range_over_median_seed_sd": float((acc[top].max() - acc[top].min()) / sd[top].median()),
        "target_sd_over_median_seed_sd": float(acc[top].std(ddof=1) / sd[top].median()),
    }
    chance_budgets = [b for b in budgets if acc[b].mean() <= CHANCE_ACCURACY]
    out: dict[str, Any] = {
        "candidates": list(acc.index),
        "spread_at_1e19": spread,
        "chance_budgets": chance_budgets,
        "designs": {},
    }
    draws = {k: (error + (sd / np.sqrt(3)) * stats.t.rvs(2, size=error.shape, random_state=rng)) for k in range(N_DRAWS)}
    for fit, target in designs(budgets):
        key = f"{fit[0]:.3g}-{fit[-1]:.3g}->{target:.3g}"
        preds = predictions(error, fit, target)
        truth = error[target]
        evidence = calibrated_target_evidence(
            {n: float(error.loc[n, target]) for n in error.index},
            {n: float(sd.loc[n, target]) for n in error.index},
            {n: 3 for n in error.index},
        )
        point = {name: mis(p, truth) for name, p in preds.items()}
        resampled: dict[str, list[float]] = {name: [] for name in preds}
        for k in range(N_DRAWS):
            e = draws[k]
            rp = predictions(e, fit, target)
            for name, p in rp.items():
                resampled[name].append(mis(p, e[target]))
        excess = {
            name: np.array(resampled[name]) - np.array(resampled["single_scale"])
            for name in ("projection", "shared_exponent")
        }
        out["designs"][key] = {
            "fit": fit,
            "target": target,
            "lever_arm": target / fit[-1],
            "mis_selection": point,
            "excess_over_single_pp": {n: 100 * (point[n] - point["single_scale"]) for n in excess},
            "excess_interval_pp": {
                n: [float(100 * np.nanpercentile(v, 2.5)), float(100 * np.nanpercentile(v, 97.5))] for n, v in excess.items()
            },
            "share_of_draws_comparator_better": {n: float(np.nanmean(v < 0)) for n, v in excess.items()},
            "resolved_pairs_at_target": int(sum(e.determined for e in evidence)),
            "pairs": len(evidence),
            "decomposition_vs_single": {
                n: {
                    k: v
                    for k, v in decomposition(_gaps(preds[n]), _gaps(preds["single_scale"]), evidence).items()
                    if not k.endswith("_pairs")
                }
                for n in ("projection", "shared_exponent")
            },
        }
    rows = list(out["designs"].values())
    lever = np.log([r["lever_arm"] for r in rows])
    out["lever_arm_trend_spearman"] = {
        n: float(stats.spearmanr(lever, [r["excess_over_single_pp"][n] for r in rows]).statistic)
        for n in ("projection", "shared_exponent")
    }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "charts": CHARTS,
        "full_ladder_metric": "OLMES10 accuracy (both axes); no other metric has the five-budget ladder",
        "assumptions": [
            "cell means resampled as mean + (sd / sqrt 3) t_2: normal seed noise, n = 3",
            "independence across cells",
            "no per-seed data; sd read from the chart's +/- 1 sd band",
            "7 candidates: candidate-level inference is meaningless, so everything is descriptive",
        ],
        "chance_accuracy": CHANCE_ACCURACY,
        "axes": {axis: axis_result(chart, rng) for axis, chart in AXES.items()},
    }
    ax = out["axes"]
    out["predictions"] = {
        "P2_projection_never_better_on_either_axis": all(
            d["excess_over_single_pp"]["projection"] >= 0 for a in ax.values() for d in a["designs"].values()
        ),
        "P2_recipe_axis_mostly_unresolved": all(
            d["resolved_pairs_at_target"] < d["pairs"] / 2 for d in ax["recipe"]["designs"].values()
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for axis, a in ax.items():
        print(axis, a["spread_at_1e19"], "chance budgets", a["chance_budgets"], "trend", a["lever_arm_trend_spearman"])
        for k, d in a["designs"].items():
            print(
                f"  {k:22s} mis { ({n: round(v, 3) for n, v in d['mis_selection'].items()}) } "
                f"excess { ({n: round(v, 1) for n, v in d['excess_over_single_pp'].items()}) } "
                f"resolved {d['resolved_pairs_at_target']}/{d['pairs']}"
            )
    print(out["predictions"])


if __name__ == "__main__":
    main()
