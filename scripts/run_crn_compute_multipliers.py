"""Task 1 of PREDICTIONS_TASK_CRN.md: is the compute-multipliers model axis already a CRN design, and how much does it buy?

The model axis is 7 recipes x 3 seed indices on one corpus (FineWeb-Edu) at 1e19
FLOPs. Every run.json records ``data_seed`` equal to the seed index, identical
across recipes, with the same corpus, tokenizer, context and batch; the data
loader itself is not in the release, so identity of the data order is probable
but not verifiable here. Initialisation cannot be shared (architectures differ).

For each metric: the intraclass correlation r of the seed-index effect across
recipes (two-way layout without replication), its 95% interval from
F(S - 1, (R - 1)(S - 1)), the implied gap-variance reduction (r) and factor
1 / (1 - r), and per recipe pair the matched- against mismatched-seed gap
variances.

Writes ``results/crn/compute_multipliers_crn.json``.
"""

from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "compute_multipliers"
OUT = REPO / "results" / "crn" / "compute_multipliers_crn.json"
PRIMARY = "native_heldout_nll"
SECONDARY = ("olmes10", "heldout7_mean_nll", "alt8")


def seed_icc(table: pd.DataFrame, level: float = 0.95) -> dict[str, Any]:
    """Intraclass correlation of the column (seed-index) effect in a rows x columns table without replication."""

    y = table.to_numpy(dtype=float)
    n_r, n_s = y.shape
    grand = y.mean()
    ss_s = n_r * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_r = n_s * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_e = ((y - grand) ** 2).sum() - ss_s - ss_r
    df_s, df_e = n_s - 1, (n_r - 1) * (n_s - 1)
    ms_s, ms_e = ss_s / df_s, ss_e / df_e
    sigma_s2 = (ms_s - ms_e) / n_r
    r_hat = sigma_s2 / (sigma_s2 + ms_e) if (sigma_s2 + ms_e) > 0 else float("nan")
    f = ms_s / ms_e
    alpha = 1 - level
    upper_q, lower_q = stats.f.ppf(1 - alpha / 2, df_s, df_e), stats.f.ppf(alpha / 2, df_s, df_e)
    theta = [(f / upper_q - 1) / n_r, (f / lower_q - 1) / n_r]
    r_ci = [t / (1 + t) if t > -1 else float("-inf") for t in theta]
    return {
        "r": r_hat,
        "r_ci": r_ci,
        "r_ci_clipped_at_0": [max(0.0, r_ci[0]), max(0.0, r_ci[1])],
        "reduction_factor": 1 / (1 - r_hat) if r_hat < 1 else float("inf"),
        "reduction_factor_ci": [1 / (1 - max(0.0, v)) if v < 1 else float("inf") for v in r_ci],
        "F": f,
        "df": [df_s, df_e],
        "p_seed_effect": float(stats.f.sf(f, df_s, df_e)),
        "sigma2_seed_index": sigma_s2,
        "sigma2_residual": ms_e,
    }


def pair_variances(table: pd.DataFrame) -> dict[str, Any]:
    """Per recipe pair: variance of the matched-seed gaps against the mismatched-seed gaps."""

    seeds = list(table.columns)
    rows = {}
    for a, b in combinations(table.index, 2):
        matched = np.array([table.loc[a, s] - table.loc[b, s] for s in seeds])
        mismatched = np.array([table.loc[a, s] - table.loc[b, t] for s, t in permutations(seeds, 2)])
        rows[f"{a}|{b}"] = {
            "matched_var": float(matched.var(ddof=1)),
            "mismatched_var": float(mismatched.var(ddof=1)),
            "ratio": float(matched.var(ddof=1) / mismatched.var(ddof=1)) if mismatched.var(ddof=1) > 0 else None,
        }
    ratios = np.array([v["ratio"] for v in rows.values() if v["ratio"] is not None])
    pooled = np.mean([v["matched_var"] for v in rows.values()]) / np.mean([v["mismatched_var"] for v in rows.values()])
    return {
        "pairs": rows,
        "pooled_matched_over_mismatched": float(pooled),
        "pooled_reduction": float(1 - pooled),
        "median_pair_ratio": float(np.median(ratios)),
        "share_of_pairs_with_matched_below_mismatched": float(np.mean(ratios < 1)),
    }


def protocol() -> dict[str, Any]:
    records = [json.loads(p.read_text()) for p in sorted((RAW / "run_json").glob("*.json"))]
    by_index: dict[int, set[int]] = {}
    for r in records:
        by_index.setdefault(r["seed"], set()).add(r["data_seed"])
    return {
        "runs": len(records),
        "data_seed_by_seed_index": {str(k): sorted(v) for k, v in by_index.items()},
        "data_seed_identical_across_recipes_per_index": all(len(v) == 1 for v in by_index.values()),
        "corpus": sorted({r["corpus"] for r in records}),
        "batch_tokens": sorted({r["batch_tokens"] for r in records}),
        "train_tokens_by_recipe": {r["vintage"]: r["tokens"] for r in records},
        "ladder_policy_by_recipe": {r["vintage"]: r["ladder_policy"] for r in records},
        "verdict": (
            "data_seed equals the seed index and is identical across all 7 recipes, with the same corpus, tokenizer, "
            "context and batch; the loader is not in the release, so data-order identity is probable but not "
            "verified; recipes train on different token counts, so at most a shared order prefix is possible; "
            "initialisation is not shared (architectures differ)"
        ),
    }


def main() -> None:
    frame = pd.read_csv(RAW / "checkpoints.csv")
    frame = frame[frame["roles"].str.contains("model_axis")]
    out: dict[str, Any] = {"protocol": protocol(), "primary_metric": PRIMARY, "metrics": {}}
    for metric in (PRIMARY, *SECONDARY):
        table = frame.pivot(index="recipe", columns="seed", values=metric)
        out["metrics"][metric] = {"icc": seed_icc(table), "pairs": pair_variances(table), "table": table.to_dict()}
    icc = out["metrics"][PRIMARY]["icc"]
    out["predictions"] = {
        "P1_r_in_0_2_to_0_7": 0.2 <= icc["r"] <= 0.7,
        "P1_interval_width_at_least_0_6": (icc["r_ci_clipped_at_0"][1] - icc["r_ci_clipped_at_0"][0]) >= 0.6,
    }
    out["gate_input"] = {"r_point": icc["r"], "meets_20pct_threshold": icc["r"] >= 0.2}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    for metric, block in out["metrics"].items():
        i, p = block["icc"], block["pairs"]
        print(
            f"{metric:20s} r={i['r']:+.2f} CI[{i['r_ci'][0]:+.2f},{i['r_ci'][1]:+.2f}] factor={i['reduction_factor']:.2f} "
            f"p_seed={i['p_seed_effect']:.3f} | pooled matched/mismatched={p['pooled_matched_over_mismatched']:.2f} "
            f"pairs<1: {p['share_of_pairs_with_matched_below_mismatched']:.2f}"
        )
    print(out["predictions"])


if __name__ == "__main__":
    main()
