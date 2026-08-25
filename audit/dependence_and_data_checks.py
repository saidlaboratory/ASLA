"""A4 (FDR under dependence) and A6 (adversarial data checks).

A4. The 300 pairwise Welch tests at the target are not independent: every pair
shares interventions with 46 others, so the test statistics are dependent.
Benjamini-Hochberg controls FDR under independence or positive regression
dependence (PRDS). Pairwise-comparison statistics built from a common pool of
group means are *not* guaranteed PRDS - contrasts sharing a group are
positively correlated, but contrasts sharing a group with opposite sign are
negatively correlated. We therefore recompute every significance count under
Benjamini-Yekutieli, which controls FDR under arbitrary dependence, and report
both. We also measure the realised correlation structure by simulation under
the global null.

A6. Data integrity: 6ND against DataDecide's own reported compute, duplicate
rows, target leakage into the fitting budgets, seed alignment across recipes,
and whether the three "seeds" per cell are genuinely distinct runs.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_EVAL = REPO / "data" / "raw" / "datadecide" / "eval_macro_avg.parquet"
RAW_PPL = REPO / "data" / "raw" / "datadecide" / "ppl_results.parquet"


# ------------------------------------------------------------------ A4


def benjamini_yekutieli(p_values: list[float | None], q: float = 0.05) -> list[bool]:
    m = sum(1 for p in p_values if p is not None and np.isfinite(p))
    if m == 0:
        return [False] * len(p_values)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return ir.benjamini_hochberg(p_values, q=q / harmonic)


def recount_under_dependence(metric: str, q: float = 0.05) -> dict[str, Any]:
    df, budgets, target = ir.design_frame(metric)
    truth = ir.truth_scores(df, target)
    proj = ir.project_scores(df, budgets, target)
    single = ir.single_scale_scores(df, budgets)
    rows = df[np.isclose(df["compute"], target)]
    by_name = {str(n): g["bpb"].to_numpy(dtype=float) for n, g in rows.groupby("intervention", sort=True)}
    out: dict[str, Any] = {"metric": metric, "q": q}
    for label, scores in (("projection", proj), ("single_scale", single)):
        pairs = ir.flips(scores, truth)
        p_values = [ir.welch_p(by_name[a], by_name[b]) for a, b in pairs]
        bh = ir.benjamini_hochberg(p_values, q=q)
        by = benjamini_yekutieli(p_values, q=q)
        # Bonferroni over all 300 pairs, the most conservative reference.
        n_all_pairs = len(truth) * (len(truth) - 1) // 2
        bonf = [bool(p is not None and p <= q / n_all_pairs) for p in p_values]
        out[label] = {
            "n_flips": len(pairs),
            "n_significant_BH": int(sum(bh)),
            "n_significant_BY": int(sum(by)),
            "n_significant_bonferroni_over_all_pairs": int(sum(bonf)),
            "harmonic_factor": float(
                np.sum(1.0 / np.arange(1, max(1, len(p_values)) + 1))
            ),
        }
    return out


def null_dependence_simulation(n_interventions: int = 25, n_seeds: int = 3, n_trials: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Realised FDR of BH and BY under the global null with shared-group dependence.

    Under the complete null every intervention has the same mean, so every
    rejection is a false discovery and the false discovery proportion equals
    1 whenever anything is rejected. The family-wise error rate is therefore
    the quantity to compare against q.
    """

    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(n_interventions), 2))
    any_bh = 0
    any_by = 0
    for _ in range(n_trials):
        samples = rng.normal(0.0, 1.0, size=(n_interventions, n_seeds))
        p_values = [ir.welch_p(samples[a], samples[b]) for a, b in pairs]
        any_bh += int(any(ir.benjamini_hochberg(p_values, q=0.05)))
        any_by += int(any(benjamini_yekutieli(p_values, q=0.05)))
    return {
        "n_interventions": n_interventions,
        "n_seeds": n_seeds,
        "n_pairs": len(pairs),
        "n_trials": n_trials,
        "familywise_error_rate_BH": any_bh / n_trials,
        "familywise_error_rate_BY": any_by / n_trials,
        "note": "under the complete null FDR == FWER, so these are directly comparable to q=0.05",
    }


# ------------------------------------------------------------------ A6


def data_integrity(metric: str) -> dict[str, Any]:
    df = pd.read_parquet(ir.TABLES[metric])
    checks: dict[str, Any] = {"metric": metric, "n_rows": int(len(df))}
    checks["duplicate_identity_rows"] = int(df.duplicated(["intervention", "compute", "seed"]).sum())
    checks["duplicate_full_rows"] = int(df.duplicated().sum())
    derived = 6.0 * df["params_n"].astype(float) * df["tokens_d"].astype(float)
    checks["compute_equals_6ND"] = bool(np.allclose(df["compute"].astype(float), derived, rtol=1e-9))
    checks["max_abs_6ND_relative_error"] = float(
        np.max(np.abs(df["compute"] - derived) / df["compute"])
    )
    cells = df.groupby(["intervention", "compute"])["seed"].nunique()
    checks["seeds_per_cell_min"] = int(cells.min())
    checks["seeds_per_cell_max"] = int(cells.max())
    checks["cells_with_fewer_than_3_seeds"] = int((cells < 3).sum())
    seed_sets = df.groupby(["intervention", "compute"])["seed"].apply(lambda s: tuple(sorted(s.unique())))
    checks["distinct_seed_labelings"] = sorted({str(v) for v in seed_sets.unique()})
    # Are the 3 "seeds" genuinely distinct runs, or duplicated values?
    identical = df.groupby(["intervention", "compute"])["bpb"].nunique()
    checks["cells_where_all_seeds_identical"] = int((identical == 1).sum())
    checks["cells_total"] = int(len(identical))
    # Target leakage: no fitting budget may equal the target.
    target = float(df[df["scale_label"] == "1B"]["compute"].iloc[0])
    fit = df[(df["compute"] < target) & (~df["scale_label"].isin(["750M", "530M"]))]
    checks["target_in_fit_budgets"] = bool(np.any(np.isclose(fit["compute"].astype(float), target)))
    checks["max_fit_budget_over_target"] = float(fit["compute"].max() / target)
    # Seed-label alignment across recipes at each scale
    by_scale = df.groupby(["scale_label", "intervention"])["seed_label"].apply(lambda s: tuple(sorted(set(s))))
    misaligned = {}
    for scale, group in by_scale.groupby(level=0):
        labelings = set(group.tolist())
        if len(labelings) > 1:
            misaligned[str(scale)] = sorted(str(x) for x in labelings)
    checks["scales_with_misaligned_seed_labels"] = misaligned
    return checks


def cross_check_raw_compute() -> dict[str, Any]:
    """Compare the harvested compute against DataDecide's own released compute column."""

    if not RAW_EVAL.exists():
        return {"available": False, "reason": f"{RAW_EVAL} not present (git-ignored cache); rerun asla harvest-datadecide"}
    raw = pd.read_parquet(RAW_EVAL, columns=["params", "data", "task", "step", "seed", "tokens", "compute"])
    raw = raw[raw["task"] == "olmes_10_macro_avg"]
    harvested = pd.read_parquet(ir.TABLES["olmes_macro_error"])
    renamed = raw.rename(
        columns={"params": "scale_label", "data": "intervention", "seed": "seed_label", "compute": "raw_compute"}
    )
    merged = harvested.merge(
        renamed,
        on=["scale_label", "intervention", "seed_label", "step"],
        how="left",
        suffixes=("", "_raw"),
    )
    matched = merged["raw_compute"].notna()
    rel = np.abs(merged.loc[matched, "compute"] - merged.loc[matched, "raw_compute"]) / merged.loc[matched, "raw_compute"]
    return {
        "available": True,
        "n_harvested_rows": int(len(harvested)),
        "n_matched_to_raw": int(matched.sum()),
        "max_relative_compute_error_vs_released": float(rel.max()) if matched.any() else None,
        "all_within_1e_9": bool(rel.max() < 1e-9) if matched.any() else None,
    }


def main() -> int:
    out: dict[str, Any] = {}
    out["fdr_under_dependence"] = [recount_under_dependence(m) for m in ir.TABLES]
    out["null_simulation"] = null_dependence_simulation()
    out["data_integrity"] = [data_integrity(m) for m in ir.TABLES]
    out["raw_compute_cross_check"] = cross_check_raw_compute()
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "a4_a6_checks.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("=== A4: significance counts under BH vs BY vs Bonferroni")
    for row in out["fdr_under_dependence"]:
        print(f"  {row['metric']}")
        for label in ("projection", "single_scale"):
            e = row[label]
            print(
                f"    {label:13s} flips={e['n_flips']:3d}  BH={e['n_significant_BH']:3d}  "
                f"BY={e['n_significant_BY']:3d}  Bonf={e['n_significant_bonferroni_over_all_pairs']:3d}"
            )
    print("\n=== A4: realised error rate under the complete null (q=0.05)")
    print(f"  {json.dumps(out['null_simulation'], indent=2)}")
    print("\n=== A6: data integrity")
    for row in out["data_integrity"]:
        print(f"  {row['metric']}: dup_identity={row['duplicate_identity_rows']} 6ND_ok={row['compute_equals_6ND']} "
              f"seeds/cell={row['seeds_per_cell_min']}-{row['seeds_per_cell_max']} "
              f"all_seeds_identical_cells={row['cells_where_all_seeds_identical']}/{row['cells_total']} "
              f"target_leak={row['target_in_fit_budgets']} misaligned_scales={list(row['scales_with_misaligned_seed_labels'])}")
    print(f"\n=== A6: raw compute cross-check\n  {json.dumps(out['raw_compute_cross_check'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
