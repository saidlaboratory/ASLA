"""PREDICTIONS_TASK_RECIPE_DEPENDENCE.md: how much the fixed-target slope depends on the recipes in the set.

Each recipe set is scored with the frozen code on that set alone (window designs, evidence, pairs and
jackknife all restricted to the set), at the pre-registered targets 150M, 300M, 530M and 1B.

Writes results/target_scoring/recipe_dependence.json. Subset results are cached; resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.confound import window_designs  # noqa: E402
from asla.analysis.fixed_target import fixed_target_slices, jackknife, target_varied_slices  # noqa: E402

CACHE = REPO / "data" / "cache" / "recipe_dependence.jsonl"
OUT = REPO / "results" / "target_scoring" / "recipe_dependence.json"
TARGETS = ("150M", "300M", "530M", "1B")
KS = (5, 10, 15, 20)
N_SUBSETS = 200
SEED = 20261031
Z = 1.959963984540054
UNIT = "points of mis-selected pairs per decade (log10) of lever arm"
FAMILIES = {
    "DCLM-Baseline": [
        "DCLM-Baseline",
        "DCLM-Baseline (QC 10%)",
        "DCLM-Baseline (QC 20%)",
        "DCLM-Baseline (QC 7%, FW2)",
        "DCLM-Baseline (QC 7%, FW3)",
        "DCLM-Baseline (QC FW 10%)",
        "DCLM-Baseline (QC FW 3%)",
    ],
    "Dolma": [
        "Dolma1.6++",
        "Dolma1.7",
        "Dolma1.7 (no Flan)",
        "Dolma1.7 (no Reddit)",
        "Dolma1.7 (no code)",
        "Dolma1.7 (no math, code)",
    ],
    "Falcon": [
        "Falcon",
        "Falcon+CC",
        "Falcon+CC (QC 10%)",
        "Falcon+CC (QC 20%)",
        "Falcon+CC (QC Orig 10%)",
        "Falcon+CC (QC Tulu 10%)",
    ],
    "DCLM/Dolma mixes": [
        "DCLM-Baseline 25% / Dolma 75%",
        "DCLM-Baseline 50% / Dolma 50%",
        "DCLM-Baseline 75% / Dolma 25%",
    ],
    "Other": ["C4", "FineWeb-Edu", "FineWeb-Pro"],
}
_FRAME: dict = {}


def frame():
    if not _FRAME:
        confound = __import__("scripts.run_confound", fromlist=["_load_frame"])
        f = confound._load_frame()
        _FRAME["f"] = f
        _FRAME["labels"] = list(f.groupby("scale_label")["compute"].first().sort_values().index)
    return _FRAME["f"], _FRAME["labels"]


def slope_for(recipes: list[str]) -> dict:
    f, labels = frame()
    sub = f[f["intervention"].isin(recipes)].reset_index(drop=True)
    ds = window_designs(sub)
    targets = [labels.index(t) for t in TARGETS]
    ft = {t: gs for t, gs in fixed_target_slices(ds, targets).items() if gs}
    tv = target_varied_slices(ds)
    jk = jackknife(ds, ft, tv, labels)["fixed_target_pooled"]
    est, se = jk["estimate"], jk["se"]
    return {
        "n_recipes": len(recipes),
        "estimate": est,
        "se": se,
        "interval": [est - Z * se, est + Z * se],
        "positive_excludes_zero": bool(est - Z * se > 0),
    }


def run_subset(job: int) -> dict:
    k = KS[job // N_SUBSETS]
    rng = np.random.default_rng([SEED, job])
    f, _ = frame()
    names = sorted(f["intervention"].astype(str).unique())
    chosen = sorted(rng.choice(names, size=k, replace=False).tolist())
    return {"job": job, "k": k, "recipes": chosen, **slope_for(chosen)}


def _read() -> list[dict]:
    if not CACHE.exists():
        return []
    return [json.loads(x) for x in CACHE.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("subsets", "summarise"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="skip cached subsets (always on)")
    args = parser.parse_args(argv)
    if args.stage == "subsets":
        confound = __import__("scripts.run_confound", fromlist=["_run_pool"])
        done = {r["job"] for r in _read()}
        jobs = [j for j in range(len(KS) * N_SUBSETS) if j not in done]
        confound._run_pool(run_subset, jobs, CACHE, min(args.workers, 2))
        return 0

    f, _ = frame()
    all_names = sorted(f["intervention"].astype(str).unique())
    assert sorted(sum(FAMILIES.values(), [])) == all_names, "families must partition the 25 recipes"
    full = slope_for(all_names)
    within = {fam: slope_for(r) for fam, r in FAMILIES.items() if len(r) >= 5}
    leave_out = {fam: slope_for([n for n in all_names if n not in r]) for fam, r in FAMILIES.items()}
    subsets = _read()
    by_k = {}
    for k in KS:
        rows = [r for r in subsets if r["k"] == k]
        est = np.array([r["estimate"] for r in rows])
        by_k[str(k)] = {
            "n_subsets": len(rows),
            "mean": float(est.mean()),
            "sd": float(est.std(ddof=1)),
            "share_positive_excluding_zero": float(np.mean([r["positive_excludes_zero"] for r in rows])),
            "share_positive": float(np.mean(est > 0)),
            "quantiles_5_50_95": [float(q) for q in np.quantile(est, [0.05, 0.5, 0.95])],
            "median_se": float(np.median([r["se"] for r in rows])),
        }
    sd_bands = {"5": (6, 15), "10": (3, 8), "15": (2, 5), "20": (1, 3)}
    share_bands = {"5": (0.10, 0.50), "10": (0.40, 0.90), "15": (0.70, 1.0), "20": (0.90, 1.0)}
    predictions = {
        "F1_all_within_family_positive": all(v["estimate"] > 0 for v in within.values()),
        "F2_at_most_one_within_family_excludes_zero": sum(v["positive_excludes_zero"] or v["interval"][1] < 0 for v in within.values()) <= 1,
        "F3_leave_one_family_out_in_8_15_and_excludes_zero": all(
            8 <= v["estimate"] <= 15 and v["interval"][0] > 0 for v in leave_out.values()
        ),
        "S1_mean_in_10_to_13_5_every_k": all(10 <= v["mean"] <= 13.5 for v in by_k.values()),
        "S2_sd_falls_with_k_and_in_bands": all(sd_bands[k][0] <= by_k[k]["sd"] <= sd_bands[k][1] for k in by_k)
        and all(by_k[str(a)]["sd"] > by_k[str(b)]["sd"] for a, b in zip(KS, KS[1:])),
        "S3_share_rises_with_k_and_in_bands": all(
            share_bands[k][0] <= by_k[k]["share_positive_excluding_zero"] <= share_bands[k][1] for k in by_k
        )
        and all(
            by_k[str(a)]["share_positive_excluding_zero"] <= by_k[str(b)]["share_positive_excluding_zero"]
            for a, b in zip(KS, KS[1:])
        ),
    }
    holds_every_family = all(v["positive_excludes_zero"] for v in within.values()) and all(
        v["interval"][0] > 0 for v in leave_out.values()
    )
    driven_by = [fam for fam, v in leave_out.items() if v["interval"][0] <= 0 or v["estimate"] < 11.7 / 2]
    too_variable = by_k["10"]["share_positive_excluding_zero"] < 0.5
    result = {
        "preregistration": "PREDICTIONS_TASK_RECIPE_DEPENDENCE.md",
        "metric": "c4_en_bits_per_token",
        "unit": UNIT,
        "targets": list(TARGETS),
        "full_set": full,
        "families": FAMILIES,
        "within_family": within,
        "leave_one_family_out": leave_out,
        "random_subsets": by_k,
        "predictions": predictions,
        "outcome": {
            "holds_within_every_family": bool(holds_every_family),
            "driven_by_family": driven_by,
            "too_variable_at_k10": bool(too_variable),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("full_set", "within_family", "leave_one_family_out", "random_subsets", "predictions", "outcome")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
