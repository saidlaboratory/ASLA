"""Task 8: validate the seed-noise model on ten seeds instead of three.

Scores N1-N4 from ``PREDICTIONS_TASK_NOISE_VALIDATION.md`` against PolyPythias
(van der Wal et al., ICLR 2025), which releases 10 seeds at each of five model
sizes with evaluated checkpoints --- as evaluation JSON, so no weights are
involved.

Every resolvability figure in this project estimates sigma from three seeds. This
is the first chance to ask what that estimate is worth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "polypythias"
DATASET = "EleutherAI/polypythias-evals"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"

# Steps present in all 50 runs, chosen to span training: early, middle, late.
STEPS = ("step1000", "step10000", "step50000")
# The accuracy metric with a single scalar per checkpoint. BLiMP is 67 subtasks
# and the bias task is not an accuracy, so lambada is the clean choice.
TASK = "lambada_openai"
METRIC = "acc,none"
SIZES = ("14m", "31m", "70m", "160m", "410m")
N_SUBSAMPLES = 2000
# A cell whose mean accuracy is at or near the floor is not measuring seed
# variance in the metric; it is measuring how often an untrained model gets a
# token right by chance. 14M at step 1000 has mean 0.0000 with a coefficient of
# variation of 3.16, which is a degenerate cell rather than a low-noise one.
MIN_MEAN_FOR_VARIANCE = 0.01
SUBSAMPLE_SIZE = 3
# Unbiased-sigma correction for the sample sd at n=3 (c4 constant).
C4_AT_THREE = 0.8862269254527580


def _listing() -> list[str]:
    result = subprocess.run(
        ["curl", "-sL", f"https://huggingface.co/api/datasets/{DATASET}?full=true"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(result.stdout)
    return [s["rfilename"] for s in payload.get("siblings", [])]


def harvest(out_dir: Path, files: list[str]) -> dict[str, Any]:
    """Download the eval JSON for the chosen steps and extract the metric.

    Each (run, step) has several result files because the suite was run in
    batches; only one carries the task we want, so all are fetched and filtered.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in files if "/" in f and f.split("/")[1] in STEPS and f.endswith(".json")]
    records: list[dict[str, Any]] = []
    for index, relative in enumerate(wanted):
        cache = out_dir / relative.replace("/", "__")
        if not cache.exists():
            subprocess.run(["curl", "-sL", f"{BASE}/{relative}", "-o", str(cache)], check=True, timeout=120)
        try:
            payload = json.loads(cache.read_text())
        except json.JSONDecodeError:
            continue
        results = payload.get("results", {})
        if TASK not in results or METRIC not in results[TASK]:
            continue
        run, step = relative.split("/")[0], relative.split("/")[1]
        size = run.split("-")[1]
        seed = int(run.rsplit("seed", 1)[1])
        records.append(
            {
                "run": run,
                "size": size,
                "seed": seed,
                "step": int(step[4:]),
                "value": float(results[TASK][METRIC]),
                "source_file": relative,
            }
        )
        if index % 200 == 0:
            print(f"  fetched {index}/{len(wanted)}", flush=True)
    return {"records": records, "n_files_considered": len(wanted)}


def _sigma(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One value per (run, step). The suite was evaluated in several passes.

    19 of 65 (run, step) pairs carry more than one result file, and they are not
    all identical --- pythia-410m-seed4 at step 50000 has 0.4473 and 0.4399 from
    two evaluation passes. Counting both as separate seeds inflates the seed
    count AND injects evaluation-rerun variance into an estimate that is supposed
    to measure training-seed variance. The first pass (earliest file name, which
    sorts by timestamp) is kept, and the disagreement is reported rather than
    averaged away.
    """

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records:
        by_key.setdefault((row["run"], row["step"]), []).append(row)

    kept: list[dict[str, Any]] = []
    spreads: list[float] = []
    duplicated = 0
    for rows in by_key.values():
        if len(rows) > 1:
            duplicated += 1
            values = [r["value"] for r in rows]
            spreads.append(max(values) - min(values))
        kept.append(sorted(rows, key=lambda r: r["source_file"])[0])

    return kept, {
        "keys_total": len(by_key),
        "keys_with_duplicates": duplicated,
        "max_rerun_spread": float(max(spreads)) if spreads else 0.0,
        "median_rerun_spread": float(np.median(spreads)) if spreads else 0.0,
        "policy": "keep the earliest evaluation pass per (run, step)",
    }


def analyse(records: list[dict[str, Any]], seed: int, min_seeds: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells: dict[tuple[str, int], list[float]] = {}
    for row in records:
        cells.setdefault((row["size"], row["step"]), []).append(row["value"])

    per_cell = []
    for (size, step), raw in sorted(cells.items()):
        values = np.asarray(sorted(raw), dtype=float)
        if values.size < min_seeds:
            continue
        full = _sigma(values)
        if full <= 0:
            continue
        degenerate = bool(values.mean() < MIN_MEAN_FOR_VARIANCE)
        draws = np.asarray([_sigma(rng.choice(values, size=SUBSAMPLE_SIZE, replace=False)) for _ in range(N_SUBSAMPLES)])
        p10, p50, p90 = (float(np.percentile(draws, q)) for q in (10, 50, 90))
        per_cell.append(
            {
                "size": size,
                "step": step,
                "n_seeds": int(values.size),
                "mean": float(values.mean()),
                "sigma_ten_seeds": full,
                "sigma_three_p10": p10,
                "sigma_three_median": p50,
                "sigma_three_p90": p90,
                "dispersion_p90_over_p10": p90 / p10 if p10 > 0 else float("inf"),
                "median_three_over_ten": p50 / full,
                "coefficient_of_variation": full / float(values.mean()) if values.mean() else None,
                "degenerate_floor_cell": degenerate,
            }
        )

    dispersions = [c["dispersion_p90_over_p10"] for c in per_cell if np.isfinite(c["dispersion_p90_over_p10"])]
    ratios = [c["median_three_over_ten"] for c in per_cell]

    # N3: does sigma shrink with model size? Use the latest common step.
    last_step = max(c["step"] for c in per_cell)
    by_size = {c["size"]: c["sigma_ten_seeds"] for c in per_cell if c["step"] == last_step}
    order = [s for s in SIZES if s in by_size]
    params = [float(s[:-1]) for s in order]
    sigmas = [by_size[s] for s in order]
    from scipy import stats

    # Spearman needs at least three points; with two sizes it is undefined and
    # reporting it as NaN alongside a verdict would invite misreading, so the
    # count is surfaced and the verdict is UNDERPOWERED by construction.
    size_rho = float(stats.spearmanr(params, sigmas).statistic) if len(order) > 2 else None

    # N4: does sigma change across training stages? Degenerate floor cells are
    # excluded: a ratio against a cell whose mean is 0.0000 measures the floor,
    # not a change in seed variance.
    usable = [c for c in per_cell if not c["degenerate_floor_cell"]]
    stage_ratios = []
    steps_seen = sorted({c["step"] for c in usable})
    if len(steps_seen) >= 2:
        early, late = steps_seen[0], steps_seen[-1]
        for size in SIZES:
            first = next((c for c in usable if c["size"] == size and c["step"] == early), None)
            final = next((c for c in usable if c["size"] == size and c["step"] == late), None)
            if first and final and first["sigma_ten_seeds"] > 0:
                stage_ratios.append(
                    {
                        "size": size,
                        "early_step": early,
                        "late_step": late,
                        "sigma_early": first["sigma_ten_seeds"],
                        "sigma_late": final["sigma_ten_seeds"],
                        "ratio_late_over_early": final["sigma_ten_seeds"] / first["sigma_ten_seeds"],
                    }
                )

    return {
        "per_cell": per_cell,
        "n_cells": len(per_cell),
        "N1_dispersion": {
            "median_p90_over_p10": float(np.median(dispersions)) if dispersions else None,
            "min": float(np.min(dispersions)) if dispersions else None,
            "max": float(np.max(dispersions)) if dispersions else None,
            "chi_squared_reference": float(np.sqrt(stats.chi2.ppf(0.9, 2) / stats.chi2.ppf(0.1, 2))),
        },
        "N2_bias": {
            "median_ratio_three_over_ten": float(np.median(ratios)) if ratios else None,
            "fraction_of_cells_underestimating": (float(np.mean([r < 1.0 for r in ratios])) if ratios else None),
            "c4_correction_at_three": C4_AT_THREE,
        },
        "N3_size_scaling": {
            "step": last_step,
            "sizes": order,
            "sigma_by_size": {s: by_size[s] for s in order},
            "spearman_size_vs_sigma": size_rho,
            "n_sizes_available": len(order),
            "verdict": (
                "UNDERPOWERED"
                if len(order) < 3
                else ("REFUTED" if size_rho is not None and size_rho < -0.9 else "CONFIRMED")
            ),
            "note": (
                "lambada_openai was evaluated on a subset of the panel: 14M and 410M carry all "
                "ten seeds, 31M and 160M carry one run each, and 70M carries none at these "
                "steps. Two sizes cannot establish a scaling trend, so N3 is underpowered on "
                "this metric rather than answered."
            ),
        },
        "N4_stage_pooling": {
            "by_size": stage_ratios,
            "excluded_degenerate_cells": [
                {"size": c["size"], "step": c["step"], "mean": c["mean"]} for c in per_cell if c["degenerate_floor_cell"]
            ],
            "median_ratio_late_over_early": (
                float(np.median([r["ratio_late_over_early"] for r in stage_ratios])) if stage_ratios else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "noise" / "noise_validation.json")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--min-seeds", type=int, default=10)
    args = parser.parse_args()

    files = _listing()
    harvested = harvest(args.raw, files)
    records, dedup = deduplicate(harvested["records"])
    if not records:
        raise SystemExit("no records extracted; check the task and metric names")

    payload: dict[str, Any] = {
        "source": {
            "dataset": DATASET,
            "paper": "van der Wal et al., PolyPythias, ICLR 2025 (arXiv:2503.09543)",
            "task": TASK,
            "metric": METRIC,
            "steps": list(STEPS),
            "n_records": len(records),
            "deduplication": dedup,
            "n_files_considered": harvested["n_files_considered"],
            "weights_downloaded": False,
        },
        "subsampling": {
            "n_subsamples": N_SUBSAMPLES,
            "subsample_size": SUBSAMPLE_SIZE,
            "seed": args.seed,
            "without_replacement": True,
        },
        "analysis": analyse(records, args.seed, args.min_seeds),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = {k: v for k, v in payload["analysis"].items() if k.startswith("N")}
    print(json.dumps(summary, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
