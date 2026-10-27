"""Power of the allocation comparisons (PREDICTIONS_TASK_FIXED_TARGET.md, Task 3).

The restated allocation claims compare projection fitted on each optimised design's
supported budgets (target 1B) with single-scale ranking at the ladder's top, scored
by expected error, with the U-statistic standard error and the C4 calibrated
random-candidate critical value. Here the same comparison runs in random-candidate
semi-synthetic worlds built as the calibration benchmark. Power is the share of
worlds whose |estimate| / standard error exceeds that critical value; the benchmark
effect is the world-mean true difference against the noiseless target.

Writes ``results/target_scoring/allocation_power.json``. Cached and resumable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import random_truth, simulate, true_mis_rate  # noqa: E402
from asla.analysis.rankers import projection_ranker, single_scale_ranker  # noqa: E402
from asla.analysis.target_scoring import expected_charges, u_statistic_test  # noqa: E402
from asla.models import FitError  # noqa: E402
from asla.provenance import Source  # noqa: E402

CACHE = REPO / "data" / "cache" / "allocation_power.jsonl"
OUT = REPO / "results" / "target_scoring" / "allocation_power.json"
N_WORLDS = 300
BANDWIDTH = 0.5
SEED = 20261029


def _designs() -> tuple[list[float], float, list[tuple[float, tuple[float, ...]]]]:
    alloc = Source.load("allocation/allocation_1B.json")
    ladder = [float(b) for b in alloc.get("ladder")]
    out = []
    for _, entry in sorted(alloc.get("sweep").items(), key=lambda kv: kv[1]["budget_fraction"]):
        runs = entry.get("scored", {}).get("decision_optimal", {}).get("rounded_runs")
        if not runs:
            continue
        support = tuple(b for b, r in zip(ladder, runs) if r >= 1)
        if len(support) >= 3:
            out.append((float(entry["budget_fraction"]), support))
    return ladder, alloc.number("target_compute"), out


def run_world(index: int) -> dict[str, Any]:
    rescoring = importlib.import_module("scripts.run_rescoring")
    calibration = importlib.import_module("scripts.run_lever_arm_calibration")
    ctx = calibration._context()
    ladder, target, designs = _designs()
    rng = np.random.default_rng([SEED, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    target_rows = truth.final[np.isclose(truth.final["compute"], target)]
    truth_gaps = rescoring._gaps(target_rows.groupby("intervention")["mu"].first())
    evidence = rescoring._evidence(observed, target)
    baseline = rescoring._gaps(single_scale_ranker(observed, tuple(ladder), target))
    right = expected_charges(baseline, evidence)
    base_true = true_mis_rate(baseline, truth_gaps)
    rows = []
    for fraction, support in designs:
        try:
            gaps = rescoring._gaps(projection_ranker(observed, support, target))
        except (FitError, ValueError):
            rows.append({"fraction": fraction, "failed": True})
            continue
        left = expected_charges(gaps, evidence)
        kernel = {p: 100 * (left[p] - right[p]) for p in left if p in right}
        test = u_statistic_test(kernel)
        rows.append(
            {
                "fraction": fraction,
                "estimate": float(test["mean"]),
                "se": float(test["standard_error"]),
                "zeta1": float(test["zeta1"]),
                "true_difference": 100 * (true_mis_rate(gaps, truth_gaps) - base_true),
            }
        )
    return {"index": index, "rows": rows}


def _read() -> list[dict[str, Any]]:
    if not CACHE.exists():
        return []
    return [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise() -> dict[str, Any]:
    worlds = _read()
    procedures = Source.load("target_scoring/calibration.json").get("calibrated")
    k = float(procedures["random_u"]["critical_value"])
    by_fraction: dict[float, list[dict[str, Any]]] = {}
    for w in worlds:
        for row in w["rows"]:
            by_fraction.setdefault(row["fraction"], []).append(row)
    out = {}
    for fraction, rows in sorted(by_fraction.items()):
        ok = [r for r in rows if not r.get("failed") and r["se"] > 0]
        reject = [abs(r["estimate"]) / r["se"] > k for r in ok]
        nonneg = [r for r in ok if r["zeta1"] >= 0]
        out[f"{fraction:g}"] = {
            "worlds": len(ok),
            "zeta1_negative_share": float(np.mean([r["zeta1"] < 0 for r in ok])),
            "power_if_zeta1_nonnegative": float(np.mean([abs(r["estimate"]) / r["se"] > k for r in nonneg]))
            if nonneg
            else None,
            "failed_fits": len(rows) - len(ok),
            "benchmark_true_difference_pp": float(np.mean([r["true_difference"] for r in ok])),
            "power_at_critical_value": float(np.mean(reject)),
        }
    record = {
        "preregistration": "PREDICTIONS_TASK_FIXED_TARGET.md (Task 3: a power calculation, no prediction)",
        "critical_value": k,
        "critical_value_source": "calibration.json calibrated.random_u (C4, primary design), as the restated claims use",
        "worlds": len(worlds),
        "by_budget_fraction": out,
        "diagnostic": (
            "The allocation comparisons are scored against Bonferroni-calibrated target evidence, so most kernel "
            "entries are zero and the unbiased candidate-level component zeta1 can be negative, which shrinks the "
            "U-statistic standard error toward zero. Power is reported as pre-registered, with the share of worlds "
            "where zeta1 < 0 and the power over the remaining worlds alongside."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarise"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="skip cached worlds (always on)")
    args = parser.parse_args(argv)
    if args.stage == "run":
        confound = importlib.import_module("scripts.run_confound")
        done = {w["index"] for w in _read()}
        confound._run_pool(run_world, [i for i in range(N_WORLDS) if i not in done], CACHE, min(args.workers, 2))
    else:
        rec = summarise()
        print({k: v["power_at_critical_value"] for k, v in rec["by_budget_fraction"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
