"""PREDICTIONS_TASK_FIXED_TARGET.md: the lever-arm effect at a fixed target.

Stages (cached, resumable; run with ``nice -n 19`` and at most two workers):

* ``real``: fixed-target slices and the determined-only target-varied slice on C4;
* ``calibration``: 200 random-candidate worlds, every statistic with its jackknife
  standard error and the worlds' true excess (all pairs and determined pairs);
* ``summarise``: critical values, intervals, predictions and the outcome, written to
  results/target_scoring/fixed_target.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import random_truth, simulate  # noqa: E402
from asla.analysis.confound import window_designs  # noqa: E402
from asla.analysis.fixed_target import (  # noqa: E402
    ADMIT_SHARE,
    admitted_targets,
    fixed_target_slices,
    jackknife,
    target_varied_slices,
    true_statistics,
)

CACHE_DIR = REPO / "data" / "cache"
REAL_CACHE = CACHE_DIR / "fixed_target_real.json"
WORLD_CACHE = CACHE_DIR / "fixed_target_worlds.jsonl"
OUT = REPO / "results" / "target_scoring" / "fixed_target.json"
N_WORLDS = 200
BANDWIDTH = 0.5
LEVEL = 0.95
SEED = 20261028
Z = 1.959963984540054
JOINT_IMPLIED = "confound.json joint model: lever coefficient minus top-rung coefficient"
UNIT = "points of mis-selected pairs per decade of lever arm (log10)"


def _labels(frame) -> list[str]:
    return list(frame.groupby("scale_label")["compute"].first().sort_values().index)


def _setup(frame, designs_, targets: list[int]):
    return fixed_target_slices(designs_, targets), target_varied_slices(designs_), _labels(frame)


def run_real() -> dict[str, Any]:
    confound = __import__("scripts.run_confound", fromlist=["_load_frame"])
    frame = confound._load_frame()
    designs_ = window_designs(frame)
    targets = admitted_targets(designs_)
    ft, tv, labels = _setup(frame, designs_, targets)
    stats_ = jackknife(designs_, ft, tv, labels)
    det_counts = {}
    for d in designs_:
        det_counts[labels[d.target]] = len(d.determined_pairs)
    record = {
        "admit_share": ADMIT_SHARE,
        "admitted_targets": [labels[t] for t in targets],
        "fixed_target_slices": {labels[t]: len(gs) for t, gs in ft.items()},
        "fixed_target_designs": int(sum(len(g) for gs in ft.values() for g in gs)),
        "target_varied_slices": len(tv),
        "target_varied_designs": int(sum(len(g) for g in tv)),
        "determined_pairs_by_target": det_counts,
        "statistics": stats_,
    }
    REAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    REAL_CACHE.write_text(json.dumps(record), encoding="utf-8")
    return record


def run_world(index: int) -> dict[str, Any]:
    calibration = __import__("scripts.run_lever_arm_calibration", fromlist=["_context"])
    ctx = calibration._context()
    real = json.loads(REAL_CACHE.read_text(encoding="utf-8"))
    rng = np.random.default_rng([SEED, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    targets_by_label = {lab: g.groupby("intervention")["mu"].first() for lab, g in truth.final.groupby("scale_label")}
    designs_ = window_designs(observed, targets_by_label)
    labels = _labels(observed)
    targets = [labels.index(lab) for lab in real["admitted_targets"]]  # the real admitted set, fixed in advance
    ft, tv, _ = _setup(observed, designs_, targets)
    return {
        "index": index,
        "statistics": jackknife(designs_, ft, tv, labels),
        "true_excess": [d.true_excess for d in designs_],
        "true_excess_determined": [
            d.true_excess_determined if d.true_excess_determined is not None else float("nan") for d in designs_
        ],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise() -> dict[str, Any]:
    confound = __import__("scripts.run_confound", fromlist=["_load_frame"])
    real = json.loads(REAL_CACHE.read_text(encoding="utf-8"))
    worlds = sorted(_read_jsonl(WORLD_CACHE), key=lambda w: w["index"])
    frame = confound._load_frame()
    template = window_designs(frame)
    labels = _labels(frame)
    targets = [labels.index(lab) for lab in real["admitted_targets"]]
    ft, tv, _ = _setup(frame, template, targets)
    true_all = np.nanmean([w["true_excess"] for w in worlds], axis=0)
    true_det = np.nanmean([w["true_excess_determined"] for w in worlds], axis=0)
    truth = true_statistics(template, true_all, true_det, ft, tv, labels)
    inference = {}
    for key, est in real["statistics"].items():
        ratios = np.array(
            [abs(w["statistics"][key]["estimate"] - truth[key]) / w["statistics"][key]["se"] for w in worlds]
        )
        ratios = ratios[np.isfinite(ratios)]
        k = float(np.quantile(ratios, LEVEL))
        used = max(k, Z)
        e, se = est["estimate"], est["se"]
        inference[key] = {
            "estimate": e,
            "se": se,
            "unit": UNIT,
            "critical_value_calibrated": k,
            "critical_value_used": used,
            "interval": [e - used * se, e + used * se],
            "interval_calibrated": [e - k * se, e + k * se],
            "excludes_zero": bool(e - used * se > 0 or e + used * se < 0),
            "benchmark_true_value": truth[key],
            "coverage_at_1_96": float(np.mean(ratios <= Z)),
            "worlds_used": int(ratios.size),
        }
    conf = json.loads((REPO / "results" / "target_scoring" / "confound.json").read_text(encoding="utf-8"))
    implied = conf["inference"]["joint_lever"]["estimate"] - conf["inference"]["joint_top"]["estimate"]
    pooled = inference["fixed_target_pooled"]
    per_target = [inference[f"fixed_target_{lab}"]["estimate"] for lab in real["admitted_targets"]]
    all_pairs = conf["inference"]["slice_lever"]["estimate"]
    det = inference["target_varied_determined"]
    predictions = {
        "F1_pooled_positive_excludes_zero": bool(pooled["estimate"] > 0 and pooled["excludes_zero"]),
        "F2_pooled_below_joint_implied": bool(pooled["estimate"] < implied),
        "F3_each_target_positive": bool(all(v > 0 for v in per_target)),
        "F4_determined_positive_excludes_zero": bool(det["estimate"] > 0 and det["excludes_zero"]),
        "F5_determined_below_all_pairs": bool(det["estimate"] < all_pairs),
    }
    if pooled["excludes_zero"] and pooled["estimate"] > 0:
        outcome = "fixed_target_positive"
    elif pooled["excludes_zero"]:
        outcome = "fixed_target_negative"
    else:
        outcome = "fixed_target_indistinguishable"
    result = {
        "preregistration": "PREDICTIONS_TASK_FIXED_TARGET.md",
        "metric": "c4_en_bits_per_token",
        "real": {k: v for k, v in real.items() if k != "statistics"},
        "inference": inference,
        "joint_model_implied_fixed_target_slope": {"value": implied, "unit": UNIT, "source": JOINT_IMPLIED},
        "target_varied_all_pairs_confound": {"value": all_pairs, "unit": UNIT},
        "calibration": {"worlds": len(worlds), "bandwidth": BANDWIDTH, "level": LEVEL},
        "simulation_reference": (
            "the confound test's semi-synthetic factor isolation scores against the noiseless truth at the target "
            "(scripts/run_confound.py synthetic_excess); no rerun needed"
        ),
        "predictions": predictions,
        "outcome": outcome,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("real", "calibration", "summarise", "time"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="skip cached worlds (always on)")
    args = parser.parse_args(argv)
    confound = __import__("scripts.run_confound", fromlist=["_run_pool"])
    if args.stage == "real":
        rec = run_real()
        print(f"real: targets {rec['admitted_targets']}, {rec['fixed_target_designs']} fixed-target designs")
    elif args.stage == "time":
        started = time.time()
        run_world(10_000)
        print(f"one world: {time.time() - started:.1f}s")
    elif args.stage == "calibration":
        done = {w["index"] for w in _read_jsonl(WORLD_CACHE)}
        todo = [i for i in range(N_WORLDS) if i not in done]
        confound._run_pool(run_world, todo, WORLD_CACHE, min(args.workers, 2))
    else:
        print(f"outcome: {summarise()['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
