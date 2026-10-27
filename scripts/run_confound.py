"""PREDICTIONS_TASK_CONFOUND.md: separate lever arm from top-rung size and rung count.

Stages (each cached one record per line under data/cache/, resumable):

* ``real``: the 220 window designs on DataDecide C4, every statistic with its
  leave-one-recipe-out jackknife standard error, and the determined share of target
  pairs at each target.
* ``calibration``: random-candidate semi-synthetic worlds (the lever-arm
  calibration's construction); each world records every statistic, its jackknife
  standard error and the worlds' true excess per design.
* ``synthetic``: fixed-candidate worlds on continuous ladders, each factor varied
  with the other two exactly fixed, under the C4 noise and recipe-specific
  misspecification profile.
* ``summarise``: critical values, intervals, effects, predictions and the outcome,
  written to results/target_scoring/confound.json.

Run with ``nice -n 19`` and at most two workers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import random_truth, simulate  # noqa: E402
from asla.analysis.confound import STATISTICS, excess_vector, jackknife, slices, statistics, window_designs  # noqa: E402

METRIC_FILE = REPO / "data" / "datadecide_runs.parquet"
CACHE_DIR = REPO / "data" / "cache"
REAL_CACHE = CACHE_DIR / "confound_real.json"
WORLD_CACHE = CACHE_DIR / "confound_worlds.jsonl"
SYNTH_CACHE = CACHE_DIR / "confound_synthetic.jsonl"
OUT = REPO / "results" / "target_scoring" / "confound.json"
N_WORLDS = 200
N_SYNTH = 300
BANDWIDTH = 0.5
LEVEL = 0.95
SEED = 20261027
Z = 1.959963984540054

# Semi-synthetic ladders (pre-registered).
SPACING = 0.35
BASE = {"rungs": 6, "log_top": 19.0, "lever": 1.0}
LEVERS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
TOPS = (17.75, 18.25, 18.75, 19.25, 19.75)
RUNGS = (3, 4, 5, 6, 7, 8)
SEEDS = 3
_DECADE_LEVER = "points of mis-selected pairs per decade (log10) of lever arm"
_DECADE_TOP = "points of mis-selected pairs per decade (log10) of top-rung compute"
_RUNG = "points of mis-selected pairs per fitted rung"
STAT_UNITS = {
    "joint_lever": _DECADE_LEVER,
    "joint_top": _DECADE_TOP,
    "joint_rungs": _RUNG,
    "slice_lever": _DECADE_LEVER,
    "slice_top": _DECADE_TOP,
    "slice_rungs": _RUNG,
}
SYNTH_UNITS = {"lever": _DECADE_LEVER, "top": _DECADE_TOP, "rungs": _RUNG}


def _load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(METRIC_FILE)
    return frame[frame["scale_label"] != "750M"].reset_index(drop=True)


# ----------------------------------------------------------------------------- real data


def run_real() -> dict[str, Any]:
    frame = _load_frame()
    all_designs = window_designs(frame)
    groups = slices(all_designs)
    stats_ = jackknife(all_designs, groups)
    order = frame.groupby("scale_label")["compute"].first().sort_values()
    determined = {}
    for d in all_designs:
        determined[str(order.index[d.target])] = d.determined_share
    record = {
        "n_designs": len(all_designs),
        "slice_counts": {k: len(v) for k, v in groups.items()},
        "slice_designs": {k: int(sum(len(g) for g in v)) for k, v in groups.items()},
        "statistics": stats_,
        "determined_share_by_target": determined,
        "designs": [
            {
                "start": str(order.index[d.start]),
                "top": str(order.index[d.top]),
                "target": str(order.index[d.target]),
                "lever": d.lever,
                "log_top": d.log_top,
                "rungs": d.rungs,
                "excess": float(np.mean(list(d.kernel.values()))),
            }
            for d in all_designs
        ],
    }
    REAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    REAL_CACHE.write_text(json.dumps(record), encoding="utf-8")
    return record


# ----------------------------------------------------------------------------- calibration worlds


def run_world(index: int) -> dict[str, Any]:
    calibration = __import__("scripts.run_lever_arm_calibration", fromlist=["_context"])
    ctx = calibration._context()
    rng = np.random.default_rng([SEED, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    targets = {label: g.groupby("intervention")["mu"].first() for label, g in truth.final.groupby("scale_label")}
    all_designs = window_designs(observed, targets)
    groups = slices(all_designs)
    stats_ = jackknife(all_designs, groups)
    return {
        "index": index,
        "statistics": stats_,
        "true_excess": [d.true_excess for d in all_designs],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_pool(func, todo: list[int], path: Path, workers: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as sink:
        futures = {pool.submit(func, i): i for i in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            sink.write(json.dumps(future.result()) + "\n")
            sink.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{path.name}: {n}/{len(todo)} in {time.time() - started:.0f}s", flush=True)


# ----------------------------------------------------------------------------- semi-synthetic factor isolation


_SYNTH: dict[str, Any] = {}


def _synth_context() -> dict[str, Any]:
    if _SYNTH:
        return _SYNTH
    arms = __import__("scripts.run_lever_arm_arms", fromlist=["_context"])
    calibration = __import__("scripts.run_lever_arm_calibration", fromlist=["_context"])
    actx, cctx = arms._context(), calibration._context()
    log_c = np.log10(actx["compute"][actx["scales"]].to_numpy(dtype=float))
    residual = pd.DataFrame(actx["arm_residual"]["B2"], index=actx["recipes"], columns=actx["scales"])
    sigma = cctx["base"].sigma
    _SYNTH.update(
        recipes=list(actx["recipes"]),
        log_c=log_c,
        params={r: tuple(cctx["params"][r]) for r in actx["recipes"]},
        residual={r: residual.loc[r].to_numpy(dtype=float) for r in actx["recipes"]},
        sigma={r: np.array([sigma[(str(r), str(s))] for s in actx["scales"]]) for r in actx["recipes"]},
    )
    return _SYNTH


def synthetic_designs() -> list[dict[str, Any]]:
    out = []
    for value in LEVERS:
        out.append({"factor": "lever", "level": value, **{**BASE, "lever": value}})
    for value in TOPS:
        out.append({"factor": "top", "level": value, **{**BASE, "log_top": value}})
    for value in RUNGS:
        out.append({"factor": "rungs", "level": value, **{**BASE, "rungs": value}})
    return out


def _truth_at(recipe: str, log_compute: np.ndarray) -> np.ndarray:
    ctx = _synth_context()
    e, a, alpha = ctx["params"][recipe]
    base = e + a * (10.0**log_compute) ** (-alpha)
    return base + np.interp(log_compute, ctx["log_c"], ctx["residual"][recipe])


def _sd_at(recipe: str, log_compute: np.ndarray) -> np.ndarray:
    ctx = _synth_context()
    return np.interp(log_compute, ctx["log_c"], ctx["sigma"][recipe])


def synthetic_excess(spec: dict[str, Any], rng: np.random.Generator | None) -> float:
    """True excess (points) of projection over single-scale for one continuous design; no noise if rng is None."""

    from itertools import combinations

    from asla.models import FitError, bpb_power_law, fit_power_law

    ctx = _synth_context()
    rung_logs = spec["log_top"] - SPACING * np.arange(spec["rungs"])[::-1]
    target_log = spec["log_top"] + spec["lever"]
    names = ctx["recipes"]
    projected, single, truth = {}, {}, {}
    for r in names:
        mu = _truth_at(r, rung_logs)
        if rng is not None:
            mu = mu + _sd_at(r, rung_logs) * rng.standard_normal((SEEDS, mu.size)).mean(axis=0)
        try:
            params = fit_power_law(10.0**rung_logs, mu)
            projected[r] = float(bpb_power_law(10.0**target_log, *params))
        except (FitError, RuntimeError, ValueError):
            projected[r] = float("nan")
        single[r] = float(mu[-1])
        truth[r] = float(_truth_at(r, np.array([target_log]))[0])

    def mis(pred: dict[str, float]) -> float:
        wrong = []
        for x, y in combinations(names, 2):
            p, t = pred[x] - pred[y], truth[x] - truth[y]
            if np.isfinite(p) and p != 0 and t != 0:
                wrong.append(np.sign(p) != np.sign(t))
        return float(np.mean(wrong))

    return 100 * (mis(projected) - mis(single))


def run_synthetic_world(index: int) -> dict[str, Any]:
    rng = np.random.default_rng([SEED, 7, index])
    return {"index": index, "excess": [synthetic_excess(spec, rng) for spec in synthetic_designs()]}


# ----------------------------------------------------------------------------- summary


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def summarise() -> dict[str, Any]:
    real = json.loads(REAL_CACHE.read_text(encoding="utf-8"))
    worlds = sorted(_read_jsonl(WORLD_CACHE), key=lambda w: w["index"])
    synth = sorted(_read_jsonl(SYNTH_CACHE), key=lambda w: w["index"])

    # True value of each statistic: the statistic of the world-mean true excess.
    frame = _load_frame()
    template = window_designs(frame)
    groups = slices(template)
    mean_true = np.mean([w["true_excess"] for w in worlds], axis=0)
    true_values = statistics(template, mean_true, groups)
    inference = {}
    for key in STATISTICS:
        ratios = np.array(
            [abs(w["statistics"][key]["estimate"] - true_values[key]) / w["statistics"][key]["se"] for w in worlds]
        )
        k = float(np.quantile(ratios, LEVEL))
        est, se = real["statistics"][key]["estimate"], real["statistics"][key]["se"]
        inference[key] = {
            "unit": STAT_UNITS[key],
            "estimate": est,
            "se": se,
            "critical_value": k,
            "ci_calibrated": [est - k * se, est + k * se],
            "ci_1_96": [est - Z * se, est + Z * se],
            "excludes_zero_calibrated": bool(est - k * se > 0 or est + k * se < 0),
            "benchmark_true_value": true_values[key],
            "coverage_at_1_96": float(np.mean(ratios <= Z)),
        }

    specs = synthetic_designs()
    per_world = np.array([w["excess"] for w in synth])
    noiseless = [synthetic_excess(spec, None) for spec in specs]
    effects = {}
    for factor in ("lever", "top", "rungs"):
        idx = [i for i, s in enumerate(specs) if s["factor"] == factor]
        x = np.array([specs[i]["level"] for i in idx], dtype=float)
        slopes = np.array([_slope(x, row[idx]) for row in per_world])
        mean = per_world[:, idx].mean(axis=0)
        effects[factor] = {
            "slope_unit": SYNTH_UNITS[factor],
            "levels": x.tolist(),
            "mean_true_excess": mean.tolist(),
            "mc_se_by_level": (per_world[:, idx].std(axis=0, ddof=1) / np.sqrt(len(per_world))).tolist(),
            "slope": _slope(x, mean),
            "slope_mc_se": float(slopes.std(ddof=1) / np.sqrt(len(slopes))),
            "noiseless_excess": [noiseless[i] for i in idx],
            "noiseless_slope": _slope(x, np.array([noiseless[i] for i in idx])),
        }
    for e in effects.values():
        e["slope_ci"] = [e["slope"] - Z * e["slope_mc_se"], e["slope"] + Z * e["slope_mc_se"]]

    det = real["determined_share_by_target"]
    smallest = min(det, key=lambda lab: frame[frame["scale_label"] == lab]["compute"].iloc[0])
    lev, top, rng_ = inference["joint_lever"], inference["joint_top"], inference["joint_rungs"]
    predictions = {
        "P1_joint_lever_positive_excludes_zero": bool(lev["estimate"] > 0 and lev["excludes_zero_calibrated"]),
        "P2_joint_top_negative": bool(top["estimate"] < 0),
        "P3_joint_rungs_includes_zero": bool(not rng_["excludes_zero_calibrated"]),
        "P4_slice_lever_positive_excludes_zero": bool(
            inference["slice_lever"]["estimate"] > 0 and inference["slice_lever"]["excludes_zero_calibrated"]
        ),
        "P5_slice_top_negative": bool(inference["slice_top"]["estimate"] < 0),
        "P6_slice_rungs_includes_zero": bool(not inference["slice_rungs"]["excludes_zero_calibrated"]),
        "P7_determined_smallest_below_1B": bool(det[smallest] < det["1B"]),
        "P8_synthetic_lever_positive_excludes_zero": bool(effects["lever"]["slope_ci"][0] > 0),
        "P9_synthetic_top_negative": bool(effects["top"]["slope"] < 0),
        "P10_synthetic_rung_effect_below_half_decade_lever": bool(
            abs(effects["rungs"]["slope"]) < abs(effects["lever"]["slope"]) * 0.5
        ),
    }

    lever_survives = lev["estimate"] > 0 and lev["excludes_zero_calibrated"] and inference["slice_lever"]["estimate"] > 0
    top_explains = (
        not lev["excludes_zero_calibrated"]
        and top["excludes_zero_calibrated"]
        and np.sign(inference["slice_top"]["estimate"]) == np.sign(top["estimate"])
    )
    outcome = "lever_arm_survives" if lever_survives else ("top_rung_explains" if top_explains else "not_separable")
    result = {
        "preregistration": "PREDICTIONS_TASK_CONFOUND.md",
        "metric": "c4_en_bits_per_token",
        "real": {k: v for k, v in real.items() if k != "statistics"},
        "inference": inference,
        "calibration": {"worlds": len(worlds), "bandwidth": BANDWIDTH, "level": LEVEL},
        "synthetic": {
            "worlds": len(synth),
            "spacing_decades": SPACING,
            "base": BASE,
            "profile": "arm A power law plus arm B2 recipe-specific residual, C4 moderated noise, three seeds",
            "effects": effects,
        },
        "predictions": predictions,
        "outcome": outcome,
        "both_contribute": bool(lever_survives and top["excludes_zero_calibrated"]),
        "deviations": [
            "Slice (ii) holds 101 designs, not the pre-registered 100: the count was enumerated from log10 compute "
            "rounded to three decimals, under which the 8M-to-20M design (lever arm 0.6994) rounds into the 0.8 bin. "
            "The pre-registered rule, round(L / 0.2) on exact values, is applied as written.",
            "Inference uses the leave-one-recipe-out jackknife; the request's leave-one-ladder-out is read as this.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("real", "calibration", "synthetic", "summarise", "time"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="skip worlds already in the cache (always on)")
    args = parser.parse_args(argv)
    workers = min(args.workers, 2)
    if args.stage == "real":
        record = run_real()
        print(f"real: {record['n_designs']} designs, slices {record['slice_counts']}")
    elif args.stage == "time":
        started = time.time()
        run_world(10_000)
        print(f"one calibration world: {time.time() - started:.1f}s")
        started = time.time()
        run_synthetic_world(10_000)
        print(f"one synthetic world: {time.time() - started:.1f}s")
    elif args.stage == "calibration":
        done = {w["index"] for w in _read_jsonl(WORLD_CACHE)}
        todo = [i for i in range(args.worlds or N_WORLDS) if i not in done]
        _run_pool(run_world, todo, WORLD_CACHE, workers)
    elif args.stage == "synthetic":
        done = {w["index"] for w in _read_jsonl(SYNTH_CACHE)}
        todo = [i for i in range(args.worlds or N_SYNTH) if i not in done]
        _run_pool(run_synthetic_world, todo, SYNTH_CACHE, workers)
    else:
        result = summarise()
        print(f"outcome: {result['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
