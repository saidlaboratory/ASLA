"""Task 1 of PREDICTIONS_TASK_FINAL_CALIBRATION.md: calibrate the lever-arm test.

Worlds are random-candidate draws of the full DataDecide ladder (750M excluded)
from the shrunk truth. In each world:

* the lever-arm test (asla.analysis.lever_arm) runs on the noisy data, exactly
  as on the real data;
* the true excess of every design is computed against the noiseless target;
* a replicate comparator ranks by the top rung of an independent noise draw of
  the same truth. Its true excess is zero at every design, and designs sharing
  a rung share its noise, so its rejection rate is the test's null behaviour
  under design overlap.

True rho is the Spearman correlation between log lever arm and each design's
true excess averaged over worlds.

Writes ``results/target_scoring/lever_arm_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import random_truth, shrink_truth, simulate, truth_from_tables  # noqa: E402
from asla.analysis.lever_arm import designs, loro_test  # noqa: E402

METRIC_FILE = REPO / "data" / "datadecide_runs.parquet"
CACHE = REPO / "data" / "cache" / "lever_arm_worlds.jsonl"
OUT = REPO / "results" / "target_scoring" / "lever_arm_calibration.json"
N_WORLDS = 300
# Null-only worlds (no projection fits) resolve the null distribution's tail. With
# 300 worlds the smallest attainable p would be 1/300, above the Holm threshold
# for the paper's claim family; this extension of the pre-registered design is
# made only for p-value resolution.
# Raised from 2000 to 8000 after the first run: the observed statistic exceeded
# every one of 2300 null draws, so its p sat at the 1/2300 floor, which is above
# the Benjamini-Yekutieli threshold for the paper's claim family. More draws only
# resolve the tail further; they cannot move a statistic that no draw reaches.
N_NULL = 8000
BANDWIDTH = 0.5
LEVEL = 0.95
SEED = 20260923
Z = 1.959963984540054
_CONTEXT: dict[str, Any] = {}


def _context() -> dict[str, Any]:
    if _CONTEXT:
        return _CONTEXT
    from asla.models import fit_power_law

    frame = pd.read_parquet(METRIC_FILE)
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    target = float(frame[frame["scale_label"] == "1B"]["compute"].iloc[0])
    base = shrink_truth(truth_from_tables(frame, frame.copy(), target, "1B", rho_ckpt={}))
    params = {}
    for recipe, group in base.final.groupby("intervention"):
        cells = group.groupby("compute")["mu"].mean()
        params[str(recipe)] = fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))
    _CONTEXT.update(frame=frame, base=base, params=params)
    return _CONTEXT


def run_null_world(index: int) -> dict[str, Any]:
    ctx = _context()
    rng = np.random.default_rng([SEED, 1, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    replicate, _ = simulate(truth, rng)
    null = loro_test(designs(observed, replicate=replicate, with_projection=False), replicate=True)
    return {
        "kind": "null",
        "index": index,
        "null_fisher_z": null["fisher_z"],
        "null_se": null["se_fisher_z"],
        "null_p": null["p_two_sided"],
    }


def run_world(index: int) -> dict[str, Any]:
    ctx = _context()
    rng = np.random.default_rng([SEED, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    replicate, _ = simulate(truth, rng)
    targets = {label: group.groupby("intervention")["mu"].first() for label, group in truth.final.groupby("scale_label")}
    world_designs = designs(observed, targets, replicate)
    main, null = loro_test(world_designs), loro_test(world_designs, replicate=True)
    return {
        "kind": "main",
        "index": index,
        "lever_arm": [d.lever_arm for d in world_designs],
        "true_excess": [d.true_excess for d in world_designs],
        "rho": main["rho"],
        "fisher_z": main["fisher_z"],
        "se": main["se_fisher_z"],
        "null_fisher_z": null["fisher_z"],
        "null_se": null["se_fisher_z"],
        "null_p": null["p_two_sided"],
    }


def _load() -> dict[tuple[str, int], dict[str, Any]]:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row.get("kind", "main"), row["index"])] = row
    return done


def summarise(cache: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    worlds = {i: row for (kind, i), row in cache.items() if kind == "main"}
    nulls = [row for (kind, _), row in cache.items() if kind == "null"] + list(worlds.values())
    rows = [worlds[i] for i in sorted(worlds)]
    log_lever = np.log(rows[0]["lever_arm"])
    mean_true = np.mean([r["true_excess"] for r in rows], axis=0)
    rho_true = float(stats.spearmanr(log_lever, mean_true).statistic)
    z_true = float(np.arctanh(np.clip(rho_true, -0.999999, 0.999999)))  # finite even if the truth is monotone

    def ratio(z: float, se: float, centre: float) -> float:
        return abs(z - centre) / se if se > 0 else float("inf")

    t_main = np.array([ratio(r["fisher_z"], r["se"], z_true) for r in rows])
    t_null = np.array([ratio(r["null_fisher_z"], r["null_se"], 0.0) for r in nulls])
    k = float(np.quantile(t_main, LEVEL))
    held_out = []
    for parity in (0, 1):
        train = np.array([t for i, t in zip(sorted(worlds), t_main) if i % 2 == parity])
        k_train = float(np.quantile(train, LEVEL))
        held_out += [t <= k_train for i, t in zip(sorted(worlds), t_main) if i % 2 != parity]
    return {
        "n_worlds": len(rows),
        "rho_true": rho_true,
        "mean_rho_hat": float(np.mean([r["rho"] for r in rows])),
        "coverage_at_1_96": float(np.mean(t_main <= Z)),
        "calibrated_critical_value": k,
        "cross_fit_coverage": float(np.mean(held_out)),
        "null_rejection_rate_at_0_05": float(np.mean([r["null_p"] <= 0.05 for r in nulls])),
        "n_null_statistics": len(nulls),
        "null_abs_t_quantiles": [float(v) for v in np.quantile(t_null, np.linspace(0, 1, 1001))],
        "main_abs_t_quantiles": [float(v) for v in np.quantile(t_main, np.linspace(0, 1, 1001))],
        "true_excess_by_design": [float(v) for v in mean_true],
    }


def real_data(summary: dict[str, Any]) -> dict[str, Any]:
    ctx = _context()
    test = loro_test(designs(ctx["frame"]))
    abs_t = abs(test["fisher_z"]) / test["se_fisher_z"]
    null = np.asarray(summary["null_abs_t_quantiles"])
    p_null = max(
        1.0 - float(np.interp(abs_t, null, np.linspace(0, 1, null.size), left=0.0, right=1.0)),
        1 / summary["n_null_statistics"],
    )
    k = summary["calibrated_critical_value"]
    return {
        "rho": test["rho"],
        "loro_se_fisher_z": test["se_fisher_z"],
        "ci_1_96": test["ci"],
        "ci_calibrated": [
            float(np.tanh(test["fisher_z"] - k * test["se_fisher_z"])),
            float(np.tanh(test["fisher_z"] + k * test["se_fisher_z"])),
        ],
        "p_normal": test["p_two_sided"],
        "abs_t": abs_t,
        "p_against_null_comparator": p_null,
        "p_floor": 1 / summary["n_null_statistics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load()
    todo = [("main", i) for i in range(N_WORLDS) if ("main", i) not in done]
    todo += [("null", i) for i in range(N_NULL) if ("null", i) not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        futures = [pool.submit(run_world if kind == "main" else run_null_world, i) for kind, i in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            sink.write(json.dumps(future.result()) + "\n")
            sink.flush()
            if count % 200 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    summary = summarise(_load())
    out = {
        "design": "full DataDecide ladder (750M excluded), C4 bits per token, shrunk truth, random-candidate worlds",
        "summary": summary,
        "real_data": real_data(summary),
    }
    out["predictions"] = {
        # As pre-registered: the LORO interval as reported (1.96); calibrated coverage is reported alongside.
        "P1_coverage_0_90_to_0_97": 0.90 <= summary["coverage_at_1_96"] <= 0.97,
        "P1_null_rate_at_most_0_08": summary["null_rejection_rate_at_0_05"] <= 0.08,
        "P1_true_rho_above_0_5": summary["rho_true"] > 0.5,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not k.endswith("quantiles") and k != "true_excess_by_design"}, indent=1
        )
    )
    print(json.dumps(out["real_data"], indent=1), out["predictions"])


if __name__ == "__main__":
    main()
