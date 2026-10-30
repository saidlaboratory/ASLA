"""Design power for the fixed-target slope test (asla.analysis.design_power), cached and resumable.

Stages:
* ``l2l``: Loss-to-loss (Brandfonbrener et al. 2024) from its paper's design only: six corpora; eight
  sweep budgets from 2e17 to 4.84e19 FLOPs (the paper gives the range and count of models, not the
  budgets, so they are taken as geometric), plus the 1e21 extrapolation budget; one run per cell; rung
  model size taken as Chinchilla-optimal, N = sqrt(C / 120). None of its data is read.
* ``grid``: a grid of designs with DataDecide's structure: n recipes in {5, 10, 15, 20, 25, 30, 40},
  S sizes in {4, ..., 10} evenly spaced in log10 compute from DataDecide's smallest rung (8.43e15) to its
  1B rung (7.06e20) with DataDecide's 100 tokens per parameter (N = sqrt(C / 600)), and 1 to 3 seeds.
  With fewer than six sizes no target has three 3-rung windows below it, so those designs have no test.
* ``summarise``: writes results/replication/design_power.json with power, false-positive rates, the
  smallest slope at 80% power, the smallest designs reaching 0.8 power, and their training compute.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.design_power import one_world, summarise  # noqa: E402

CACHE = REPO / "data" / "cache" / "design_power.jsonl"
OUT = REPO / "results" / "replication" / "design_power.json"
SEED = 20261101
L2L_WORLDS = 1000
GRID_WORLDS = 100
N_GRID = (5, 10, 15, 20, 25, 30, 40)
S_GRID = (4, 5, 6, 7, 8, 9, 10)
SEED_GRID = (1, 2, 3)
DD_LOW, DD_HIGH = float(np.log10(8.43e15)), float(np.log10(7.06e20))
L2L_BUDGETS = np.linspace(np.log10(2e17), np.log10(4.84e19), 8).tolist() + [21.0]
# Hardware: dense BF16 tensor peak with FP32 accumulation, and the model FLOPs utilisation assumed.
HARDWARE = {
    "H100 SXM": {"peak_flops": 989.4e12, "mfu": 0.40},
    "RTX 3090": {"peak_flops": 71.0e12, "mfu": 0.30},
}


def ladder(sizes: int) -> list[float]:
    return np.linspace(DD_LOW, DD_HIGH, sizes).tolist()


def cells() -> list[tuple[int, int, int]]:
    return [(n, s, k) for s, n, k in itertools.product(S_GRID, N_GRID, SEED_GRID) if s >= 6]


def job_key(job: dict) -> str:
    return f"{job['stage']}|{job.get('cell')}|{job['world']}"


def run_job(job: dict) -> dict:
    rng = np.random.default_rng([SEED, zlib.crc32(job_key(job).encode())])
    if job["stage"] == "l2l":
        w = one_world(6, L2L_BUDGETS, lambda c: np.sqrt(c / 120), 1, rng)
    else:
        n, s, k = job["cell"]
        w = one_world(n, ladder(s), lambda c: np.sqrt(c / 600), k, rng)
    return {"key": job_key(job), "stage": job["stage"], "cell": job.get("cell"), "result": w}


def _read() -> list[dict]:
    if not CACHE.exists():
        return []
    return [json.loads(x) for x in CACHE.read_text(encoding="utf-8").splitlines() if x.strip()]


def _run(jobs: list[dict], workers: int) -> None:
    confound = __import__("scripts.run_confound", fromlist=["_run_pool"])
    done = {r["key"] for r in _read()}
    todo = [j for j in jobs if job_key(j) not in done]
    confound._run_pool(run_job, todo, CACHE, min(workers, 2))


def compute_cost(n: int, s: int, k: int) -> dict:
    flops = float(n * k * np.sum(10.0 ** np.asarray(ladder(s))))
    return {
        "training_flops": flops,
        "gpu_hours": {name: flops / (h["peak_flops"] * h["mfu"]) / 3600 for name, h in HARDWARE.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("l2l", "grid", "summarise"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="skip cached jobs (always on)")
    args = parser.parse_args(argv)
    if args.stage == "l2l":
        _run([{"stage": "l2l", "world": w} for w in range(L2L_WORLDS)], args.workers)
        return 0
    if args.stage == "grid":
        _run([{"stage": "grid", "cell": list(c), "world": w} for c in cells() for w in range(GRID_WORLDS)], args.workers)
        return 0
    rows = _read()
    out = {
        "status": "design power for a future experiment; DataDecide C4 structure and noise",
        "unit": "points of mis-selected pairs per decade (log10) of lever arm",
        "hardware_assumptions": HARDWARE,
        "compute_note": "training FLOPs = recipes x seeds x sum of rung budgets (target rung included); C = 6 N D",
    }
    l2l = [r["result"] for r in rows if r["stage"] == "l2l"]
    if l2l:
        out["loss_to_loss"] = {
            "design": {
                "recipes": 6,
                "budgets_log10_flops": L2L_BUDGETS,
                "budget_assumption": "eight sweep budgets geometric from 2e17 to 4.84e19 (the paper states the range); plus 1e21",
                "seeds": 1,
                "rung_model_size": "Chinchilla-optimal, N = sqrt(C / 120)",
                "data_read": "none; design from arXiv:2411.12925 section 3.2",
            },
            **summarise(l2l),
        }
    grid = {}
    for c in cells():
        res = [r["result"] for r in rows if r["stage"] == "grid" and r["cell"] == list(c)]
        if res:
            grid["n{}_S{}_seeds{}".format(*c)] = {"recipes": c[0], "sizes": c[1], "seeds": c[2], **summarise(res), **compute_cost(*c)}
    for s in S_GRID:
        if s < 6:
            for n, k in itertools.product(N_GRID, SEED_GRID):
                grid[f"n{n}_S{s}_seeds{k}"] = {"recipes": n, "sizes": s, "seeds": k, "note": "no target has three 3-rung windows: no test", **compute_cost(n, s, k)}
    if grid:
        out["grid"] = grid
        smallest = {}
        for framing in ("fixed_candidate", "random_candidate"):
            for beta in ("11.7", "5.85"):
                ok = [
                    v for v in grid.values()
                    if framing in v and v[framing].get(f"power_at_{beta}", 0) >= 0.8
                ]
                ok.sort(key=lambda v: v["training_flops"])
                smallest[f"{framing}_at_{beta}"] = (
                    {k: ok[0][k] for k in ("recipes", "sizes", "seeds", "training_flops", "gpu_hours")}
                    | {"power": ok[0][framing][f"power_at_{beta}"]}
                    if ok
                    else None
                )
        out["smallest_designs_by_compute"] = smallest
        smallest2 = {}
        for framing in ("fixed_candidate", "random_candidate"):
            for beta in ("11.7", "5.85"):
                ok = sorted(
                    [v for v in grid.values() if framing in v and v["seeds"] >= 2 and v[framing].get(f"power_at_{beta}", 0) >= 0.8],
                    key=lambda v: v["training_flops"],
                )
                smallest2[f"{framing}_at_{beta}"] = (
                    {k: ok[0][k] for k in ("recipes", "sizes", "seeds", "training_flops", "gpu_hours")}
                    | {"power": ok[0][framing][f"power_at_{beta}"]}
                    if ok
                    else None
                )
        out["smallest_designs_by_compute_at_least_2_seeds"] = smallest2
        out["caveats"] = [
            "With one seed, target evidence uses the simulation's true noise SD (an oracle); a real one-seed "
            "experiment must borrow it, so one-seed power is optimistic. With two or more seeds the frozen rule "
            "estimates it from the seeds.",
            "Power at a given slope is a location shift: the simulated DataDecide-structure truths have mean true "
            "slopes of about 2 to 4 points per decade, so power at +11.7 and +5.85 assumes the slope's sampling "
            "error does not depend on its size.",
            "100 worlds per design: power near 0.8 carries a Monte Carlo SE of about 0.04.",
            "Training compute counts every rung, including the target rung, which dominates it.",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if "loss_to_loss" in out:
        print("L2L", json.dumps({k: v for k, v in out["loss_to_loss"].items() if k != "design"}, indent=1))
    if "smallest_designs_by_compute" in out:
        print(json.dumps(out["smallest_designs_by_compute"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
