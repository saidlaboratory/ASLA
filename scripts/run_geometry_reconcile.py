"""Why does the design-power simulator's slope (2 to 4 per decade) differ from DataDecide's (+11.7)?

A chain of simulator specs (asla.analysis.sim_design) moves from the confound benchmark's lever-arm-only
variation (+7.80) and DataDecide's exact fixed-target windows to the last round's grid designs, changing one
factor per step. Each step runs 200 worlds through scripts/sim_runner.py (subprocesses with a per-job
timeout, checkpointed as they complete).

Writes results/replication/geometry_reconciliation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.sim_runner import read_cache, run_jobs  # noqa: E402

CACHE = REPO / "data" / "cache" / "geometry_reconcile.jsonl"
OUT = REPO / "results" / "replication" / "geometry_reconciliation.json"
WORLDS, CHUNK, SEED = 200, 25, 20261102
DD_LADDER = [15.926, 16.336, 16.641, 16.77, 17.094, 17.187, 17.341, 18.291, 18.76, 19.128, 19.753, 20.175, 20.849]
LOW, HIGH = DD_LADDER[0], DD_LADDER[-1]


def even(n: int) -> list[float]:
    return np.linspace(LOW, HIGH, n).tolist()


def chain() -> list[tuple[str, str, dict]]:
    """(step id, description, spec); each spec changes one thing from the one before (noted)."""

    lever_only = {
        "statistic": "lever_only",
        "candidates": "fixed",
        "truth": "specific",
        "noise": "per_recipe",
        "tpp": "datadecide",
        "seeds": 3,
        "ladder": (19.0 - 0.35 * np.arange(6)[::-1]).tolist(),
        "levers": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    }
    s1 = {
        "statistic": "fixed_target",
        "candidates": "fixed",
        "truth": "specific",
        "noise": "per_recipe",
        "tpp": "datadecide",
        "seeds": 3,
        "ladder": DD_LADDER,
        "targets": [9, 10, 11, 12],
        "admit": "all",
    }
    steps = [
        ("R0", "confound benchmark's lever-arm-only variation (reference +7.80)", lever_only),
        ("R1", "fixed-target statistic on DataDecide's exact ladder and targets (150M to 1B), same truth and noise", s1),
        ("R2", "truth: add the common-mode residual (full residual, as the grid)", {**s1, "truth": "full"}),
        ("R3", "candidates: 25 random perturbed draws (as the grid)", {**s1, "truth": "full", "candidates": "random", "n": 25}),
        ("R4", "noise: median relative SD by model size (as the grid)", {**s1, "truth": "full", "candidates": "random", "n": 25, "noise": "by_size"}),
        ("R5", "targets admitted by world determinacy (as the grid)", {**s1, "truth": "full", "candidates": "random", "n": 25, "noise": "by_size", "admit": "determinacy"}),
    ]
    g0 = steps[-1][2]
    g1 = {**g0, "ladder": even(13)}
    steps += [
        ("G1", "geometry: size spacing even in log compute (13 rungs, same range, top four targets)", g1),
        ("G2", "geometry: compute range covered (the grid spans DataDecide's 4M to 1B range: unchanged)", dict(g1)),
        ("G3", "geometry: tokens per parameter constant at 100 (DataDecide drifts to 85)", {**g1, "tpp": 100}),
        ("G4", "geometry: target rule of the grid (every rung with five or more below)", {**g1, "tpp": 100, "targets": "auto"}),
    ]
    g4 = steps[-1][2]
    for s in (10, 8, 7, 6):
        steps.append((f"G5_S{s}", f"geometry: rung count {s} (evenly spaced over the same range)", {**g4, "ladder": even(s)}))
    g7 = {**g4, "ladder": even(7)}
    steps += [
        ("N1", "grid cell n=10, S=7, 3 seeds (recipes 25 to 10)", {**g7, "n": 10}),
        ("N2", "grid cell n=10, S=7, 1 seed (seeds 3 to 1)", {**g7, "n": 10, "seeds": 1}),
    ]
    return steps


def jobs() -> list[dict]:
    out = []
    for step, _, spec in chain():
        for start in range(0, WORLDS, CHUNK):
            out.append({"key": f"{step}|{start}", "spec": spec, "worlds": [start, start + CHUNK], "seed": SEED})
    return out


def summarise() -> dict:
    cache = read_cache(CACHE)
    steps = []
    for step, desc, spec in chain():
        rows, timeouts = [], 0
        for start in range(0, WORLDS, CHUNK):
            r = cache.get(f"{step}|{start}")
            if r is None or "results" not in r:
                timeouts += int(bool(r and r.get("timeout")))
                continue
            rows += r["results"]
        trues = np.array([r["true"] for r in rows if r.get("true") is not None], dtype=float)
        ests = np.array([r["estimate"] for r in rows if r.get("estimate") is not None], dtype=float)
        lr = [r["lever_range"] for r in rows if r.get("lever_range")]
        steps.append(
            {
                "step": step,
                "change": desc,
                "worlds": len(rows),
                "jobs_timed_out": timeouts,
                "mean_true_slope": float(trues.mean()) if trues.size else None,
                "mc_se": float(trues.std(ddof=1) / np.sqrt(trues.size)) if trues.size > 1 else None,
                "sd_true_slope": float(trues.std(ddof=1)) if trues.size > 1 else None,
                "mean_estimate": float(ests.mean()) if ests.size else None,
                "lever_range_median": [float(np.median([x[0] for x in lr])), float(np.median([x[1] for x in lr]))] if lr else None,
                "spec": {k: v for k, v in spec.items() if k != "ladder"} | {"ladder_rungs": len(spec["ladder"])},
            }
        )
    out = {
        "unit": "points of mis-selected pairs per decade (log10) of lever arm",
        "reference": {"confound_lever_only": 7.80, "datadecide_fixed_target_real": 11.67, "grid_mean_true_slopes": "2 to 4"},
        "worlds_per_step": WORLDS,
        "steps": steps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        run_jobs(jobs(), CACHE, workers=workers, timeout=1200)
    res = summarise()
    for s in res["steps"]:
        m = s["mean_true_slope"]
        print(f"{s['step']:7s} {('%.2f' % m) if m is not None else '  -  '} +/- {('%.2f' % s['mc_se']) if s['mc_se'] else '-'}  est {('%.2f' % s['mean_estimate']) if s['mean_estimate'] is not None else '-'}  levers {s['lever_range_median']}  {s['change']}")


if __name__ == "__main__":
    main()
