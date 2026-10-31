"""Is the low slope on the even grid real, or an artifact of interpolating recipe-specific misfit?

Every step keeps rungs on DataDecide's own sizes (750M excluded), where the misfit is measured, except the
interpolation steps, which compare a ladder on DataDecide sizes with an evenly spaced ladder of the same rung
count and range whose truth is interpolated between sizes. The simulator, noise and candidates are the
design-power ones (random perturbed recipes, full residual, noise by size, determinacy admission, 3 seeds,
25 recipes), so each step changes exactly one thing. Targets are rungs at or above DataDecide's 150M compute
with five or more rungs below.

Runs through scripts/sim_runner.py with one worker. Writes results/replication/geometry_reconciliation_v2.json.
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

CACHE = REPO / "data" / "cache" / "geometry_reconcile_v2.jsonl"
OUT = REPO / "results" / "replication" / "geometry_reconciliation_v2.json"
WORLDS, CHUNK, SEED = 160, 20, 20261104
NODES = [15.926, 16.336, 16.641, 16.77, 17.094, 17.187, 17.341, 18.291, 18.76, 19.128, 19.753, 20.175, 20.849]
LABELS = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "1B"]
MIN_TARGET = 19.128


def nodes(top: int, sizes: int, start: int = 0) -> list[float]:
    wanted = np.linspace(NODES[start], NODES[top], sizes)
    chosen: list[int] = []
    for w in wanted:
        free = [i for i in range(start, top + 1) if i not in chosen]
        chosen.append(min(free, key=lambda i: abs(NODES[i] - w)))
    return [NODES[i] for i in sorted(chosen)]


def base_spec(ladder: list[float], **extra) -> dict:
    targets = [t for t in range(5, len(ladder)) if ladder[t] >= MIN_TARGET - 1e-9]
    return {
        "statistic": "fixed_target",
        "candidates": "random",
        "n": 25,
        "truth": "full",
        "noise": "by_size",
        "tpp": "datadecide",
        "seeds": 3,
        "ladder": ladder,
        "targets": targets,
        "admit": "determinacy",
        "lam": 1.0,
        **extra,
    }


def chain() -> list[tuple[str, str, dict]]:
    full = NODES
    steps = [
        ("B0", "all 13 DataDecide sizes, targets 150M to 1B (the reference)", base_spec(full)),
        ("B0_fixed", "the same with DataDecide's 25 recipes unperturbed (candidates fixed)", base_spec(full, candidates="fixed")),
        ("TPP", "tokens per parameter constant at 100 in the noise model", base_spec(full, tpp=100)),
    ]
    for s in (10, 8, 7, 6):
        steps.append((f"RUNGS_{s}", f"rung count {s}: DataDecide sizes spread evenly up to 1B", base_spec(nodes(12, s))))
    steps += [
        ("SPACING_7_top", "spacing at 7 rungs: the 7 largest consecutive DataDecide sizes (20M to 1B)", base_spec(NODES[6:13])),
        ("RANGE_7_from14M", "compute range at 7 rungs: DataDecide sizes from 14M (not 4M) to 1B", base_spec(nodes(12, 7, start=4))),
        ("TARGET_8_530M", "target size: 8 DataDecide sizes up to 530M", base_spec(nodes(11, 8))),
        ("TARGET_8_300M", "target size: 8 DataDecide sizes up to 300M", base_spec(nodes(10, 8))),
    ]
    for s in (13, 10, 7):
        even = np.linspace(NODES[0], NODES[12], s).tolist()
        sp = base_spec(even)
        if s == 13:
            sp["targets"] = [9, 10, 11, 12]  # the same four targets as the reference
        steps.append((f"INTERP_{s}", f"interpolation: {s} evenly spaced rungs, truth interpolated between sizes (same count and range)", sp))
    return steps


def jobs() -> list[dict]:
    return [
        {"key": f"{step}|{s}", "spec": spec, "worlds": [s, s + CHUNK], "seed": SEED}
        for step, _, spec in chain()
        for s in range(0, WORLDS, CHUNK)
    ]


def summarise() -> dict:
    cache = read_cache(CACHE)
    rows_out = []
    for step, desc, spec in chain():
        rows = []
        for s in range(0, WORLDS, CHUNK):
            r = cache.get(f"{step}|{s}")
            if r and "results" in r:
                rows += r["results"]
        t = np.array([r["true"] for r in rows if r.get("true") is not None], dtype=float)
        rows_out.append(
            {
                "step": step,
                "change": desc,
                "rungs": len(spec["ladder"]),
                "ladder_log10_flops": [round(x, 3) for x in spec["ladder"]],
                "worlds": len(rows),
                "mean_true_slope": float(t.mean()) if t.size else None,
                "mc_se": float(t.std(ddof=1) / np.sqrt(t.size)) if t.size > 1 else None,
                "lever_range_median": [
                    float(np.median([r["lever_range"][0] for r in rows if r.get("lever_range")])),
                    float(np.median([r["lever_range"][1] for r in rows if r.get("lever_range")])),
                ] if any(r.get("lever_range") for r in rows) else None,
            }
        )
    by = {r["step"]: r for r in rows_out}
    pairs = {}
    node_for = {13: "B0", 10: "RUNGS_10", 7: "RUNGS_7"}
    for s, node_step in node_for.items():
        a, b = by.get(node_step), by.get(f"INTERP_{s}")
        if a and b and a["mean_true_slope"] is not None and b["mean_true_slope"] is not None:
            pairs[str(s)] = {
                "on_datadecide_sizes": a["mean_true_slope"],
                "interpolated_even": b["mean_true_slope"],
                "difference": b["mean_true_slope"] - a["mean_true_slope"],
                "difference_mc_se": float(np.hypot(a["mc_se"], b["mc_se"])),
            }
    out = {
        "unit": "points of mis-selected pairs per decade (log10) of lever arm",
        "note": "+7.80 is the lever-arm-only statistic (target varied on one synthetic window), not the fixed-target slope",
        "steps": rows_out,
        "interpolation_effect_by_rung_count": pairs,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_jobs(jobs(), CACHE, workers=1, timeout=1500)
    res = summarise()
    for r in res["steps"]:
        m = r["mean_true_slope"]
        print(f"{r['step']:16s} {('%.2f' % m) if m is not None else '-':>6} +/- {('%.2f' % r['mc_se']) if r['mc_se'] else '-'}  {r['change']}")
    print(json.dumps(res["interpolation_effect_by_rung_count"], indent=1))


if __name__ == "__main__":
    main()
