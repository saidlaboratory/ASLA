"""Power of the fixed-target slope test without the location-shift assumption.

For each design, the recipe-specific misspecification amplitude lambda is set so that the design's own
worlds generate the slope being tested (their mean true fixed-target slope equals beta), then power is the
rejection rate of the frozen test in those worlds:

* random-candidate: the share of worlds whose estimate - 1.96 SE > 0;
* fixed-candidate: the same share among worlds whose own true slope is within 20% of beta.

Each is reported with its binomial Monte Carlo SE. lambda is found by evaluating the mean true slope at
lambda in {0.25, 0.5, 1, 2, 3, 4, 6, 8} (40 worlds each) and interpolating linearly; designs that do not reach beta by
lambda = 8 are marked unreachable.

Stages:
* ``task2``: last round's designs (grid geometry: 4M to 1B, evenly spaced, 100 tokens per parameter).
* ``task3``: top targets 300M, 530M and 1B; 6, 8 or 10 sizes; 2 or 3 seeds; for each geometry and beta, the
  smallest recipe count in {5, 10, 15, 20, 25, 30, 40} reaching 0.8 in each framing. Targets are restricted
  to rungs at or above DataDecide's 150M compute (the smallest DataDecide target with half its pairs
  determined), and must pass the world determinacy rule. lambda is calibrated per geometry at 15 recipes.
* ``summarise``: results/replication/direct_power.json.

Jobs run through scripts/sim_runner.py: one subprocess per chunk of worlds, a per-job timeout, and a
checkpoint after every completed job.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.sim_runner import read_cache, run_jobs  # noqa: E402

CACHE = REPO / "data" / "cache" / "direct_power.jsonl"
OUT = REPO / "results" / "replication" / "direct_power.json"
SEED = 20261103
Z = 1.959963984540054
LAMS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CAL_WORLDS, POWER_WORLDS, CHUNK = 40, 200, 20
BETAS = (11.7, 5.85)
LOW = 15.926  # DataDecide 4M
TOPS = {"300M": 19.753, "530M": 20.175, "1B": 20.849}
MIN_TARGET = 19.128  # DataDecide 150M compute
TASK2 = [(10, 7, 1), (15, 9, 1), (20, 7, 1), (10, 8, 2), (10, 10, 2), (15, 7, 3), (15, 10, 3)]
N_GRID = (5, 10, 15, 20, 25, 30, 40)
S3, SEEDS3 = (6, 8, 10), (2, 3)
HARDWARE = {"H100 SXM": {"peak_flops": 989.4e12, "mfu": 0.40}, "RTX 3090": {"peak_flops": 71.0e12, "mfu": 0.30}}


def spec(n: int, sizes: int, seeds: int, top: float, lam: float, restrict: bool) -> dict:
    ladder = np.linspace(LOW, top, sizes).tolist()
    targets = [t for t in range(5, sizes) if (not restrict) or ladder[t] >= MIN_TARGET - 1e-9]
    return {
        "statistic": "fixed_target",
        "candidates": "random",
        "n": n,
        "truth": "full",
        "noise": "by_size",
        "tpp": 100,
        "seeds": seeds,
        "ladder": ladder,
        "targets": targets,
        "admit": "determinacy",
        "lam": lam,
    }


def chunked(key: str, sp: dict, worlds: int) -> list[dict]:
    return [{"key": f"{key}|{s}", "spec": sp, "worlds": [s, min(s + CHUNK, worlds)], "seed": SEED} for s in range(0, worlds, CHUNK)]


def collect(cache: dict, key: str, worlds: int) -> list[dict]:
    rows = []
    for s in range(0, worlds, CHUNK):
        r = cache.get(f"{key}|{s}")
        if r and "results" in r:
            rows += r["results"]
    return rows


def lam_for(cache: dict, base_key: str, beta: float) -> tuple[float | None, list[float]]:
    means = []
    for lam in LAMS:
        rows = [r for r in collect(cache, f"{base_key}|lam{lam:g}", CAL_WORLDS) if r.get("true") is not None]
        means.append(float(np.mean([r["true"] for r in rows])) if rows else np.nan)
    m = np.array(means)
    for i in range(len(LAMS) - 1):
        a, b = m[i], m[i + 1]
        if np.isfinite(a) and np.isfinite(b) and (a - beta) * (b - beta) <= 0 and a != b:
            return float(LAMS[i] + (beta - a) * (LAMS[i + 1] - LAMS[i]) / (b - a)), means
    return None, means


def power_of(rows: list[dict], beta: float) -> dict:
    ok = [r for r in rows if r.get("estimate") is not None]
    n_all = len(rows)
    rej = np.array([r["se"] > 0 and r["estimate"] - Z * r["se"] > 0 for r in ok])
    p_rc = float(rej.sum() / n_all) if n_all else None
    near = [r for r in ok if abs(r["true"] - beta) <= 0.2 * beta]
    rej_fc = np.array([r["se"] > 0 and r["estimate"] - Z * r["se"] > 0 for r in near])
    p_fc = float(rej_fc.mean()) if near else None
    return {
        "worlds": n_all,
        "random_candidate": p_rc,
        "random_candidate_mc_se": float(np.sqrt(p_rc * (1 - p_rc) / n_all)) if n_all else None,
        "fixed_candidate": p_fc,
        "fixed_candidate_worlds": len(near),
        "fixed_candidate_mc_se": float(np.sqrt(p_fc * (1 - p_fc) / len(near))) if near else None,
        "mean_true_slope": float(np.mean([r["true"] for r in ok])) if ok else None,
        "median_se": float(np.median([r["se"] for r in ok])) if ok else None,
        "median_determined_share_top": float(np.median([max(map(float, r["determined_share"].values())) for r in ok if r.get("determined_share")])) if ok else None,
    }


def compute_cost(n: int, sizes: int, seeds: int, top: float) -> dict:
    flops = float(n * seeds * np.sum(10.0 ** np.linspace(LOW, top, sizes)))
    return {"training_flops": flops, "gpu_hours": {k: flops / (h["peak_flops"] * h["mfu"]) / 3600 for k, h in HARDWARE.items()}}


def task2_cal_jobs() -> list[dict]:
    jobs = []
    for n, s, k in TASK2:
        for lam in LAMS:
            jobs += chunked(f"t2|n{n}S{s}k{k}|lam{lam:g}", spec(n, s, k, TOPS["1B"], lam, restrict=False), CAL_WORLDS)
    return jobs


def task2_power_jobs(cache: dict) -> list[dict]:
    jobs = []
    for n, s, k in TASK2:
        base = f"t2|n{n}S{s}k{k}"
        jobs += chunked(f"{base}|natural", spec(n, s, k, TOPS["1B"], 1.0, restrict=False), POWER_WORLDS)
        for beta in BETAS:
            lam, _ = lam_for(cache, base, beta)
            if lam is not None:
                jobs += chunked(f"{base}|beta{beta:g}", spec(n, s, k, TOPS["1B"], lam, restrict=False), POWER_WORLDS)
    return jobs


def geometries():
    return list(itertools.product(TOPS, S3, SEEDS3))


def task3_cal_jobs() -> list[dict]:
    jobs = []
    for top, s, k in geometries():
        for lam in LAMS:
            jobs += chunked(f"t3|{top}S{s}k{k}|lam{lam:g}", spec(15, s, k, TOPS[top], lam, restrict=True), CAL_WORLDS)
    return jobs


def task3_power_jobs(cache: dict, n: int) -> list[dict]:
    jobs = []
    for top, s, k in geometries():
        base = f"t3|{top}S{s}k{k}"
        for beta in BETAS:
            lam, _ = lam_for(cache, base, beta)
            if lam is None:
                continue
            key = f"{base}|beta{beta:g}|n{n}"
            # Skip recipe counts above one that already reached 0.8 in both framings.
            done = False
            for smaller in N_GRID:
                if smaller >= n:
                    break
                p = power_of(collect(cache, f"{base}|beta{beta:g}|n{smaller}", POWER_WORLDS), beta)
                if p["worlds"] == POWER_WORLDS and (p["random_candidate"] or 0) >= 0.8 and (p["fixed_candidate"] or 0) >= 0.8:
                    done = True
                    break
            if not done:
                jobs += chunked(key, spec(n, s, k, TOPS[top], lam, restrict=True), POWER_WORLDS)
    return jobs


def summarise() -> dict:
    cache = read_cache(CACHE)
    shift = json.loads((REPO / "results" / "replication" / "design_power.json").read_text(encoding="utf-8"))["grid"]
    out = {
        "unit": "points of mis-selected pairs per decade (log10) of lever arm",
        "method": "lambda scales the recipe-specific misspecification so each design's worlds generate the tested slope",
        "hardware_assumptions": HARDWARE,
        "timeouts": sum(1 for v in cache.values() if v.get("timeout")),
        "failures": sum(1 for v in cache.values() if v.get("failed")),
    }
    t2 = {}
    for n, s, k in TASK2:
        base = f"t2|n{n}S{s}k{k}"
        row = {"recipes": n, "sizes": s, "seeds": k}
        nat = power_of(collect(cache, f"{base}|natural", POWER_WORLDS), 0.0)
        row["natural"] = {"predicted_slope": nat["mean_true_slope"], "median_se": nat["median_se"], "median_determined_share_top": nat["median_determined_share_top"]}
        nat_rows = collect(cache, f"{base}|natural", POWER_WORLDS)
        ok = [r for r in nat_rows if r.get("estimate") is not None]
        p = float(np.mean([r["se"] > 0 and r["estimate"] - Z * r["se"] > 0 for r in ok])) * len(ok) / max(len(nat_rows), 1) if nat_rows else None
        row["natural"]["power_at_predicted_slope"] = p
        row["natural"]["power_mc_se"] = float(np.sqrt(p * (1 - p) / len(nat_rows))) if nat_rows else None
        sh = shift.get(f"n{n}_S{s}_seeds{k}", {})
        for beta in BETAS:
            lam, means = lam_for(cache, base, beta)
            d = power_of(collect(cache, f"{base}|beta{beta:g}", POWER_WORLDS), beta) if lam is not None else None
            row[f"beta_{beta:g}"] = {
                "lambda": lam,
                "calibration_mean_true_by_lambda": dict(zip(map(str, LAMS), means)),
                "direct": d,
                "shift": {
                    "fixed_candidate": sh.get("fixed_candidate", {}).get(f"power_at_{beta:g}"),
                    "random_candidate": sh.get("random_candidate", {}).get(f"power_at_{beta:g}"),
                },
            }
        t2[base.split("|")[1]] = row
    out["task2"] = t2
    t3 = {}
    for top, s, k in geometries():
        base = f"t3|{top}S{s}k{k}"
        for beta in BETAS:
            lam, means = lam_for(cache, base, beta)
            for n in N_GRID:
                rows = collect(cache, f"{base}|beta{beta:g}|n{n}", POWER_WORLDS)
                if not rows:
                    continue
                t3[f"{top}_S{s}_seeds{k}_beta{beta:g}_n{n}"] = {
                    "top": top, "sizes": s, "seeds": k, "beta": beta, "recipes": n, "lambda": lam,
                    **power_of(rows, beta), **compute_cost(n, s, k, TOPS[top]),
                }
    out["task3"] = t3
    cheapest = {}
    for beta in BETAS:
        for framing in ("random_candidate", "fixed_candidate"):
            ok = [v for v in t3.values() if v["beta"] == beta and (v[framing] or 0) >= 0.8 and v["worlds"] == POWER_WORLDS]
            ok.sort(key=lambda v: v["training_flops"])
            cheapest[f"{framing}_at_{beta:g}"] = ok[0] if ok else None
    out["task3_cheapest"] = cheapest
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "summarise"
    if stage == "task2":
        run_jobs(task2_cal_jobs(), CACHE, workers=1, timeout=1500)
        run_jobs(task2_power_jobs(read_cache(CACHE)), CACHE, workers=1, timeout=1500)
    elif stage == "task3":
        run_jobs(task3_cal_jobs(), CACHE, workers=1, timeout=1500)
        for n in N_GRID:
            run_jobs(task3_power_jobs(read_cache(CACHE), n), CACHE, workers=1, timeout=1500)
    res = summarise()
    print(json.dumps({k: res[k] for k in ("timeouts", "failures")}))


if __name__ == "__main__":
    main()
