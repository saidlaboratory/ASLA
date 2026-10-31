"""PREDICTIONS_TASK_CURVE.md: validate the corrected recipe model, estimate the excess curve, build the untuned spec.

Stages, run in order (each resumable; simulator jobs go through scripts/sim_runner.py with one worker, a per-job
timeout and a checkpoint per completed job):

* ``validate``: V1, V2, the plateau check and the generative sanity check; writes
  results/replication/curve_validation.json and stops the round if V1 or V2 fails.
* ``curve``: the real-data curve (binned and saturating, jackknife, coincidence test) with its calibration;
  writes results/target_scoring/excess_curve.json.
* ``spec``: the design search at multiplier 1; writes results/replication/untuned_spec.json.
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

CACHE = REPO / "data" / "cache" / "curve_round.jsonl"
VALIDATION_OUT = REPO / "results" / "replication" / "curve_validation.json"
CURVE_OUT = REPO / "results" / "target_scoring" / "excess_curve.json"
SPEC_OUT = REPO / "results" / "replication" / "untuned_spec.json"
SEED = 20261105
Z = 1.959963984540054
CHUNK = 20
NODES = [15.926, 16.336, 16.641, 16.77, 17.094, 17.187, 17.341, 18.291, 18.76, 19.128, 19.753, 20.175, 20.849]
LABELS = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "1B"]
FULL_TARGETS = [9, 10, 11, 12]
REAL_SUBSETS = {"10": (11.61, 1.71), "15": (11.72, 0.97)}
REF_FULL, REF_FULL_SE = 11.96, 0.05
TOP_INDEX = {"300M": 10, "530M": 11, "1B": 12}
SIZES = (6, 8, 10)
N_GRID = (5, 10, 15, 20, 25, 30, 40)
SPEC_WORLDS = 150
HARDWARE = {"H100 SXM": {"peak_flops": 989.4e12, "mfu": 0.40}, "RTX 3090": {"peak_flops": 71.0e12, "mfu": 0.30}}


def node_ladder(top: str, sizes: int) -> list[float]:
    t = TOP_INDEX[top]
    wanted = np.linspace(NODES[0], NODES[t], sizes)
    chosen: list[int] = []
    for w in wanted:
        free = [i for i in range(t + 1) if i not in chosen]
        chosen.append(min(free, key=lambda i: abs(NODES[i] - w)))
    return [NODES[i] for i in sorted(chosen)]


def base(**kw) -> dict:
    return {
        "statistic": "fixed_target",
        "candidates": "corrected",
        "n": 25,
        "truth": "full",
        "noise": "by_size",
        "tpp": "datadecide",
        "seeds": 3,
        "ladder": NODES,
        "targets": FULL_TARGETS,
        "admit": "all",
        "lam": 1.0,
        **kw,
    }


def jobs_for(name: str, spec: dict, worlds: int) -> list[dict]:
    return [{"key": f"{name}|{s}", "spec": spec, "worlds": [s, min(s + CHUNK, worlds)], "seed": SEED} for s in range(0, worlds, CHUNK)]


def collect(cache: dict, name: str, worlds: int) -> list[dict]:
    rows = []
    for s in range(0, worlds, CHUNK):
        r = cache.get(f"{name}|{s}")
        if r and "results" in r:
            rows += r["results"]
    return rows


# ------------------------------------------------------------------ validation


def validation_specs() -> list[tuple[str, dict, int]]:
    out = [
        ("V1", base(), 200),
        ("V2_k10", base(n=10), 200),
        ("V2_k15", base(n=15), 200),
        ("GEN25", base(candidates="generative"), 100),
    ]
    for lam in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        out.append((f"PLATEAU_full|lam{lam:g}", base(lam=lam), 60))
        for top in ("300M", "1B"):
            lad = node_ladder(top, 8)
            targets = [t for t in range(5, 8) if lad[t] >= NODES[9] - 1e-9]
            out.append((f"PLATEAU_{top}_S8|lam{lam:g}", base(n=15, ladder=lad, targets=targets, admit="determinacy", lam=lam), 60))
    return out


def summarise_validation() -> dict:
    cache = read_cache(CACHE)
    res: dict = {"preregistration": "PREDICTIONS_TASK_CURVE.md", "unit": "points per decade (log10) of lever arm"}

    def stats_of(name: str, worlds: int, field: str) -> tuple[float, float, float, int]:
        v = np.array([r[field] for r in collect(cache, name, worlds) if r.get(field) is not None], dtype=float)
        return float(v.mean()), float(v.std(ddof=1)), float(v.std(ddof=1) / np.sqrt(len(v))), len(v)

    m, sd, se, n = stats_of("V1", 200, "true")
    combined = float(np.hypot(se, REF_FULL_SE))
    res["V1"] = {"mean_true_slope": m, "mc_se": se, "worlds": n, "reference": REF_FULL,
                 "within_2_combined_se": bool(abs(m - REF_FULL) <= 2 * combined), "combined_se": combined}
    for k in ("10", "15"):
        m, sd, se, n = stats_of(f"V2_k{k}", 200, "estimate")
        real_m, real_sd = REAL_SUBSETS[k]
        res[f"V2_k{k}"] = {"mean_estimate": m, "sd_estimate": sd, "mc_se": se, "worlds": n, "real_mean": real_m, "real_sd": real_sd,
                           "mean_within_1": bool(abs(m - real_m) <= 1.0), "sd_ratio": sd / real_sd,
                           "sd_ratio_in_0_67_to_1_5": bool(0.67 <= sd / real_sd <= 1.5)}
    m, sd, se, n = stats_of("GEN25", 100, "true")
    res["generative_sanity_25"] = {"mean_true_slope": m, "mc_se": se, "worlds": n, "reference": REF_FULL}
    plateau = {}
    for ladder_name in ("full", "300M_S8", "1B_S8"):
        curve = {}
        for lam in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            v = [r["true"] for r in collect(cache, f"PLATEAU_{ladder_name}|lam{lam:g}", 60) if r.get("true") is not None]
            curve[f"{lam:g}"] = float(np.mean(v)) if v else None
        vals = {k: v for k, v in curve.items() if v is not None}
        peak = max(vals, key=vals.get) if vals else None
        plateau[ladder_name] = {"mean_true_by_lambda": curve, "max": vals.get(peak) if peak else None, "lambda_at_max": peak,
                                "plateau_before_6": bool(peak is not None and float(peak) < 6)}
    res["plateau"] = plateau
    res["pass"] = bool(
        res["V1"]["within_2_combined_se"]
        and all(res[f"V2_k{k}"]["mean_within_1"] and res[f"V2_k{k}"]["sd_ratio_in_0_67_to_1_5"] for k in ("10", "15"))
    )
    VALIDATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


# ------------------------------------------------------------------ the curve


def real_curve() -> dict:
    from asla.analysis import excess_curve
    from asla.analysis.confound import window_designs

    confound = __import__("scripts.run_confound", fromlist=["_load_frame"])
    frame = confound._load_frame()
    ds = [d for d in window_designs(frame) if d.target in FULL_TARGETS]
    jk = excess_curve.jackknife(ds, FULL_TARGETS)
    lever = {str(t): sorted(d.lever for d in ds if d.target == t) for t in FULL_TARGETS}
    return {"jackknife": jk, "designs": [{"target": d.target, "lever": d.lever, "rungs": d.rungs, "excess": float(np.mean(list(d.kernel.values())))} for d in ds], "levers": lever, "_ds": ds}


def summarise_curve() -> dict:
    from asla.analysis import excess_curve

    cache = read_cache(CACHE)
    real = real_curve()
    jk = real["jackknife"]
    worlds = collect(cache, "CURVE_CAL", 200)
    ks = {}
    for key in jk["estimate"]:
        ratios = []
        for w in worlds:
            e, s_, t = w["estimate"].get(key), w["se"].get(key), w["true"].get(key)
            if e is not None and s_ and t is not None and np.isfinite([e, s_, t]).all() and s_ > 0:
                ratios.append(abs(e - t) / s_)
        ks[key] = float(np.quantile(ratios, 0.95)) if len(ratios) >= 20 else None
    bin_ks = [v for k, v in ks.items() if k.startswith("bin_") and v is not None]
    inflation = max(1.0, (float(np.median(bin_ks)) / Z) ** 2) if bin_ks else 1.0
    stats_out = {}
    for key, est in jk["estimate"].items():
        se = jk["se"].get(key)
        k_used = max(ks.get(key) or Z, Z)
        stats_out[key] = {
            "estimate": est,
            "se": se,
            "critical_value_calibrated": ks.get(key),
            "critical_value_used": k_used,
            "interval": [est - k_used * se, est + k_used * se] if (se is not None and np.isfinite(se) and np.isfinite(est)) else None,
        }
    coin = excess_curve.coincidence(jk, FULL_TARGETS, inflation=inflation)
    departures = excess_curve.max_departure(real["_ds"], jk["estimate"], FULL_TARGETS)
    coincide = bool(coin["binned"]["p"] > 0.05 and coin["binned_max_gap"] <= 3.0)
    est = jk["estimate"]
    s_by_target = [est[f"s_{t}"] for t in FULL_TARGETS]
    predictions = {
        "C1_initial_slopes_fall_with_target": bool(all(a > b for a, b in zip(s_by_target, s_by_target[1:]))),
        "C2_binned_coincidence_rejects": bool(coin["binned"]["p"] < 0.05),
        "C3_tau_below_bound_for_two_targets": bool(sum(est[f"tau_{t}"] < 20.0 - 1e-6 for t in FULL_TARGETS) >= 2),
    }
    out = {
        "preregistration": "PREDICTIONS_TASK_CURVE.md",
        "metric": "c4_en_bits_per_token",
        "unit": "excess in points of mis-selected pairs; lever arm in decades (log10); s in points per decade",
        "targets": {str(t): LABELS[t] for t in FULL_TARGETS},
        "bins_decades": list(excess_curve.BINS),
        "form": "excess = s * tau * (1 - exp(-L / tau)); s in [-100, 100], tau in [0.1, 20]",
        "statistics": stats_out,
        "calibration_worlds": len(worlds),
        "wald_inflation": inflation,
        "coincidence": coin,
        "curve_departure_from_common": departures,
        "coincide": coincide,
        "predictions": predictions,
        "designs": real["designs"],
    }
    CURVE_OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------ the spec


def spec_designs() -> list[tuple[str, str, int, int, dict]]:
    out = []
    for top in TOP_INDEX:
        for s in SIZES:
            lad = node_ladder(top, s)
            targets = [t for t in range(5, s) if lad[t] >= NODES[9] - 1e-9]
            for n in N_GRID:
                out.append((f"SPEC|{top}S{s}|n{n}", top, s, n, base(n=n, ladder=lad, targets=targets, admit="determinacy")))
    return out


def design_power(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("estimate") is not None]
    n_all = len(rows)
    trues = np.array([r["true"] for r in ok], dtype=float)
    rej = np.array([bool(r["se"] > 0 and r["estimate"] - Z * r["se"] > 0) for r in ok])
    pred = float(trues.mean()) if ok else None
    p_rc = float(rej.sum() / n_all) if n_all else None
    out = {
        "worlds": n_all,
        "tested_share": len(ok) / n_all if n_all else None,
        "predicted_slope": pred,
        "predicted_slope_mc_interval": [pred - Z * trues.std(ddof=1) / np.sqrt(len(trues)), pred + Z * trues.std(ddof=1) / np.sqrt(len(trues))] if len(trues) > 1 else None,
        "predicted_slope_spread_2_5_97_5": [float(np.quantile(trues, 0.025)), float(np.quantile(trues, 0.975))] if len(trues) > 1 else None,
        "random_candidate": p_rc,
        "random_candidate_mc_se": float(np.sqrt(p_rc * (1 - p_rc) / n_all)) if n_all else None,
    }
    if ok and pred is not None and pred != 0:
        near = np.abs(trues - pred) <= 0.2 * abs(pred)
        p = float(rej[near].mean()) if near.any() else None
        out.update({"fixed_candidate": p, "fixed_candidate_worlds": int(near.sum()),
                    "fixed_candidate_mc_se": float(np.sqrt(p * (1 - p) / near.sum())) if near.any() else None})
    else:
        out.update({"fixed_candidate": None, "fixed_candidate_worlds": 0, "fixed_candidate_mc_se": None})
    out["valid_random"] = bool((out["tested_share"] or 0) >= 0.9)
    out["valid_fixed"] = bool(out["valid_random"] and out["fixed_candidate_worlds"] >= 40)
    return out


def curve_implied_slope(ladder: list[float], targets: list[int], curve: dict, coincide: bool) -> float | None:
    from asla.analysis import excess_curve
    from asla.analysis.confound import MIN_RUNGS, WindowDesign, _within
    from asla.analysis.fixed_target import fixed_target_slices

    ds = []
    for t in targets:
        for f in range(MIN_RUNGS - 1, t):
            for s in range(0, f - MIN_RUNGS + 2):
                ds.append(WindowDesign(start=s, top=f, target=t, lever=ladder[t] - ladder[f], log_top=ladder[f], rungs=f - s + 1))
    groups = [g for gs in fixed_target_slices(ds, targets).values() for g in gs]
    if not groups:
        return None
    vals = []
    for d in ds:
        dd_index = min(range(len(NODES)), key=lambda i: abs(NODES[i] - ladder[d.target]))
        key = "common" if coincide or f"s_{dd_index}" not in curve else str(dd_index)
        sl = curve["common_s"] if key == "common" else curve[f"s_{dd_index}"]
        tau = curve["common_tau"] if key == "common" else curve[f"tau_{dd_index}"]
        vals.append(float(excess_curve.saturating(np.array([d.lever]), sl, tau)[0]))
    return float(_within(np.array(vals), np.array([d.lever for d in ds]), groups))


def run_spec() -> None:
    by_geom: dict[tuple[str, int], list[tuple[str, str, int, int, dict]]] = {}
    for item in spec_designs():
        by_geom.setdefault((item[1], item[2]), []).append(item)
    for geom, items in by_geom.items():
        for name, top, s, n, sp in items:
            run_jobs(jobs_for(name, sp, SPEC_WORLDS), CACHE, workers=1, timeout=1800)
            p = design_power(collect(read_cache(CACHE), name, SPEC_WORLDS))
            if p["valid_fixed"] and (p["random_candidate"] or 0) >= 0.8 and (p["fixed_candidate"] or 0) >= 0.8:
                print(f"{geom}: both framings reach 0.8 at n={n}; stopping this geometry", flush=True)
                break


def summarise_spec() -> dict:
    cache = read_cache(CACHE)
    curve = json.loads(CURVE_OUT.read_text(encoding="utf-8")) if CURVE_OUT.exists() else None
    cvals = {k: v["estimate"] for k, v in curve["statistics"].items()} if curve else {}
    designs = {}
    for name, top, s, n, sp in spec_designs():
        rows = collect(cache, name, SPEC_WORLDS)
        if not rows:
            continue
        lad = sp["ladder"]
        flops = float(n * 3 * np.sum(10.0 ** np.asarray(lad)))
        d = design_power(rows) | {
            "top": top, "sizes": s, "recipes": n, "seeds": 3, "ladder_sizes": [LABELS[NODES.index(x)] for x in lad],
            "targets": [LABELS[NODES.index(lad[t])] for t in sp["targets"]],
            "curve_implied_slope": curve_implied_slope(lad, sp["targets"], cvals, bool(curve and curve["coincide"])) if curve else None,
            "training_flops": flops,
            "gpu_hours": {k: flops / (h["peak_flops"] * h["mfu"]) / 3600 for k, h in HARDWARE.items()},
        }
        if n == 25:
            d["note"] = "25 recipes drawn without replacement is always DataDecide's 25: no recipe variation"
        designs[name.split("|", 1)[1]] = d
    cheapest = {}
    for framing, valid_key in (("random_candidate", "valid_random"), ("fixed_candidate", "valid_fixed")):
        ok = sorted([v for v in designs.values() if v[valid_key] and (v[framing] or 0) >= 0.8], key=lambda v: v["training_flops"])
        cheapest[framing] = ok[0] if ok else None
        if not ok:
            best = sorted([v for v in designs.values() if v[valid_key]], key=lambda v: -(v[framing] or 0))
            cheapest[f"{framing}_best_below_0_8"] = best[0] if best else None
    out = {
        "preregistration": "PREDICTIONS_TASK_CURVE.md",
        "unit": "points per decade (log10) of lever arm",
        "hardware_assumptions": HARDWARE,
        "designs": designs,
        "cheapest": cheapest,
        "prediction_D1_valid_random_0_8_within_25_with_300M": bool(
            cheapest["random_candidate"] is not None and cheapest["random_candidate"]["recipes"] <= 25 and cheapest["random_candidate"]["top"] == "300M"
        ),
    }
    SPEC_OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    stage = sys.argv[1]
    if stage == "validate":
        jobs = [j for name, sp, w in validation_specs() for j in jobs_for(name, sp, w)]
        run_jobs(jobs, CACHE, workers=1, timeout=1800)
        res = summarise_validation()
        print(json.dumps({k: res[k] for k in ("V1", "V2_k10", "V2_k15", "generative_sanity_25", "pass")}, indent=1))
        print(json.dumps(res["plateau"], indent=1))
    elif stage == "curve":
        run_jobs(jobs_for("CURVE_CAL", base(statistic="curve"), 200), CACHE, workers=1, timeout=1800)
        res = summarise_curve()
        print(json.dumps({k: res[k] for k in ("coincidence", "coincide", "curve_departure_from_common", "predictions", "wald_inflation")}, indent=1))
    elif stage == "spec":
        run_spec()
        res = summarise_spec()
        print(json.dumps({k: (None if v is None else {a: v.get(a) for a in ("top", "sizes", "recipes", "predicted_slope", "random_candidate", "random_candidate_mc_se", "fixed_candidate", "fixed_candidate_mc_se", "fixed_candidate_worlds", "training_flops")}) for k, v in res["cheapest"].items()}, indent=1))


if __name__ == "__main__":
    main()
