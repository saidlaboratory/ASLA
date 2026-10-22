"""PREDICTIONS_TASK_NOISE_BRACKET.md, Tasks 2 and 3: leaderboard resolution at 1e22; power for the class comparison.

Writes ``results/external/leaderboard_resolution.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402

OUT = REPO / "results" / "external" / "leaderboard_resolution.json"
BRACKET = REPO / "results" / "external" / "noise_bracket.json"
DRAWS = 1000
SEED = 20261022
DD_1B = 0.000511266156200176
DELPHI_BUDGETS = (3e18, 9e18, 3e19, 1.8e20)
DELPHI_TARGETS = (1e21, 1e22, 1e23)
RUNGS = ("130m", "300m", "520m", "1_2b")
DD_MATCH = {"130m": "150M", "300m": "300M", "520m": "530M", "1_2b": "1B"}
POWER_DIFF = 0.04
POWER_DRAWS = 2000


# --- Task 2 ----------------------------------------------------------------------------------


def bootstrap(pts: dict[str, list[dict[str, float]]], sigma: np.ndarray, fixed: dict[str, float]) -> dict[str, Any]:
    members = sorted(pts)
    rng = np.random.default_rng(SEED)
    noise = 1 + sigma[None, None, :] * rng.standard_normal((DRAWS, len(members), len(RUNGS)))
    jobs = {f"{s}|{i}": [{"x": q["x"], "y": q["y"] * noise[i, j, r]} for r, q in enumerate(pts[s])]
            for i in range(DRAWS) for j, s in enumerate(members)}  # fmt: skip
    res = al.patched_predictions(jobs)
    f = np.array([[np.nan if res[f"{s}|{i}"][0] is None else res[f"{s}|{i}"][0] for s in members] for i in range(DRAWS)])
    ok = np.all(np.isfinite(f), axis=1)
    f = f[ok]
    holds = {}
    for a, b in combinations(members, 2):
        sign = np.sign(fixed[a] - fixed[b])
        holds[(a, b)] = float(np.mean(np.sign(f[:, members.index(a)] - f[:, members.index(b)]) == sign))
    order = sorted(members, key=lambda s: fixed[s])
    ranks = np.argsort(f, axis=1)
    topk = {}
    for k in range(1, len(members)):
        target = {members.index(s) for s in order[:k]}
        topk[k] = float(np.mean([set(r[:k]) == target for r in ranks]))
    robust = [list(p) for p, v in holds.items() if v >= 0.95]
    return {
        "draws_used": int(ok.sum()),
        "forecast_sd": {s: float(f[:, j].std(ddof=1)) for j, s in enumerate(members)},
        "pairs_holding_95": len(robust),
        "robust_pairs": robust,
        "largest_top_k_holding_95": max([k for k, v in topk.items() if v >= 0.95], default=0),
        # every top-j for j <= k holds: the pre-registered readout above is satisfied trivially at k = n - 1
        # whenever last place never moves
        "largest_contiguous_top_k_holding_95": next((k - 1 for k in sorted(topk) if topk[k] < 0.95), len(topk)),
        "top_k_hold_share": {str(k): v for k, v in topk.items() if k <= 5},
    }


def delphi_check(fixed: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gaps = np.diff(sorted(fixed.values()))
    median_gap, median_forecast = float(np.median(gaps)), float(np.median(list(fixed.values())))
    for recipe, table in (("adamh", "delphi-ladder"), ("cautious_adamc", "first-ladder")):
        d = pd.read_parquet(REPO / "data" / "raw" / "delphi" / table / "train.parquet")
        iso, opt, held = d[d["kind"] == "isoflop"], d[d["kind"] == "optimum"], d[d["kind"] == "held_out"]
        points: dict[str, list[dict[str, float]]] = {"lowest_loss_run": [], "loss_star": []}
        for c in DELPHI_BUDGETS:
            s = iso[np.isclose(iso["gflops_bucket_flops"], c)]
            best = s.loc[s["loss"].idxmin()]
            points["lowest_loss_run"].append({"x": float(best["total_gflops"]) * 1e9, "y": float(best["loss"])})
            o = opt[np.isclose(opt["C_flops"], c)].iloc[0]
            points["loss_star"].append({"x": c, "y": float(o["loss_star"])})
        pred = al.patched_predictions(points, DELPHI_TARGETS)
        actual = {}
        for t in DELPHI_TARGETS:
            h = held[np.isclose(held["C_nom"], t)]
            if recipe == "cautious_adamc" and t == 1e23:
                continue  # its "1e23" run has C = 6.9e22 and a loss above its own 1e22 run
            if len(h):
                actual[t] = {"loss": float(h["loss"].mean()), "runs": int(len(h)), "C": float(h["C_flops"].mean())}
        rows: dict[str, dict[str, Any]] = {}
        for kind, p in pred.items():
            rows[kind] = {}
            for t, v in zip(DELPHI_TARGETS, p):
                if t not in actual or v is None:
                    continue
                rel = (v - actual[t]["loss"]) / actual[t]["loss"]
                rows[kind][f"{t:.0e}"] = {"forecast": v, "actual": actual[t]["loss"], "relative_error": rel,
                                          "in_median_neighbour_gaps": abs(rel) * median_forecast / median_gap}  # fmt: skip
        out[recipe] = {"points": points, "actual": {f"{k:.0e}": v for k, v in actual.items()}, "errors": rows}
    out["leaderboard_median_neighbour_gap_bpb"] = median_gap
    out["leaderboard_median_forecast_bpb"] = median_forecast
    out["label"] = "absolute error, largely common across recipes: it bounds forecast accuracy, not ranking accuracy"
    return out


# --- Task 3 ----------------------------------------------------------------------------------


def listing() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    runs = json.loads((rig.RAW / "runs.json").read_text())
    by: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    unlabelled = []
    for r in runs:
        if "/" not in r["run_name"]:
            unlabelled.append(r["run_name"])
            continue
        prefix, label = r["run_name"].rsplit("/", 1)
        label = {"1.2b": "1_2b"}.get(label.lower(), label.lower())
        by[prefix][label] = {"x": float(r["model_flops"]), "y": float(r["eval_paloma_c4_en_bpb"])}
    rows = [
        {"prefix": p, "budgets": sorted(v), "model_flops": {k: q["x"] for k, q in v.items()}} for p, v in sorted(by.items())
    ]
    rows += [{"prefix": n, "budgets": [], "note": "no budget label"} for n in sorted(unlabelled)]
    cells = {(p, b): v[b] for p, v in by.items() for b in v if set(RUNGS) <= set(v)}
    return rows, cells


def dd_reorder(f: str, t: str) -> dict[str, Any]:
    from asla.analysis.moderation import fit_prior, moderate

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    fit = frame[frame["scale_label"] == f].groupby("intervention")["bpb"].mean().to_dict()
    tcells = frame[frame["scale_label"] == t].groupby("intervention")["bpb"]
    target, s2, n = tcells.mean().to_dict(), tcells.var(ddof=1).to_dict(), int(tcells.count().min())
    prior = fit_prior(list(s2.values()), n - 1)
    mod = {r: float(moderate(v, prior)) for r, v in s2.items()}
    se = {(a, b): float(np.sqrt(mod[a] / n + mod[b] / n)) for a, b in combinations(sorted(target), 2)}
    r = ref.reorder(fit, target, se)
    return {"determined": r["determined"], "reordered": int(r["reordered"]), "rate": r["rate"]}


def marin_reorder(cells: dict, pairs: list[tuple[str, str]], f: str, t: str, cv: float) -> dict[str, Any]:
    det = wrong = 0
    for a, b in pairs:
        g = cells[(a, t)]["y"] - cells[(b, t)]["y"]
        se = cv * float(np.hypot(cells[(a, t)]["y"], cells[(b, t)]["y"]))
        if abs(g) > al.Z * se:
            det += 1
            wrong += int(np.sign(cells[(a, f)]["y"] - cells[(b, f)]["y"]) != np.sign(g))
    return {"pairs": len(pairs), "determined": det, "reordered": wrong, "rate": wrong / det if det else None}


def fisher(m: dict[str, Any], d: dict[str, Any]) -> float | None:
    if not m["determined"]:
        return None
    table = [[m["reordered"], m["determined"] - m["reordered"]], [d["reordered"], d["determined"] - d["reordered"]]]
    return float(stats.fisher_exact(table, alternative="greater").pvalue)


def power_n(p0: float, n0: int, diff: float) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    grid = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]
    curve = {}
    for n in grid:
        x1 = rng.binomial(n, p0 + diff, POWER_DRAWS)
        x0 = rng.binomial(n0, p0, POWER_DRAWS)
        hits = [stats.fisher_exact([[a, n - a], [b, n0 - b]], alternative="greater").pvalue < 0.05 for a, b in zip(x1, x0)]
        curve[n] = float(np.mean(hits))
    need = next((n for n in grid if curve[n] >= 0.8), None)
    return {"p0": p0, "n0": n0, "difference": diff, "power_by_n": {str(k): v for k, v in curve.items()}, "n_for_80pct": need}


def main() -> None:
    bracket = json.loads(BRACKET.read_text())
    sig = bracket["e_reruns"]["sigma_by_rung"]
    members, cells, _ = rig.load_cells()
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in RUNGS] for s in members}
    fixed = al.patched_forecasts(pts)
    form_core = {tuple(sorted(p)) for p in json.loads((REPO / "results" / "external" / "marin_aligned.json").read_text())[
        "task2"]["two_forms"]["invariant_pair_list"]}  # fmt: skip
    levels = {
        "0_0511pct": np.full(4, DD_1B),
        "task1_bound": np.array([sig[b] for b in RUNGS]),
        "0_43pct": np.full(4, al.BOUND_THREE),
    }
    t2a = {}
    for name, sigma in levels.items():
        b = bootstrap(pts, sigma, fixed)
        noise_pairs = {tuple(sorted(p)) for p in b["robust_pairs"]}
        b["intersection_with_form_core_175"] = len(noise_pairs & form_core)
        b["form_core_pairs"] = len(form_core)
        t2a[name] = b
    out: dict[str, Any] = {
        "preregistration": "PREDICTIONS_TASK_NOISE_BRACKET.md",
        "task2a": {
            "levels": t2a,
            "fixed_forecast": fixed,
            "note": "noise only: a lower bound on total forecast uncertainty (form, solver and extrapolation "
            "uncertainty are not included)",
        },  # fmt: skip
        "task2b": delphi_check(fixed),
    }

    rows, cells22 = listing()
    subs = sorted({k[0] for k in cells22})
    family = {s: s.split("/")[0] for s in subs}
    designs: dict[str, dict[str, Any]] = {}
    pooled = {"determined": 0, "reordered": 0}
    for f, t in combinations(RUNGS, 2):
        groups: dict[tuple[float, float], list[str]] = defaultdict(list)
        for s in subs:
            groups[(cells22[(s, f)]["x"], cells22[(s, t)]["x"])].append(s)
        genuine = [(a, b) for g in groups.values() for a, b in combinations(sorted(g), 2) if family[a] != family[b]]
        dd = dd_reorder(DD_MATCH[f], DD_MATCH[t])
        m = marin_reorder(cells22, genuine, f, t, DD_1B)
        mb = marin_reorder(cells22, genuine, f, t, sig[t])
        designs[f"{f}->{t}"] = {"genuine_pairs": len(genuine), "flops_groups": len(groups), "marin_dd1b_sd": m,
                                "marin_task1_bound": mb, "datadecide_matched": dd,
                                "fisher_p_one_sided_dd1b": fisher(m, dd),
                                "fisher_p_one_sided_bound": fisher(mb, dd)}  # fmt: skip
        pooled["determined"] += m["determined"]
        pooled["reordered"] += m["reordered"]
    primary_dd = designs["520m->1_2b"]["datadecide_matched"]
    power = {
        "primary_530M_1B": power_n(primary_dd["rate"], primary_dd["determined"], POWER_DIFF),
        "sensitivity_300M_1B": power_n(designs["300m->1_2b"]["datadecide_matched"]["rate"],
                                       designs["300m->1_2b"]["datadecide_matched"]["determined"], POWER_DIFF),
    }  # fmt: skip
    out["task3"] = {
        "listing": rows,
        "links": len(json.loads((rig.RAW / "runs.json").read_text())),
        "four_rung_submissions": subs,
        "designs": designs,
        "pooled_over_designs_not_independent": {**pooled, "rate": pooled["reordered"] / pooled["determined"]},
        "power": power,
    }
    b = t2a["task1_bound"]
    gap = out["task2b"]["adamh"]["errors"]["lowest_loss_run"].get("1e+22", {}).get("in_median_neighbour_gaps")
    out["predictions"] = {
        "PL1_fewer_than_100_pairs_hold": b["pairs_holding_95"] < 100,
        "PL1_top_k_0_or_1": b["largest_top_k_holding_95"] <= 1,
        "PL2_delphi_1e22_error_over_10_gaps": gap is not None and gap > 10,
        "PP1_n_at_least_150": (power["primary_530M_1B"]["n_for_80pct"] or np.inf) >= 150,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
