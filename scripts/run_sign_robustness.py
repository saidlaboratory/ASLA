"""Sign robustness across geometries, and what drives the design dependence (PREDICTIONS_TASK_SIGN_ROBUSTNESS.md).

Task 1 enumerates every valid geometry (a subset of DataDecide's sizes topped by a target with at least five
sizes below it) and computes the frozen fixed-target slope and its jackknife SE on each, through
asla.analysis.geometry_scan (checked against the frozen path in tests/test_geometry_scan.py). Results are
checkpointed one geometry per line, so an interrupted run resumes. Task 2 computes per-size influence at each
target, a Shapley decomposition of two designs' slopes, and a per-window split of their gap to the
curve-implied slope.

Run niced, one process: nice -n 19 .venv/bin/python scripts/run_sign_robustness.py [scan|dependence|summary]
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from asla.analysis import excess_curve  # noqa: E402
from asla.analysis.geometry_scan import Scanner  # noqa: E402
from asla.models import fit_power_law, bpb_power_law  # noqa: E402

OUT = REPO / "results" / "target_scoring"
CHECKPOINT = OUT / "sign_robustness_geometries.jsonl"
TARGET_LABELS = ("150M", "300M", "530M", "1B")
MIN_BELOW = 5
Z = 1.959963984540054
DESIGNS = {
    "300M_8": ["4M", "6M", "14M", "20M", "60M", "90M", "150M", "300M"],
    "1B_6": ["4M", "10M", "60M", "90M", "300M", "1B"],
}


def scanner() -> Scanner:
    import importlib

    return Scanner(importlib.import_module("scripts.run_confound")._load_frame())


def geometries(sc: Scanner) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    tix = [sc.labels.index(t) for t in TARGET_LABELS]
    out = []
    for top in tix:
        below = range(top)
        for k in range(MIN_BELOW, top + 1):
            for sub in combinations(below, k):
                sizes = (*sub, top)
                targets = tuple(t for t in tix if t in sizes and sum(s < t for s in sizes) >= MIN_BELOW)
                out.append((sizes, targets))
    return out


def run_scan() -> None:
    sc = scanner()
    done = set()
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            done.add(tuple(json.loads(line)["sizes"]))
    todo = [g for g in geometries(sc) if tuple(sc.labels[i] for i in g[0]) not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with CHECKPOINT.open("a", encoding="utf-8") as fh:
        for j, (sizes, targets) in enumerate(todo):
            r = sc.slope(sizes, targets)
            rec = {
                "sizes": [sc.labels[i] for i in sizes],
                "targets": [sc.labels[t] for t in targets],
                "estimate": r["estimate"],
                "se": r["se"],
                "n_designs": r["n_designs"],
            }
            fh.write(json.dumps(rec) + "\n")
            if j % 100 == 0:
                fh.flush()
                print(f"{j}/{len(todo)}  kernels cached {len(sc._kernel)}", flush=True)


def shares(rows: list[dict]) -> dict:
    est = np.array([r["estimate"] for r in rows], dtype=float)
    se = np.array([r["se"] for r in rows], dtype=float)
    ok = np.isfinite(est) & np.isfinite(se)
    est, se = est[ok], se[ok]
    n = len(est)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "share_positive": float(np.mean(est > 0)),
        "share_positive_excluding_zero": float(np.mean(est - Z * se > 0)),
        "share_negative_excluding_zero": float(np.mean(est + Z * se < 0)),
        "quantiles": {q: float(np.quantile(est, float(q))) for q in ("0.0", "0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975", "1.0")},
        "median_se": float(np.median(se)),
    }


def summarise_scan() -> dict:
    rows = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()]
    seen = {}
    for r in rows:
        seen[tuple(r["sizes"])] = r
    rows = list(seen.values())
    overall = shares(rows)
    by_top = {t: shares([r for r in rows if r["sizes"][-1] == t]) for t in TARGET_LABELS}
    counts = sorted({len(r["sizes"]) for r in rows})
    by_count = {str(k): shares([r for r in rows if len(r["sizes"]) == k]) for k in counts}
    by_nt = {str(k): shares([r for r in rows if len(r["targets"]) == k]) for k in sorted({len(r["targets"]) for r in rows})}
    pse = [by_count[str(k)]["share_positive_excluding_zero"] for k in counts]
    top_pse = {t: by_top[t]["share_positive_excluding_zero"] for t in TARGET_LABELS}
    from scipy import stats

    rho = stats.spearmanr(counts, pse).statistic
    predictions = {
        "R1_share_positive_85_to_100": bool(0.85 <= overall["share_positive"] <= 1.0),
        "R2_share_positive_excluding_zero_60_to_90": bool(0.60 <= overall["share_positive_excluding_zero"] <= 0.90),
        "R3_share_negative_excluding_zero_0_to_5": bool(overall["share_negative_excluding_zero"] <= 0.05),
        "R4_lowest_for_1B_highest_for_150M": bool(
            min(top_pse, key=top_pse.get) == "1B" and max(top_pse, key=top_pse.get) == "150M"
        ),
        "R5_rises_with_size_count_monotone": bool(all(b >= a for a, b in zip(pse, pse[1:]))),
    }
    out = {
        "preregistration": "PREDICTIONS_TASK_SIGN_ROBUSTNESS.md",
        "metric": "c4_en_bits_per_token",
        "unit": "fixed-target pooled slope, points of mis-selected pairs per decade of lever arm",
        "interval": "estimate +/- 1.96 leave-one-recipe-out jackknife SE",
        "note": "geometries overlap heavily; shares describe robustness and are not a pooled test",
        "n_geometries": len(rows),
        "overall": overall,
        "by_top_target": by_top,
        "by_size_count": by_count,
        "by_number_of_targets": by_nt,
        "spearman_size_count_vs_positive_excluding_zero": float(rho),
        "predictions": predictions,
        "most_negative": sorted(rows, key=lambda r: r["estimate"])[:10],
    }
    (OUT / "sign_robustness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def size_diagnostics(sc: Scanner) -> dict:
    frame = sc._frame
    full = list(range(len(sc.labels)))
    resid = {}
    for r in sc.recipes:
        p = fit_power_law(sc.compute, sc.means[r])
        resid[r] = np.log(sc.means[r]) - np.log(bpb_power_law(sc.compute, *p))
    mat = np.array([resid[r] for r in sc.recipes])
    specific = mat - mat.mean(axis=0)
    rel = frame.groupby(["scale_label", "intervention"])["bpb"].agg(lambda v: np.std(v, ddof=1) / np.mean(v))
    tpp = frame.groupby("scale_label")["tokens_per_param"].first()
    out = {}
    for i in full:
        lab = sc.labels[i]
        out[lab] = {
            "misfit_sd_log_loss": float(np.std(specific[:, i], ddof=1)),
            "median_relative_seed_sd": float(np.median(rel.loc[lab].to_numpy())),
            "tokens_per_param": float(tpp.loc[lab]),
            "gap_below_decades": float(sc.log_c[i] - sc.log_c[i - 1]) if i > 0 else None,
            "gap_above_decades": float(sc.log_c[i + 1] - sc.log_c[i]) if i + 1 < len(full) else None,
        }
    return out


def curve_params() -> dict:
    curve = json.loads((OUT / "excess_curve.json").read_text(encoding="utf-8"))
    return {k: v["estimate"] for k, v in curve["statistics"].items()}


def gap_split(sc: Scanner, sizes: list[int], targets: list[int], cvals: dict) -> dict:
    r = sc.slope(sizes, targets)
    ds, x, y, groups = r["designs"], r["lever"], r["excess"], r["groups"]
    c = np.array([float(excess_curve.saturating(np.array([x[i]]), cvals[f"s_{t}"], cvals[f"tau_{t}"])[0]) for i, (w, t) in enumerate(ds)])
    den = sum(float(((x[g] - x[g].mean()) ** 2).sum()) for g in groups)
    terms = {}
    for g in groups:
        xc = x[g] - x[g].mean()
        for j, i in enumerate(g):
            terms[i] = float(xc[j] * (y[i] - c[i]) / den)
    implied = Scanner.pooled_slope(c, x, groups)
    by_slice: dict[str, float] = {}
    by_top: dict[str, float] = {}
    for i, v in terms.items():
        w, t = ds[i]
        by_slice[f"{sc.labels[t]}|{len(w)} rungs"] = by_slice.get(f"{sc.labels[t]}|{len(w)} rungs", 0.0) + v
        by_top[sc.labels[w[-1]]] = by_top.get(sc.labels[w[-1]], 0.0) + v
    return {
        "slope": r["estimate"],
        "se": r["se"],
        "curve_implied": implied,
        "gap": r["estimate"] - implied,
        "sum_of_terms": float(sum(terms.values())),
        "by_slice": by_slice,
        "by_top_rung": by_top,
    }


def run_dependence() -> dict:
    sc = scanner()
    lab = sc.labels.index
    diag = size_diagnostics(sc)
    influence = {}
    flagged: set[str] = set()
    for tl in TARGET_LABELS:
        t = lab(tl)
        full = list(range(t + 1))
        base = sc.slope(full, [t])
        rows = {}
        for s in range(t):
            r = sc.slope([i for i in full if i != s], [t])
            infl = (r["estimate"] - base["estimate"]) / base["se"]
            flip = bool(np.sign(r["estimate"]) != np.sign(base["estimate"]))
            flag = bool(abs(infl) > 1 or flip)
            if flag:
                flagged.add(sc.labels[s])
            rows[sc.labels[s]] = {"slope_without": r["estimate"], "influence_se": float(infl), "sign_flip": flip, "flagged": flag}
        influence[tl] = {"full_slope": base["estimate"], "full_se": base["se"], "drop": rows}
    cvals = curve_params()
    decomposition = {}
    for name, labels in DESIGNS.items():
        sizes = [lab(s) for s in labels]
        top = sizes[-1]
        targets = [t for t in (lab(x) for x in TARGET_LABELS) if t in sizes and sum(s < t for s in sizes) >= MIN_BELOW]
        removed = [i for i in range(top) if i not in sizes]
        cache: dict[frozenset, float] = {}

        def value(extra: frozenset) -> float:
            if extra not in cache:
                cache[extra] = sc.slope(sorted([*sizes, *extra]), targets)["estimate"]
            return cache[extra]

        m = len(removed)
        shap = {}
        for s in removed:
            others = [o for o in removed if o != s]
            tot = 0.0
            for k in range(m):
                wgt = factorial(k) * factorial(m - k - 1) / factorial(m)
                for sub in combinations(others, k):
                    a = frozenset(sub)
                    # contribution to (design - full): removing s, given the others removed beyond sub
                    tot += wgt * (value(frozenset(removed) - a - {s}) - value(frozenset(removed) - a))
            shap[sc.labels[s]] = tot
        design_slope = value(frozenset())
        full_slope = value(frozenset(removed))
        mags = sorted((abs(v) for v in shap.values()), reverse=True)
        top2 = sum(mags[:2]) / sum(mags) if sum(mags) > 0 else float("nan")
        decomposition[name] = {
            "sizes": labels,
            "targets": [sc.labels[t] for t in targets],
            "full_ladder_slope": full_slope,
            "design_slope": design_slope,
            "change": design_slope - full_slope,
            "shapley_removed_sizes": shap,
            "shapley_sum": float(sum(shap.values())),
            "top2_share_of_abs": float(top2),
            "concentrated": bool(top2 >= 0.6),
            "gap_split": gap_split(sc, sizes, targets, cvals),
        }
    top_sizes = {s for d in decomposition.values() for s in sorted(d["shapley_removed_sizes"], key=lambda k: -abs(d["shapley_removed_sizes"][k]))[:2]}
    out = {
        "preregistration": "PREDICTIONS_TASK_SIGN_ROBUSTNESS.md",
        "unit": "slopes in points per decade; influence in SE units of the full-ladder slope",
        "influence": influence,
        "flagged_sizes": sorted(flagged, key=lab),
        "decomposition": decomposition,
        "size_diagnostics": diag,
        "diagnosed_sizes": sorted(flagged | top_sizes, key=lab),
        "predictions": {
            "I1_20M_and_60M_flagged": bool({"20M", "60M"} <= flagged),
            "I2_both_designs_concentrated": bool(all(d["concentrated"] for d in decomposition.values())),
        },
    }
    (OUT / "design_dependence.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        run_scan()
    elif cmd == "summary":
        print(json.dumps(summarise_scan()["predictions"], indent=2))
    elif cmd == "dependence":
        print(json.dumps(run_dependence()["predictions"], indent=2))
