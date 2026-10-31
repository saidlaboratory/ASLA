"""Config steps in DataDecide's ladder and the lever-arm effect (PREDICTIONS_TASK_CONFIG_STEPS.md).

(b) misfit at step sizes, (c) within-regime against spanning windows at fixed targets and across last round's
geometries, (d) the step-term projection. One niced process; the geometry scan checkpoints one geometry per
line so an interrupted run resumes.

Run: nice -n 19 .venv/bin/python scripts/run_config_steps.py [main|geometries|summary]
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from asla.analysis.geometry_scan import Scanner  # noqa: E402
from asla.analysis.step_fit import jk_se, step_kernel, subset_stat  # noqa: E402
from asla.models import bpb_power_law, fit_power_law  # noqa: E402

OUT = REPO / "results" / "target_scoring"
MAIN = OUT / "config_steps_main.json"
GEOM = OUT / "config_steps_geometries.jsonl"
Z = 1.959963984540054
TARGETS = ("150M", "300M", "530M", "1B")
PRIMARY = {"A": ["4M", "6M", "8M", "10M", "14M", "16M"], "B": ["20M", "60M", "90M"], "C": ["150M", "300M", "530M", "1B"]}
DEPTH = {"D8": PRIMARY["A"], "D16a": PRIMARY["B"], "D12": ["150M"], "D16b": ["300M", "530M", "1B"]}
STEP_SIZES = {"primary": ["16M", "20M", "90M", "150M"], "depth": ["16M", "20M", "90M", "150M", "300M"]}


def scanner() -> Scanner:
    import importlib

    return Scanner(importlib.import_module("scripts.run_confound")._load_frame())


def regime_of(sc: Scanner, regimes: dict) -> list[str]:
    m = {s: k for k, v in regimes.items() for s in v}
    return [m[l] for l in sc.labels]


def boundaries_in(sc: Scanner, reg: list[str], w: tuple[int, ...]) -> list[float]:
    return [float(sc.compute[w[i]]) for i in range(1, len(w)) if reg[w[i]] != reg[w[i - 1]]]


def summary(st: dict) -> dict:
    e, se = st["estimate"], st["se"]
    ok = np.isfinite(e)
    return {
        "estimate": float(e) if ok else None,
        "se": float(se) if ok else None,
        "interval": [float(e - Z * se), float(e + Z * se)] if ok else None,
        "n_designs": st["n_designs"],
        "n_slices": st["n_slices"],
    }


def contrast(a: dict, b: dict, fn) -> dict:
    est = fn(a["estimate"], b["estimate"])
    loo = fn(a["loo"], b["loo"])
    if not np.isfinite(est):
        return {"estimate": None, "se": None, "interval": None}
    se = jk_se(loo)
    return {"estimate": float(est), "se": se, "interval": [float(est - Z * se), float(est + Z * se)]}


# ------------------------------------------------------------------ (b) misfit


def misfit(sc: Scanner) -> dict:
    frame = sc._frame
    resid = []
    for r in sc.recipes:
        p = fit_power_law(sc.compute, sc.means[r])
        resid.append(np.log(sc.means[r]) - np.log(bpb_power_law(sc.compute, *p)))
    spec = np.array(resid) - np.mean(resid, axis=0)
    raw = spec.std(axis=0, ddof=1)
    # variance of the seed mean in log loss, per recipe and size, averaged over recipes
    vbar = []
    for lab in sc.labels:
        g = frame[frame["scale_label"] == lab].groupby("intervention")["bpb"]
        vbar.append(float(np.mean([np.var(np.log(v), ddof=1) / len(v) for _, v in g])))
    vbar = np.array(vbar)
    corrected = np.sqrt(np.maximum(0.0, raw**2 - vbar))
    out = {"per_size": {l: {"raw": float(raw[i]), "seed_mean_sd": float(np.sqrt(vbar[i])), "corrected": float(corrected[i])} for i, l in enumerate(sc.labels)}}
    for name, steps in STEP_SIZES.items():
        idx = [sc.labels.index(s) for s in steps]
        res = {}
        for kind, m in (("corrected", corrected), ("raw", raw)):
            obs = m[idx].mean() - np.delete(m, idx).mean()
            null = [m[list(c)].mean() - np.delete(m, list(c)).mean() for c in combinations(range(len(m)), len(idx))]
            res[kind] = {"difference": float(obs), "p_one_sided": float(np.mean(np.array(null) >= obs - 1e-15)), "subsets": len(null)}
        out[name] = {"step_sizes": steps, **res}
    return out


# ------------------------------------------------------------------ (c) and (d)


def window_sets(sc: Scanner, reg: list[str], t: int) -> dict:
    ds = sc.designs(range(len(sc.labels)), [t])
    within = [d for d in ds if len({reg[i] for i in d[0]}) == 1]
    spanning = [d for d in ds if d not in within]
    a_only = [d for d in within if reg[d[0][0]] == reg[0]]
    return {"all": ds, "within": within, "spanning": spanning, "a_only": a_only}


def part_c(sc: Scanner, regimes: dict) -> dict:
    reg = regime_of(sc, regimes)
    tix = [sc.labels.index(t) for t in TARGETS]
    sets = {t: window_sets(sc, reg, t) for t in tix}
    pooled = {k: [d for t in tix for d in sets[t][k]] for k in ("all", "within", "spanning", "a_only")}
    stats = {"pooled": {k: subset_stat(sc, v) for k, v in pooled.items()}}
    for t in tix:
        stats[sc.labels[t]] = {k: subset_stat(sc, v) for k, v in sets[t].items()}
    out = {}
    for key, st in stats.items():
        out[key] = {
            "W_within": summary(st["within"]),
            "W2_regime_A_only": summary(st["a_only"]),
            "S_spanning": summary(st["spanning"]),
            "all": summary(st["all"]),
            "S_minus_W": contrast(st["spanning"], st["within"], lambda a, b: a - b),
        }
    F = stats["pooled"]["all"]
    W = stats["pooled"]["within"]
    out["F_reference"] = summary(F)
    out["r_W_over_F"] = contrast(W, F, lambda a, b: a / b)
    out["r2_W2_over_F"] = contrast(stats["pooled"]["a_only"], F, lambda a, b: a / b)
    out["outcome"] = outcome(W, F)
    out["outcome_W2"] = outcome(stats["pooled"]["a_only"], F)
    return out


def outcome(W: dict, F: dict) -> str:
    w, se, f = W["estimate"], W["se"], F["estimate"]
    if not np.isfinite(w):
        return "no slice"
    lo, hi = w - Z * se, w + Z * se
    if lo > 0:
        return "O1 persists at a similar size" if w / f >= 0.75 else "O2 persists but shrinks"
    if hi < 0.5 * f:
        return "O3 vanishes"
    return "O4 inconclusive"


def part_d(sc: Scanner) -> dict:
    reg = regime_of(sc, PRIMARY)
    tix = [sc.labels.index(t) for t in TARGETS]
    keep, kern_step, excluded = [], [], 0
    for t in tix:
        for d in sc.designs(range(len(sc.labels)), [t]):
            b = boundaries_in(sc, reg, d[0])
            if not b:
                keep.append(d)
                kern_step.append(sc.kernel(*d))
            elif len(d[0]) >= 3 + len(b) + 1:
                keep.append(d)
                kern_step.append(step_kernel(sc, d[0], d[1], b))
            else:
                excluded += 1
    D = subset_stat(sc, keep, np.array(kern_step))
    Fs = subset_stat(sc, keep)
    return {
        "D_step_fit": summary(D),
        "F_same_frozen": summary(Fs),
        "share_1_minus_D_over_F_same": contrast(D, Fs, lambda a, b: 1 - a / b),
        "difference_D_minus_F_same": contrast(D, Fs, lambda a, b: a - b),
        "spanning_windows_excluded_for_dof": excluded,
        "windows_used": len(keep),
    }


def run_main() -> dict:
    sc = scanner()
    out = {
        "misfit": misfit(sc),
        "within_vs_spanning_primary": part_c(sc, PRIMARY),
        "within_vs_spanning_depth": part_c(sc, DEPTH),
        "step_term": part_d(sc),
    }
    MAIN.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_geometries() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from run_sign_robustness import geometries

    sc = scanner()
    reg = regime_of(sc, PRIMARY)
    done = set()
    if GEOM.exists():
        done = {tuple(json.loads(l)["sizes"]) for l in GEOM.read_text(encoding="utf-8").splitlines()}
    with GEOM.open("a", encoding="utf-8") as fh:
        for j, (sizes, targets) in enumerate(geometries(sc)):
            key = tuple(sc.labels[i] for i in sizes)
            if key in done:
                continue
            ds = [d for d in sc.designs(sizes, targets) if len({reg[i] for i in d[0]}) == 1]
            st = subset_stat(sc, ds)
            ok = np.isfinite(st["estimate"])
            fh.write(json.dumps({"sizes": list(key), "estimate": float(st["estimate"]) if ok else None, "se": float(st["se"]) if ok else None, "n_designs": st["n_designs"]}) + "\n")
            if j % 500 == 0:
                fh.flush()
                print(j, flush=True)


def summarise() -> dict:
    main = json.loads(MAIN.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in GEOM.read_text(encoding="utf-8").splitlines()]
    tested = [r for r in rows if r["estimate"] is not None]
    e = np.array([r["estimate"] for r in tested])
    s = np.array([r["se"] for r in tested])
    geo = {
        "geometries": len(rows),
        "tested": len(tested),
        "share_positive": float(np.mean(e > 0)) if len(e) else None,
        "share_positive_excluding_zero": float(np.mean(e - Z * s > 0)) if len(e) else None,
        "share_negative_excluding_zero": float(np.mean(e + Z * s < 0)) if len(e) else None,
        "median": float(np.median(e)) if len(e) else None,
        "median_se": float(np.median(s)) if len(e) else None,
    }
    pc, pd_ = main["within_vs_spanning_primary"], main["step_term"]
    share_d = pd_["share_1_minus_D_over_F_same"]
    out = {
        "preregistration": "PREDICTIONS_TASK_CONFIG_STEPS.md",
        "metric": "c4_en_bits_per_token",
        "unit": "slopes in points of mis-selected pairs per decade of lever arm; misfit in log loss",
        **main,
        "within_regime_geometries": geo,
        "steps_contribute": bool(
            pc["outcome"].startswith(("O2", "O3")) or (share_d["interval"] is not None and share_d["interval"][0] > 0)
        ),
        "predictions": {
            "M1_corrected_misfit_not_concentrated": bool(main["misfit"]["primary"]["corrected"]["p_one_sided"] >= 0.05),
            "M2_raw_misfit_not_concentrated": bool(main["misfit"]["primary"]["raw"]["p_one_sided"] >= 0.05),
            "W1_primary_outcome_O4": pc["outcome"].startswith("O4"),
            "W2_regime_A_interval_includes_zero": bool(
                pc["pooled"]["W2_regime_A_only"]["interval"] is None
                or pc["pooled"]["W2_regime_A_only"]["interval"][0] <= 0 <= pc["pooled"]["W2_regime_A_only"]["interval"][1]
            ),
            "D1_step_share_interval_includes_zero": bool(share_d["interval"] is not None and share_d["interval"][0] <= 0 <= share_d["interval"][1]),
            "G1_within_regime_positive_share_60_to_90": bool(geo["share_positive"] is not None and 0.6 <= geo["share_positive"] <= 0.9),
        },
    }
    (OUT / "config_steps.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    if cmd == "main":
        run_main()
    elif cmd == "geometries":
        run_geometries()
    elif cmd == "summary":
        print(json.dumps(summarise()["predictions"], indent=2))
