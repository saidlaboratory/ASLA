"""PREDICTIONS_TASK_NOISE_BRACKET.md, Task 1: narrow the single-run noise bracket from public data.

(a) Delphi isoFLOP sweeps, (b) Fantastic Optimizers one-parameter slices, (c) the same leave-one-out
procedure on DataDecide where the seed SD is known, (d) the bound per speedrun size, (e) Marin reruns.

Writes ``results/external/noise_bracket.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import make_smoothing_spline

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402
from scripts.run_marin_noise import N_NULL, SEED, both_rung, determined  # noqa: E402

OUT = REPO / "results" / "external" / "noise_bracket.json"
DELPHI = {"adamh": "delphi-ladder", "cautious_adamc": "first-ladder"}
DELPHI_BUDGETS = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)
SIZE_TO_DELPHI = {"130m": 3e18, "300m": 9e18, "520m": 3e19, "1_2b": 1.8e20}
FO_SIZE = {"130m": "130m", "300m": "300m", "520m": "520m", "1_2b": "1.2b"}
DD_SIZES = ("60M", "90M", "150M", "300M", "530M", "750M", "1B")
NEAR_OPTIMUM = 0.02
FLEX: dict[str, int] = {"linear": 2, "quadratic": 3, "cubic": 4, "spline": 5}  # minimum training points


# --- leave-one-out ---------------------------------------------------------------------------


def fitter(kind: str) -> Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    if kind == "spline":

        def spline(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
            order = np.argsort(x)
            return make_smoothing_spline(x[order], y[order])

        return spline
    degree = {"linear": 1, "quadratic": 2, "cubic": 3}[kind]

    def poly(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        coef = np.polyfit(x, y, degree)
        return lambda z: np.polyval(coef, z)

    return poly


def loo(x: np.ndarray, y: np.ndarray, kind: str) -> list[float] | None:
    """Relative leave-one-out errors (y - yhat) / yhat, or None if the fit is not identified."""

    n = len(x)
    if n - 1 < FLEX[kind] or (kind == "spline" and len(np.unique(x)) < n):
        return None
    out = []
    fit = fitter(kind)
    for i in range(n):
        keep = np.arange(n) != i
        try:
            yhat = float(fit(x[keep], y[keep])(x[i]))
        except ValueError:
            return None
        out.append((y[i] - yhat) / yhat)
    return out


def residual(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, int] | None:
    """(sum of squared relative residuals, df) for the polynomial flexibilities."""

    if kind == "spline" or len(x) <= FLEX[kind]:
        return None
    yhat = fitter(kind)(x, y)(x)
    return float(np.sum(((y - yhat) / yhat) ** 2)), len(x) - FLEX[kind]


def summarise(errors: list[float], res: list[tuple[float, int]]) -> dict[str, Any]:
    e = np.asarray(errors)
    df = sum(r[1] for r in res)
    return {
        "loo_rms": float(np.sqrt(np.mean(e**2))) if e.size else None,
        "loo_n": int(e.size),
        "residual_sd": float(np.sqrt(sum(r[0] for r in res) / df)) if df else None,
        "residual_df": int(df),
    }


def curves_summary(curves: list[tuple[np.ndarray, np.ndarray]], kinds: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for kind in kinds:
        errors: list[float] = []
        res: list[tuple[float, int]] = []
        used = 0
        for x, y in curves:
            e = loo(x, y, kind)
            if e is not None:
                errors += e
                used += 1
            r = residual(x, y, kind)
            if r is not None:
                res.append(r)
        out[kind] = {**summarise(errors, res), "curves_used": used, "curves": len(curves)}
    return out


# --- sources ---------------------------------------------------------------------------------


def delphi_sweeps() -> dict[float, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[float, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for table in DELPHI.values():
        d = pd.read_parquet(REPO / "data" / "raw" / "delphi" / table / "train.parquet")
        iso = d[d["kind"] == "isoflop"]
        for c in DELPHI_BUDGETS:
            s = iso[np.isclose(iso["gflops_bucket_flops"], c)]
            out[c].append((np.log(s["params"].to_numpy(float)), s["loss"].to_numpy(float)))
    return out


def coordinate(param: str, values: np.ndarray) -> np.ndarray | None:
    if param in ("beta1", "beta2"):
        return np.log(1 - values) if np.all(values < 1) else None
    if np.all(values > 0):
        return np.log(values)
    return values.astype(float)


def fo_slices() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in sorted((REPO / "data" / "raw" / "fantastic_optimizers").glob("*/*/*/result.json")):
        size = path.parts[-3]
        d = json.loads(path.read_text())
        best, base = d.get("best_config") or {}, d.get("baseline") or {}
        if base.get("loss") is None:
            continue
        by: dict[str, list[tuple[Any, float]]] = defaultdict(list)
        for ab in d.get("ablations") or []:
            if ab.get("loss") is not None:
                by[ab["param"]].append((ab["value"], float(ab["loss"])))
        for param, rows in by.items():
            if param not in best:
                continue
            pts = [(best[param], float(base["loss"]))] + rows
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pts):
                continue
            v = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts])
            keep = y <= y.min() * (1 + NEAR_OPTIMUM)
            x = coordinate(param, v[keep])
            if x is None or keep.sum() < 3:
                continue
            out[size].append((x, y[keep]))
    return out


def datadecide() -> dict[str, Any]:
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"].isin(DD_SIZES)]
    curves = []
    pos: list[np.ndarray] = []
    for (_, _), g in frame.groupby(["intervention", "seed"]):
        g = g.set_index("scale_label").loc[list(DD_SIZES)]
        curves.append((np.log(g["params_n"].to_numpy(float)), g["bpb"].to_numpy(float)))
        pos.append(np.arange(len(DD_SIZES)))
    cells = frame.groupby(["intervention", "scale_label"])["bpb"]
    var_rel = (cells.var(ddof=1) / cells.mean() ** 2).unstack()[list(DD_SIZES)]
    sd_by_size = {s: float(np.sqrt(var_rel[s].mean())) for s in DD_SIZES}
    sd_all = float(np.sqrt(var_rel.to_numpy().mean()))
    out: dict[str, Any] = {"curves": len(curves), "seed_sd_rel_by_size": sd_by_size, "seed_sd_rel_all": sd_all, "flex": {}}
    for kind in FLEX:
        errs = [loo(x, y, kind) for x, y in curves]
        by_size: dict[str, list[float]] = defaultdict(list)
        flat = []
        for e in errs:
            if e is None:
                continue
            flat += e
            for i, v in enumerate(e):
                by_size[DD_SIZES[i]].append(v)
        rms = float(np.sqrt(np.mean(np.square(flat)))) if flat else None
        out["flex"][kind] = {
            "loo_rms": rms,
            "loo_n": len(flat),
            "ratio_all": rms / sd_all if rms else None,
            "ratio_by_size": {
                s: float(np.sqrt(np.mean(np.square(by_size[s])))) / sd_by_size[s]
                for s in ("300M", "530M", "1B")
                if by_size[s]
            },
        }
    return out


def cross_hardware() -> dict[str, Any]:
    runs = {r["run_name"]: r for r in json.loads((rig.RAW / "runs.json").read_text())}
    pairs = []
    for size in ("75m", "150m", "300m"):
        g, t = runs.get(f"parallel_llama/old_scaling/gpu/{size}"), runs.get(f"parallel_llama/old_scaling/tpu/{size}")
        if g and t:
            a, b = g["eval_paloma_c4_en_bpb"], t["eval_paloma_c4_en_bpb"]
            pairs.append({"size": size, "gpu": a, "tpu": b, "same_model_flops": g["model_flops"] == t["model_flops"],
                          "relative_sd": abs(a - b) / (np.sqrt(2) * (a + b) / 2)})  # fmt: skip
    pooled = float(np.sqrt(np.mean([p["relative_sd"] ** 2 for p in pairs]))) if pairs else None
    return {"pairs": pairs, "pooled_relative_sd": pooled, "df": len(pairs),
            "caveat": "same model size, FLOPs and description; configuration identity unverified (hardware may also "
            "change precision or kernels)"}  # fmt: skip


def bound_for(summary: dict[str, Any], ratios: dict[str, float | None]) -> dict[str, Any]:
    usable = {k: v for k, v in summary.items() if v["loo_rms"] is not None and v["loo_n"] >= 3}
    best = min(usable, key=lambda k: usable[k]["loo_rms"])
    rms, n = usable[best]["loo_rms"], usable[best]["loo_n"]
    upper = rms * np.sqrt(n / stats.chi2.ppf(0.05, n))
    ratio = ratios.get(best)
    if ratio is not None and ratio < 1:
        upper /= ratio
    return {"flexibility": best, "loo_rms": rms, "loo_n": n, "upper_95": float(upper),
            "calibrated_estimate": rms / ratio if ratio else None, "datadecide_ratio": ratio}  # fmt: skip


# --- (e) reruns -------------------------------------------------------------------------------


def both_rung_two_sigma(members: list[str], cells: dict, pairs: set, s_fit: float, s_tgt: float) -> dict[str, Any]:
    fit = {s: cells[(s, "520m")]["y"] for s in members}
    tgt = {s: cells[(s, "1_2b")]["y"] for s in members}
    fs = determined(fit, al.masked_se(fit, s_fit, pairs))
    ts = determined(tgt, al.masked_se(tgt, s_tgt, pairs))
    b = both_rung(fs, ts)
    target_det = [k for k in ts if ts[k] != 0]
    target_reorder = [k for k in target_det if np.sign(fit[k[0]] - fit[k[1]]) != ts[k]]
    return {"pairs": len(pairs), "determined_at_both": b["determined_at_both"], "genuine": b["genuine_reorderings"],
            "genuine_rate": b["rate_among_both_determined"], "target_determined": len(target_det),
            "target_determined_reordered": len(target_reorder),
            "target_rate": len(target_reorder) / len(target_det) if target_det else None}  # fmt: skip


def sign_null(observed: np.ndarray, noiseless: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    sims = noiseless[None] * (1 + sigma[None, None, :] * rng.standard_normal((N_NULL, *noiseless.shape)))
    obs = st.changes_with_tolerance(st.pair_gaps(observed), st.TAU_PRIMARY)
    null = st.null_row(st.changes_with_tolerance(st.pair_gaps(sims), st.TAU_PRIMARY))
    return {"observed": st.row(obs), "null": null,
            "inside_or_below": bool(obs.mean() <= null["mean_95"][1])}  # fmt: skip


def main() -> None:
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_NOISE_BRACKET.md"}
    sweeps = delphi_sweeps()
    out["a_delphi"] = {f"{c:g}": curves_summary(v, ("quadratic", "cubic", "spline")) for c, v in sweeps.items()}
    slices = fo_slices()
    out["b_fantastic_optimizers"] = {s: curves_summary(v, ("linear", "quadratic", "cubic")) for s, v in slices.items()}
    out["c_datadecide"] = datadecide()
    ratios = {k: v["ratio_all"] for k, v in out["c_datadecide"]["flex"].items()}
    out["cross_hardware"] = cross_hardware()

    bounds: dict[str, Any] = {}
    for size, c in SIZE_TO_DELPHI.items():
        src = {"delphi": bound_for(out["a_delphi"][f"{c:g}"], ratios),
               "fantastic_optimizers": bound_for(out["b_fantastic_optimizers"][FO_SIZE[size]], ratios)}  # fmt: skip
        best = min(src, key=lambda k: src[k]["upper_95"])
        calibrated = min(v["calibrated_estimate"] for v in src.values() if v["calibrated_estimate"])
        bounds[size] = {"sources": src, "bound": src[best]["upper_95"], "from": best, "calibrated_estimate_min": calibrated}
    out["d_bounds"] = {
        "by_size": bounds,
        "caveat": "speedrun architectures may be noisier than Delphi's or Fantastic Optimizers'; the bound is the "
        "tighter of two leave-one-out upper limits, each including lack of fit",
    }

    members, cells, _ = rig.load_cells()
    groups = al.flops_groups(members, cells)
    group_of = {s: g for g, v in groups.items() for s in v}
    all_pairs = set(combinations(sorted(members), 2))
    within = {p for p in all_pairs if group_of[p[0]] == group_of[p[1]]}
    lq = set(combinations(sorted(groups["llama_qwen"]), 2))
    near_dup = within - lq
    xs = al.common_x(members, cells)
    design_cells = {"named_rungs": cells, "C2": al.aligned_cells(members, cells, xs, al.loglog_fit, rig.BUDGETS),
                    "C3": al.aligned_cells(members, cells, xs, al.three_fit, rig.BUDGETS)}  # fmt: skip
    sig = {b: bounds[b]["bound"] for b in rig.BUDGETS}
    levels = {
        "bound": (sig["520m"], sig["1_2b"]),
        "datadecide_1b": (0.000511266156200176,) * 2,
        "bound_0_43pct": (al.BOUND_THREE,) * 2,
    }
    e: dict[str, Any] = {}
    for lvl, (sf, stg) in levels.items():
        e[lvl] = {
            "named_rungs": both_rung_two_sigma(members, cells, all_pairs, sf, stg),
            "W": both_rung_two_sigma(members, cells, within, sf, stg),
            "W_LQ": both_rung_two_sigma(members, cells, lq, sf, stg),
            "near_duplicate_within_groups": both_rung_two_sigma(members, cells, near_dup, sf, stg),
            "C2": both_rung_two_sigma(members, design_cells["C2"], all_pairs, sf, stg),
            "C3": both_rung_two_sigma(members, design_cells["C3"], all_pairs, sf, stg),
        }
    sigma_vec = np.array([sig[b] for b in rig.BUDGETS])
    nulls = {}
    for name in ("named_rungs", "C2"):
        dc = design_cells[name]
        obs = np.array([[dc[(s, b)]["y"] for b in rig.BUDGETS] for s in members])
        x = np.array([[dc[(s, b)]["x"] for b in rig.BUDGETS] for s in members])
        noiseless = np.array([al.loglog_fit(np.array([cells[(s, b)]["x"] for b in rig.BUDGETS]),
                                            np.array([cells[(s, b)]["y"] for b in rig.BUDGETS]))(x[i])
                              for i, s in enumerate(members)])  # fmt: skip
        nulls[name] = sign_null(obs, noiseless, sigma_vec)
    out["e_reruns"] = {"sigma_by_rung": sig, "both_rung": e, "sign_change_null_primary_tau": nulls,
                       "near_duplicate_pairs": len(near_dup)}  # fmt: skip

    bvals = [bounds[s]["bound"] for s in bounds]
    nd = e["bound"]["near_duplicate_within_groups"]
    out["predictions"] = {
        "PB1_bound_in_0_10_to_0_50_pct": all(0.001 <= v <= 0.005 for v in bvals),
        "PB2_ratio_in_2_to_10": 2 <= (ratios.get(bounds["1_2b"]["sources"]["delphi"]["flexibility"]) or 0) <= 10,
        "PB2_calibrated_in_0_02_to_0_20_pct": all(0.0002 <= bounds[s]["calibrated_estimate_min"] <= 0.002 for s in bounds),
        "PB3_cross_hardware_in_0_02_to_0_15_pct": out["cross_hardware"]["pooled_relative_sd"] is not None
        and 0.0002 <= out["cross_hardware"]["pooled_relative_sd"] <= 0.0015,
        "PB4_near_duplicate_genuine_at_most_3": nd["genuine"] <= 3,
        "PB4_llama_qwen_genuine_at_most_1": e["bound"]["W_LQ"]["genuine"] <= 1,
        "PB4_most_near_duplicates_not_both_determined": nd["determined_at_both"] < nd["pairs"] / 2,
        "PB5_sign_changes_inside_or_below_null": all(v["inside_or_below"] for v in nulls.values()),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
