"""PREDICTIONS_TASK_ALIGNED.md: the Marin analyses at aligned compute, two-form uncertainty, the leaderboard fix.

Writes ``results/external/marin_aligned.json``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_leaderboard_core as core  # noqa: E402
from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402
from scripts.run_marin_noise import B0_FACTORS, C0_FRACTIONS, DD_LADDER, N_NULL, SEED, both_rung, determined  # noqa: E402

OUT = REPO / "results" / "external" / "marin_aligned.json"
PATCH = REPO / "scripts" / "patches" / "regression_ts_fix.patch"
Z = 1.959963984540054
BOUND_THREE = 0.004283575224018509
FIT_LABELS = ("130m", "300m", "520m")
TARGET = "1_2b"
TAUS = (1e-9, st.TAU_PRIMARY)


# --- fits ------------------------------------------------------------------------------------


def loglog_fit(x: np.ndarray, y: np.ndarray) -> Any:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return lambda z: np.exp(intercept + slope * np.log(z))


def three_fit(x: np.ndarray, y: np.ndarray) -> Any:
    """Global least-squares optimum of a x^-b + c (leaderboard bounds), log a, tight tolerances, 18 starts."""

    lx = np.log(x)
    lower, upper = [np.log(1e-10), 1e-6, 0.0], [np.log(1e30), 2.0, 0.999 * y.min()]
    best = None
    for f in C0_FRACTIONS:
        for k in B0_FACTORS:
            a0, b0, c0 = core.start(x, y, f, k)
            q = np.clip([np.log(a0), b0, c0], lower, upper)
            res = optimize.least_squares(
                lambda r: np.exp(r[0] - r[1] * lx) + r[2] - y,
                q,
                bounds=(lower, upper),
                max_nfev=100_000,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15,
            )
            sse = float(np.sum(res.fun**2))
            if best is None or sse < best[0]:
                best = (sse, res.x)
    assert best is not None
    la, b, c = best[1]
    return lambda z: np.exp(la - b * np.log(z)) + c


FITS = {"C2": loglog_fit, "C3": three_fit}


# --- designs ---------------------------------------------------------------------------------


def flops_groups(members: list[str], cells: dict) -> dict[str, list[str]]:
    """Submissions sharing an identical FLOPs ladder, named by a known member."""

    sig: dict[tuple, list[str]] = {}
    for s in members:
        sig.setdefault(tuple(cells[(s, b)]["x"] for b in rig.BUDGETS), []).append(s)
    names = {
        "adamw_llama_scaling": "llama_qwen",
        "hackable_transformer_attn_gate/lr_x1": "attn_gate",
        "parallel_llama": "attn_sink_parallel",
    }
    out = {}
    for v in sig.values():
        (label,) = [names[k] for k in names if k in v]
        out[label] = sorted(v)
    return out


def common_x(members: list[str], cells: dict, clip: bool = True) -> dict[str, float]:
    lo = max(min(cells[(s, b)]["x"] for b in rig.BUDGETS) for s in members)
    hi = min(max(cells[(s, b)]["x"] for b in rig.BUDGETS) for s in members)
    med = {b: float(np.median([cells[(s, b)]["x"] for s in members])) for b in rig.BUDGETS}
    return {b: float(np.clip(v, lo, hi)) if clip else v for b, v in med.items()}


def aligned_cells(members: list[str], cells: dict, xs: dict[str, float], fit: Any, labels: tuple[str, ...]) -> dict:
    """y * f(x*) / f(x), f fitted on ``labels`` of the submission's own cells; x replaced by x*."""

    out = {}
    for s in members:
        x = np.array([cells[(s, b)]["x"] for b in labels])
        y = np.array([cells[(s, b)]["y"] for b in labels])
        f = fit(x, y)
        for b in labels:
            c = cells[(s, b)]
            out[(s, b)] = {"x": xs[b], "y": float(c["y"] * f(xs[b]) / f(c["x"])), "train_flops": c["train_flops"]}
    return out


# --- analyses --------------------------------------------------------------------------------


def masked_se(values: dict[str, float], cv: float, pairs: set) -> dict[tuple[str, str], float]:
    return {
        (a, b): cv * float(np.hypot(values[a], values[b])) if (a, b) in pairs else np.inf
        for a, b in combinations(sorted(values), 2)
    }


def reorder_rates(members: list[str], cells: dict, pairs: set, cvs: dict[str, float]) -> dict[str, Any]:
    fit = {s: cells[(s, "520m")]["y"] for s in members}
    tgt = {s: cells[(s, TARGET)]["y"] for s in members}
    out = {}
    for name, cv in cvs.items():
        r = ref.reorder(fit, tgt, masked_se(tgt, cv, pairs))
        fs, ts = determined(fit, masked_se(fit, cv, pairs)), determined(tgt, masked_se(tgt, cv, pairs))
        b = both_rung(fs, ts)
        out[name] = {
            "pairs": len(pairs),
            "determined": r["determined"],
            "reordered": r["reordered"],
            "rate": r["rate"] if r["determined"] else None,
            "both_rung_determined": b["determined_at_both"],
            "both_rung_genuine": b["genuine_reorderings"],
            "both_rung_rate": b["rate_among_both_determined"],
        }
    return out


def heldout(members: list[str], fit_cells: dict, truth_cells: dict, pairs: set, groups: dict | None) -> dict[str, Any]:
    proj = rig.projections(
        members, {**fit_cells, **{k: v for k, v in truth_cells.items() if k[1] == TARGET}}, FIT_LABELS, TARGET
    )
    truth = rig.gaps({s: truth_cells[(s, TARGET)]["y"] for s in members})
    single = rig.gaps({s: fit_cells[(s, "520m")]["y"] for s in members})
    keys = [k for k in truth if k in pairs]
    out: dict[str, Any] = {"pairs": len(keys), "single_wrong": rig.wrong(single, truth, keys), "forms": {}}
    for form, values in proj.items():
        pg = rig.gaps(values)
        kern = {
            k: 100.0 * (float(np.sign(pg[k]) != np.sign(truth[k])) - float(np.sign(single[k]) != np.sign(truth[k])))
            for k in keys
            if np.isfinite(pg[k])
        }
        row: dict[str, Any] = {
            "wrong": rig.wrong(pg, truth, keys),
            "failed": sorted(s for s, v in values.items() if not np.isfinite(v)),
        }
        if groups is None:
            row["u_test"] = rig.u_test(kern) if len(kern) == len(keys) else None
        else:
            per = {}
            for g, mem in groups.items():
                sub = {k: v for k, v in kern.items() if k[0] in mem and k[1] in mem}
                if len(mem) >= 4 and len(sub) == len(mem) * (len(mem) - 1) // 2:
                    per[g] = {**rig.u_test(sub), "pairs": len(sub)}
            w = {g: v["pairs"] for g, v in per.items()}
            num = sum(w[g] * per[g]["mean_pp"] for g in per)
            den = np.sqrt(sum((w[g] * per[g]["se_pp"]) ** 2 for g in per))
            row["u_test_by_group"] = per
            row["u_test_combined"] = {"mean_pp": num / sum(w.values()), "abs_z": float(abs(num) / den) if den > 0 else None}
        out["forms"][form] = row
    return out


def dd_means() -> np.ndarray:
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    return frame.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()[list(DD_LADDER)].to_numpy(float)


def sign_table(observed: np.ndarray, noiseless: np.ndarray, mask: np.ndarray, dd: np.ndarray) -> dict[str, Any]:
    g_obs, g_fit, g_dd = st.pair_gaps(observed)[mask], st.pair_gaps(noiseless)[mask], st.pair_gaps(dd)
    sims = {}
    for name, sigma in st.NULL_SIGMAS.items():
        rng = np.random.default_rng(SEED)
        sims[name] = st.pair_gaps(noiseless[None] * (1 + sigma * rng.standard_normal((N_NULL, *noiseless.shape))))[:, mask]
    out = {}
    for tau in TAUS:
        t: dict[str, Any] = {
            "observed": st.row(st.changes_with_tolerance(g_obs, tau)),
            "noiseless": st.row(st.changes_with_tolerance(g_fit, tau)),
            "datadecide_c4": st.row(st.changes_with_tolerance(g_dd, tau)),
        }
        for name, s in sims.items():
            t[f"null_{name}"] = st.null_row(st.changes_with_tolerance(s, tau))
        dd = t["datadecide_c4"]["mean"]
        t["observed_over_datadecide"] = t["observed"]["mean"] / dd if dd else None
        out[f"{tau:.6g}"] = t
    return out


def fisher_vs_dd(marin: dict[str, Any], dd: dict[str, Any]) -> dict[str, Any]:
    if not marin["determined"]:
        return {"survives": False, "reason": "no determined pairs"}
    table = [
        [marin["reordered"], marin["determined"] - marin["reordered"]],
        [dd["reordered"], dd["determined"] - dd["reordered"]],
    ]
    p = float(stats.fisher_exact(table, alternative="greater").pvalue)
    diff = marin["rate"] - dd["rate"]
    return {
        "marin_rate": marin["rate"],
        "dd_rate": dd["rate"],
        "difference_pp": 100 * diff,
        "fisher_p_one_sided": p,
        "survives": bool(diff >= 0.05 and p < 0.05),
    }


# --- Task 3 ----------------------------------------------------------------------------------


def patched_predictions(
    points: dict[str, list[dict[str, float]]], targets: tuple[float, ...] = (rig.FORECAST_X,)
) -> dict[str, list[float | None]]:
    """Apply the committed patch to a copy of their regression.ts and run the patched function at ``targets``."""

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(rig.NODE_DIR / "regression.ts", d / "regression.ts")
        shutil.copy(rig.NODE_DIR / "package.json", d / "package.json")
        (d / "node_modules").symlink_to(rig.NODE_DIR / "node_modules")
        subprocess.run(["patch", "-s", str(d / "regression.ts"), str(PATCH)], check=True)
        jobs = [{"id": s, "points": p, "targets": list(targets)} for s, p in points.items()]
        proc = subprocess.run(
            ["node", str(REPO / "scripts" / "marin_lm_options.mjs"), str(d)],
            input=json.dumps({"fits": jobs}),
            capture_output=True,
            text=True,
            check=True,
        )
    return {f["id"]: f["reference_predictions"] for f in json.loads(proc.stdout)["fits"]}


def patched_forecasts(points: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    return {k: v[0] for k, v in patched_predictions(points).items()}  # type: ignore[misc]


def main() -> None:
    members, cells, _ = rig.load_cells()
    cvs_full = rig.borrowed_cvs()
    cvs = {k: v["cv"] for k, v in cvs_full.items()}
    cvs_all = dict(cvs, empirical_three_parameter_0_43pct=BOUND_THREE)
    groups = flops_groups(members, cells)
    group_of = {s: g for g, v in groups.items() for s in v}
    all_pairs = set(combinations(sorted(members), 2))
    within = {p for p in all_pairs if group_of[p[0]] == group_of[p[1]]}
    lq = groups["llama_qwen"]
    lq_pairs = set(combinations(sorted(lq), 2))
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_ALIGNED.md", "groups": groups}

    # (a) offsets
    beta = {
        s: float(
            np.polyfit(
                np.log([cells[(s, b)]["x"] for b in rig.BUDGETS]), np.log([cells[(s, b)]["y"] for b in rig.BUDGETS]), 1
            )[0]
        )
        for s in members
    }
    offsets: dict[str, Any] = {}
    for b in rig.BUDGETS:
        rows = []
        for a, c in sorted(all_pairs - within):
            ya, xa, xc = cells[(a, b)]["y"], cells[(a, b)]["x"], cells[(c, b)]["x"]
            delta = ya * ((xc / xa) ** beta[a] - 1)
            gap = ya - cells[(c, b)]["y"]
            rows.append(
                (abs(delta), abs(delta) / abs(gap) if gap else np.inf, {k: abs(delta) / (v * ya) for k, v in cvs.items()})
            )
        dist = np.array([r[0] for r in rows])
        ratio = np.array([r[1] for r in rows])
        offsets[b] = {
            "cross_group_pairs": len(rows),
            "abs_delta_bpb_median_p90": [float(np.median(dist)), float(np.percentile(dist, 90))],
            "over_gap_median_p90": [float(np.median(ratio)), float(np.percentile(ratio, 90))],
            "share_delta_exceeds_gap": float(np.mean(ratio > 1)),
            "over_borrowed_sd_median_p90": {
                k: [float(np.median([r[2][k] for r in rows])), float(np.percentile([r[2][k] for r in rows], 90))]
                for k in cvs
            },
        }
    out["task1a"] = {
        "flops": {s: {b: cells[(s, b)]["x"] for b in rig.BUDGETS} for s in members},
        "loglog_slope": beta,
        "offsets_by_rung": offsets,
        "MA3": {
            "median_over_each_borrowed_sd_at_least_5": all(
                v[0] >= 5 for v in offsets[TARGET]["over_borrowed_sd_median_p90"].values()
            ),
            "median_over_gap_in_0_3_to_3": 0.3 <= offsets[TARGET]["over_gap_median_p90"][0] <= 3,
        },
    }

    # (b) designs
    xs = common_x(members, cells)
    xs_raw = common_x(members, cells, clip=False)
    designs: dict[str, dict[str, Any]] = {
        "named_rungs": {"members": members, "cells": cells, "fit_cells": cells, "pairs": all_pairs, "groups": None},
        "W": {
            "members": members,
            "cells": cells,
            "fit_cells": cells,
            "pairs": within,
            "groups": {g: v for g, v in groups.items()},
        },
        "W_LQ": {"members": lq, "cells": cells, "fit_cells": cells, "pairs": lq_pairs, "groups": None},
    }
    for name, fit in FITS.items():
        designs[name] = {
            "members": members,
            "cells": aligned_cells(members, cells, xs, fit, rig.BUDGETS),
            "fit_cells": aligned_cells(members, cells, xs, fit, FIT_LABELS),
            "pairs": all_pairs,
            "groups": None,
        }
    dd = dd_means()
    dd_class = ref.datadecide_c4()
    iu = np.triu_indices(len(members), 1)
    pair_index = [(members[i], members[j]) for i, j in zip(*iu)]
    results: dict[str, Any] = {}
    for name, d in designs.items():
        mem = d["members"]
        obs = np.array([[d["cells"][(s, b)]["y"] for b in rig.BUDGETS] for s in members])
        x_design = np.array([[d["cells"][(s, b)]["x"] for b in rig.BUDGETS] for s in members])
        noiseless = np.empty_like(obs)
        for i, s in enumerate(members):
            f = loglog_fit(
                np.array([cells[(s, b)]["x"] for b in rig.BUDGETS]), np.array([cells[(s, b)]["y"] for b in rig.BUDGETS])
            )
            noiseless[i] = f(x_design[i])
        mask = np.array([tuple(sorted(p)) in d["pairs"] for p in pair_index])
        rates = reorder_rates(mem, d["cells"], {p for p in d["pairs"] if p[0] in mem and p[1] in mem}, cvs_all)
        results[name] = {
            "pairs": len(d["pairs"]),
            "reorder": rates,
            "class_vs_datadecide": {
                k: fisher_vs_dd(rates["datadecide_c4_1B"], dd_class[k]) for k in ("300M->1B", "530M->1B")
            },
            "heldout": heldout(members, d["fit_cells"], d["cells"], d["pairs"], d["groups"]),
            "sign_changes": sign_table(obs, noiseless, mask, dd),
        }
    # C-drop sensitivity: unclipped median, out-of-range cells dropped (C2 alignment)
    keep = {
        s: {
            b
            for b in rig.BUDGETS
            if min(cells[(s, q)]["x"] for q in rig.BUDGETS) <= xs_raw[b] <= max(cells[(s, q)]["x"] for q in rig.BUDGETS)
        }
        for s in members
    }
    drop_cells = aligned_cells(members, cells, xs_raw, loglog_fit, rig.BUDGETS)
    complete = [s for s in members if {"520m", TARGET} <= keep[s]]
    results["C_drop"] = {
        "coverage": {b: sum(b in keep[s] for s in members) for b in rig.BUDGETS},
        "submissions_complete_at_520m_and_1_2b": len(complete),
        "reorder": reorder_rates(complete, drop_cells, set(combinations(sorted(complete), 2)), cvs_all),
    }
    named = results["named_rungs"]["reorder"]["datadecide_c4_1B"]["rate"]
    named_diff = results["named_rungs"]["class_vs_datadecide"]["300M->1B"]["difference_pp"]
    designs_ = ("W", "W_LQ", "C2", "C3")
    rate = {k: results[k]["reorder"]["datadecide_c4_1B"]["rate"] for k in designs_}
    out["task1"] = {
        "common_x": xs,
        "median_x_unclipped": xs_raw,
        "clip_shift_ratio": {b: xs[b] / xs_raw[b] for b in rig.BUDGETS},
        "designs": results,
        "datadecide_class": {k: {kk: vv for kk, vv in v.items() if kk != "raw_welch_0_05"} for k, v in dd_class.items()},
        "predictions": {
            "UA1_rate_at_most_half_named": {k: bool(rate[k] is not None and rate[k] <= named / 2) for k in designs_},
            "UA2_class_difference_shrinks": {
                k: bool(results[k]["class_vs_datadecide"]["300M->1B"].get("difference_pp", -np.inf) < named_diff)
                for k in designs_
            },
            "MA1_falls_less_than_a_third_and_above_10pct": {
                k: bool(rate[k] is not None and rate[k] > 0.10 and rate[k] > named * 2 / 3) for k in designs_
            },
            "MA2_class_difference_survives": {
                k: results[k]["class_vs_datadecide"]["300M->1B"]["survives"] for k in designs_
            },
            "MA4_projection_at_least_single_errors": {
                k: all(v["wrong"] >= results[k]["heldout"]["single_wrong"] for v in results[k]["heldout"]["forms"].values())
                for k in designs_
            },
            "MA5_sign_changes_5x_datadecide": {
                k: (results[k]["sign_changes"][f"{st.TAU_PRIMARY:.6g}"]["observed_over_datadecide"] or np.inf) >= 5
                for k in designs_
            },
            "MA5_two_or_more_at_most_5pct_within": {
                k: results[k]["sign_changes"][f"{st.TAU_PRIMARY:.6g}"]["observed"]["share_two_or_more"] <= 0.05
                for k in ("W", "W_LQ")
            },
        },
    }

    # Task 2
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rig.BUDGETS] for s in members}
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in members}
    optimum = st.global_optimum(members, raw)
    loglog = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in members}
    shared = rig.shared_exponent(raw, {s: rig.FORECAST_X for s in members})
    two = core.invariant_core({"three_parameter_global_optimum": optimum, "two_parameter_loglog": loglog}, members)
    three = core.invariant_core(
        {"three_parameter_global_optimum": optimum, "two_parameter_loglog": loglog, "shared_exponent": shared}, members
    )
    out["task2"] = {
        "reason": "the shared exponent imposes one scaling exponent on every submission, assuming away the differences "
        "in scaling an architecture leaderboard exists to detect; it tests offsets given a common slope, not each "
        "submission's own curve",
        "two_forms": two,
        "three_forms": three,
        "MF1": {
            "at_least_150": two["invariant_pairs"] >= 150,
            "k_in_1_to_3": 1 <= two["largest_invariant_top_k"] <= 3,
            "top1_muonh": two["distinct_top1"] == ["muonh_qwen3_scaling"],
        },
    }

    # Task 3
    displayed_res = ref.lm([{"id": s, "points": pts[s], "targets": [rig.FORECAST_X]} for s in members])
    displayed = {s: displayed_res[s]["reference_predictions"][0] for s in members}
    fixed = patched_forecasts(pts)
    variants = {
        "x_over_1e18_error_tolerance_1e-15": st.reach(
            st.lm_starts(pts, members, st.SCALE, error_tolerance=1e-15), optimum, members
        ),
        "x_over_1e18_y_times_100": st.reach(st.lm_starts(pts, members, st.SCALE, y_scale=100.0), optimum, members),
    }
    changed = [
        [a, b]
        for a, b in combinations(sorted(members), 2)
        if np.sign(displayed[a] - displayed[b]) != np.sign(fixed[a] - fixed[b])
    ]
    rigour = json.loads((REPO / "results" / "external" / "marin_rigour.json").read_text())["task_1"]
    qualified = rigour.get("qualified", {})
    out["task3"] = {
        "patch": str(PATCH.relative_to(REPO)),
        "regression_ts_sha256": rig.sha256(rig.NODE_DIR / "regression.ts"),
        "change": "computePowerLawWithOffset fits x / 1e18 (the returned intercept is converted back to raw-x units, "
        "a * 1e18^b, so predictWithPowerLaw and every caller are unchanged) and uses errorTolerance 1e-15 instead of "
        "1e-8. Equivalent: x / 1e18 and y * 100.",
        "verification": {
            "patched_function_default_start_at_optimum": int(
                sum(abs(fixed[s] - optimum[s]) <= st.SAME_BPB for s in members)
            ),
            "patched_max_abs_minus_optimum_bpb": max(abs(fixed[s] - optimum[s]) for s in members),
            "variants_13_starts_cap_200": {
                k: {kk: v[kk] for kk in ("fits", "fits_reaching_optimum", "max_departure_bpb")} for k, v in variants.items()
            },
        },
        "qualified_all": all(qualified.values()) if qualified else None,
        "displayed_vs_fixed": {
            "changed_pairs": changed,
            "count": len(changed),
            "kendall": ref.kendall(displayed, fixed),
            "top3_displayed": sorted(members, key=lambda s: displayed[s])[:3],
            "top3_fixed": sorted(members, key=lambda s: fixed[s])[:3],
            "ranking_displayed": sorted(members, key=lambda s: displayed[s]),
            "ranking_fixed": sorted(members, key=lambda s: fixed[s]),
        },
    }
    dv = out["task3"]["displayed_vs_fixed"]
    out["task3"]["MX1"] = {
        "eleven_changed": dv["count"] == 11,
        "top1_unchanged": dv["top3_displayed"][0] == dv["top3_fixed"][0],
        "adamh_replaced_by_default_x4": "adamh_llama_scaling" in dv["top3_displayed"]
        and "hackable_transformer_attn_sink/attnsink/splash/default_x4" in dv["top3_fixed"]
        and "adamh_llama_scaling" not in dv["top3_fixed"],
    }
    out["task1"]["predictions_note"] = (
        "MA4 covers the aligned designs W, W_LQ, C2 and C3 only. At named rungs the shared exponent has fewer "
        "point errors than single-scale (43 against 46 on the 21; 48 against 52 on the official 22), and no "
        "projection form beats single-scale significantly."
    )
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
