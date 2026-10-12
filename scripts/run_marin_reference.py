"""PREDICTIONS_TASK_REFERENCE.md: class-dependent crossover, the leaderboard's convergence defect, family inference.

Task 1 compares single-scale's determined-pair reorder rate between Marin's
speedrun ladder (520m -> 1.2b, Paloma C4-EN BPB, single runs, borrowed noise) and
DataDecide at matched parameter ratios (C4 bits per token with 3 seeds; Signal
and Noise bits per byte, single runs). Task 2 measures how the leaderboard's
unconverged fit changes its forecasts and the held-out comparison. Task 3 tests
at the family level. Task 4 records the lever-arm mapping and the C4-calibrated
values applied to Marin.

Writes ``results/external/marin_reference.json``.
"""

from __future__ import annotations

import json
import subprocess
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

from scripts import run_marin_rigour as rig  # noqa: E402

OUT = REPO / "results" / "external" / "marin_reference.json"
NODE_DIR = REPO / "data" / "cache" / "marin_node"
Z = 1.959963984540054
MULTIPLES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
GAP_BINS = (1.96, 4.0, 8.0, 16.0, np.inf)
CAPS = (50, 100, 200, 500, 1000, 100_000)
INITS = {
    "default": (None, None),
    "c0.5_b0.05": (0.5, 0.05),
    "c0.5_b0.3": (0.5, 0.3),
    "c0.9_b0.05": (0.9, 0.05),
    "c0.9_b0.3": (0.9, 0.3),
}
DD_DESIGNS = {"300M->1B": ("300M", "1B"), "530M->1B": ("530M", "1B")}
TOP4_DD = ("150M", "300M", "530M", "1B")


# --- Task 1 ---------------------------------------------------------------------------------


def reorder(
    fit: dict[str, float], target: dict[str, float], se: dict[tuple[str, str], float], m: float = 1.0
) -> dict[str, Any]:
    """Determined-pair reorder rate, and the same rate by gap bin (gap over SE)."""

    det, wrong, bins = 0, 0, defaultdict(lambda: [0, 0])
    pairs_out = {}
    for a, b in combinations(sorted(target), 2):
        g, s = target[a] - target[b], m * se[(a, b)]
        f = fit[a] - fit[b]
        if g == 0 or f == 0 or not abs(g) > Z * s:
            continue
        det += 1
        bad = np.sign(f) != np.sign(g)
        wrong += bad
        ratio = abs(g) / s
        for lo, hi in zip(GAP_BINS, GAP_BINS[1:]):
            if lo <= ratio < hi:
                bins[f"[{lo:g},{hi:g})"][0] += 1
                bins[f"[{lo:g},{hi:g})"][1] += int(bad)
        pairs_out[(a, b)] = bool(bad)
    return {
        "determined": det,
        "reordered": int(wrong),
        "rate": wrong / det if det else float("nan"),
        "by_gap_bin": {
            k: {"determined": v[0], "reordered": v[1], "rate": v[1] / v[0] if v[0] else None} for k, v in bins.items()
        },
        "_pairs": pairs_out,
    }


def marin_suite() -> dict[str, Any]:
    members, cells, _ = rig.load_cells()
    cvs = rig.borrowed_cvs()
    fit = {s: cells[(s, "520m")]["y"] for s in members}
    target = {s: cells[(s, "1_2b")]["y"] for s in members}
    out = {"n": len(members), "parameter_ratio": "1.2b / 520m", "by_borrowed": {}}
    for name, entry in cvs.items():
        se = {(a, b): entry["cv"] * float(np.hypot(target[a], target[b])) for a, b in combinations(sorted(target), 2)}
        base = reorder(fit, target, se)
        sweep = {f"{m:g}": reorder(fit, target, se, m) for m in MULTIPLES}
        out["by_borrowed"][name] = {
            "cv": entry["cv"],
            "rate": {k: v for k, v in base.items() if k != "_pairs"},
            "noise_multiples": {k: {"determined": v["determined"], "rate": v["rate"]} for k, v in sweep.items()},
            "_pairs": base["_pairs"],
        }
    out["_cells"], out["_members"] = cells, members
    return out


def datadecide_c4() -> dict[str, Any]:
    from asla.analysis.moderation import evidence_from_cells, fit_prior, moderate

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"]
    out = {}
    for name, (f, t) in DD_DESIGNS.items():
        fit = frame[frame["scale_label"] == f].groupby("intervention")["bpb"].mean().to_dict()
        tcells = frame[frame["scale_label"] == t].groupby("intervention")["bpb"]
        target = tcells.mean().to_dict()
        s2 = tcells.var(ddof=1).to_dict()
        n = int(tcells.count().min())
        prior = fit_prior(list(s2.values()), n - 1)
        mod = {r: float(moderate(v, prior)) for r, v in s2.items()}
        se = {(a, b): float(np.sqrt(mod[a] / n + mod[b] / n)) for a, b in combinations(sorted(target), 2)}
        res = reorder(fit, target, se)
        welch = evidence_from_cells({str(r): g.tolist() for r, g in tcells}, alpha=0.05, multiplicity="none")
        welch_det = [(e.left, e.right) for e in welch if e.determined]
        welch_wrong = sum(np.sign(fit[a] - fit[b]) != np.sign(target[a] - target[b]) for a, b in welch_det)
        out[name] = {
            **{k: v for k, v in res.items() if k != "_pairs"},
            "raw_welch_0_05": {
                "determined": len(welch_det),
                "reordered": int(welch_wrong),
                "rate": welch_wrong / len(welch_det),
            },
        }
    return out


def signal_and_noise() -> dict[str, Any]:
    tw = __import__("scripts.transfer_worlds", fromlist=["context"])
    ctx = tw.context("sn")
    table = ctx["frame"].groupby(["intervention", "scale_label"])["bpb"].mean().unstack()
    out = {}
    for name, (f, t) in DD_DESIGNS.items():
        fit, target = table[f].to_dict(), table[t].to_dict()
        sig = ctx["sigma_table"][t]
        se = {(a, b): float(np.hypot(sig[a], sig[b])) for a, b in combinations(sorted(target), 2)}
        res = reorder(fit, target, se)
        out[name] = {k: v for k, v in res.items() if k != "_pairs"}
    return out


def slopes_loglog(points: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    return {s: float(np.polyfit(np.log([p[0] for p in v]), np.log([p[1] for p in v]), 1)[0]) for s, v in points.items()}


def mechanism(marin: dict[str, Any]) -> dict[str, Any]:
    from asla.models import fit_power_law

    cells, members = marin["_cells"], marin["_members"]
    m_points = {s: [(cells[(s, b)]["x"], cells[(s, b)]["y"]) for b in rig.BUDGETS] for s in members}
    m_slope = slopes_loglog(m_points)
    m_level = {s: float(np.log(cells[(s, "520m")]["y"])) for s in members}
    m_alpha = {}
    for s, pts in m_points.items():
        try:
            m_alpha[s] = float(fit_power_law(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))[2])
        except Exception:  # noqa: BLE001
            m_alpha[s] = float("nan")
    m_ratio = float(np.median([cells[(s, "1_2b")]["x"] / cells[(s, "520m")]["x"] for s in members]))

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"]
    means = frame.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()
    compute = frame.groupby("scale_label")["compute"].first()
    d_points = {r: [(float(compute[s]), float(means.loc[r, s])) for s in TOP4_DD] for r in means.index}
    d_slope = slopes_loglog(d_points)
    d_level = {r: float(np.log(means.loc[r, "530M"])) for r in means.index}
    full = [s for s in compute.sort_values().index]
    d_alpha = {
        r: float(fit_power_law(compute[full].to_numpy(float), means.loc[r, full].to_numpy(float))[2]) for r in means.index
    }
    d_ratio = float(compute["1B"] / compute["530M"])

    def index(slope: dict[str, float], level: dict[str, float], ratio: float) -> dict[str, float]:
        sd_s, sd_l = float(np.std(list(slope.values()), ddof=1)), float(np.std(list(level.values()), ddof=1))
        return {
            "sd_slope": sd_s,
            "sd_level": sd_l,
            "slope_over_level": sd_s / sd_l,
            "crossover_index": sd_s * np.log(ratio) / sd_l,
        }

    reordered = {k: v for k, v in marin["by_borrowed"]["datadecide_c4_1B"]["_pairs"].items()}
    dslope_bad = [abs(m_slope[a] - m_slope[b]) for (a, b), bad in reordered.items() if bad]
    dslope_ok = [abs(m_slope[a] - m_slope[b]) for (a, b), bad in reordered.items() if not bad]
    mw = stats.mannwhitneyu(dslope_bad, dslope_ok, alternative="greater") if dslope_bad and dslope_ok else None
    return {
        "marin": {**index(m_slope, m_level, m_ratio), "compute_ratio_target_over_fit": m_ratio,
                  "alpha_three_param_sd": float(np.nanstd(list(m_alpha.values()), ddof=1)),
                  "alpha_three_param_failed": int(np.isnan(list(m_alpha.values())).sum())},
        "datadecide_c4": {**index(d_slope, d_level, d_ratio), "compute_ratio_target_over_fit": d_ratio,
                          "alpha_three_param_sd": float(np.std(list(d_alpha.values()), ddof=1))},
        "marin_reordered_vs_not_abs_delta_slope": {
            "borrowed_sd": "datadecide_c4_1B",
            "median_reordered": float(np.median(dslope_bad)) if dslope_bad else None,
            "median_not_reordered": float(np.median(dslope_ok)) if dslope_ok else None,
            "n_reordered": len(dslope_bad),
            "n_not_reordered": len(dslope_ok),
            "mann_whitney_p_one_sided": float(mw.pvalue) if mw else None,
        },
    }  # fmt: skip


def structure(marin: dict[str, Any]) -> dict[str, Any]:
    members = marin["_members"]
    family = {s: s.split("/")[0] if "/" in s else s for s in members}
    out = {}
    for name, block in marin["by_borrowed"].items():
        within = [bad for (a, b), bad in block["_pairs"].items() if family[a] == family[b]]
        cross = [bad for (a, b), bad in block["_pairs"].items() if family[a] != family[b]]
        out[name] = {
            "within": {
                "determined": len(within),
                "reordered": int(sum(within)),
                "rate": float(np.mean(within)) if within else None,
            },
            "cross": {
                "determined": len(cross),
                "reordered": int(sum(cross)),
                "rate": float(np.mean(cross)) if cross else None,
            },
        }
    return out


# --- Task 2 ---------------------------------------------------------------------------------


def lm(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    proc = subprocess.run(
        ["node", str(REPO / "scripts" / "marin_lm_options.mjs"), str(NODE_DIR)],
        input=json.dumps({"fits": jobs}),
        capture_output=True,
        text=True,
        check=True,
    )
    return {f["id"]: f for f in json.loads(proc.stdout)["fits"]}


def kendall(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(a)
    return float(stats.kendalltau([a[k] for k in keys], [b[k] for k in keys]).statistic)


def convergence(members: list[str], cells: dict) -> dict[str, Any]:
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rig.BUDGETS] for s in members}
    forecasts: dict[str, dict[str, float]] = {}
    exact_match = True
    for init, (c_mult, b0) in INITS.items():
        for cap in CAPS:
            jobs = []
            for s in members:
                y_min = min(p["y"] for p in pts[s])
                jobs.append({"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "maxIterations": cap,
                             "cInit": c_mult * y_min if c_mult else None, "bInit": b0})  # fmt: skip
            res = lm(jobs)
            forecasts[f"{init}|{cap}"] = {s: res[s]["predictions"][0] for s in members}
            if init == "default" and cap == 200:
                exact_match = all(res[s]["predictions"][0] == res[s]["reference_predictions"][0] for s in members)
    as_impl, conv = forecasts["default|200"], forecasts["default|100000"]
    python = {s: rig.python_marin([(p["x"], p["y"]) for p in pts[s]], rig.FORECAST_X) for s in members}
    order_impl = sorted(members, key=lambda s: as_impl[s])
    order_conv = sorted(members, key=lambda s: conv[s])
    changes = sum(np.sign(as_impl[a] - as_impl[b]) != np.sign(conv[a] - conv[b]) for a, b in combinations(members, 2))
    shift_over_gap = {}
    for i, s in enumerate(order_impl):
        neighbours = [abs(as_impl[s] - as_impl[o]) for o in order_impl[max(0, i - 1) : i + 2] if o != s]
        shift_over_gap[s] = abs(conv[s] - as_impl[s]) / min(neighbours) if min(neighbours) > 0 else float("inf")
    rankings = {k: sorted(members, key=lambda s: v[s]) for k, v in forecasts.items()}
    optima = optimum_check(members, pts)
    return {
        "reimplementation_matches_their_code_exactly_at_defaults": exact_match,
        "as_implemented_vs_converged": {
            "kendall_tau": kendall(as_impl, conv),
            "pairwise_order_changes": int(changes),
            "pairs": len(members) * (len(members) - 1) // 2,
            "top1_changes": order_impl[0] != order_conv[0],
            "top3_set_changes": set(order_impl[:3]) != set(order_conv[:3]),
            "top3_as_implemented": order_impl[:3],
            "top3_converged": order_conv[:3],
            "median_shift_over_adjacent_gap": float(np.median(list(shift_over_gap.values()))),
            "max_shift_over_adjacent_gap": float(max(shift_over_gap.values())),
            "converged_vs_python_least_squares_max_abs": float(max(abs(conv[s] - python[s]) for s in members)),
        },
        "optimum_check": optima,
        "as_implemented_vs_least_squares_optimum": {
            "kendall_tau": kendall(as_impl, python),
            "pairwise_order_changes": int(
                sum(np.sign(as_impl[a] - as_impl[b]) != np.sign(python[a] - python[b]) for a, b in combinations(members, 2))
            ),
            "top1_changes": order_impl[0] != sorted(members, key=lambda s: python[s])[0],
            "top3_set_changes": set(order_impl[:3]) != set(sorted(members, key=lambda s: python[s])[:3]),
            "top3_least_squares_optimum": sorted(members, key=lambda s: python[s])[:3],
            "median_shift_over_adjacent_gap": float(np.median(_shift_over_gap(order_impl, as_impl, python))),
        },
        "kendall_tau_vs_converged": {k: kendall(v, conv) for k, v in forecasts.items()},
        "top1_by_setting": {k: v[0] for k, v in rankings.items()},
        "top3_by_setting": {k: v[:3] for k, v in rankings.items()},
    }


def optimum_check(members: list[str], pts: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Which starting points reach the least-squares optimum when run to 100,000 iterations."""

    def sse(params: dict[str, float] | None, s: str) -> float:
        if not params:
            return float("inf")
        return float(
            sum((params["intercept"] * q["x"] ** params["slope"] + params["asymptote"] - q["y"]) ** 2 for q in pts[s])
        )

    by_init = {}
    for init, (c_mult, b0) in INITS.items():
        jobs = [
            {"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "maxIterations": CAPS[-1],
             "cInit": c_mult * min(q["y"] for q in pts[s]) if c_mult else None, "bInit": b0}
            for s in members
        ]  # fmt: skip
        res = lm(jobs)
        by_init[init] = {s: sse(res[s]["params"], s) for s in members}
    python = {s: rig.converged_sse([(q["x"], q["y"]) for q in pts[s]]) for s in members}
    best = {s: min([python[s], *(v[s] for v in by_init.values())]) for s in members}
    return {
        "at_best_sse_within_1pct": {
            **{k: int(sum(v[s] <= 1.01 * best[s] for s in members)) for k, v in by_init.items()},
            "python_least_squares": int(sum(python[s] <= 1.01 * best[s] for s in members)),
        },
        "median_sse_over_best": {k: float(np.median([v[s] / best[s] for s in members])) for k, v in by_init.items()},
        "reading": "starts with c0 = 0.9 min y get stuck far above the optimum; the leaderboard's LM even at 100,000 "
        "iterations reaches the least-squares optimum (within 1%) in only some submissions, while a Python "
        "least-squares fit reaches it in all; the optimum is the reference in as_implemented_vs_least_squares_optimum",
    }


def _shift_over_gap(order: list[str], before: dict[str, float], after: dict[str, float]) -> list[float]:
    out = []
    for i, s in enumerate(order):
        neighbours = [abs(before[s] - before[o]) for o in order[max(0, i - 1) : i + 2] if o != s]
        nearest = min(neighbours)
        out.append(abs(after[s] - before[s]) / nearest if nearest > 0 else float("inf"))
    return out


def heldout_by_cap(members: list[str], cells: dict) -> dict[str, Any]:
    truth = rig.gaps({s: cells[(s, "1_2b")]["y"] for s in members})
    out = {}
    for cap in CAPS:
        jobs = [
            {"id": s, "points": [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in ("130m", "300m", "520m")],
             "targets": [cells[(s, "1_2b")]["x"]], "maxIterations": cap}
            for s in members
        ]  # fmt: skip
        res = lm(jobs)
        pred = {s: (res[s]["predictions"][0] if res[s]["predictions"][0] is not None else float("nan")) for s in members}
        out[str(cap)] = rig.wrong(rig.gaps(pred), truth)
    exact = rig.wrong(
        rig.gaps(
            {
                s: rig.exact_three(
                    [(cells[(s, b)]["x"], cells[(s, b)]["y"]) for b in ("130m", "300m", "520m")], cells[(s, "1_2b")]["x"]
                )
                for s in members
            }
        ),
        truth,
    )
    return {
        "wrong_pairs_by_cap": out,
        "exact_three_parameter": exact,
        "single_scale": rig.wrong(rig.gaps({s: cells[(s, "520m")]["y"] for s in members}), truth),
        "early_stopping_explains_gap": abs(out[str(CAPS[-1])] - exact) <= 10,
        "share_of_gap_from_early_stopping": (out[str(CAPS[-1])] - out["200"]) / (exact - out["200"]),
    }


# --- Task 3 ---------------------------------------------------------------------------------


def family_inference(members: list[str], cells: dict) -> dict[str, Any]:
    from asla.analysis.target_scoring import jackknife_standard_error, u_statistic_test

    family = {s: s.split("/")[0] if "/" in s else s for s in members}
    fams = sorted(set(family.values()))
    truth = rig.gaps({s: cells[(s, "1_2b")]["y"] for s in members})
    single = rig.gaps({s: cells[(s, "520m")]["y"] for s in members})
    proj = rig.projections(members, cells, ("130m", "300m", "520m"), "1_2b")
    out = {"units": len(fams), "unit_pairs": len(fams) * (len(fams) - 1) // 2, "forms": {}}
    for form, values in proj.items():
        pg = rig.gaps(values)
        kernel: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (a, b), g in truth.items():
            fa, fb = family[a], family[b]
            if fa == fb or g == 0:
                continue
            key = tuple(sorted((fa, fb)))
            kernel[key].append(
                100.0 * (float(np.sign(pg[(a, b)]) != np.sign(g)) - float(np.sign(single[(a, b)]) != np.sign(g)))
            )
        k = {key: float(np.mean(v)) for key, v in kernel.items()}
        test = u_statistic_test(k)
        jk = jackknife_standard_error(k)
        t = abs(test["mean"]) / test["standard_error"] if test["standard_error"] > 0 else float("inf")
        out["forms"][form] = {
            "mean_pp": float(test["mean"]),
            "u_se_pp": float(test["standard_error"]),
            "jackknife_se_pp": float(jk),
            "abs_t_u": float(t),
            "abs_t_jackknife": float(abs(test["mean"]) / jk) if jk > 0 else None,
            "significant_1_96": bool(t > Z),
            "significant_2_99": bool(t > rig.K_CONSERVATIVE),
        }
    out["power_note"] = "13 independent units (78 family pairs): low power; a null here is weak evidence of no difference"
    return out


def main() -> None:
    marin = marin_suite()
    dd = datadecide_c4()
    sn = signal_and_noise()
    dd_rate = dd["530M->1B"]["rate"]
    dd_rate_300 = dd["300M->1B"]["rate"]
    falls = {}
    for name, block in marin["by_borrowed"].items():
        below = [float(m) for m, v in block["noise_multiples"].items() if v["rate"] <= dd_rate]
        below300 = [float(m) for m, v in block["noise_multiples"].items() if v["rate"] <= dd_rate_300]
        falls[name] = {"to_c4_530M": min(below) if below else None, "to_c4_300M": min(below300) if below300 else None}
    members, cells = marin["_members"], marin["_cells"]
    task1 = {
        "marin": {k: {kk: vv for kk, vv in v.items() if kk != "_pairs"} for k, v in marin["by_borrowed"].items()},
        "datadecide_c4_bits_per_token": dd,
        "signal_and_noise_bits_per_byte": sn,
        "multiple_at_which_marin_falls_to_datadecide": falls,
        "ratio_marin_over_datadecide_530M": {k: v["rate"]["rate"] / dd_rate for k, v in marin["by_borrowed"].items()},
        "mechanism": mechanism(marin),
        "structure": structure(marin),
        "caveats": [
            "Marin submissions are curated entries, not a designed sample of recipes",
            "noise may depend on architecture; Marin's is borrowed from other suites",
            "single runs at both rungs on Marin and on Signal and Noise",
        ],
    }
    task2 = {"forecast_1e22": convergence(members, cells), "held_out_1_2b": heldout_by_cap(members, cells)}
    task3 = family_inference(members, cells)
    rigour = json.loads((REPO / "results" / "external" / "marin_rigour.json").read_text())
    task4 = {
        "lever_arm_designs": {
            k: {"lever_arm": v["lever_arm"], "excess_pp": v["excess_pp"]} for k, v in rigour["task_3"]["designs"].items()
        },
        "trend_reading": (
            "uninformative: three of four lever arms are nearly identical (about 4.9, 5.1, 5.1); not a refutation"
        ),
        "c4_calibrated_values_applied_elsewhere": [
            {
                "where": "results/external/marin_rigour.json task_5 u_statistic_tests (k = 2.99 from C4)",
                "metric_profile": "not estimable (Marin has one run per cell)",
                "action": "flagged; no Marin worlds exist to recalibrate in",
            }
        ],
        "own_calibration_elsewhere": (
            "OLMES (own worlds); published re-tests (OLMES macro worlds); lever-arm test (per metric); C4 on C4"
        ),
    }
    det_rate = {k: v["rate"]["rate"] for k, v in marin["by_borrowed"].items()}
    mech = task1["mechanism"]
    out = {
        "preregistration": "PREDICTIONS_TASK_REFERENCE.md",
        "task_1": task1,
        "task_2": task2,
        "task_3": task3,
        "task_4": task4,
        "predictions": {
            "PR1_at_least_5x": all(r >= 5 * dd_rate for r in det_rate.values()),
            "PR1_survives_3x_noise": all(
                v["noise_multiples"]["3"]["rate"] >= 5 * dd_rate for v in marin["by_borrowed"].values()
            ),
            "PR2_marin_index_larger": mech["marin"]["crossover_index"] > mech["datadecide_c4"]["crossover_index"],
            "PR3_reordered_larger_delta_slope": bool(
                mech["marin_reordered_vs_not_abs_delta_slope"]["mann_whitney_p_one_sided"] is not None
                and mech["marin_reordered_vs_not_abs_delta_slope"]["mann_whitney_p_one_sided"] < 0.05
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "marin": {k: v["rate"] for k, v in task1["marin"].items()},
                "dd": {k: {kk: v[kk] for kk in ("determined", "reordered", "rate")} for k, v in dd.items()},
                "sn": {k: {kk: v[kk] for kk in ("determined", "reordered", "rate")} for k, v in sn.items()},
            },
            indent=1,
            default=str,
        )
    )
    print(json.dumps(falls), json.dumps(task1["ratio_marin_over_datadecide_530M"]))
    print(json.dumps(mech, indent=1))
    print(json.dumps(task1["structure"], indent=1))
    print(json.dumps(task2, indent=1, default=str)[:3500])
    print(json.dumps(task3, indent=1))
    print(out["predictions"])


if __name__ == "__main__":
    main()
