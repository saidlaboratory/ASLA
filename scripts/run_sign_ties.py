"""PREDICTIONS_TASK_SIGN_TIES.md: sign-change ties; form uncertainty vs deployed fragility; LM non-convergence.

Writes ``results/external/sign_ties.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_leaderboard_core as core  # noqa: E402
from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts.run_marin_noise import B0_FACTORS, C0_FRACTIONS, DD_LADDER, N_NULL, SEED  # noqa: E402

OUT = REPO / "results" / "external" / "sign_ties.json"
TAU_PRIMARY = 1.959963984540054 * np.sqrt(2) * 0.000511
TAU_TIE = 1e-9
TAUS = (1e-9, 1e-4, 5e-4, 1e-3, TAU_PRIMARY, 0.003, 0.006, 0.012)
NULL_SIGMAS = {"empirical_two_parameter_1.36pct": 0.013590422048154386,
               "empirical_three_parameter_0.43pct": 0.004283575224018509,
               "borrowed_datadecide_1b_0.0511pct_added": 0.000511}  # fmt: skip
REASONABLE_C0 = (0.25, 0.5, 0.7, 0.9)
SCALE = 1e18
SAME_BPB = 1e-5


# --- Task 1 ---------------------------------------------------------------------------------


def pair_gaps(values: np.ndarray) -> np.ndarray:
    """values (..., candidates, rungs) > 0 -> log gaps (..., pairs, rungs), upper-triangle pair order."""

    lv = np.log(values)
    iu = np.triu_indices(values.shape[-2], 1)
    return lv[..., iu[0], :] - lv[..., iu[1], :]


def changes_with_tolerance(gaps: np.ndarray, tau: float) -> np.ndarray:
    """Sign changes between consecutive rungs whose |gap| exceeds tau; rungs at or below tau are dropped."""

    signs = np.where(np.abs(gaps) > tau, np.sign(gaps), 0.0)
    last = np.zeros(gaps.shape[:-1])
    count = np.zeros(gaps.shape[:-1], dtype=int)
    for r in range(gaps.shape[-1]):
        s = signs[..., r]
        count += (s != 0) & (last != 0) & (s != last)
        last = np.where(s != 0, s, last)
    return count


def row(changes: np.ndarray) -> dict[str, Any]:
    return {"mean": float(changes.mean()), "share_two_or_more": float(np.mean(changes >= 2)),
            "distribution": {str(k): float(np.mean(changes == k)) for k in range(4)}}  # fmt: skip


def null_row(changes: np.ndarray) -> dict[str, Any]:
    means, multi = changes.mean(axis=1), (changes >= 2).mean(axis=1)
    return {"mean_95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "share_two_or_more_95": [float(np.percentile(multi, 2.5)), float(np.percentile(multi, 97.5))],
            "mean_of_means": float(means.mean())}  # fmt: skip


def task1(members: list[str], cells: dict) -> dict[str, Any]:
    y = np.array([[cells[(s, b)]["y"] for b in rig.BUDGETS] for s in members])
    x = np.array([[cells[(s, b)]["x"] for b in rig.BUDGETS] for s in members])
    common_x = np.exp(np.log(x).mean(axis=0))
    own, common = np.empty_like(y), np.empty_like(y)
    for i in range(len(members)):
        slope, intercept = np.polyfit(np.log(x[i]), np.log(y[i]), 1)
        own[i] = np.exp(intercept + slope * np.log(x[i]))
        common[i] = np.exp(intercept + slope * np.log(common_x))
    sims = {}
    for name, sigma in NULL_SIGMAS.items():
        rng = np.random.default_rng(SEED)
        sims[name] = pair_gaps(own[None] * (1 + sigma * rng.standard_normal((N_NULL, *own.shape))))
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    dd = frame.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()[list(DD_LADDER)].to_numpy(float)
    g = {"marin_observed": pair_gaps(y), "noiseless_own_x": pair_gaps(own), "noiseless_common_x": pair_gaps(common),
         "datadecide_c4": pair_gaps(dd)}  # fmt: skip

    table: dict[str, dict[str, Any]] = {}
    for tau in TAUS:
        t: dict[str, Any] = {k: row(changes_with_tolerance(v, tau)) for k, v in g.items()}
        for name, s in sims.items():
            t[f"null_{name}"] = null_row(changes_with_tolerance(s, tau))
        t["marin_over_datadecide_mean"] = (
            t["marin_observed"]["mean"] / t["datadecide_c4"]["mean"] if t["datadecide_c4"]["mean"] else None
        )
        table[f"{tau:.6g}"] = t

    iu = np.triu_indices(len(members), 1)
    tie = changes_with_tolerance(g["noiseless_own_x"], TAU_TIE)
    two = []
    for p in np.where(tie >= 2)[0]:
        a, b = members[iu[0][p]], members[iu[1][p]]
        gap = g["noiseless_own_x"][p]
        two.append({
            "pair": [a, b],
            "log_gaps_own_x": gap.tolist(),
            "log_gaps_common_x": g["noiseless_common_x"][p].tolist(),
            "min_abs_gap": float(np.abs(gap).min()),
            "flops_ratio_a_over_b": (x[iu[0][p]] / x[iu[1][p]]).tolist(),
        })  # fmt: skip
    ratios = x[iu[0]] / x[iu[1]]
    # exploratory: submissions sharing an identical FLOPs ladder form groups; within-group gaps are aligned in x
    same = np.all(np.isclose(ratios, 1.0, rtol=1e-9), axis=1)
    labels = {tuple(np.round(r, 9)) for r in x / x[0]}
    split = {}
    for tau in (TAU_TIE, TAU_PRIMARY):
        obs = changes_with_tolerance(g["marin_observed"], tau)
        own = changes_with_tolerance(g["noiseless_own_x"], tau)
        split[f"{tau:.6g}"] = {
            "within_group": {"pairs": int(same.sum()), "observed": row(obs[same]), "noiseless_own_x": row(own[same])},
            "across_groups": {"pairs": int((~same).sum()), "observed": row(obs[~same]), "noiseless_own_x": row(own[~same])},
        }
    return {
        "taus": [f"{t:.6g}" for t in TAUS],
        "tau_primary": TAU_PRIMARY,
        "common_x_per_rung": common_x.tolist(),
        "table": table,
        "exploratory_flops_group_split": {"groups": len(labels), "by_tau": split},
        "noiseless_own_x_two_change_pairs_at_tie_tolerance": two,
        "flops_ratio_spread_across_rungs": {
            "all_pairs_median_max_over_min": float(np.median(ratios.max(axis=1) / ratios.min(axis=1))),
            "two_change_pairs_median_max_over_min": float(
                np.median([max(t["flops_ratio_a_over_b"]) / min(t["flops_ratio_a_over_b"]) for t in two])
            )
            if two
            else None,
        },  # fmt: skip
        "predictions": {
            "PT1_tie_tolerance_keeps_21_two_change_pairs": len(two) == 21,
            "PT2_common_x_has_no_multi_change_pairs": table[f"{TAU_TIE:.6g}"]["noiseless_common_x"]["share_two_or_more"]
            == 0.0,
            "PT3_marin_mean_at_least_0_30": table[f"{TAU_PRIMARY:.6g}"]["marin_observed"]["mean"] >= 0.30,
            "PT3_datadecide_mean_at_most_0_04": table[f"{TAU_PRIMARY:.6g}"]["datadecide_c4"]["mean"] <= 0.04,
            "PT3_ratio_at_least_5": (table[f"{TAU_PRIMARY:.6g}"]["marin_over_datadecide_mean"] or np.inf) >= 5,
            "PT4_below_1_36pct_null_band": table[f"{TAU_PRIMARY:.6g}"]["marin_observed"]["mean"]
            < table[f"{TAU_PRIMARY:.6g}"]["null_empirical_two_parameter_1.36pct"]["mean_95"][0],
        },
    }


# --- Task 2 and 3 ---------------------------------------------------------------------------


def degenerate(params: dict[str, Any] | None, ymin: float) -> str | None:
    if params is None:
        return "no fit returned"
    b, c = -params["slope"], params["asymptote"]
    if c >= (1 - 1e-4) * 0.999 * ymin:
        return "floor at its bound"
    if b >= 2 * (1 - 1e-4):
        return "exponent at its upper bound"
    if b <= 1e-5:
        return "exponent at its lower bound"
    return None


def lm_starts(
    pts: dict[str, list[dict[str, float]]],
    members: list[str],
    scale: float | dict[str, float],
    cap: int = 200,
    y_scale: float = 1.0,
    error_tolerance: float | None = None,
) -> dict:
    """Their LM from the default start and the 12 reasonable starts, x divided by scale (y times y_scale)."""

    sc = {s: scale[s] if isinstance(scale, dict) else scale for s in members}
    scaled = {s: [{"x": q["x"] / sc[s], "y": q["y"] * y_scale} for q in pts[s]] for s in members}
    starts: dict[str, dict[str, dict[str, float | None]]] = {"default": {s: {} for s in members}}
    for frac in REASONABLE_C0:
        for factor in B0_FACTORS:
            starts[f"c0={frac},b0x{factor}"] = {}
            for s in members:
                xx, yy = np.array([q["x"] for q in pts[s]]), np.array([q["y"] for q in pts[s]])
                _, b0, c0 = core.start(xx, yy, frac, factor)
                starts[f"c0={frac},b0x{factor}"][s] = {"cInit": c0 * y_scale, "bInit": b0}
    out = {}
    for tag, init in starts.items():
        base = {"targets": None, "maxIterations": cap, "errorTolerance": error_tolerance}
        jobs = [dict(base, id=s, points=scaled[s], targets=[rig.FORECAST_X / sc[s]], **init[s]) for s in members]
        res = ref.lm(jobs)
        out[tag] = {
            s: {"forecast": None if res[s]["predictions"][0] is None else res[s]["predictions"][0] / y_scale,
                "reference_forecast": None if res[s]["reference_predictions"][0] is None
                else res[s]["reference_predictions"][0] / y_scale,
                "params": res[s]["params"],
                "degenerate": degenerate(res[s]["params"], min(q["y"] for q in scaled[s]))}
            for s in members
        }  # fmt: skip
    return out


def global_optimum(members: list[str], raw: dict) -> dict[str, float]:
    out = {}
    for s in members:
        x, y = np.array([p[0] for p in raw[s]]), np.array([p[1] for p in raw[s]])
        fits = [core.least_squares_log_a(x, y, core.start(x, y, f, k)) for f in C0_FRACTIONS for k in B0_FACTORS]
        out[s] = min(fits, key=lambda t: t[1])[0]
    return out


def reach(runs: dict, optimum: dict[str, float], members: list[str]) -> dict[str, Any]:
    ok = {t: [s for s in members if r[s]["forecast"] is not None and abs(r[s]["forecast"] - optimum[s]) <= SAME_BPB]
          for t, r in runs.items()}  # fmt: skip
    total = len(runs) * len(members)
    return {
        "fits": total,
        "fits_reaching_optimum": sum(len(v) for v in ok.values()),
        "share_reaching_optimum": sum(len(v) for v in ok.values()) / total,
        "submissions_reaching_optimum_from_every_start": sum(all(s in v for v in ok.values()) for s in members),
        "default_start_submissions_reaching_optimum": len(ok["default"]),
        "by_start": {t: len(v) for t, v in ok.items()},
        "max_departure_bpb": max(abs((r[s]["forecast"] or np.inf) - optimum[s]) for r in runs.values() for s in members),
        "degenerate_fits": {t: {s: r[s]["degenerate"] for s in members if r[s]["degenerate"]} for t, r in runs.items()
                            if any(r[s]["degenerate"] for s in members)},
    }  # fmt: skip


def main() -> None:
    members, cells, _ = rig.load_cells()
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_SIGN_TIES.md", "task1": task1(members, cells)}

    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rig.BUDGETS] for s in members}
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in members}
    optimum = global_optimum(members, raw)
    default_opt = {s: rig.python_marin(raw[s], rig.FORECAST_X) for s in members}
    loglog = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in members}
    shared = rig.shared_exponent(raw, {s: rig.FORECAST_X for s in members})
    form = core.invariant_core({"three_parameter_global_optimum": optimum, "two_parameter_loglog": loglog,
                                "shared_exponent": shared}, members)  # fmt: skip

    deployed_runs = lm_starts(pts, members, 1.0)
    clean = {t: {s: r[s]["forecast"] for s in members} for t, r in deployed_runs.items()
             if not any(r[s]["degenerate"] for s in members)}  # fmt: skip
    deployed = core.invariant_core(clean, members) if clean else None
    excluded = {t: {s: r[s]["degenerate"] for s in members if r[s]["degenerate"]} for t, r in deployed_runs.items()
                if any(r[s]["degenerate"] for s in members)}  # fmt: skip
    out["task2"] = {
        "global_optimum_matches_python_marin_max_bpb": max(abs(optimum[s] - default_opt[s]) for s in members),
        "a_form_uncertainty": form,
        "b_deployed_fragility": {
            "starts": list(deployed_runs),
            "starts_ranked": list(clean),
            "starts_excluded_degenerate": excluded,
            "core": deployed,
            "unscaled_reach": reach(deployed_runs, optimum, members),
        },
        "predictions": {
            "PF1_form_pairs_at_least_140": form["invariant_pairs"] >= 140,
            "PF1_form_top_k_is_1": form["largest_invariant_top_k"] == 1,
            "PF1_form_top1_muonh": form["distinct_top1"] == ["muonh_qwen3_scaling"],
            "PF2_deployed_fewer_pairs": deployed is not None and deployed["invariant_pairs"] < form["invariant_pairs"],
            "PF2_deployed_top1_not_invariant": deployed is not None and len(deployed["distinct_top1"]) >= 2,
        },
    }

    scaled_runs = lm_starts(pts, members, SCALE)
    scaled = reach(scaled_runs, optimum, members)
    default_scaled = {s: scaled_runs["default"][s]["reference_forecast"] for s in members}
    unmodified_ok = sum(abs(default_scaled[s] - optimum[s]) <= SAME_BPB for s in members)
    t3: dict[str, Any] = {
        "their_fit_input": "raw x: computePowerLawWithOffset uses p.x directly; the 1e22 forecast uses model_flops",
        "change": f"x -> x / {SCALE:g} before computePowerLawWithOffset; forecast at {rig.FORECAST_X / SCALE:g}",
        "unmodified_function_default_start_reaching_optimum": int(unmodified_ok),
        "options_runner_matches_unmodified_at_default": max(
            abs(scaled_runs["default"][s]["forecast"] - default_scaled[s]) for s in members),
        "reach": scaled,
        "kendall_default_start_vs_optimum": ref.kendall(default_scaled, optimum),
        "ranking_default_start": sorted(members, key=lambda s: default_scaled[s]),
        "ranking_optimum": sorted(members, key=lambda s: optimum[s]),
        "a_range": [min(r[s]["params"]["intercept"] for r in scaled_runs.values() for s in members if r[s]["params"]),
                    max(r[s]["params"]["intercept"] for r in scaled_runs.values() for s in members if r[s]["params"])],
    }  # fmt: skip
    if scaled["submissions_reaching_optimum_from_every_start"] < len(members):
        per_fit = {s: min(q["x"] for q in pts[s]) for s in members}
        t3["secondary_scale_own_min_x"] = reach(lm_starts(pts, members, per_fit), optimum, members)
    fixed_runs = lm_starts(pts, members, SCALE, error_tolerance=1e-15)
    fixed_default = {s: fixed_runs["default"][s]["forecast"] for s in members}
    t3["candidate_fix_exploratory"] = {
        "change": "in computePowerLawWithOffset: fit x / 1e18 (forecast at x / 1e18) and errorTolerance 1e-15 instead "
        "of 1e-8; everything else unchanged. Equivalent: x / 1e18 and y * 100 (forecast / 100)",
        "status": "exploratory; the pre-registered change (x / 1e18 alone) is not sufficient",
        "kendall_default_start_vs_optimum": ref.kendall(fixed_default, optimum),
        "max_abs_forecast_minus_optimum_bpb": max(abs(fixed_default[s] - optimum[s]) for s in members),
        "ranking": sorted(members, key=lambda s: fixed_default[s]),
    }
    t3["exploratory_not_preregistered"] = {
        "why": "the library keeps a new point only if SSE falls by more than errorTolerance (1e-8, absolute); minimum "
        "SSEs here are about 4e-8 to 1e-4, so final improvements below 1e-8 are discarded",
        "x_scaled_cap_100000": reach(lm_starts(pts, members, SCALE, cap=100_000), optimum, members),
        "x_scaled_y_times_100_cap_200": reach(lm_starts(pts, members, SCALE, y_scale=100.0), optimum, members),
        "x_scaled_error_tolerance_1e-15_cap_200": reach(
            lm_starts(pts, members, SCALE, error_tolerance=1e-15), optimum, members
        ),
        "unscaled_error_tolerance_1e-15_cap_200": reach(
            lm_starts(pts, members, 1.0, error_tolerance=1e-15), optimum, members
        ),
        "iterations_used_x_scaled_cap_200": sorted(
            {r[s]["params"]["iterations"] for r in scaled_runs.values() for s in members if r[s]["params"]}
        ),
    }
    t3["fix_recorded"] = scaled["submissions_reaching_optimum_from_every_start"] == len(members)
    t3["predictions"] = {
        "PR1_all_21_from_default": unmodified_ok == len(members),
        "PR1_at_least_90pct_fits": scaled["share_reaching_optimum"] >= 0.9,
        "PR1_ranking_equals_optimum": t3["kendall_default_start_vs_optimum"] == 1.0,
    }
    out["task3"] = t3
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
