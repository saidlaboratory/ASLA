"""PREDICTIONS_TASK_MARIN_NOISE_2.md, Task 2: the robust core of the Marin speedrun leaderboard at 1e22.

56 rankings: the three-parameter form from 18 starting values (c0 x b0, as in run_marin_noise.task2) under their
LM capped at 200, their LM at 100,000 and the least-squares optimum (scipy trust region) from the same start; plus
two-parameter log-log and shared exponent. A sensitivity set adds the as-implemented default and the exact
three-parameter fit (asla.models.fit_power_law), 58 in all.

Writes ``results/external/leaderboard_core.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts.run_marin_noise import B0_FACTORS, C0_FRACTIONS  # noqa: E402

OUT = REPO / "results" / "external" / "leaderboard_core.json"
SAME_FORECAST_BPB = 1e-6
SAME_SSE_REL = 1e-3


def start(x: np.ndarray, y: np.ndarray, frac: float, factor: float) -> tuple[float, float, float]:
    c0 = frac * y.min()
    slope, intercept = np.polyfit(np.log(x), np.log(y - c0), 1)
    return float(np.exp(intercept)), float(np.clip(factor * -slope, 0.01, 2.0)), float(c0)


def least_squares_from(x: np.ndarray, y: np.ndarray, p0: tuple[float, float, float]) -> tuple[float, float]:
    """(forecast at 1e22, SSE) of the leaderboard's form and bounds, trust region from p0."""

    lower, upper = [1e-10, 1e-6, 0.0], [1e30, 2.0, 0.999 * y.min()]
    p = np.clip(p0, lower, upper)
    res = optimize.least_squares(lambda q: q[0] * x ** (-q[1]) + q[2] - y, p, bounds=(lower, upper), max_nfev=20000)
    a, b, c = res.x
    return float(a * rig.FORECAST_X ** (-b) + c), float(np.sum(res.fun**2))


def least_squares_log_a(x: np.ndarray, y: np.ndarray, p0: tuple[float, float, float]) -> tuple[float, float]:
    """Diagnostic solver: the same form and bounds with log a and tight tolerances (a spans 1e-10 to 1e30)."""

    lx = np.log(x)
    lower, upper = [np.log(1e-10), 1e-6, 0.0], [np.log(1e30), 2.0, 0.999 * y.min()]
    q = np.clip([np.log(p0[0]), p0[1], p0[2]], lower, upper)
    res = optimize.least_squares(lambda r: np.exp(r[0] - r[1] * lx) + r[2] - y, q, bounds=(lower, upper),
                                 max_nfev=100_000, ftol=1e-15, xtol=1e-15, gtol=1e-15)  # fmt: skip
    return float(np.exp(res.x[0] - res.x[1] * np.log(rig.FORECAST_X)) + res.x[2]), float(np.sum(res.fun**2))


def multistart(runs: dict[str, dict[str, tuple[float, float]]], optimum: dict[str, float], members: list[str]) -> dict:
    off = {s: [t for t in runs if abs(runs[t][s][0] - optimum[s]) > SAME_FORECAST_BPB] for s in members}
    worse = {
        s: [t for t in off[s] if runs[t][s][1] > min(runs[u][s][1] for u in runs) * (1 + SAME_SSE_REL)] for s in members
    }
    orders = {tuple(sorted(members, key=lambda s: runs[t][s][0])) for t in runs}
    counts: dict[str, int] = {}
    for v in off.values():
        for t in v:
            counts[t] = counts.get(t, 0) + 1
    return {
        "starts": len(runs),
        "submissions_reaching_optimum_from_every_start": sum(not v for v in off.values()),
        "starts_off_optimum_per_submission": {s: len(v) for s, v in off.items()},
        "off_optimum_starts_with_higher_sse": sum(len(v) for v in worse.values()),
        "off_optimum_starts_total": sum(len(v) for v in off.values()),
        "off_optimum_count_by_start": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "max_forecast_departure_bpb": max(abs(runs[t][s][0] - optimum[s]) for t in runs for s in members),
        "distinct_rankings": len(orders),
        "same_ranking_from_all_starts": len(orders) == 1,
    }


def invariant_core(rankings: dict[str, dict[str, float]], members: list[str]) -> dict[str, Any]:
    signs = {(a, b): {int(np.sign(r[a] - r[b])) for r in rankings.values()} for a, b in combinations(sorted(members), 2)}
    invariant = [p for p, s in signs.items() if len(s) == 1]
    orders = [sorted(members, key=lambda s: r[s]) for r in rankings.values()]
    k_max = 0
    for k in range(1, len(members) + 1):
        if len({frozenset(o[:k]) for o in orders}) == 1:
            k_max = k
        else:
            break
    invariant_k = [k for k in range(1, len(members) + 1) if len({frozenset(o[:k]) for o in orders}) == 1]
    # interchangeable groups: connected components of non-invariant pairs
    parent = {s: s for s in members}

    def find(s: str) -> str:
        while parent[s] != s:
            s = parent[s]
        return s

    for p, sg in signs.items():
        if len(sg) > 1:
            parent[find(p[0])] = find(p[1])
    comps: dict[str, list[str]] = {}
    for s in members:
        comps.setdefault(find(s), []).append(s)
    median_rank = {s: float(np.median([o.index(s) + 1 for o in orders])) for s in members}
    groups = sorted((sorted(v, key=lambda s: median_rank[s]) for v in comps.values()), key=lambda g: median_rank[g[0]])
    return {
        "rankings": len(rankings),
        "pairs": len(signs),
        "invariant_pairs": len(invariant),
        "largest_invariant_top_k": k_max,
        "all_invariant_k": invariant_k,
        "invariant_pair_list": [
            [a, b] if next(iter(signs[(a, b)])) < 0 else [b, a] for a, b in invariant
        ],  # [better, worse]
        "rank_range": {
            s: [min(o.index(s) + 1 for o in orders), max(o.index(s) + 1 for o in orders)]
            for s in sorted(members, key=lambda s: median_rank[s])
        },  # fmt: skip
        "top1_share": {s: sum(o[0] == s for o in orders) / len(orders) for s in sorted({o[0] for o in orders})},
        "top1_by_ranking": {k: min(members, key=lambda s: r[s]) for k, r in rankings.items()},
        "distinct_top1": sorted({o[0] for o in orders}),
        "groups_by_median_rank": [
            {
                "members": g,
                "median_ranks": [median_rank[s] for s in g],
                "rank_range": [
                    min(o.index(s) + 1 for o in orders for s in g),
                    max(o.index(s) + 1 for o in orders for s in g),
                ],
            }
            for g in groups
        ],  # fmt: skip
    }


def main() -> None:
    members, cells, _ = rig.load_cells()
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rig.BUDGETS] for s in members}
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in members}
    xs = {s: np.array([p[0] for p in raw[s]]) for s in members}
    ys = {s: np.array([p[1] for p in raw[s]]) for s in members}

    rankings: dict[str, dict[str, float]] = {}
    ls_runs: dict[str, dict[str, tuple[float, float]]] = {}
    for frac in C0_FRACTIONS:
        for factor in B0_FACTORS:
            tag = f"c0={frac},b0x{factor}"
            starts = {s: start(xs[s], ys[s], frac, factor) for s in members}
            jobs = [{"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "cInit": starts[s][2],
                     "bInit": starts[s][1]} for s in members]  # fmt: skip
            for cap in (200, 100_000):
                res = ref.lm([dict(j, maxIterations=cap) for j in jobs])
                rankings[f"lm{cap}|{tag}"] = {
                    s: res[s]["predictions"][0] if res[s]["predictions"][0] is not None else np.inf for s in members
                }
            ls_runs[tag] = {s: least_squares_from(xs[s], ys[s], starts[s]) for s in members}
            rankings[f"least_squares|{tag}"] = {s: ls_runs[tag][s][0] for s in members}
    rankings["two_parameter_loglog"] = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in members}
    rankings["shared_exponent"] = rig.shared_exponent(raw, {s: rig.FORECAST_X for s in members})
    assert len(rankings) == 56

    impl_res = ref.lm([{"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "maxIterations": 200} for s in members])
    impl = {s: impl_res[s]["predictions"][0] for s in members}
    optimum = {s: rig.python_marin(raw[s], rig.FORECAST_X) for s in members}
    exact = {s: rig.exact_three(raw[s], rig.FORECAST_X) for s in members}
    sensitivity = dict(rankings, as_implemented_default=impl, exact_three_parameter=exact)

    # least squares from all 18 starts: the pre-registered solver, and a well-scaled diagnostic solver
    log_a_runs = {t: {s: least_squares_log_a(xs[s], ys[s], start(xs[s], ys[s], f, k)) for s in members}
                  for t, (f, k) in ((f"c0={f},b0x{k}", (f, k)) for f in C0_FRACTIONS for k in B0_FACTORS)}  # fmt: skip
    ls_check: dict[str, Any] = {
        "optimum": "rig.python_marin (default start); its SSE equals the minimum over all 36 multistart fits",
        "same_optimum_criterion": f"forecast within {SAME_FORECAST_BPB} BPB",
        "trust_region_default_tolerances_preregistered": multistart(ls_runs, optimum, members),
        "trust_region_log_a_tight_tolerances_diagnostic": multistart(log_a_runs, optimum, members),
    }
    bmin = {s: min(r[s][1] for r in (*ls_runs.values(), *log_a_runs.values())) for s in members}
    opt_sse = {s: rig.converged_sse(raw[s]) for s in members}
    ls_check["optimum_sse_equals_multistart_minimum"] = bool(
        all(opt_sse[s] <= bmin[s] * (1 + SAME_SSE_REL) for s in members)
    )

    # monotone distortion: as implemented (LM, cap 200, default start) -> least-squares optimum
    rank_impl = {s: i + 1 for i, s in enumerate(sorted(members, key=lambda s: impl[s]))}
    discordant = [
        {"pair": [a, b], "impl_ranks": [rank_impl[a], rank_impl[b]], "impl_rank_distance": abs(rank_impl[a] - rank_impl[b]),
         "impl_gap_bpb": impl[a] - impl[b], "optimum_gap_bpb": optimum[a] - optimum[b]}
        for a, b in combinations(members, 2)
        if np.sign(impl[a] - impl[b]) != np.sign(optimum[a] - optimum[b])
    ]  # fmt: skip
    rho = float(stats.spearmanr([impl[s] for s in members], [optimum[s] for s in members]).statistic)
    shift = {s: optimum[s] - impl[s] for s in members}
    dist = [d["impl_rank_distance"] for d in discordant]
    monotone: dict[str, Any] = {
        "spearman_forecasts": rho,
        "kendall_forecasts": ref.kendall(impl, optimum),
        "pearson_forecasts": float(np.corrcoef([impl[s] for s in members], [optimum[s] for s in members])[0, 1]),
        "spearman_forecast_vs_shift": float(
            stats.spearmanr([impl[s] for s in members], [shift[s] for s in members]).statistic
        ),
        "discordant_pairs": len(discordant),
        "discordant_rank_distance_counts": {str(k): dist.count(k) for k in sorted(set(dist))},
        "discordant_share_rank_distance_le_2": float(np.mean([d <= 2 for d in dist])) if dist else None,
        "discordant": sorted(discordant, key=lambda d: d["impl_rank_distance"]),
        "approximately_monotone": bool(rho >= 0.95 and (np.mean([d <= 2 for d in dist]) if dist else 1.0) > 0.5),
    }  # fmt: skip
    monotone["reading"] = (
        "approximately monotone: a monotone map of forecasts preserves every pairwise order, so differential shifts "
        "of median 8.7x the nearest-neighbour gap (marin_noise.json) that are close to order-preserving change only the "
        f"{len(discordant)} orders listed, which sit where the map is locally non-monotone"
        if monotone["approximately_monotone"]
        else "not approximately monotone under the pre-registered criterion; the monotone-distortion reading is not recorded"
    )

    discordant_per_submission: dict[str, int] = {}
    for d in discordant:
        for m in d["pair"]:
            discordant_per_submission[m] = discordant_per_submission.get(m, 0) + 1
    monotone["discordant_per_submission"] = dict(sorted(discordant_per_submission.items(), key=lambda kv: -kv[1]))

    core = invariant_core(rankings, members)
    verified = {k: v for k, v in rankings.items() if not k.startswith("least_squares|")}
    verified["least_squares_optimum"] = optimum
    four_forms = {
        "as_implemented_default": impl,
        "least_squares_optimum": optimum,
        "two_parameter_loglog": rankings["two_parameter_loglog"],
        "shared_exponent": rankings["shared_exponent"],
    }
    out: dict[str, Any] = {
        "preregistration": "PREDICTIONS_TASK_MARIN_NOISE_2.md",
        "core_56": core,
        "core_58_sensitivity": invariant_core(sensitivity, members),
        "core_three_parameter_only_54": invariant_core({k: v for k, v in rankings.items() if "|" in k}, members),
        "core_39_least_squares_as_verified_optimum": invariant_core(verified, members),
        "core_four_forms_default_starts": invariant_core(four_forms, members),
        "core_lm_cap200_18_starts": invariant_core({k: v for k, v in rankings.items() if k.startswith("lm200|")}, members),
        "least_squares_from_18_starts": ls_check,
        "monotone_distortion": monotone,
        "predictions": {
            "PM4_core_small": "no threshold was pre-registered; invariant share "
            f"{core['invariant_pairs']}/{core['pairs']} is reported and judged in the report",
            "PM4_top1_invariant": core["largest_invariant_top_k"] >= 1 and core["distinct_top1"] == ["muonh_qwen3_scaling"],
            "PM4_top3_not_invariant": 3 not in core["all_invariant_k"],
        },
        "rankings": rankings,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    for k in [k for k in out if k.startswith("core_")]:
        c = out[k]
        print(
            k,
            c["invariant_pairs"],
            "/",
            c["pairs"],
            "k",
            c["largest_invariant_top_k"],
            c["all_invariant_k"],
            c["distinct_top1"],
        )
        for g in c["groups_by_median_rank"]:
            print("   ", g["rank_range"], g["members"])
    print(json.dumps(ls_check, indent=1))
    print(json.dumps({k: v for k, v in monotone.items() if k != "discordant"}, indent=1))
    for d in monotone["discordant"]:
        print("  ", d["impl_ranks"], d["pair"], round(d["impl_gap_bpb"], 5), round(d["optimum_gap_bpb"], 5))
    print(out["predictions"])


if __name__ == "__main__":
    main()
