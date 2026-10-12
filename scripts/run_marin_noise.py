"""PREDICTIONS_TASK_MARIN_NOISE.md: noise or genuine reordering on Marin; the leaderboard at 1e22.

Writes ``results/external/marin_noise.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402

OUT = REPO / "results" / "external" / "marin_noise.json"
Z = 1.959963984540054
N_NULL = 2000
SEED = 20261013
DD_LADDER = ("150M", "300M", "530M", "1B")
C0_FRACTIONS = (0.0, 0.25, 0.5, 0.7, 0.9, 0.99)
B0_FACTORS = (0.5, 1.0, 2.0)


# --- Task 1 ---------------------------------------------------------------------------------


def empirical_bound(members: list[str], cells: dict) -> dict[str, Any]:
    from asla.models import FitError, bpb_power_law, fit_power_law

    two, three, df_two, df_three, failed = 0.0, 0.0, 0, 0, []
    for s in members:
        x = np.array([cells[(s, b)]["x"] for b in rig.BUDGETS])
        y = np.array([cells[(s, b)]["y"] for b in rig.BUDGETS])
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        fit2 = np.exp(intercept + slope * np.log(x))
        two += float(np.sum(((y - fit2) / fit2) ** 2))
        df_two += len(x) - 2
        try:
            fit3 = np.array(bpb_power_law(x, *fit_power_law(x, y)))
            three += float(np.sum(((y - fit3) / fit3) ** 2))
            df_three += len(x) - 3
        except (FitError, RuntimeError, ValueError):
            failed.append(s)
    return {
        "two_parameter": {"sigma_rel": float(np.sqrt(two / df_two)), "df": df_two},
        "three_parameter": {"sigma_rel": float(np.sqrt(three / df_three)), "df": df_three, "failed": failed},
        "note": "upper bound on seed noise: the residual scatter includes lack of fit",
    }


def determined(values: dict[str, float], se: dict[tuple[str, str], float]) -> dict[tuple[str, str], int]:
    """Sign of each pair determined at this rung (0 if undetermined)."""

    out = {}
    for a, b in combinations(sorted(values), 2):
        g = values[a] - values[b]
        out[(a, b)] = int(np.sign(g)) if abs(g) > Z * se[(a, b)] else 0
    return out


def both_rung(fit_sign: dict, target_sign: dict) -> dict[str, Any]:
    both = [k for k in target_sign if fit_sign[k] != 0 and target_sign[k] != 0]
    genuine = [k for k in both if fit_sign[k] != target_sign[k]]
    return {
        "determined_at_both": len(both),
        "genuine_reorderings": len(genuine),
        "rate_among_both_determined": len(genuine) / len(both) if both else None,
        "rate_among_all_pairs": len(genuine) / len(target_sign),
        "_genuine": genuine,
    }


def marin_both(members: list[str], cells: dict, rel: dict[str, float]) -> dict[str, Any]:
    fit = {s: cells[(s, "520m")]["y"] for s in members}
    tgt = {s: cells[(s, "1_2b")]["y"] for s in members}
    out = {}
    for name, cv in rel.items():
        se_f = {(a, b): cv * float(np.hypot(fit[a], fit[b])) for a, b in combinations(sorted(fit), 2)}
        se_t = {(a, b): cv * float(np.hypot(tgt[a], tgt[b])) for a, b in combinations(sorted(tgt), 2)}
        fs, ts = determined(fit, se_f), determined(tgt, se_t)
        res = both_rung(fs, ts)
        target_reorders = [k for k in ts if ts[k] != 0 and np.sign(fit[k[0]] - fit[k[1]]) != ts[k]]
        res["target_determined_reorderings"] = len(target_reorders)
        res["of_which_determined_at_520m"] = int(sum(fs[k] != 0 for k in target_reorders))
        res.pop("_genuine")
        out[name] = res
    return out


def datadecide_both() -> dict[str, Any]:
    from asla.analysis.moderation import fit_prior, moderate

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"]

    def rung(label: str) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
        cells = frame[frame["scale_label"] == label].groupby("intervention")["bpb"]
        means = cells.mean().to_dict()
        s2 = cells.var(ddof=1).to_dict()
        n = int(cells.count().min())
        prior = fit_prior(list(s2.values()), n - 1)
        mod = {r: float(moderate(v, prior)) for r, v in s2.items()}
        return means, {(a, b): float(np.sqrt(mod[a] / n + mod[b] / n)) for a, b in combinations(sorted(means), 2)}

    out = {}
    for name, (f, t) in ref.DD_DESIGNS.items():
        fm, fse = rung(f)
        tm, tse = rung(t)
        res = both_rung(determined(fm, fse), determined(tm, tse))
        res.pop("_genuine")
        out[name] = res
    return out


def sign_changes(values: np.ndarray) -> np.ndarray:
    """values: candidates x rungs (or sims x candidates x rungs); changes per pair, upper triangle order."""

    n = values.shape[-2]
    iu = np.triu_indices(n, 1)
    diff = values[..., :, None, :] - values[..., None, :, :]
    signs = np.sign(diff[..., iu[0], iu[1], :])
    return np.sum((signs[..., 1:] != signs[..., :-1]) & (signs[..., 1:] != 0) & (signs[..., :-1] != 0), axis=-1)


def distribution(changes: np.ndarray) -> dict[str, float]:
    return {str(k): float(np.mean(changes == k)) for k in range(4)}


def sign_pattern(members: list[str], cells: dict, sigma_rel: float) -> dict[str, Any]:
    y = np.array([[cells[(s, b)]["y"] for b in rig.BUDGETS] for s in members])
    x = np.array([[cells[(s, b)]["x"] for b in rig.BUDGETS] for s in members])
    observed = sign_changes(y)
    one = np.where(observed == 1)[0]
    iu = np.triu_indices(len(members), 1)
    log_gap = np.log(y[iu[0]]) - np.log(y[iu[1]])
    monotone = [bool(np.all(np.diff(log_gap[i]) > 0) or np.all(np.diff(log_gap[i]) < 0)) for i in one]
    fitted = np.empty_like(y)
    for i in range(len(members)):
        slope, intercept = np.polyfit(np.log(x[i]), np.log(y[i]), 1)
        fitted[i] = np.exp(intercept + slope * np.log(x[i]))
    rng = np.random.default_rng(SEED)
    sims = fitted[None] * (1 + sigma_rel * rng.standard_normal((N_NULL, *fitted.shape)))
    null = sign_changes(sims)
    null_mean = null.mean(axis=1)
    null_multi = (null >= 2).mean(axis=1)
    obs_mean, obs_multi = float(observed.mean()), float(np.mean(observed >= 2))
    return {
        "observed_distribution": distribution(observed),
        "observed_mean_changes": obs_mean,
        "observed_share_two_or_more": obs_multi,
        "single_change_pairs": int(len(one)),
        "single_change_monotone_gap": int(sum(monotone)),
        "null": {
            "sigma_rel": sigma_rel,
            "smooth_fit_changes_without_noise": distribution(sign_changes(fitted)),
            "mean_changes_95": [float(np.percentile(null_mean, 2.5)), float(np.percentile(null_mean, 97.5))],
            "share_two_or_more_95": [float(np.percentile(null_multi, 2.5)), float(np.percentile(null_multi, 97.5))],
            "mean_distribution": {k: float(np.mean(null == int(k))) for k in "0123"},
        },
        "observed_inside_null_band": bool(
            np.percentile(null_mean, 2.5) <= obs_mean <= np.percentile(null_mean, 97.5)
            and np.percentile(null_multi, 2.5) <= obs_multi <= np.percentile(null_multi, 97.5)
        ),
    }


def datadecide_pattern() -> dict[str, Any]:
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    means = frame.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()[list(DD_LADDER)]
    changes = sign_changes(means.to_numpy(float))
    return {"rungs": list(DD_LADDER), "distribution": distribution(changes), "mean_changes": float(changes.mean())}


# --- Task 2 ---------------------------------------------------------------------------------


def task2(members: list[str], cells: dict) -> dict[str, Any]:
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rig.BUDGETS] for s in members}
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in members}
    as_impl = ref.lm([{"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "maxIterations": 200} for s in members])
    impl = {s: as_impl[s]["predictions"][0] for s in members}
    optimum = {s: rig.python_marin(raw[s], rig.FORECAST_X) for s in members}
    order = sorted(members, key=lambda s: impl[s])
    gaps = {}
    for i, s in enumerate(order):
        neighbours = [abs(impl[s] - impl[o]) for o in order[max(0, i - 1) : i + 2] if o != s]
        gaps[s] = min(neighbours)
    shift = {s: optimum[s] - impl[s] for s in members}
    common = float(np.mean(list(shift.values())))
    differential = {s: shift[s] - common for s in members}
    finite = [s for s in members if gaps[s] > 0]
    common_over_gap = [abs(common) / gaps[s] for s in finite]
    diff_over_gap = [abs(differential[s]) / gaps[s] for s in finite]
    changed = [
        [a, b, impl[a] - impl[b], optimum[a] - optimum[b]]
        for a, b in combinations(members, 2)
        if np.sign(impl[a] - impl[b]) != np.sign(optimum[a] - optimum[b])
    ]
    top_impl, top_opt = order[:3], sorted(members, key=lambda s: optimum[s])[:3]

    settings = []
    for frac in C0_FRACTIONS:
        for factor in B0_FACTORS:
            jobs = []
            for s in members:
                y = np.array([q["y"] for q in pts[s]])
                x = np.array([q["x"] for q in pts[s]])
                c0 = frac * y.min()
                slope = np.polyfit(np.log(x), np.log(y - c0), 1)[0]
                b0 = float(np.clip(factor * -slope, 0.01, 2.0))
                jobs.append({"id": s, "points": pts[s], "targets": [rig.FORECAST_X], "cInit": c0, "bInit": b0})
            for cap in (200, 100_000):
                res = ref.lm([dict(j, maxIterations=cap) for j in jobs])
                f = {s: res[s]["predictions"][0] if res[s]["predictions"][0] is not None else np.inf for s in members}
                rank = sorted(members, key=lambda s: f[s])
                settings.append({"c0_fraction": frac, "b0_factor": factor, "cap": cap, "top1": rank[0], "top3": rank[:3]})
    by_cap = {}
    for cap in (200, 100_000):
        rows = [r for r in settings if r["cap"] == cap]
        by_cap[str(cap)] = {
            "settings": len(rows),
            "top1_differs_from_default_share": float(np.mean([r["top1"] != top_impl[0] for r in rows])),
            "top3_set_differs_from_default_share": float(np.mean([set(r["top3"]) != set(top_impl) for r in rows])),
            "distinct_top1": sorted({r["top1"] for r in rows}),
        }

    loglog = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in members}
    shared = rig.shared_exponent(raw, {s: rig.FORECAST_X for s in members})
    forms = {
        "as_implemented": impl,
        "least_squares_optimum": optimum,
        "two_parameter_loglog": loglog,
        "shared_exponent": shared,
    }
    tau = {
        f"{a}|{b}": float(stats.kendalltau([forms[a][s] for s in members], [forms[b][s] for s in members]).statistic)
        for a, b in combinations(forms, 2)
    }
    tops = {k: sorted(members, key=lambda s: v[s])[:3] for k, v in forms.items()}
    return {
        "reference": "least-squares optimum (Python, converged in all 21)",
        "shift_decomposition": {
            "common_shift_bpb": common,
            "median_common_over_gap": float(np.median(common_over_gap)),
            "median_differential_over_gap": float(np.median(diff_over_gap)),
            "max_differential_over_gap": float(np.max(diff_over_gap)),
            "submissions_with_differential_over_gap_above_1": int(sum(v > 1 for v in diff_over_gap)),
            "common_explains_few_order_changes": bool(np.median(diff_over_gap) <= 1 < np.median(common_over_gap)),
        },
        "changed_pairs": {"count": len(changed), "pairs": changed},
        "top3_as_implemented": top_impl,
        "top3_least_squares_optimum": top_opt,
        "initialisation_range": {"settings": settings, "by_cap": by_cap},
        "forms": {
            "kendall_tau": tau,
            "top3": tops,
            "top3_stable_across_forms": len({tuple(sorted(v)) for v in tops.values()}) == 1,
        },
    }


def main() -> None:
    members, cells, repeats = rig.load_cells()
    bound = empirical_bound(members, cells)
    cvs = rig.borrowed_cvs()
    rel = {k: v["cv"] for k, v in cvs.items()}
    rel["empirical_two_parameter"] = bound["two_parameter"]["sigma_rel"]
    rel["empirical_three_parameter"] = bound["three_parameter"]["sigma_rel"]
    out = {
        "preregistration": "PREDICTIONS_TASK_MARIN_NOISE.md",
        "task_1a_replicates": {
            "found": False,
            "searched": [
                "speedrun runs.json (118 records): no prefix-and-size repeats",
                "Delphi public tables: seed replicates only at 1e21 and 1e22 (3.4B and 9.7B parameters)",
                "marin, marin-experiments, marin-logbooks repositories: no seed-sweep experiments",
                "W&B: not accessible (no API key on this machine)",
            ],
            "speedrun_repeats": repeats,
        },
        "task_1b_empirical_bound": {**bound, "borrowed": rel},
        "task_1c_both_rung": {"marin_520m_to_1_2b": marin_both(members, cells, rel), "datadecide_c4": datadecide_both()},
        "task_1d_sign_pattern": {
            "marin": sign_pattern(members, cells, bound["two_parameter"]["sigma_rel"]),
            "marin_exploratory_null_at_three_parameter_bound": {
                "label": "exploratory, not pre-registered: the same null at the tighter three-parameter bound",
                **sign_pattern(members, cells, bound["three_parameter"]["sigma_rel"]),
            },
            "datadecide_c4": datadecide_pattern(),
        },
        "task_2": task2(members, cells),
    }
    emp = out["task_1c_both_rung"]["marin_520m_to_1_2b"]["empirical_two_parameter"]
    out["predictions"] = {
        "PN1_most_reorderings_not_determined_at_both_rungs": emp["genuine_reorderings"]
        < emp["target_determined_reorderings"] / 2
        if emp["target_determined_reorderings"]
        else True,
        "PN2_sign_changes_match_noise_null": out["task_1d_sign_pattern"]["marin"]["observed_inside_null_band"],
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("task_1b_empirical_bound", "task_1c_both_rung", "task_1d_sign_pattern")},
            indent=1,
            default=str,
        )
    )
    t2 = out["task_2"]
    print(
        json.dumps(
            {k: t2[k] for k in ("shift_decomposition", "top3_as_implemented", "top3_least_squares_optimum", "forms")},
            indent=1,
        )
    )
    print(json.dumps(t2["initialisation_range"]["by_cap"], indent=1), t2["changed_pairs"]["count"])
    print(out["predictions"])


if __name__ == "__main__":
    main()
