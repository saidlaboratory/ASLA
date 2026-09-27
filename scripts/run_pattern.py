"""Tasks 1-4 of PREDICTIONS_TASK_PATTERN.md: the misspecification-pattern property (exploratory).

* Task 1: alignment of the recipe-specific misspecification at the target with
  the true target ordering (a) and the top-fitted-rung ordering (b), plus an
  observable in-sample proxy (c), for 12 own-world cells and the 18 swap
  configurations, against which rule wins.
* Task 2: C4 sweep of alignment at measured recipe-specific amplitude, in the
  physical region only, at the short, primary and long designs.
* Task 3: whether (b) and (c) track (a), against the crossover rate.
* Task 4: correct-prob's long design: single-scale against chance, and the
  truth's monotonicity.

Writes ``results/target_scoring/pattern.json`` and
``results/target_scoring/figures/alignment_sweep.png``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import gaps, true_mis_rate  # noqa: E402
from scripts import transfer_worlds as tw  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "pattern.json"
FIGURE = REPO / "results" / "target_scoring" / "figures" / "alignment_sweep.png"
SWEEP = tuple(np.round(np.linspace(-1.0, 1.0, 9), 2))
N_SWEEP_WORLDS = 100
N_CP_WORLDS = 150
N_BOOT = 2000
SEED = 20260930
C4_DESIGNS = {"short": ("14M", "16M"), "primary": ("300M", "530M"), "long": ("8M", "1B")}


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return float(stats.spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)).statistic)


def in_sample_residual(table: pd.DataFrame, compute: pd.Series, design: tuple[str, str]) -> pd.Series:
    """Per recipe: value at the top fitted rung minus its power-law fit on rungs <= f, centred across recipes."""

    from asla.models import FitError, bpb_power_law, fit_power_law

    fit_top = design[0]
    order = sorted(table.columns, key=lambda s: float(compute[s]))
    rungs = order[: order.index(fit_top) + 1]
    x = np.array([float(compute[s]) for s in rungs])
    out = {}
    for recipe in table.index:
        y = table.loc[recipe, rungs].to_numpy(dtype=float)
        try:
            params = fit_power_law(x, y)
            out[recipe] = float(y[-1] - bpb_power_law(float(compute[fit_top]), *params))
        except (FitError, RuntimeError, ValueError):
            out[recipe] = float("nan")
    series = pd.Series(out)
    return series - series.mean()


def crossover_rate(a: pd.Series, b: pd.Series) -> float:
    ga, gb = gaps(a), gaps(b)
    scored = [(ga[k], gb[k]) for k in ga if ga[k] != 0 and gb[k] != 0]
    return float(np.mean([np.sign(x) != np.sign(y) for x, y in scored]))


def alignment(metric: str, design: tuple[str, str]) -> dict[str, Any]:
    ctx = tw.context(metric)
    fit_top, target = design
    mu, spec = ctx["mu"], ctx["specific_table"]
    delta = spec[target]
    observed = ctx["frame"].groupby(["intervention", "scale_label"])["bpb"].mean().unstack()[ctx["scales"]]
    observed = observed.loc[ctx["recipes"]]
    residual_truth = in_sample_residual(mu, ctx["compute"], design)
    residual_obs = in_sample_residual(observed, ctx["compute"], design)
    return {
        "design": list(design),
        "a_with_target": _spearman(delta, mu[target]),
        "b_with_top_rung_truth": _spearman(delta, mu[fit_top]),
        "b_with_top_rung_observed": _spearman(delta, observed[fit_top]),
        "c_proxy_truth": _spearman(residual_truth, mu[fit_top]),
        "c_proxy_observed": _spearman(residual_obs, observed[fit_top]),
        "crossover_rate_f_to_t": crossover_rate(mu[fit_top], mu[target]),
    }


def swap_alignment(name: str) -> float:
    swap = importlib.import_module("scripts.run_structure_swap")
    base, sources = swap.configurations()[name]
    truth, _ = tw.hybrid(base, sources)
    src = sources.get("iv_specific", sources.get("iv", base))
    s_ctx = tw.context(src)
    b = tw.context(base)
    delta = (s_ctx["specific_table"] / s_ctx["g"] * b["g"]).loc[b["recipes"], tw.GAP_SCALE]
    return _spearman(delta, truth[tw.GAP_SCALE])


# --- Task 2 -----------------------------------------------------------------------------------


def _rank_one(ctx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = ctx["residual"]
    u, sv, vt = np.linalg.svd(residual, full_matrices=False)
    loadings, shape = u[:, 0] * sv[0], vt[0]
    if loadings.mean() < 0:
        loadings, shape = -loadings, -shape
    remainder = residual - np.outer(loadings, shape)
    return loadings, shape, remainder


def _standardise(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    return v / v.std(ddof=1)


def swept_truth(c: float, design: tuple[str, str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """C4 truth with recipe-specific loadings whose target misspecification has Pearson correlation c with y0."""

    ctx = tw.context("c4")
    loadings, shape, remainder = _rank_one(ctx)
    t = ctx["scales"].index(design[1])
    y0 = (ctx["parametric"] + ctx["common_table"])[design[1]].to_numpy()
    z_y = _standardise(y0)
    centred = loadings - loadings.mean()
    perp = centred - (centred @ z_y) / (z_y @ z_y) * z_y
    z_perp = _standardise(perp)
    sd_a = float(centred.std(ddof=1))
    sign = float(np.sign(shape[t]))

    def delta(theta: float) -> np.ndarray:
        b = sd_a * sign * (np.cos(theta) * z_y + np.sin(theta) * z_perp)
        return b * shape[t] + remainder[:, t]

    def corr(theta: float) -> float:
        return float(np.corrcoef(delta(theta), y0)[0, 1])

    grid = np.linspace(0.0, np.pi, 2001)
    values = np.array([corr(th) for th in grid])
    k = int(np.argmin(np.abs(values - c)))
    theta = float(grid[k])
    lo, hi = max(k - 1, 0), min(k + 1, len(grid) - 1)
    if (values[lo] - c) * (values[hi] - c) < 0:
        theta = float(optimize.brentq(lambda th: corr(th) - c, grid[lo], grid[hi]))
    b = sd_a * sign * (np.cos(theta) * z_y + np.sin(theta) * z_perp)
    mean_loading = float(loadings.mean())
    matrix = np.outer(np.full_like(loadings, mean_loading), shape) + np.outer(b, shape) + remainder
    truth = ctx["parametric"] + pd.DataFrame(matrix, index=ctx["recipes"], columns=ctx["scales"])
    d = pd.Series(delta(theta), index=ctx["recipes"])
    info = {
        "requested_c": c,
        "achieved_pearson_with_y0": corr(theta),
        "achieved_spearman_with_true_target": _spearman(d, truth[design[1]]),
        "loading_sd": float(np.std(b, ddof=1)),
        "measured_loading_sd": sd_a,
    }
    return truth, info


def reach_of_preregistered(design: tuple[str, str]) -> list[float]:
    """Range of Pearson correlation the pre-registered construction (rank-one loadings, E fixed) can reach."""

    lo = swept_truth(-1.0, design)[1]["achieved_pearson_with_y0"]
    hi = swept_truth(1.0, design)[1]["achieved_pearson_with_y0"]
    return [lo, hi]


def permuted_truth(c: float, design: tuple[str, str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """C4 truth whose measured recipe-specific residual rows are reassigned among recipes.

    Deviation from the pre-registration (documented in the output): recipes are
    scored by c z_y + sqrt(1 - c^2) z_perp, with z_y the standardised target
    values before the recipe-specific part and z_perp the standardised
    component of the measured target residual orthogonal to it; the row with
    the k-th smallest residual at the target goes to the recipe with the k-th
    smallest score. The set of rows is unchanged, so the amplitude is exactly
    the measured one; only the alignment changes.
    """

    ctx = tw.context("c4")
    spec = ctx["specific_table"].to_numpy()
    t = ctx["scales"].index(design[1])
    y0 = (ctx["parametric"] + ctx["common_table"])[design[1]].to_numpy()
    z_y = _standardise(y0)
    delta = spec[:, t]
    centred = delta - delta.mean()
    z_perp = _standardise(centred - (centred @ z_y) / (z_y @ z_y) * z_y)
    score = c * z_y + np.sqrt(max(0.0, 1 - c**2)) * z_perp
    rows_by_delta = np.argsort(delta, kind="stable")
    recipes_by_score = np.argsort(score, kind="stable")
    permuted = np.empty_like(spec)
    permuted[recipes_by_score] = spec[rows_by_delta]
    matrix = ctx["common"] + permuted
    truth = ctx["parametric"] + pd.DataFrame(matrix, index=ctx["recipes"], columns=ctx["scales"])
    d = pd.Series(permuted[:, t], index=ctx["recipes"])
    info = {
        "requested_c": c,
        "achieved_spearman_with_y0": _spearman(d, pd.Series(y0, index=ctx["recipes"])),
        "achieved_spearman_with_true_target": _spearman(d, truth[design[1]]),
        "specific_rms": float(np.sqrt(np.mean(permuted**2))),
        "measured_specific_rms": float(np.sqrt(np.mean(spec**2))),
    }
    return truth, info


def sweep_world(c: float, design_name: str, index: int) -> float:
    ctx = tw.context("c4")
    design = C4_DESIGNS[design_name]
    truth, _ = permuted_truth(c, design)
    rng = np.random.default_rng([SEED, list(C4_DESIGNS).index(design_name), index])
    return tw.world_excess(truth, ctx["base"].sigma, ctx, {design_name: design}, rng)[design_name]


# --- Task 4 -----------------------------------------------------------------------------------


def cp_long_world(index: int) -> tuple[float, float]:
    ctx = tw.context("cp")
    design = tw.ladder_designs(ctx["scales"], ctx["compute"])["long"]
    rates: dict[str, tuple[float, float]] = {}
    rng = np.random.default_rng([SEED, 9, index])
    tw.world_excess(ctx["mu"], ctx["base"].sigma, ctx, {"long": design}, rng, rates=rates)
    return rates["long"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    # Task 1 and 3.
    transfer = json.loads((REPO / "results" / "target_scoring" / "dimensionless_transfer.json").read_text())
    swap_json = json.loads((REPO / "results" / "target_scoring" / "structure_swap.json").read_text())
    own: dict[str, Any] = {}
    for m in tw.METRICS:
        ctx = tw.context(m)
        for d, design in tw.ladder_designs(ctx["scales"], ctx["compute"]).items():
            entry = alignment(m, design)
            truth = transfer["ground_truth"][m][d]
            entry["own_world_sign"] = truth["sign"]
            entry["own_world_excess_pp"] = truth["excess_pp"]
            own[f"{m}|{d}"] = entry

    def lines_up(a: float, sign: str) -> bool | None:
        if sign not in ("wins", "loses", "projection wins", "projection loses"):
            return None
        return (a > 0) == ("wins" in sign)

    own_tests = [lines_up(e["a_with_target"], e["own_world_sign"]) for e in own.values()]
    swaps = {}
    for name, row in swap_json["configurations"].items():
        a = swap_alignment(name)
        swaps[name] = {
            "a_with_target": a,
            "excess_pp": row["excess_pp"],
            "sign": row["sign"],
            "lines_up": lines_up(a, row["sign"]),
        }
    swap_tests = [v["lines_up"] for v in swaps.values()]
    measurability = {
        key: {
            "sign_b_truth_matches_a": np.sign(e["b_with_top_rung_truth"]) == np.sign(e["a_with_target"]),
            "abs_diff_b_truth_a": abs(e["b_with_top_rung_truth"] - e["a_with_target"]),
            "sign_b_observed_matches_a": np.sign(e["b_with_top_rung_observed"]) == np.sign(e["a_with_target"]),
            "sign_c_observed_matches_a": np.sign(e["c_proxy_observed"]) == np.sign(e["a_with_target"]),
            "sign_c_truth_matches_a": np.sign(e["c_proxy_truth"]) == np.sign(e["a_with_target"]),
            "crossover_rate": e["crossover_rate_f_to_t"],
        }
        for key, e in own.items()
    }
    measurability = {
        k: {kk: (bool(vv) if isinstance(vv, np.bool_) else vv) for kk, vv in v.items()} for k, v in measurability.items()
    }

    # Task 2.
    sweep_info = {d: {f"{c:g}": permuted_truth(c, design)[1] for c in SWEEP} for d, design in C4_DESIGNS.items()}
    ctx = tw.context("c4")
    physical = {
        d: {f"{c:g}": tw.physical(permuted_truth(c, design)[0], ctx["sigma_table"], "c4") for c in SWEEP}
        for d, design in C4_DESIGNS.items()
    }
    reach = {d: reach_of_preregistered(design) for d, design in C4_DESIGNS.items()}
    jobs = [(c, d, i) for d in C4_DESIGNS for c in SWEEP for i in range(N_SWEEP_WORLDS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(sweep_world, *zip(*jobs), chunksize=20))
        cp_rates = list(pool.map(cp_long_world, range(N_CP_WORLDS), chunksize=20))
    boot = rng.integers(0, N_SWEEP_WORLDS, size=(N_BOOT, N_SWEEP_WORLDS))
    grid: dict[str, Any] = {}
    for d in C4_DESIGNS:
        rows = {}
        for c in SWEEP:
            x = np.array([v for (cc, dd, _), v in zip(jobs, values) if cc == c and dd == d])
            lo, hi = np.nanpercentile(np.nanmean(x[boot], axis=1), [2.5, 97.5])
            rows[f"{c:g}"] = {
                "excess_pp": float(np.nanmean(x)),
                "ci": [float(lo), float(hi)],
                "sign": "wins" if hi < 0 else ("loses" if lo > 0 else "unresolved"),
                "physical": bool(physical[d][f"{c:g}"]["tolerant_monotone"] and physical[d][f"{c:g}"]["in_range"]),
                **sweep_info[d][f"{c:g}"],
            }
        phys = [(float(k), r["excess_pp"]) for k, r in rows.items() if r["physical"]]
        cs, ex = (np.array(v) for v in zip(*phys)) if phys else (np.array([]), np.array([]))
        crossings = [c0 + (0 - e0) * (c1 - c0) / (e1 - e0) for (c0, e0), (c1, e1) in pairwise(phys) if e0 * e1 < 0]
        achieved = [(r["achieved_spearman_with_true_target"], r["excess_pp"]) for r in rows.values() if r["physical"]]
        achieved_crossings = [
            a0 + (0 - e0) * (a1 - a0) / (e1 - e0) for (a0, e0), (a1, e1) in pairwise(achieved) if e0 * e1 < 0
        ]
        grid[d] = {
            "sign_changes_at_achieved_alignment": achieved_crossings,
            "points": rows,
            "physical_points": len(phys),
            "spearman_excess_vs_c": float(stats.spearmanr(cs, ex).statistic) if len(phys) >= 3 else None,
            "sign_changes_at_c": crossings,
        }
    measured = {d: own[f"c4|{d}"]["a_with_target"] for d in C4_DESIGNS}

    # Task 4.
    cp = tw.context("cp")
    long_design = tw.ladder_designs(cp["scales"], cp["compute"])["long"]
    proj_rates, single_rates = (np.array(v) for v in zip(*cp_rates))
    f_rung, t_rung = long_design
    mu = cp["mu"]
    step = mu.diff(axis=1).iloc[:, 1:]
    cell_noise = float(np.sqrt(np.mean(cp["sigma_table"][f_rung] ** 2)) / np.sqrt(cp["n_seeds"]))
    task4 = {
        "design": list(long_design),
        "chance_rate": 0.5,
        "single_scale_rate": float(single_rates.mean()),
        "single_scale_rate_ci": [
            float(v)
            for v in np.percentile(single_rates[boot[:, : len(single_rates)] % len(single_rates)].mean(axis=1), [2.5, 97.5])
        ],
        "projection_rate": float(proj_rates.mean()),
        "noiseless_single_scale_rate": true_mis_rate(gaps(mu[f_rung]), gaps(mu[t_rung])),
        "between_recipe_sd_at_top_rung": float(mu[f_rung].std(ddof=1)),
        "cell_mean_noise_at_top_rung": cell_noise,
        "spread_to_noise_at_top_rung": float(mu[f_rung].std(ddof=1) / cell_noise),
        "truth_monotonicity": {
            "strict_violations_by_rung": {k: int(v) for k, v in (step > 0).sum(axis=0).items() if v},
            "tolerant_violations_by_rung": {
                k: int(v) for k, v in (step > cp["sigma_table"].iloc[:, 1:]).sum(axis=0).items() if v
            },
            "max_rise_over_sd": float((step / cp["sigma_table"].iloc[:, 1:]).max().max()),
        },
    }
    resolved_own = [t for t in own_tests if t is not None]
    resolved_swap = [t for t in swap_tests if t is not None]
    low_cross = [v for v in measurability.values() if v["crossover_rate"] < 0.10]
    out = {
        "label": "exploratory: characterises the mechanism of metric-specificity, not a decision rule",
        "definitions": {
            "a": "Spearman(recipe-specific residual at the target, true target values)",
            "b": "Spearman(the same residual, top-fitted-rung values; truth and observed)",
            "c": "Spearman(in-sample residual at the top fitted rung from a fit on rungs <= f, top-rung values)",
        },
        "task_1": {
            "own_world": own,
            "own_world_lines_up": {"tests": len(resolved_own), "agree": int(sum(resolved_own))},
            "swaps": swaps,
            "swaps_lines_up": {"tests": len(resolved_swap), "agree": int(sum(resolved_swap))},
        },
        "task_2": {
            "measured_c4_alignment": measured,
            "deviation": (
                "the pre-registered construction (rank-one loadings varied, remainder E fixed) cannot span the range: "
                "the remainder dominates the recipe-specific residual at the target. Its reachable Pearson range is "
                "reported; the sweep instead reassigns the measured recipe-specific residual rows among recipes, "
                "holding the amplitude exactly"
            ),
            "preregistered_construction_reach": reach,
            "sweep": grid,
        },
        "task_3": {
            "per_cell": measurability,
            "b_truth_sign_agrees_where_crossovers_below_10pct": [v["sign_b_truth_matches_a"] for v in low_cross],
            "c_observed_sign_agrees": int(sum(v["sign_c_observed_matches_a"] for v in measurability.values())),
            "b_observed_sign_agrees": int(sum(v["sign_b_observed_matches_a"] for v in measurability.values())),
        },
        "task_4": task4,
    }
    out["predictions"] = {
        "P1_own_world_80pct": sum(resolved_own) >= 0.8 * len(resolved_own),
        "P1_swaps_80pct": sum(resolved_swap) >= 0.8 * len(resolved_swap),
        "P2_monotone_decreasing_each_design": all(
            g["spearman_excess_vs_c"] is not None and g["spearman_excess_vs_c"] <= -0.9 for g in grid.values()
        ),
        "P2_sign_change_in_neutral_band_each_design": all(
            any(-0.25 <= x <= 0.25 for x in g["sign_changes_at_achieved_alignment"]) for g in grid.values()
        ),
        "P3_1_b_agrees_where_crossovers_rare": all(v["sign_b_truth_matches_a"] for v in low_cross),
        "P3_2_proxy_agrees_at_most_8_of_12": out["task_3"]["c_observed_sign_agrees"] <= 8,
        "P4_single_scale_near_chance": task4["single_scale_rate"] >= 0.40,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    _figure(grid, measured)
    print(json.dumps(out["predictions"], indent=1))


def _figure(grid: dict[str, Any], measured: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (d, g) in zip(axes, grid.items()):
        pts = g["points"]
        cs = [float(k) for k in pts]
        ex = [p["excess_pp"] for p in pts.values()]
        lo = [p["excess_pp"] - p["ci"][0] for p in pts.values()]
        hi = [p["ci"][1] - p["excess_pp"] for p in pts.values()]
        colors = ["k" if p["physical"] else "lightgrey" for p in pts.values()]
        ax.errorbar(cs, ex, yerr=[lo, hi], fmt="none", ecolor="grey", lw=0.8)
        ax.scatter(cs, ex, c=colors, zorder=3, s=25)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(measured[d], color="C0", ls="--", lw=1, label="measured C4 (a)")
        ax.set_title(f"{d} {C4_DESIGNS[d][0]} -> {C4_DESIGNS[d][1]}")
        ax.set_xlabel("alignment c (misspecification vs target values)")
    axes[0].set_ylabel("projection excess over single-scale (pp)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
