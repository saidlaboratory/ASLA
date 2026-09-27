"""Tasks 1-3 of PREDICTIONS_TASK_REGIME_MAP.md: the lever-arm mechanism, the regime map, the ensemble dose-response.

Fixed-candidate worlds on the full ladder (750M excluded), with truth built from
the per-recipe power-law fit mu_P, the residual's common mode mean(a) m, and
its recipe-specific part (a - mean(a)) m + E:

* map cells (s, lambda): mu_P + common + lambda * specific, seed sd times s;
* ensemble cells (lambda): mu_P + lambda * R, measured noise;
* OLMES cells: OLMES macro error's own shrunk truth at its measured noise.

World i draws the same standard normals in every cell of a kind (common random
numbers), so cells differ only in their parameters. The s = 0 row is exact:
every ranker runs once on the truth.

Writes ``results/target_scoring/regime_map.json`` and
``results/target_scoring/figures/regime_map.png``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import (  # noqa: E402
    Truth,
    evidence_at_target,
    gaps,
    shrink_truth,
    simulate,
    true_mis_rate,
    truth_from_tables,
)
from asla.analysis.lever_arm import designs  # noqa: E402
from asla.analysis.target_scoring import expected_charges, u_statistic_components, u_statistic_variance  # noqa: E402

CACHE = REPO / "data" / "cache" / "regime_map.jsonl"
OUT = REPO / "results" / "target_scoring" / "regime_map.json"
FIGURE = REPO / "results" / "target_scoring" / "figures" / "regime_map.png"
BASE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
ENSEMBLE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
N_WORLDS = 150
N_BOOT = 2000
PRIMARY_FIT, PRIMARY_TARGET = "300M", "530M"
CALIBRATED_K = 2.99  # results/target_scoring/calibration.json, random-candidate U-statistic
FAMILIES = ("power_law", "saturating", "damped_power_law")
SEED = 20260927
STREAM = {"map": 7, "ens": 8, "olmes": 9}
_CONTEXT: dict[str, dict[str, Any]] = {}


def _frame(metric: str) -> pd.DataFrame:
    if metric == "c4":
        return importlib.import_module("scripts.run_lever_arm_calibration")._context()["frame"]
    built = importlib.import_module("scripts.run_rescoring")._load_checkpoint_builder()("olmes_macro_error", "530M", "1B")
    frame = built["final_table"]
    return frame[frame["scale_label"] != "750M"].reset_index(drop=True)


def build_context(frame: pd.DataFrame, base: Truth | None = None) -> dict[str, Any]:
    """Shrunk truth, per-recipe power-law fits, residual decomposition and the practitioner coordinates.

    ``base`` supplies the truth directly (for a table without seed replicates);
    by default it is the shrunk truth estimated from ``frame``.
    """

    from asla.models import fit_power_law

    if base is None:
        target = float(frame[frame["scale_label"] == "1B"]["compute"].iloc[0])
        base = shrink_truth(truth_from_tables(frame, frame.copy(), target, "1B", rho_ckpt={}))
    compute = base.final.groupby("scale_label")["compute"].first().sort_values()
    scales = list(compute.index)
    table = base.final.groupby(["intervention", "scale_label"])["mu"].first().unstack()[scales]
    recipes = list(table.index)
    params = {r: fit_power_law(compute.to_numpy(dtype=float), table.loc[r].to_numpy(dtype=float)) for r in recipes}
    parametric = pd.DataFrame(
        [[params[r][0] + params[r][1] * compute[s] ** (-params[r][2]) for s in scales] for r in recipes],
        index=recipes,
        columns=scales,
    )
    residual = (table - parametric).to_numpy()
    u, sv, vt = np.linalg.svd(residual, full_matrices=False)
    loadings, shape = u[:, 0] * sv[0], vt[0]
    if loadings.mean() < 0:
        loadings, shape = -loadings, -shape
    common = np.outer(np.full_like(loadings, loadings.mean()), shape)
    specific = residual - common
    spread = float(np.sqrt(np.mean(table.var(axis=0, ddof=1))))
    noise = float(np.sqrt(np.mean([v**2 for v in base.sigma.values()])))
    rms_shape = float(np.sqrt(np.mean(shape**2)))
    coordinates = {
        "between_recipe_spread_T": spread,
        "seed_sd_rms": noise,
        "loading_sd": float(np.std(loadings, ddof=1)),
        "loading_cv": float(np.std(loadings, ddof=1) / abs(np.mean(loadings))),
        "shape_rms": rms_shape,
        "x_noise": noise / spread,
        "y_recipe_specific": float(np.std(loadings, ddof=1)) * rms_shape / spread,
        "alt_cv_times_residual_rms": float(np.std(loadings, ddof=1) / abs(np.mean(loadings)))
        * float(np.sqrt(np.mean(residual**2)))
        / spread,
        "alt_specific_rms": float(np.sqrt(np.mean(specific**2))) / spread,
        "rank_one_share": float(sv[0] ** 2 / np.sum(sv**2)),
    }
    return {
        "base": base,
        "recipes": recipes,
        "scales": scales,
        "compute": compute,
        "parametric": parametric,
        "common": common,
        "specific": specific,
        "residual": residual,
        "coordinates": coordinates,
    }


def context(metric: str) -> dict[str, Any]:
    if metric not in _CONTEXT:
        _CONTEXT[metric] = build_context(_frame(metric))
    return _CONTEXT[metric]


def truth_table(ctx: dict[str, Any], kind: str, lam: float) -> pd.DataFrame:
    """Expected values per recipe and scale for a cell."""

    if kind == "map":
        extra = ctx["common"] + lam * ctx["specific"]
    elif kind == "ens":
        extra = lam * ctx["residual"]
    elif kind == "olmes":
        extra = ctx["residual"]
    elif kind == "A":
        extra = np.zeros_like(ctx["residual"])
    else:
        raise ValueError(kind)
    return ctx["parametric"] + pd.DataFrame(extra, index=ctx["recipes"], columns=ctx["scales"])


def make_truth(ctx: dict[str, Any], mu: pd.DataFrame, noise_scale: float) -> Truth:
    lookup = mu.stack()

    def fill(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["mu"] = [lookup[(r, s)] for r, s in zip(out["intervention"], out["scale_label"])]
        return out

    base: Truth = ctx["base"]
    sigma = {k: v * noise_scale for k, v in base.sigma.items()}
    return replace(base, final=fill(base.final), ckpt=fill(base.ckpt), sigma=sigma)


def primary(ctx: dict[str, Any], observed: pd.DataFrame, mu: pd.DataFrame, with_ensemble: bool) -> dict[str, Any]:
    """True excesses at the primary design, the kernel's Hoeffding components, and the ensemble's family weights."""

    from asla.analysis.ensemble import ensemble_report
    from asla.analysis.rankers import projection_ranker, single_scale_ranker

    compute = ctx["compute"]
    fit_max, target = float(compute[PRIMARY_FIT]), float(compute[PRIMARY_TARGET])
    budgets = tuple(float(c) for c in compute if c <= fit_max)
    table = observed[(observed["compute"] <= fit_max) | np.isclose(observed["compute"], target)]
    truth_gaps = gaps(mu[PRIMARY_TARGET])
    preds = {
        "single": gaps(single_scale_ranker(table, budgets, target)),
        "projection": gaps(projection_ranker(table, budgets, target)),
    }
    out: dict[str, Any] = {}
    if with_ensemble:
        report = ensemble_report(table, budgets, target, None)
        preds["ensemble"] = gaps(pd.Series({r: e["point"] for r, e in report.items()}))
        out["weights"] = {
            r: {f: float(e["families"].get(f, {}).get("weight", 0.0)) for f in FAMILIES} for r, e in report.items()
        }
    mis = {k: true_mis_rate(v, truth_gaps) for k, v in preds.items()}
    out["projection_true_excess"] = 100 * (mis["projection"] - mis["single"])
    if with_ensemble:
        out["ensemble_true_excess"] = 100 * (mis["ensemble"] - mis["single"])
    if observed.groupby(["intervention", "compute"])["bpb"].std().max() > 0:
        evidence = evidence_at_target(table, PRIMARY_TARGET)
        left = expected_charges(preds["projection"], evidence)
        right = expected_charges(preds["single"], evidence)
        comp = u_statistic_components({p: 100 * (left[p] - right[p]) for p in left if p in right})
        out["zeta1"], out["zeta2"] = comp["zeta1"], comp["zeta2"]
    return out


def run_world(metric: str, kind: str, s: float, lam: float, index: int) -> dict[str, Any]:
    ctx = context(metric)
    mu = truth_table(ctx, kind, lam)
    truth = make_truth(ctx, mu, s)
    rng = np.random.default_rng([SEED, STREAM[kind], index])
    observed, _ = simulate(truth, rng)
    targets = {label: mu[label] for label in ctx["scales"]}
    world = designs(observed, targets)
    return {
        "kind": kind,
        "s": s,
        "lam": lam,
        "index": index,
        "true_excess": [d.true_excess for d in world],
        **primary(ctx, observed, mu, with_ensemble=kind == "ens"),
    }


def noiseless_designs(ctx: dict[str, Any], mu: pd.DataFrame) -> list[dict[str, Any]]:
    """Every design run once on the truth: per-rule error counts and the excess (the noiseless column)."""

    from asla.models import FitError, bpb_power_law, fit_power_law

    scales, compute = ctx["scales"], ctx["compute"]
    rows = []
    for t in range(3, len(scales)):
        truth_gaps = gaps(mu[scales[t]])
        for f in range(2, t):
            budgets = compute.iloc[: f + 1].to_numpy(dtype=float)
            projected = {}
            for name in mu.index:
                try:
                    params = fit_power_law(budgets, mu.loc[name].iloc[: f + 1].to_numpy(dtype=float))
                    projected[name] = float(bpb_power_law(float(compute.iloc[t]), *params))
                except (FitError, RuntimeError, ValueError):
                    projected[name] = float("nan")
            counts = {}
            for rule, values in (("projection", pd.Series(projected)), ("single", mu[scales[f]])):
                pred = gaps(values)
                scored = [(p, truth_gaps[k]) for k, p in pred.items() if np.isfinite(p) and p != 0 and truth_gaps[k] != 0]
                counts[rule] = (sum(np.sign(p) != np.sign(q) for p, q in scored), len(scored))
            rows.append(
                {
                    "fit": scales[f],
                    "target": scales[t],
                    "lever_arm": float(compute.iloc[t] / compute.iloc[f]),
                    "projection_wrong": int(counts["projection"][0]),
                    "projection_scored": int(counts["projection"][1]),
                    "single_wrong": int(counts["single"][0]),
                    "single_scored": int(counts["single"][1]),
                    "excess": 100
                    * (counts["projection"][0] / counts["projection"][1] - counts["single"][0] / counts["single"][1]),
                }
            )
    return rows


def noiseless_ensemble(ctx: dict[str, Any], mu: pd.DataFrame) -> dict[str, Any]:
    """The ensemble on the truth itself at the primary design: excess and per-recipe family weights."""

    frame = ctx["base"].final.copy()
    frame["bpb"] = [mu.loc[r, s] for r, s in zip(frame["intervention"], frame["scale_label"])]
    return primary(ctx, frame, mu, with_ensemble=True)


def candidates_needed(zeta1: float, zeta2: float, delta: float = 2.0, k: float = CALIBRATED_K) -> int | None:
    """Candidates for power 0.8 at a true difference of ``delta`` points; None if not reached below 5000."""

    for n in range(4, 5000):
        variance = u_statistic_variance(zeta1, zeta2, n)
        if variance <= 0:
            return n
        se = float(np.sqrt(variance))
        if stats.norm.cdf(delta / se - k) + stats.norm.cdf(-delta / se - k) >= 0.8:
            return n
    return None


def crossings(xs: list[float], ys: list[float]) -> list[float]:
    """Zeros of the piecewise-linear interpolant through (xs, ys)."""

    out = []
    for (x0, y0), (x1, y1) in pairwise(zip(xs, ys)):
        if y0 == 0:
            out.append(x0)
        elif y0 * y1 < 0:
            out.append(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    if ys and ys[-1] == 0:
        out.append(xs[-1])
    return out


def ensemble_verdict(excess: list[float], se: list[float], weights: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    """The pre-registered rule: family switching, smooth peak, or indeterminate."""

    span = max(excess) - min(excess)
    steps = []
    for i in range(len(excess) - 1):
        jump = excess[i + 1] - excess[i]
        combined = float(np.hypot(se[i], se[i + 1]))
        recipes = weights[i].keys()
        modal_changes = sum(
            max(weights[i][r], key=weights[i][r].get) != max(weights[i + 1][r], key=weights[i + 1][r].get) for r in recipes
        )
        big_shift = sum(any(abs(weights[i + 1][r][f] - weights[i][r][f]) >= 0.3 for f in FAMILIES) for r in recipes)
        steps.append(
            {
                "jump": jump,
                "jump_over_combined_se": jump / combined if combined > 0 else float("inf"),
                "jump_over_half_range": abs(jump) / (span / 2) if span > 0 else 0.0,
                "recipes_changing_modal_family": int(modal_changes),
                "recipes_with_weight_shift_0_3": int(big_shift),
            }
        )
    switching = [
        i
        for i, st in enumerate(steps)
        if st["jump_over_half_range"] > 1
        and abs(st["jump_over_combined_se"]) > 4
        and (st["recipes_changing_modal_family"] >= 3 or st["recipes_with_weight_shift_0_3"] >= 3)
    ]
    smooth = all(st["jump_over_half_range"] <= 1 and st["recipes_with_weight_shift_0_3"] == 0 for st in steps)
    verdict = "family_switching" if switching else ("smooth_peak" if smooth else "indeterminate")
    return {"steps": steps, "switching_steps": switching, "verdict": verdict}


def _load() -> dict[tuple[str, float, float, int], dict[str, Any]]:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["kind"], row["s"], row["lam"], row["index"])] = row
    return done


def grids(c4: dict[str, Any], olmes: dict[str, Any]) -> tuple[list[float], list[float]]:
    """The base grid, extended by doubling on an axis until OLMES's coordinate is bracketed."""

    s_grid, lam_grid = list(BASE_GRID), list(BASE_GRID)
    s_o = olmes["x_noise"] / c4["x_noise"]
    lam_o = olmes["y_recipe_specific"] / c4["y_recipe_specific"]
    while s_grid[-1] < s_o:
        s_grid.append(2 * s_grid[-1])
    while lam_grid[-1] < lam_o:
        lam_grid.append(2 * lam_grid[-1])
    return s_grid, lam_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    c4, olmes = context("c4"), context("olmes")
    s_grid, lam_grid = grids(c4["coordinates"], olmes["coordinates"])
    print("grid s", s_grid, "lambda", lam_grid, flush=True)
    jobs = [("c4", "map", s, lam, i) for s in s_grid if s > 0 for lam in lam_grid for i in range(N_WORLDS)]
    jobs += [("c4", "ens", 1.0, lam, i) for lam in ENSEMBLE_GRID for i in range(N_WORLDS)]
    jobs += [("olmes", "olmes", 1.0, 1.0, i) for i in range(N_WORLDS)]
    done = _load()
    todo = [j for j in jobs if j[1:] not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
            futures = [pool.submit(run_world, *j) for j in todo]
            for count, future in enumerate(as_completed(futures), start=1):
                sink.write(json.dumps(future.result()) + "\n")
                sink.flush()
                if count % 100 == 0:
                    print(f"  {count}/{len(todo)}", flush=True)
    summarise(_load(), c4, olmes, s_grid, lam_grid)


def summarise(
    cache: dict[tuple[str, float, float, int], dict[str, Any]],
    c4: dict[str, Any],
    olmes: dict[str, Any],
    s_grid: list[float],
    lam_grid: list[float],
) -> None:
    rng = np.random.default_rng(SEED)
    boot_index = rng.integers(0, N_WORLDS, size=(N_BOOT, N_WORLDS))
    lever = np.array([row["lever_arm"] for row in noiseless_designs(c4, truth_table(c4, "A", 0.0))])
    order = np.argsort(lever)
    primary_index = int(
        next(
            i
            for i, row in enumerate(noiseless_designs(c4, truth_table(c4, "A", 0.0)))
            if row["fit"] == PRIMARY_FIT and row["target"] == PRIMARY_TARGET
        )
    )
    tercile = np.array_split(order, 3)
    design_sets = {
        "short": [int(order[0])],
        "primary": [primary_index],
        "long": [int(order[-1])],
        "low_tercile": [int(i) for i in tercile[0]],
        "high_tercile": [int(i) for i in tercile[-1]],
    }
    log_lever = np.log(lever)

    # 1a: noiseless arm A, error counts per design.
    arm_a = noiseless_designs(c4, truth_table(c4, "A", 0.0))
    single_wrong = np.array([r["single_wrong"] for r in arm_a])
    task_1a = {
        "definition": (
            "truth table mu(r, s) used as the data (no seeds, no noise); for each of the 55 designs (top fitted rung f, "
            "target t, t >= 3, 2 <= f < t) projection fits E + A C^-alpha to mu(r, rungs 0..f) and evaluates at C_t, "
            "single-scale ranks by mu(r, f); each rule's mis-selection rate is the fraction of the 300 pairs whose "
            "predicted sign differs from sign(mu(r,t) - mu(r',t)), dropping zero or non-finite gaps; excess = 100 x "
            "(projection - single-scale); the column is the Spearman rho of excess on log(C_t / C_f) across designs"
        ),
        "designs": arm_a,
        "max_projection_wrong": int(max(r["projection_wrong"] for r in arm_a)),
        "designs_with_projection_wrong": int(sum(r["projection_wrong"] > 0 for r in arm_a)),
        "single_wrong_spearman_with_log_lever": float(stats.spearmanr(log_lever, single_wrong).statistic),
        "single_wrong_range": [int(single_wrong.min()), int(single_wrong.max())],
        "excess_rho_recomputed": float(stats.spearmanr(log_lever, [r["excess"] for r in arm_a]).statistic),
    }

    # 1b: the map.
    def cell_matrix(kind: str, s: float, lam: float) -> np.ndarray:
        return np.array([cache[(kind, s, lam, i)]["true_excess"] for i in range(N_WORLDS)], dtype=float)

    exact_row = {
        lam: np.array([r["excess"] for r in noiseless_designs(c4, truth_table(c4, "map", lam))]) for lam in lam_grid
    }
    means: dict[str, np.ndarray] = {}
    boots: dict[str, np.ndarray] = {}
    cells: dict[str, Any] = {}
    for name, idx in design_sets.items():
        grid = np.zeros((len(s_grid), len(lam_grid)))
        boot = np.zeros((N_BOOT, len(s_grid), len(lam_grid)))
        for a, s in enumerate(s_grid):
            for b, lam in enumerate(lam_grid):
                if s == 0:
                    grid[a, b] = boot[:, a, b] = float(exact_row[lam][idx].mean())
                else:
                    per_world = cell_matrix("map", s, lam)[:, idx].mean(axis=1)
                    grid[a, b] = per_world.mean()
                    boot[:, a, b] = per_world[boot_index].mean(axis=1)
        means[name], boots[name] = grid, boot
    for a, s in enumerate(s_grid):
        for b, lam in enumerate(lam_grid):
            entry: dict[str, Any] = {"s": s, "lambda": lam}
            for name in design_sets:
                value = float(means[name][a, b])
                if s == 0:
                    lo = hi = value
                else:
                    lo, hi = (float(v) for v in np.percentile(boots[name][:, a, b], [2.5, 97.5]))
                entry[name] = {
                    "excess_pp": value,
                    "ci": [lo, hi],
                    "sign": "wins" if hi < 0 else ("loses" if lo > 0 else "unresolved"),
                }
            if s == 0:
                entry["slope_on_log_lever"] = float(np.polyfit(log_lever, exact_row[lam], 1)[0])
            else:
                entry["slope_on_log_lever"] = float(np.polyfit(log_lever, cell_matrix("map", s, lam).mean(axis=0), 1)[0])
            cells[f"s={s:g},lambda={lam:g}"] = entry

    def contour(name: str, grid: np.ndarray) -> dict[str, Any]:
        return {
            "lambda_star_by_s": {f"{s:g}": crossings(lam_grid, list(grid[a])) for a, s in enumerate(s_grid)},
            "s_star_by_lambda": {f"{lam:g}": crossings(s_grid, list(grid[:, b])) for b, lam in enumerate(lam_grid)},
        }

    contours = {}
    for name in ("short", "primary", "long"):
        point = contour(name, means[name])
        intervals: dict[str, Any] = {"lambda_star_by_s": {}, "s_star_by_lambda": {}}
        for axis, labels, getter in (
            ("lambda_star_by_s", s_grid, lambda g, a: crossings(lam_grid, list(g[a]))),
            ("s_star_by_lambda", lam_grid, lambda g, b: crossings(s_grid, list(g[:, b]))),
        ):
            for pos, label in enumerate(labels):
                found = [getter(boots[name][k], pos) for k in range(N_BOOT)]
                firsts = [f[0] for f in found if f]
                intervals[axis][f"{label:g}"] = {
                    "share_of_bootstraps_with_crossing": len(firsts) / N_BOOT,
                    "ci_first_crossing": [float(np.percentile(firsts, 2.5)), float(np.percentile(firsts, 97.5))]
                    if firsts
                    else None,
                }
        contours[name] = {"point": point, "bootstrap": intervals}
    winning = {name: int((means[name] < 0).sum()) for name in ("short", "primary", "long")}
    significant_wins = [
        (key, name)
        for key, entry in cells.items()
        for name in ("short", "primary", "long")
        if entry[name]["sign"] == "wins" or (entry["s"] == 0 and entry[name]["excess_pp"] < 0)
    ]
    corner = all(cells[key]["s"] <= 0.5 and cells[key]["lambda"] <= 0.5 for key, _ in significant_wins)
    zero_row = {name: int((means[name][0] < 0).sum()) for name in ("short", "primary", "long")}

    # 1c: placement.
    s_o = olmes["coordinates"]["x_noise"] / c4["coordinates"]["x_noise"]
    lam_o = olmes["coordinates"]["y_recipe_specific"] / c4["coordinates"]["y_recipe_specific"]

    def interp_lambda(grid: np.ndarray, lam: float) -> np.ndarray:
        return np.array([np.interp(lam, lam_grid, grid[a]) for a in range(len(s_grid))])

    def interp_s(grid: np.ndarray, s: float) -> np.ndarray:
        return np.array([np.interp(s, s_grid, grid[:, b]) for b in range(len(lam_grid))])

    olmes_rows = [cache[("olmes", 1.0, 1.0, i)] for i in range(N_WORLDS)]
    olmes_excess = np.array([r["true_excess"] for r in olmes_rows], dtype=float)
    placement: dict[str, Any] = {
        "c4": {"coordinates": c4["coordinates"], "map_position": [1.0, 1.0]},
        "olmes": {"coordinates": olmes["coordinates"], "map_position": [s_o, lam_o]},
        "by_design": {},
    }
    for name in ("short", "primary", "long"):
        grid = means[name]
        by: dict[str, Any] = {}
        for metric, (s_m, lam_m) in (("c4", (1.0, 1.0)), ("olmes", (s_o, lam_o))):
            column = interp_lambda(grid, lam_m)
            row = interp_s(grid, s_m)
            s_star = crossings(s_grid, list(column))
            lam_star = crossings(lam_grid, list(row))
            map_value = float(np.interp(s_m, s_grid, column))
            by[metric] = {
                "map_excess_pp": map_value,
                "map_sign": "wins" if map_value < 0 else "loses",
                "s_star_at_metric_lambda": s_star,
                "lambda_star_at_metric_s": lam_star,
                "noise_ratio_to_boundary": s_m / s_star[0] if s_star else None,
                "lambda_ratio_to_boundary": lam_m / lam_star[0] if lam_star else None,
            }
        per_world = olmes_excess[:, design_sets[name]].mean(axis=1)
        lo, hi = np.percentile(per_world[boot_index].mean(axis=1), [2.5, 97.5])
        by["olmes_worlds"] = {
            "excess_pp": float(per_world.mean()),
            "ci": [float(lo), float(hi)],
            "sign": "wins" if hi < 0 else ("loses" if lo > 0 else "unresolved"),
        }
        placement["by_design"][name] = by

    # Task 2: ensemble dose-response.
    arms = json.loads((REPO / "results" / "target_scoring" / "lever_arm_arms.json").read_text())["arms"]
    existing = {
        arm: {
            "excess_pp": a["ensemble_true_excess_pp"],
            "ci": [
                a["ensemble_true_excess_pp"] - 1.959963984540054 * a["ensemble_true_excess_se"],
                a["ensemble_true_excess_pp"] + 1.959963984540054 * a["ensemble_true_excess_se"],
            ],
        }
        for arm, a in arms.items()
    }
    dose, weight_means, se_list, ex_list = {}, [], [], []
    for lam in ENSEMBLE_GRID:
        rows = [cache[("ens", 1.0, lam, i)] for i in range(N_WORLDS)]
        ens = np.array([r["ensemble_true_excess"] for r in rows])
        recipes = sorted(rows[0]["weights"])
        mean_w = {r: {f: float(np.mean([row["weights"][r][f] for row in rows])) for f in FAMILIES} for r in recipes}
        modal = {
            r: max(FAMILIES, key=lambda f, r=r: sum(max(row["weights"][r], key=row["weights"][r].get) == f for row in rows))
            for r in recipes
        }
        exact = noiseless_ensemble(c4, truth_table(c4, "ens", lam))
        se = float(ens.std(ddof=1) / np.sqrt(len(ens)))
        dose[f"{lam:g}"] = {
            "ensemble_excess_pp": float(ens.mean()),
            "ci": [float(ens.mean() - 1.96 * se), float(ens.mean() + 1.96 * se)],
            "se": se,
            "projection_excess_pp": float(np.mean([r["projection_true_excess"] for r in rows])),
            "mean_weights": mean_w,
            "modal_top_family": modal,
            "top_family_counts": {f: sum(m == f for m in modal.values()) for f in FAMILIES},
            "noiseless_ensemble_excess_pp": exact["ensemble_true_excess"],
            "noiseless_top_family": {r: max(w, key=w.get) for r, w in exact["weights"].items()},
            "noiseless_weights": exact["weights"],
        }
        weight_means.append(mean_w)
        se_list.append(se)
        ex_list.append(float(ens.mean()))
    verdict = ensemble_verdict(ex_list, se_list, weight_means)
    noiseless_switches = [
        sum(
            dose[f"{a:g}"]["noiseless_top_family"][r] != dose[f"{b:g}"]["noiseless_top_family"][r]
            for r in dose[f"{a:g}"]["noiseless_top_family"]
        )
        for a, b in pairwise(ENSEMBLE_GRID)
    ]
    one = ENSEMBLE_GRID.index(1.0)
    peak_vs_neighbours = [
        (ex_list[one] - ex_list[j]) / float(np.hypot(se_list[one], se_list[j])) for j in (one - 1, one + 1)
    ]

    # Task 3: candidates across the recipe-specific grid at measured noise.
    by_lambda = {}
    for lam in lam_grid:
        rows = [cache[("map", 1.0, lam, i)] for i in range(N_WORLDS)]
        z1, z2 = float(np.mean([r["zeta1"] for r in rows])), float(np.mean([r["zeta2"] for r in rows]))
        by_lambda[f"{lam:g}"] = {"zeta1": z1, "zeta2": z2, "candidates_for_2pp": candidates_needed(z1, z2)}
    arm_rows = [
        json.loads(line)
        for line in (REPO / "data" / "cache" / "lever_arm_arms.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line)["arm"] == "B3_1"
    ]
    per_world_n = np.array([candidates_needed(r["zeta1"], r["zeta2"]) or 5000 for r in arm_rows], dtype=float)
    expected = json.loads((REPO / "results" / "target_scoring" / "expected_error.json").read_text())
    real = expected["regimes"]["c4_en_bits_per_token/primary_4M-300M_gate530M"]["significance"]["u_statistic"]
    real_n = candidates_needed(real["zeta1"], real["zeta2"])
    sim_z1 = np.array([r["zeta1"] for r in arm_rows])
    sim_z2 = np.array([r["zeta2"] for r in arm_rows])
    lo_n, hi_n = (float(v) for v in np.percentile(per_world_n, [2.5, 97.5]))
    reconciliation = {
        "real_data": {
            "source": "expected_error.json regimes.c4_en_bits_per_token/primary_4M-300M_gate530M, one realisation",
            "zeta1": real["zeta1"],
            "zeta2": real["zeta2"],
            "candidates_for_2pp": real_n,
        },
        "simulated_B3_1": {
            "source": "lever_arm_arms.jsonl arm B3_1, 150 fixed-candidate worlds on the shrunk truth",
            "zeta1_mean": float(sim_z1.mean()),
            "zeta2_mean": float(sim_z2.mean()),
            "candidates_from_mean_zeta": candidates_needed(float(sim_z1.mean()), float(sim_z2.mean())),
            "per_world_candidates_median": float(np.median(per_world_n)),
            "per_world_candidates_95": [lo_n, hi_n],
            "real_zeta1_percentile": float(stats.percentileofscore(sim_z1, real["zeta1"])),
            "real_zeta2_percentile": float(stats.percentileofscore(sim_z2, real["zeta2"])),
        },
        "real_inside_simulated_95": bool(lo_n <= (real_n or 5000) <= hi_n),
    }
    reconciliation["quote"] = (
        "real-data figure, with the simulated per-world 95% range as its uncertainty"
        if reconciliation["real_inside_simulated_95"]
        else "real-data figure, with the discrepancy to the simulation reported"
    )
    n_seq = [by_lambda[f"{lam:g}"]["candidates_for_2pp"] or 5000 for lam in lam_grid]

    # Task 5: attribution bases.
    attribution = json.loads((REPO / "results" / "target_scoring" / "lever_arm_arms.json").read_text())["attribution"]
    model_share = attribution["share_of_real_slope"]["B3_1_total"]
    task5 = {
        "components_share_of_B3_1_model_slope": attribution["share_of_B3_1_slope"],
        "model_slope_share_of_real_slope": model_share,
        "components_share_of_real_slope": {k: v * model_share for k, v in attribution["share_of_B3_1_slope"].items()},
        "unexplained_share_of_real_slope": attribution["share_of_real_slope"]["unexplained"],
        "B3_1_model_slope_pp_per_log": arms["B3_1"]["true_slope_pp_per_log"],
        "real_slope_pp_per_log": attribution["real_slope_pp_per_log"],
    }

    out = {
        "design": "fixed-candidate worlds, full ladder (750M excluded), common random numbers within each cell kind",
        "grid": {"s": s_grid, "lambda": lam_grid, "worlds_per_cell": N_WORLDS},
        "designs": {k: {"indices": v, "lever_arm": [float(lever[i]) for i in v]} for k, v in design_sets.items()},
        "task_1a": task_1a,
        "task_1b": {
            "cells": cells,
            "contours": contours,
            "winning_cells_point_estimate": winning,
            "winning_cells_noise_free_row": zero_row,
        },
        "task_1c": placement,
        "task_2": {
            "existing_arms_with_intervals": existing,
            "dose_response": dose,
            "verdict": verdict,
            "noiseless_top_family_changes_between_adjacent": noiseless_switches,
            "peak_at_1_over_neighbours_in_combined_se": peak_vs_neighbours,
        },
        "task_3": {"at_measured_noise_by_lambda": by_lambda, "reconciliation": reconciliation},
        "task_5": task5,
    }
    primary_c4, primary_o = placement["by_design"]["primary"]["c4"], placement["by_design"]["primary"]["olmes"]
    out["predictions"] = {
        "P1a_projection_at_most_1_wrong": task_1a["max_projection_wrong"] <= 1,
        "P1a_single_wrong_rho_at_least_0_8": task_1a["single_wrong_spearman_with_log_lever"] >= 0.8,
        "P1b1_wins_only_in_low_corner": corner,
        "P1b2_winning_region_shrinks_with_L": winning["short"] >= winning["primary"] >= winning["long"],
        "P1b3_noise_free_row_does_not_shrink": zero_row["long"] >= zero_row["short"],
        "P1c1_both_lose_at_primary": primary_c4["map_sign"] == "loses" and primary_o["map_sign"] == "loses",
        "P1c2_olmes_noisier_than_c4": s_o > 1,
        "P1c3_olmes_worlds_match_map_sign": all(
            placement["by_design"][n]["olmes_worlds"]["sign"] == placement["by_design"][n]["olmes"]["map_sign"]
            for n in ("short", "primary", "long")
        ),
        "P2_family_switching": verdict["verdict"] == "family_switching"
        and all(v > 4 for v in peak_vs_neighbours)
        and bool({one - 1, one} & set(verdict["switching_steps"])),
        "P3_1_monotone_in_lambda": all(a <= b for a, b in pairwise(n_seq)),
        "P3_2_real_inside_simulated_95": reconciliation["real_inside_simulated_95"],
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _figure(means, s_grid, lam_grid, (s_o, lam_o), [lever[design_sets[n][0]] for n in ("short", "primary", "long")])
    print(json.dumps(out["predictions"], indent=1))


def _figure(
    means: dict[str, np.ndarray], s_grid: list[float], lam_grid: list[float], olmes: tuple[float, float], levers: list[float]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, name, lever in zip(axes, ("short", "primary", "long"), levers):
        grid = means[name]
        bound = float(np.nanmax(np.abs(grid))) or 1.0
        x = np.arange(len(lam_grid))
        y = np.arange(len(s_grid))
        image = ax.imshow(grid, origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.contour(x, y, grid, levels=[0.0], colors="k", linewidths=1.5)
        for a in range(len(s_grid)):
            for b in range(len(lam_grid)):
                ax.text(b, a, f"{grid[a, b]:+.1f}", ha="center", va="center", fontsize=7)
        ax.plot(np.interp(1.0, lam_grid, x), np.interp(1.0, s_grid, y), "k*", mfc="none", ms=14, label="C4")
        ax.plot(np.interp(olmes[1], lam_grid, x), np.interp(olmes[0], s_grid, y), "ko", mfc="none", ms=10, label="OLMES")
        ax.set_xticks(x, [f"{v:g}" for v in lam_grid])
        ax.set_yticks(y, [f"{v:g}" for v in s_grid])
        ax.set_xlabel("recipe-specific amplitude (x measured C4)")
        ax.set_title(f"{name}, L = {lever:.3g}")
        fig.colorbar(image, ax=ax, shrink=0.8, label="projection excess (pp)")
    axes[0].set_ylabel("seed-noise scale (x measured C4)")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
