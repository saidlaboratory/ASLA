"""Task 1 of PREDICTIONS_TASK_LEVER_GAP.md: what produces the lever-arm effect?

Fixed-candidate worlds on the full DataDecide ladder (C4, 750M excluded), with
the real 25 recipes' structure and the moderated noise model. Truth varies by
arm:

* A: each recipe's power-law fit to its shrunk curve (variance only);
* B1: A plus the common-mode misspecification, mean(a) m(C);
* B2: A plus the recipe-specific part, (a_r - mean(a)) m(C) + E_r(C);
* B3(lambda): A plus lambda times the full residual R = a m^T + E.

B3(1) is the real shrunk truth, the 1a check. R = a m^T + E is the rank-one SVD of
the residual matrix. Every world also records the ensemble-versus-single-scale
true difference at the primary design, and the variance components of the
projection kernel there.

Writes ``results/target_scoring/lever_arm_arms.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import Truth, evidence_at_target, gaps, simulate, true_mis_rate  # noqa: E402
from asla.analysis.lever_arm import designs, loro_test  # noqa: E402
from asla.analysis.target_scoring import expected_charges, u_statistic_components, u_statistic_variance  # noqa: E402

CACHE = REPO / "data" / "cache" / "lever_arm_arms.jsonl"
OUT = REPO / "results" / "target_scoring" / "lever_arm_arms.json"
ARMS = ("A", "B1", "B2", "B3_0.5", "B3_1", "B3_2")
N_WORLDS = 150
N_BOOT = 2000
PRIMARY_FIT, PRIMARY_TARGET = "300M", "530M"
CALIBRATED_K = 2.99  # results/target_scoring/calibration.json, random-candidate U-statistic
SEED = 20260926
_CONTEXT: dict[str, Any] = {}


def _context() -> dict[str, Any]:
    if _CONTEXT:
        return _CONTEXT
    calibration = __import__("scripts.run_lever_arm_calibration", fromlist=["_context"])
    ctx = calibration._context()
    base: Truth = ctx["base"]
    table = base.final.groupby(["intervention", "scale_label"])["mu"].first().unstack()
    compute = base.final.groupby("scale_label")["compute"].first()
    scales = list(compute.sort_values().index)
    table = table[scales]
    recipes = list(table.index)
    parametric = pd.DataFrame(
        [
            [ctx["params"][r][0] + ctx["params"][r][1] * compute[s] ** (-ctx["params"][r][2]) for s in scales]
            for r in recipes
        ],
        index=recipes,
        columns=scales,
    )
    residual = table - parametric
    u, sv, vt = np.linalg.svd(residual.to_numpy(), full_matrices=False)
    loadings, shape = u[:, 0] * sv[0], vt[0]
    if loadings.mean() < 0:
        loadings, shape = -loadings, -shape
    common = np.outer(np.full_like(loadings, loadings.mean()), shape)
    rank_one = np.outer(loadings, shape)
    remainder = residual.to_numpy() - rank_one
    arm_residual = {
        "A": np.zeros_like(rank_one),
        "B1": common,
        "B2": rank_one - common + remainder,
        "B3_0.5": 0.5 * residual.to_numpy(),
        "B3_1": residual.to_numpy(),
        "B3_2": 2.0 * residual.to_numpy(),
    }
    _CONTEXT.update(
        base=base,
        recipes=recipes,
        scales=scales,
        compute=compute,
        parametric=parametric,
        arm_residual=arm_residual,
        loading_cv=float(np.std(loadings, ddof=1) / abs(np.mean(loadings))),
        rank_one_share=float(sv[0] ** 2 / np.sum(sv**2)),
        frame=ctx["frame"],
    )
    return _CONTEXT


def arm_truth(arm: str) -> Truth:
    ctx = _context()
    mu = ctx["parametric"] + pd.DataFrame(ctx["arm_residual"][arm], index=ctx["recipes"], columns=ctx["scales"])
    lookup = mu.stack()

    def fill(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["mu"] = [lookup[(r, s)] for r, s in zip(out["intervention"], out["scale_label"])]
        return out

    base: Truth = ctx["base"]
    return replace(base, final=fill(base.final), ckpt=fill(base.ckpt))


def _primary(observed: pd.DataFrame, truth: Truth) -> dict[str, Any]:
    from asla.analysis.rankers import ensemble_ranker, projection_ranker, single_scale_ranker

    ctx = _context()
    fit_max = float(ctx["compute"][PRIMARY_FIT])
    target = float(ctx["compute"][PRIMARY_TARGET])
    budgets = tuple(float(c) for c in ctx["compute"] if c <= fit_max)
    table = observed[(observed["compute"] <= fit_max) | np.isclose(observed["compute"], target)]
    truth_target = truth.final[truth.final["scale_label"] == PRIMARY_TARGET].groupby("intervention")["mu"].first()
    truth_gaps = gaps(truth_target)
    preds = {
        "single": gaps(single_scale_ranker(table, budgets, target)),
        "projection": gaps(projection_ranker(table, budgets, target)),
        "ensemble": gaps(ensemble_ranker(table, budgets, target)),
    }
    evidence = evidence_at_target(table, PRIMARY_TARGET)
    base_charges = expected_charges(preds["single"], evidence)
    proj_charges = expected_charges(preds["projection"], evidence)
    kernel = {p: 100 * (proj_charges[p] - base_charges[p]) for p in proj_charges if p in base_charges}
    comp = u_statistic_components(kernel)
    mis = {k: true_mis_rate(v, truth_gaps) for k, v in preds.items()}
    return {
        "ensemble_true_excess": 100 * (mis["ensemble"] - mis["single"]),
        "projection_true_excess": 100 * (mis["projection"] - mis["single"]),
        "zeta1": comp["zeta1"],
        "zeta2": comp["zeta2"],
    }


def run_world(arm: str, index: int) -> dict[str, Any]:
    truth = arm_truth(arm)
    rng = np.random.default_rng([SEED, ARMS.index(arm), index])
    observed, _ = simulate(truth, rng)
    targets = {label: g.groupby("intervention")["mu"].first() for label, g in truth.final.groupby("scale_label")}
    world_designs = designs(observed, targets)
    test = loro_test(world_designs)
    return {
        "arm": arm,
        "index": index,
        "lever_arm": [d.lever_arm for d in world_designs],
        "true_excess": [d.true_excess for d in world_designs],
        "rho_hat": test["rho"],
        **_primary(observed, truth),
    }


def noiseless(arm: str) -> dict[str, Any]:
    """Every ranker run once on the truth itself: the lever-arm effect with no noise."""

    from asla.models import FitError, bpb_power_law, fit_power_law

    ctx = _context()
    truth = arm_truth(arm)
    mu = truth.final.groupby(["intervention", "scale_label"])["mu"].first().unstack()[ctx["scales"]]
    compute = ctx["compute"][ctx["scales"]]
    names = list(mu.index)
    rows = []
    for t in range(3, len(ctx["scales"])):
        target = ctx["scales"][t]
        truth_gaps = gaps(mu[target])
        for f in range(2, t):
            budgets = compute.iloc[: f + 1].to_numpy(dtype=float)
            projected = {}
            for name in names:
                try:
                    params = fit_power_law(budgets, mu.loc[name].iloc[: f + 1].to_numpy(dtype=float))
                    projected[name] = float(bpb_power_law(float(compute[target]), *params))
                except (FitError, RuntimeError, ValueError):
                    projected[name] = float("nan")
            single = mu[ctx["scales"][f]]
            excess = 100 * (true_mis_rate(gaps(pd.Series(projected)), truth_gaps) - true_mis_rate(gaps(single), truth_gaps))
            rows.append((float(compute[target] / compute.iloc[f]), excess))
    lever, excess = np.array(rows).T
    return {
        "rho": float(stats.spearmanr(np.log(lever), excess).statistic),
        "slope": float(np.polyfit(np.log(lever), excess, 1)[0]),
    }


def _load() -> dict[tuple[str, int], dict[str, Any]]:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["arm"], row["index"])] = row
    return done


def summarise_arm(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    log_lever = np.log(rows[0]["lever_arm"])
    excess = np.array([r["true_excess"] for r in rows])

    def stats_of(block: np.ndarray) -> tuple[float, float]:
        mean = block.mean(axis=0)
        return float(stats.spearmanr(log_lever, mean).statistic), float(np.polyfit(log_lever, mean, 1)[0])

    rho, slope = stats_of(excess)
    boots = [stats_of(excess[rng.integers(0, len(rows), len(rows))]) for _ in range(N_BOOT)]
    rhos, slopes = np.array(boots).T
    zeta1 = float(np.mean([r["zeta1"] for r in rows]))
    zeta2 = float(np.mean([r["zeta2"] for r in rows]))
    needed = None
    for n in range(4, 5000):
        se = float(np.sqrt(u_statistic_variance(zeta1, zeta2, n)))
        if stats.norm.cdf(2.0 / se - CALIBRATED_K) + stats.norm.cdf(-2.0 / se - CALIBRATED_K) >= 0.8:
            needed = n
            break
    ens = np.array([r["ensemble_true_excess"] for r in rows])
    return {
        "n_worlds": len(rows),
        "true_rho": rho,
        "true_rho_ci": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
        "true_slope_pp_per_log": slope,
        "true_slope_ci": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "mean_rho_hat": float(np.mean([r["rho_hat"] for r in rows])),
        "ensemble_true_excess_pp": float(ens.mean()),
        "ensemble_true_excess_se": float(ens.std(ddof=1) / np.sqrt(len(ens))),
        "projection_true_excess_primary_pp": float(np.mean([r["projection_true_excess"] for r in rows])),
        "zeta1": zeta1,
        "zeta2": zeta2,
        "candidates_for_2pp_power_0_8": needed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load()
    todo = [(arm, i) for arm in ARMS for i in range(N_WORLDS) if (arm, i) not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        futures = [pool.submit(run_world, arm, i) for arm, i in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            sink.write(json.dumps(future.result()) + "\n")
            sink.flush()
            if count % 100 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    cache = _load()
    rng = np.random.default_rng(SEED)
    ctx = _context()
    arms = {}
    for arm in ARMS:
        rows = [cache[(arm, i)] for i in range(N_WORLDS)]
        arms[arm] = {**summarise_arm(rows, rng), "noiseless": noiseless(arm)}
    real = designs(ctx["frame"])
    real_log = np.log([d.lever_arm for d in real])
    real_excess = np.array([np.mean(list(d.kernel.values())) for d in real])
    real_slope = float(np.polyfit(real_log, real_excess, 1)[0])
    s = {arm: arms[arm]["true_slope_pp_per_log"] for arm in ARMS}
    attribution = {
        "real_slope_pp_per_log": real_slope,
        "real_rho": float(stats.spearmanr(real_log, real_excess).statistic),
        "share_of_B3_1_slope": {
            "A_variance": s["A"] / s["B3_1"],
            "B1_common_mode_increment": (s["B1"] - s["A"]) / s["B3_1"],
            "B2_recipe_specific_increment": (s["B2"] - s["A"]) / s["B3_1"],
            "interaction_remainder": (s["B3_1"] - s["B1"] - s["B2"] + s["A"]) / s["B3_1"],
        },
        "share_of_real_slope": {
            "A_variance": s["A"] / real_slope,
            "B3_1_total": s["B3_1"] / real_slope,
            "unexplained": (real_slope - s["B3_1"]) / real_slope,
        },
    }
    out = {
        "design": "fixed-candidate worlds, full C4 ladder (750M excluded), moderated noise; arms differ only in truth",
        "decomposition": {"loading_cv": ctx["loading_cv"], "rank_one_share_of_residual_ss": ctx["rank_one_share"]},
        "arms": arms,
        "attribution": attribution,
        "random_candidate_true_rho": json.loads(
            (REPO / "results" / "target_scoring" / "lever_arm_calibration.json").read_text()
        )["summary"]["rho_true"],
    }
    r = arms
    out["predictions"] = {
        "P1a_fixed_true_rho_at_least_0_80": r["B3_1"]["true_rho"] >= 0.80,
        "P1b1_common_mode_cancels": abs(r["B1"]["true_rho"] - r["A"]["true_rho"]) <= 0.10,
        "P1b2_recipe_specific_carries_rise": (r["B2"]["true_rho"] - r["A"]["true_rho"] >= 0.10)
        and (s["B2"] - s["A"]) >= 2 * (s["B1"] - s["A"]),
        "P1b3_monotone_in_lambda": r["A"]["true_rho"]
        <= r["B3_0.5"]["true_rho"]
        <= r["B3_1"]["true_rho"]
        <= r["B3_2"]["true_rho"]
        and s["A"] <= s["B3_0.5"] <= s["B3_1"] <= s["B3_2"],
        "P1b4_variance_alone_0_3_to_0_8": 0.3 <= r["A"]["true_rho"] <= 0.8,
        "P1b5_loading_cv_0_05_to_0_35": 0.05 <= ctx["loading_cv"] <= 0.35,
        "P1c1_ensemble_worse_in_A_by_2pp": r["A"]["ensemble_true_excess_pp"] >= 2.0,
        "P1c2_B3_1_needs_90_to_200": r["B3_1"]["candidates_for_2pp_power_0_8"] is not None
        and 90 <= r["B3_1"]["candidates_for_2pp_power_0_8"] <= 200,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm in ARMS:
        a = arms[arm]
        print(
            f"{arm:7s} rho {a['true_rho']:+.2f} [{a['true_rho_ci'][0]:+.2f},{a['true_rho_ci'][1]:+.2f}] "
            f"slope {a['true_slope_pp_per_log']:+.2f} noiseless rho {a['noiseless']['rho']:+.2f} "
            f"ens {a['ensemble_true_excess_pp']:+.2f} N2pp {a['candidates_for_2pp_power_0_8']}"
        )
    print(json.dumps(out["decomposition"]), json.dumps(attribution, indent=1), out["predictions"])


if __name__ == "__main__":
    main()
