"""Tasks 2 and 3 of PREDICTIONS_TASK_CALIBRATED_INFERENCE.md: two estimands, end-to-end calibration.

Task 2 (real data, C4 primary design, fit 4M-300M, target 530M):

* **Fixed-candidate** estimand: conditional on these 25 recipes. Uncertainty is
  seed noise only, obtained by parametric resampling of every cell and
  checkpoint from the moderated noise model around the observed means. The
  interval is estimate +/- 1.96 x the resampling SE.
* **Random-candidate** estimand: generalizing to new recipes. The U-statistic
  test.

Task 3 (semi-synthetic worlds from asla.analysis.calibration): coverage and bias
of both procedures against the true mis-selection difference, and power at
planted effects. Shared-exponent and EB shrinkage get a delete-one-candidate
jackknife that refits the whole ranker.

Declared simplification: EB shrinkage uses its strength frozen at the real-data
estimate, fitted on the training interventions exactly as in the paper. Its
calibration therefore omits the variability of estimating the strength.

Usage::

    python scripts/run_calibration.py --resume          # continue from data/cache
    python scripts/run_calibration.py --workers 6

Writes ``results/target_scoring/calibration.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import (  # noqa: E402
    Truth,
    compare,
    evidence_at_target,
    gaps,
    jackknife_se,
    predict,
    random_truth,
    simulate,
    truth_from_tables,
)

CACHE = REPO / "data" / "cache" / "calibration_worlds.jsonl"
OUT = REPO / "results" / "target_scoring" / "calibration.json"
METRIC = "c4_en_bits_per_token"
# The metric the benchmark is built on; other metrics transport its critical values.
CALIBRATED_METRIC = "c4_en_bits_per_token"
MAX_FIT, TARGET_LABEL = "300M", "530M"
BASELINE = "single_scale"
CHECKPOINT_RANKERS = ("checkpoint_augmented",)
JACKKNIFE_RANKERS = ("shared_exponent", "eb_shrinkage")
PLANTED_RUNGS = ("150M", "90M", "60M", "20M")
BANDWIDTH = 0.5
COUPLING = ("300M", "530M", 0.30)
N_FIXED, N_RANDOM, N_MC_FIXED, N_MC_RANDOM, N_COUPLED = 150, 300, 400, 400, 300
B_WORLD, B_REAL = 50, 400
SEED = 20260923
Z = 1.959963984540054

_CONTEXT: dict[str, Any] = {}


def _context() -> dict[str, Any]:
    """Tables, frozen rankers and the base truth; built once per process."""

    if _CONTEXT:
        return _CONTEXT
    from asla.analysis.checkpoints import make_checkpoint_ranker
    from asla.analysis.fits import normalize_budgets
    from asla.analysis.pooled import fit_shrunk, make_shrinkage_ranker
    from asla.analysis.rankers import ensemble_ranker, projection_ranker, single_scale_ranker
    from asla.models import fit_power_law
    from asla.provenance import Source

    rescoring = importlib.import_module("scripts.run_rescoring")
    built = rescoring._load_checkpoint_builder()(METRIC, MAX_FIT, TARGET_LABEL)
    final, ckpt = built["final_table"], built["ckpt_table"]
    target = float(built["target"])
    budgets = normalize_budgets(built["budgets"], target=target)
    training = Source.load("task_b/task_b.json").get("shrinkage_strength_protocol.training_interventions")
    strength_rows = final[final["intervention"].astype(str).isin(training)]
    strength = fit_shrunk(
        final, budgets, strength=None, n_boot=40, rng=np.random.default_rng(0), strength_df=strength_rows
    ).shrinkage
    rung_compute = {str(label): float(final[final["scale_label"] == label]["compute"].iloc[0]) for label in PLANTED_RUNGS}

    def at_rung(label: str):  # noqa: ANN202
        def ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
            return single_scale_ranker(df, tuple(b for b in budgets if b <= rung_compute[label] * 1.0001), target)

        return ranker

    rankers = {
        BASELINE: single_scale_ranker,
        "projection": projection_ranker,
        "shared_exponent": importlib.import_module("scripts.run_rescoring").shared_exponent_ranker,
        "eb_shrinkage": make_shrinkage_ranker(strength, n_boot=2),
        "ensemble": ensemble_ranker,
        "checkpoint_augmented": make_checkpoint_ranker("ar1"),
        **{f"single_scale_{label}": at_rung(label) for label in PLANTED_RUNGS},
    }
    base = truth_from_tables(final, ckpt, target, TARGET_LABEL)
    params = {}
    for recipe, group in base.final.groupby("intervention"):
        cells = group.groupby("compute")["mu"].mean()
        params[str(recipe)] = fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))
    _CONTEXT.update(
        final=final,
        ckpt=ckpt,
        target=target,
        budgets=budgets,
        rankers=rankers,
        base=base,
        params=params,
        strength=float(strength),
    )
    return _CONTEXT


def _procedure(
    truth: Truth, final: pd.DataFrame, ckpt: pd.DataFrame, rng: np.random.Generator, n_boot: int, jackknife: bool
) -> dict[str, Any]:
    ctx = _context()
    evidence = evidence_at_target(final, TARGET_LABEL)
    preds = predict(final, ckpt, ctx["rankers"], CHECKPOINT_RANKERS, ctx["budgets"], ctx["target"])
    truth_gaps = gaps(truth.target_truth())
    out = {name: vars(c) for name, c in compare(preds, evidence, truth_gaps, BASELINE, with_interval=True).items()}
    if jackknife:
        for name in JACKKNIFE_RANKERS:
            out[name]["jk_se"] = jackknife_se(
                final, ckpt, ctx["rankers"], CHECKPOINT_RANKERS, name, BASELINE, evidence, ctx["budgets"], ctx["target"]
            )
    if n_boot:
        plug = truth_from_tables(final, ckpt, ctx["target"], TARGET_LABEL, rho_ckpt=ctx["base"].rho_ckpt)
        plug_gaps = gaps(plug.target_truth())
        draws: dict[str, list[float]] = {name: [] for name in out}
        for _ in range(n_boot):
            f, c = simulate(plug, rng)
            p = predict(f, c, ctx["rankers"], CHECKPOINT_RANKERS, ctx["budgets"], ctx["target"])
            for name, comp in compare(
                p, evidence_at_target(f, TARGET_LABEL), plug_gaps, BASELINE, with_interval=False
            ).items():
                draws[name].append(comp.estimate)
        for name in out:
            out[name]["boot_se"] = float(np.std(draws[name], ddof=1))
    return out


def run_world(kind: str, index: int) -> dict[str, Any]:
    ctx = _context()
    # ``random_jk`` regenerates exactly the ``random`` world (same seed) to add the kernel jackknife.
    seed_kind = "random" if kind == "random_jk" else kind
    rng = np.random.default_rng([SEED, ["fixed", "random", "mc_fixed", "mc_random", "coupled"].index(seed_kind), index])
    truth = (
        ctx["base"] if kind in ("fixed", "mc_fixed", "coupled") else random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    )
    final, ckpt = simulate(truth, rng, coupling=COUPLING if kind == "coupled" else None)
    if kind == "random_jk":
        evidence = evidence_at_target(final, TARGET_LABEL)
        preds = predict(final, ckpt, ctx["rankers"], CHECKPOINT_RANKERS, ctx["budgets"], ctx["target"])
        comps = compare(preds, evidence, gaps(truth.target_truth()), BASELINE, with_interval=True)
        return {
            "kind": kind,
            "index": index,
            "comparisons": {n: {"estimate": c.estimate, "kjk_se": c.kjk_se, "u_se": c.u_se} for n, c in comps.items()},
        }
    if kind in ("mc_fixed", "mc_random", "coupled"):
        evidence = evidence_at_target(final, TARGET_LABEL)
        preds = predict(final, ckpt, ctx["rankers"], CHECKPOINT_RANKERS, ctx["budgets"], ctx["target"])
        comps = compare(preds, evidence, gaps(truth.target_truth()), BASELINE, with_interval=False)
        result = {name: {"estimate": c.estimate, "true_difference": c.true_difference} for name, c in comps.items()}
    else:
        result = _procedure(truth, final, ckpt, rng, n_boot=B_WORLD if kind == "fixed" else 0, jackknife=True)
    return {"kind": kind, "index": index, "comparisons": result}


def real_data(rng: np.random.Generator) -> dict[str, Any]:
    """Task 2: both estimands on the real C4 data."""

    from scipy import stats

    ctx = _context()
    base: Truth = ctx["base"]
    result = _procedure(base, ctx["final"], ctx["ckpt"], rng, n_boot=B_REAL, jackknife=True)
    out = {}
    for name, row in result.items():
        entry = {
            "estimate_pp": row["estimate"],
            "kernel_jackknife_se": row["kjk_se"],
            "random_candidate": {"ci_low": row["u_low"], "ci_high": row["u_high"], "se": row["u_se"]},
            "fixed_candidate": {
                "se": row["boot_se"],
                "ci_low": row["estimate"] - Z * row["boot_se"],
                "ci_high": row["estimate"] + Z * row["boot_se"],
                "p_two_sided": float(2 * stats.norm.sf(abs(row["estimate"]) / row["boot_se"]))
                if row["boot_se"] > 0
                else None,
            },
        }
        entry["random_candidate"]["p_two_sided"] = (
            float(2 * stats.norm.sf(abs(row["estimate"]) / row["u_se"])) if row["u_se"] > 0 else None
        )
        if "jk_se" in row:
            entry["random_candidate_jackknife"] = {
                "se": row["jk_se"],
                "ci_low": row["estimate"] - Z * row["jk_se"],
                "ci_high": row["estimate"] + Z * row["jk_se"],
                "p_two_sided": float(2 * stats.norm.sf(abs(row["estimate"]) / row["jk_se"])) if row["jk_se"] > 0 else None,
            }
        out[name] = entry
    return {"design": f"{METRIC}, fit to {MAX_FIT}, target {TARGET_LABEL}", "bootstrap_draws": B_REAL, "comparisons": out}


def apply_calibration(real: dict[str, Any], calibrated: dict[str, Any]) -> dict[str, Any]:
    """Calibrated intervals and p-values for the real-data comparisons, under both estimands."""

    fixed = calibrated["fixed_bootstrap"]
    random_label = calibrated["random_adopted"]
    random = calibrated[random_label]
    for name, entry in real["comparisons"].items():
        estimate = entry["estimate_pp"]
        se_f = entry["fixed_candidate"]["se"]
        if random_label == "random_jackknife":
            se_r = (
                entry["random_candidate_jackknife"]["se"]
                if "random_candidate_jackknife" in entry
                else entry["kernel_jackknife_se"]
            )
        else:
            se_r = entry["random_candidate"]["se"]
        entry["calibrated_fixed_candidate"] = {
            "se": se_f,
            "critical_value": fixed["critical_value"],
            "ci_low": estimate - fixed["critical_value"] * se_f,
            "ci_high": estimate + fixed["critical_value"] * se_f,
            "p_two_sided": calibrated_p(abs(estimate) / se_f, fixed["null_quantiles_of_abs_t"], fixed["n_statistics"])
            if se_f > 0
            else None,
        }
        entry["calibrated_random_candidate"] = {
            "se": se_r,
            "se_method": random_label,
            "critical_value": random["critical_value"],
            "ci_low": estimate - random["critical_value"] * se_r,
            "ci_high": estimate + random["critical_value"] * se_r,
            "p_two_sided": calibrated_p(abs(estimate) / se_r, random["null_quantiles_of_abs_t"], random["n_statistics"])
            if se_r > 0
            else None,
        }
    return real


def _load_cache() -> dict[tuple[str, int], dict[str, Any]]:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["kind"], row["index"])] = row
    return done


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["comparisons"])
    names = sorted(by_kind["fixed"][0])

    def values(kinds: tuple[str, ...], name: str, field: str) -> np.ndarray:
        return np.array([w[name][field] for k in kinds for w in by_kind.get(k, []) if name in w and field in w[name]])

    out: dict[str, Any] = {}
    for name in names:
        theta_f = float(values(("fixed", "mc_fixed"), name, "true_difference").mean())
        theta_r = float(values(("random", "mc_random"), name, "true_difference").mean())
        fixed = by_kind["fixed"]
        random = by_kind["random"]
        entry: dict[str, Any] = {
            "theta_fixed": theta_f,
            "theta_random": theta_r,
            "bias_fixed": float(values(("fixed", "mc_fixed"), name, "estimate").mean() - theta_f),
            "bias_random": float(values(("random", "mc_random"), name, "estimate").mean() - theta_r),
            "fixed_candidate_coverage": float(
                np.mean([abs(w[name]["estimate"] - theta_f) <= Z * w[name]["boot_se"] for w in fixed])
            ),
            "fixed_candidate_power": float(np.mean([abs(w[name]["estimate"]) > Z * w[name]["boot_se"] for w in fixed])),
            "random_candidate_coverage_u": float(
                np.mean([w[name]["u_low"] <= theta_r <= w[name]["u_high"] for w in random])
            ),
            "random_candidate_power_u": float(np.mean([w[name]["u_low"] > 0 or w[name]["u_high"] < 0 for w in random])),
            "u_interval_coverage_of_fixed_theta_in_fixed_worlds": float(
                np.mean([w[name]["u_low"] <= theta_f <= w[name]["u_high"] for w in fixed])
            ),
            "n_fixed_worlds": len(fixed),
            "n_random_worlds": len(random),
        }
        if name in JACKKNIFE_RANKERS:
            entry["random_candidate_coverage_jackknife"] = float(
                np.mean([abs(w[name]["estimate"] - theta_r) <= Z * w[name]["jk_se"] for w in random])
            )
            entry["random_candidate_power_jackknife"] = float(
                np.mean([abs(w[name]["estimate"]) > Z * w[name]["jk_se"] for w in random])
            )
        out[name] = entry
    coupled = by_kind.get("coupled", [])
    sensitivity = {}
    for name in names:
        independent = values(("mc_fixed",), name, "estimate").mean()
        sensitivity[name] = float(np.mean([w[name]["estimate"] for w in coupled]) - independent) if coupled else None
    return {"by_comparator": out, "coupling_sensitivity_shift_pp": sensitivity, "coupling": list(COUPLING)}


CALIBRATION_LEVEL = 0.95
NULL_GRID = 2001


def _quantile_grid(values: np.ndarray) -> list[float]:
    finite_cap = np.where(np.isfinite(values), values, np.nanmax(values[np.isfinite(values)]) * 10)
    return [float(v) for v in np.quantile(finite_cap, np.linspace(0, 1, NULL_GRID))]


def calibrated_procedures(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Addendum 2: simulation-calibrated critical values, the U-versus-jackknife choice, cross-fit coverage."""

    by_kind: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], {})[row["index"]] = row["comparisons"]
    names = sorted(summary["by_comparator"])
    theta_f = {n: summary["by_comparator"][n]["theta_fixed"] for n in names}
    theta_r = {n: summary["by_comparator"][n]["theta_random"] for n in names}
    mismatch = max(
        abs(by_kind["random"][i][n]["estimate"] - by_kind["random_jk"][i][n]["estimate"])
        for i in by_kind["random_jk"]
        for n in names
    )
    if mismatch > 1e-9:
        raise ValueError(f"random_jk worlds do not reproduce the random worlds (max estimate gap {mismatch})")

    def jk_se(i: int, n: str) -> float:
        return by_kind["random"][i][n]["jk_se"] if n in JACKKNIFE_RANKERS else by_kind["random_jk"][i][n]["kjk_se"]

    def ratio(dev: float, se: float) -> float:
        return abs(dev) / se if se > 0 else (0.0 if dev == 0 else float("inf"))

    procedures = {
        "fixed_bootstrap": {
            i: {n: (w[n]["estimate"] - theta_f[n], w[n]["boot_se"]) for n in names} for i, w in by_kind["fixed"].items()
        },
        "random_u": {
            i: {n: (w[n]["estimate"] - theta_r[n], w[n]["u_se"]) for n in names} for i, w in by_kind["random"].items()
        },
        "random_jackknife": {
            i: {n: (w[n]["estimate"] - theta_r[n], jk_se(i, n)) for n in names} for i, w in by_kind["random"].items()
        },
    }
    out: dict[str, Any] = {"level": CALIBRATION_LEVEL, "pooled_over": names}
    for label, worlds in procedures.items():
        stats_all = np.array([ratio(d, se) for w in worlds.values() for d, se in w.values()])
        k = float(np.quantile(stats_all, CALIBRATION_LEVEL))
        half_widths = np.array([k * se for w in worlds.values() for _, se in w.values()])
        held_out: dict[str, float] = {}
        for n in names:
            covered = []
            for parity in (0, 1):
                train = np.array([ratio(*w[m]) for i, w in worlds.items() if i % 2 == parity for m in names])
                k_train = float(np.quantile(train, CALIBRATION_LEVEL))
                covered += [ratio(*w[n]) <= k_train for i, w in worlds.items() if i % 2 != parity]
            held_out[n] = float(np.mean(covered))
        # Power at the calibrated critical value: how often the interval excludes zero.
        est_by_comparator = {
            n: [(w[n][0] + (theta_f[n] if label == "fixed_bootstrap" else theta_r[n]), w[n][1]) for w in worlds.values()]
            for n in names
        }
        power = {n: float(np.mean([ratio(e, se) > k for e, se in est_by_comparator[n]])) for n in names}
        out[label] = {
            "power_at_critical_value": power,
            "critical_value": k,
            "median_half_width_pp": float(np.median(half_widths)),
            "coverage_at_1_96": {
                n: float(np.mean([ratio(*w[n]) <= 1.959963984540054 for w in worlds.values()])) for n in names
            },
            "cross_fit_coverage": held_out,
            "cross_fit_within_0_93_0_97": all(0.93 <= v <= 0.97 for v in held_out.values()),
            "null_quantiles_of_abs_t": _quantile_grid(stats_all),
            "n_statistics": int(stats_all.size),
        }
    u, jk = out["random_u"], out["random_jackknife"]
    out["random_adopted"] = "random_jackknife" if jk["median_half_width_pp"] <= u["median_half_width_pp"] else "random_u"
    return out


def calibrated_p(abs_t: float, null_quantiles: list[float], n_statistics: int) -> float:
    """P(|T| >= abs_t) under the calibrated null distribution of the studentized deviation.

    Floored at 1/n_statistics: the simulation cannot resolve a smaller tail
    probability than one in the number of simulated statistics.
    """

    grid = np.asarray(null_quantiles)
    probability_below = float(np.interp(abs_t, grid, np.linspace(0, 1, grid.size), left=0.0, right=1.0))
    return max(1.0 - probability_below, 1.0 / n_statistics)


def predictions(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["by_comparator"]
    c1_names = ["projection", "ensemble", "checkpoint_augmented"] + [f"single_scale_{r}" for r in PLANTED_RUNGS]
    c1 = {n: rows[n]["random_candidate_coverage_u"] for n in c1_names}
    c2_u = {n: rows[n]["random_candidate_coverage_u"] for n in JACKKNIFE_RANKERS}
    c2_jk = {n: rows[n]["random_candidate_coverage_jackknife"] for n in JACKKNIFE_RANKERS}
    c3 = {n: rows[n]["fixed_candidate_coverage"] for n in rows}
    c4 = {n: max(abs(rows[n]["bias_fixed"]), abs(rows[n]["bias_random"])) for n in rows}
    return {
        "C1_u_statistic_coverage": {"measured": c1, "confirmed": all(0.90 <= v <= 0.97 for v in c1.values())},
        "C2_coupled_rankers": {
            "u_statistic": c2_u,
            "jackknife": c2_jk,
            "confirmed": all(v < 0.90 for v in c2_u.values()) and all(v >= 0.90 for v in c2_jk.values()),
        },
        "C3_fixed_candidate_coverage": {"measured": c3, "confirmed": all(0.90 <= v <= 0.97 for v in c3.values())},
        "C4_expected_error_unbiased": {"max_abs_bias_pp": c4, "confirmed": all(v < 0.5 for v in c4.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load_cache()
    plan = [("fixed", i) for i in range(N_FIXED)] + [("random", i) for i in range(N_RANDOM)]
    plan += [("mc_fixed", i) for i in range(N_MC_FIXED)] + [("mc_random", i) for i in range(N_MC_RANDOM)]
    plan += [("coupled", i) for i in range(N_COUPLED)]
    plan += [("random_jk", i) for i in range(N_RANDOM)]
    todo = [task for task in plan if task not in done]
    print(f"{len(done)} worlds cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        futures = {pool.submit(run_world, kind, index): (kind, index) for kind, index in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if count % 50 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    rows = list(_load_cache().values())
    summary = summarise(rows)
    ctx = _context()
    out = {
        "design": {
            "metric": METRIC,
            "fit_to": MAX_FIT,
            "target": TARGET_LABEL,
            "n_candidates": int(ctx["final"]["intervention"].nunique()),
            "worlds": {
                "fixed": N_FIXED,
                "random": N_RANDOM,
                "mc_fixed": N_MC_FIXED,
                "mc_random": N_MC_RANDOM,
                "coupled": N_COUPLED,
            },
            "bootstrap_draws_per_fixed_world": B_WORLD,
            "random_world_bandwidth": BANDWIDTH,
            "eb_strength_frozen_at": ctx["strength"],
            "checkpoint_autocorrelation": dict(ctx["base"].rho_ckpt),
            "noise": "Gaussian, moderated per-(recipe, scale) sd; AR(1) along checkpoints; independent across scales",
            "truth": "observed seed-mean trajectories (fixed); smoothed bootstrap over recipes (random)",
        },
        "real_data": real_data(np.random.default_rng([SEED, 99])),
        "calibration": summary,
    }
    out["predictions"] = predictions(summary)
    out["calibrated"] = calibrated_procedures(rows, summary)
    out["real_data"] = apply_calibration(out["real_data"], out["calibrated"])
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["predictions"], indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
