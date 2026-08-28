"""Task 1: evaluate checkpoint-augmented fitting against PREDICTIONS_TASK_CHECKPOINTS.md.

Produces results/checkpoints/checkpoint_study.json with a verdict for each
pre-registered prediction (C1-C5), computed by the criteria fixed before any
ranker was implemented. Figures are generated from that JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.audit import audit_with_ci, resample_runs_by_cell  # noqa: E402
from asla.analysis.checkpoints import (  # noqa: E402
    autocorrelation_summary,
    fit_checkpoint_augmented,
    make_checkpoint_ranker,
)
from asla.analysis.fits import project_ranking, truth_ranking  # noqa: E402
from asla.analysis.pooled import fit_shared_exponent, project, shared_exponent_ranker  # noqa: E402
from asla.analysis.rankers import ensemble_ranker, make_projection_ranker, single_scale_ranker  # noqa: E402
from asla.cli import _write_json_atomically  # noqa: E402
from asla.data.sources import datadecide as dd  # noqa: E402

CACHE = "data/raw/datadecide"
METRIC = "c4_en_bits_per_token"
LADDER = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M"]
# Pre-registered thresholds (PREDICTIONS_TASK_CHECKPOINTS.md).
C1_MIN_RELATIVE = 0.25
C1_MIN_ABSOLUTE_POINTS = 0.5
C3_RATIO = 1.5
C3_MIN_LONG_ARM_REMOVED = 3


def build(metric: str, max_fit: str, target_label: str = "1B") -> dict[str, Any]:
    """Final-checkpoint and checkpoint-augmented tables for one design."""

    eval_df = dd.load_eval_macro(CACHE)
    ppl_df = dd.load_ppl(CACHE) if dd.METRICS[metric]["source"] == "ppl" else None
    final = dd.build_runs_table(eval_df, ppl_df, metric=metric, checkpoints="final")
    augmented = dd.build_runs_table(eval_df, ppl_df, metric=metric, checkpoints="all")
    final = final[final["scale_label"] != "750M"].reset_index(drop=True)
    augmented = augmented[augmented["scale_label"] != "750M"].reset_index(drop=True)
    target = float(final[final["scale_label"] == target_label]["compute"].iloc[0])
    keep = LADDER[: LADDER.index(max_fit) + 1]
    ladder = final[final["scale_label"].isin(keep) & (final["compute"] < target)]
    budgets = tuple(sorted(float(b) for b in ladder["compute"].unique()))
    max_budget = max(budgets)
    ckpt_fit = augmented[augmented["compute"] <= max_budget]
    # the augmented evaluation table: all fitting checkpoints plus the target rows
    ckpt_table = pd.concat([ckpt_fit, final[np.isclose(final["compute"], target)]], ignore_index=True)
    final_table = final[final["compute"].isin(budgets) | np.isclose(final["compute"], target)]
    return {
        "final_table": final_table.reset_index(drop=True),
        "ckpt_table": ckpt_table.reset_index(drop=True),
        "budgets": budgets,
        "target": target,
        "max_fit": max_fit,
        "n_ladder_points": len(budgets),
        "n_checkpoint_points": int(ckpt_fit.groupby("intervention")["compute"].nunique().min()),
    }


def flips(scores: pd.Series, truth: pd.Series) -> set[tuple[str, str]]:
    names = sorted(set(scores.index.astype(str)) & set(truth.index.astype(str)))
    out = set()
    for a, b in itertools.combinations(names, 2):
        pg, tg = float(scores[a] - scores[b]), float(truth[a] - truth[b])
        if pg == 0.0 or tg == 0.0:
            continue
        if np.sign(pg) != np.sign(tg):
            out.add((a, b))
    return out


def checkpoint_plus_pooling_ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
    """Composition: shared exponent fitted on all checkpoints below the target."""

    available = sorted(float(c) for c in df["compute"].unique() if float(c) < float(target))
    params = fit_shared_exponent(df, tuple(available))
    return project(params, target)


def evaluate(design: dict[str, Any], n_boot: int, seed: int, include_ensemble: bool) -> dict[str, Any]:
    started = time.time()
    final_table, ckpt_table = design["final_table"], design["ckpt_table"]
    budgets, target = design["budgets"], design["target"]
    truth = truth_ranking(final_table, target)

    final_rankers: dict[str, Any] = {
        "single_scale": single_scale_ranker,
        "plain_projection": make_projection_ranker("compute_power_law"),
        "shared_exponent": shared_exponent_ranker,
    }
    if include_ensemble:
        final_rankers["ensemble"] = ensemble_ranker
    ckpt_rankers: dict[str, Any] = {
        "checkpoint_naive": make_checkpoint_ranker("naive"),
        "checkpoint_ar1": make_checkpoint_ranker("ar1"),
        "checkpoint_plus_pooling": checkpoint_plus_pooling_ranker,
    }
    audit_final = audit_with_ci(
        final_table,
        budgets,
        target,
        final_rankers,
        n_boot,
        np.random.default_rng(seed),
        estimand="pairwise_decisions",
    )
    ckpt_budgets = tuple(sorted(float(c) for c in ckpt_table["compute"].unique() if c < target))
    audit_ckpt = audit_with_ci(
        ckpt_table,
        ckpt_budgets,
        target,
        ckpt_rankers,
        n_boot,
        np.random.default_rng(seed),
        estimand="pairwise_decisions",
    )

    plain_flips = flips(project_ranking(final_table, budgets, target), truth)
    rankers: dict[str, Any] = {}
    for name, ranker, table, bud in (
        [(n, r, final_table, budgets) for n, r in final_rankers.items()]
        + [(n, r, ckpt_table, ckpt_budgets) for n, r in ckpt_rankers.items()]
    ):
        audit = audit_final if name in final_rankers else audit_ckpt
        scores = ranker(table, bud, target)
        f = flips(scores, truth)
        rankers[name] = {
            "mis_selection": audit["rankers"][name]["mis_selection_rate"],
            "n_flips": len(f),
            "n_removed_vs_plain": len(plain_flips - f),
            "n_added_vs_plain": len(f - plain_flips),
            "net_removed_vs_plain": len(plain_flips - f) - len(f - plain_flips),
        }
    _, diagnostics = fit_checkpoint_augmented(ckpt_table[ckpt_table["compute"] < target], ckpt_budgets)
    return {
        "max_fit": design["max_fit"],
        "target": target,
        "lever_arm": target / max(budgets),
        "n_ladder_points": design["n_ladder_points"],
        "n_checkpoint_points": design["n_checkpoint_points"],
        "n_boot": n_boot,
        "rankers": rankers,
        "autocorrelation": autocorrelation_summary(diagnostics),
        "elapsed_seconds": time.time() - started,
    }


def bootstrap_interaction(designs: dict[str, dict[str, Any]], builds: dict[str, dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    """C3: rho_C = I(long arm) / I(short arm) for the checkpoint ranker."""

    rng = np.random.default_rng(seed)
    ranker = make_checkpoint_ranker("ar1")

    def removed(build_info: dict[str, Any], ckpt_table: pd.DataFrame) -> int:
        target, budgets = build_info["target"], build_info["budgets"]
        truth = truth_ranking(build_info["final_table"], target)
        plain = flips(project_ranking(build_info["final_table"], budgets, target), truth)
        bud = tuple(sorted(float(c) for c in ckpt_table["compute"].unique() if c < target))
        got = flips(ranker(ckpt_table, bud, target), truth)
        return len(plain - got) - len(got - plain)

    long_build, short_build = builds["long"], builds["short"]
    point_long = removed(long_build, long_build["ckpt_table"])
    point_short = removed(short_build, short_build["ckpt_table"])
    ratios: list[float] = []
    for _ in range(n_boot):
        try:
            i_long = removed(long_build, resample_runs_by_cell(long_build["ckpt_table"], rng))
            i_short = removed(short_build, resample_runs_by_cell(short_build["ckpt_table"], rng))
        except Exception:  # noqa: BLE001
            continue
        if i_short > 0:
            ratios.append(i_long / i_short)
    arr = np.asarray(ratios, dtype=float)
    return {
        "I_long_arm": point_long,
        "I_short_arm": point_short,
        "ratio_point": (point_long / point_short) if point_short > 0 else None,
        "ratio_lo": float(np.percentile(arr, 2.5)) if len(arr) > 10 else None,
        "ratio_hi": float(np.percentile(arr, 97.5)) if len(arr) > 10 else None,
        "n_draws": len(arr),
    }


def judge(results: dict[str, Any]) -> dict[str, Any]:
    primary = results["designs"]["primary"]
    r = primary["rankers"]
    plain = r["plain_projection"]["mis_selection"]["point"]
    ckpt = r["checkpoint_ar1"]["mis_selection"]
    single = r["single_scale"]["mis_selection"]
    pool = r["shared_exponent"]["mis_selection"]["point"]
    both = r["checkpoint_plus_pooling"]["mis_selection"]["point"]

    relative = (plain - ckpt["point"]) / plain if plain > 0 else 0.0
    absolute = 100.0 * (plain - ckpt["point"])
    separated = bool(ckpt["hi"] is not None and ckpt["hi"] < plain)
    if relative >= C1_MIN_RELATIVE and ckpt["point"] < plain:
        c1_verdict = "CONFIRMED"
    elif abs(absolute) < C1_MIN_ABSOLUTE_POINTS and not separated:
        c1_verdict = "UNDERPOWERED"
    else:
        c1_verdict = "REFUTED"
    c1 = {
        "verdict": c1_verdict,
        "plain": plain,
        "checkpoint_ar1": ckpt,
        "relative_improvement": relative,
        "absolute_improvement_points": absolute,
        "separated_from_plain": separated,
        "criterion": (
            f"CONFIRMED at >= {C1_MIN_RELATIVE:.0%} relative; "
            f"UNDERPOWERED if < {C1_MIN_ABSOLUTE_POINTS} points absolute and not separated from plain"
        ),
    }

    beats_single = bool(ckpt["lo"] is not None and ckpt["lo"] < single["point"] and ckpt["point"] < single["point"])
    c2 = {
        "verdict": "REFUTED" if beats_single else "CONFIRMED",
        "single_scale": single,
        "checkpoint_ar1": ckpt,
        "criterion": "refuted if the checkpoint CI lower bound falls below single-scale's point estimate",
        "note": "predicted at 65% confidence in the pre-registration",
    }

    inter = results.get("interaction")
    if inter is None:
        c3 = {"verdict": "NOT_RUN"}
    else:
        ratio, hi = inter["ratio_point"], inter["ratio_hi"]
        if inter["I_short_arm"] == 0:
            verdict = "CONFIRMED" if inter["I_long_arm"] >= C3_MIN_LONG_ARM_REMOVED else "UNDERPOWERED"
        elif inter["I_long_arm"] < C3_MIN_LONG_ARM_REMOVED:
            verdict = "UNDERPOWERED"
        elif ratio is not None and ratio >= C3_RATIO:
            verdict = "CONFIRMED"
        elif ratio is not None and hi is not None and hi < C3_RATIO:
            verdict = "FAILED"
        else:
            verdict = "UNDERPOWERED"
        c3 = {
            "verdict": verdict,
            **inter,
            "criterion": f"rho_C >= {C3_RATIO}; FAILED only if the point estimate and CI upper bound are both below",
        }

    g_ckpt = (plain - ckpt["point"]) / plain if plain > 0 else 0.0
    g_pool = (plain - pool) / plain if plain > 0 else 0.0
    g_both = (plain - both) / plain if plain > 0 else 0.0
    best_single_lever = max(g_ckpt, g_pool)
    both_interval = r["checkpoint_plus_pooling"]["mis_selection"]
    best_name = "shared_exponent" if g_pool >= g_ckpt else "checkpoint_ar1"
    best_interval = r[best_name]["mis_selection"]
    separable = not (both_interval["lo"] > best_interval["hi"] or best_interval["lo"] > both_interval["hi"])
    if g_both <= best_single_lever and separable:
        c4_verdict = "UNDERPOWERED"
    elif g_both <= best_single_lever:
        c4_verdict = "REFUTED_REDUNDANT"
    elif g_both >= g_ckpt + g_pool:
        c4_verdict = "REFUTED_INDEPENDENT"
    else:
        c4_verdict = "CONFIRMED_SUBADDITIVE"
    c4 = {
        "verdict": c4_verdict,
        "G_checkpoint": g_ckpt,
        "G_pooling": g_pool,
        "G_both": g_both,
        "sum_of_parts": g_ckpt + g_pool,
        "best_single_lever": best_single_lever,
        "cis_overlap_with_best_single": separable,
        "criterion": "CONFIRMED if max(parts) < G_both < sum(parts)",
    }

    naive = r["checkpoint_naive"]["mis_selection"]
    corrected = r["checkpoint_ar1"]["mis_selection"]
    naive_width = (naive["hi"] - naive["lo"]) if naive["hi"] is not None else None
    corrected_width = (corrected["hi"] - corrected["lo"]) if corrected["hi"] is not None else None
    differs = bool(abs(naive["point"] - corrected["point"]) > 1e-9)
    c5 = {
        "verdict": "CONFIRMED" if differs else "REFUTED_NO_OP",
        "naive": naive,
        "corrected": corrected,
        "naive_ci_width": naive_width,
        "corrected_ci_width": corrected_width,
        "point_estimates_differ": differs,
        "autocorrelation": primary["autocorrelation"],
        "criterion": "the correction must change the fit; a no-op means the trap was not a trap here",
    }

    return {
        "C1_beats_plain": c1,
        "C2_beats_single_scale": c2,
        "C3_lever_arm_interaction": c3,
        "C4_composition_with_pooling": c4,
        "C5_correlation_correction_matters": c5,
        "mechanism_damaged": bool(c1["verdict"] == "REFUTED"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/checkpoints")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    n_boot = 3 if args.fast else 80
    n_boot_ratio = 3 if args.fast else 40

    builds = {
        "primary": build(METRIC, "300M"),
        "long": build(METRIC, "150M"),
        "short": build(METRIC, "530M"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "fast": args.fast,
        "seed": args.seed,
        "metric": METRIC,
        "counts": {"n_boot": n_boot, "n_boot_ratio": n_boot_ratio},
        "designs": {},
    }
    for label, built in builds.items():
        results["designs"][label] = evaluate(
            built, n_boot=n_boot, seed=args.seed, include_ensemble=(label == "primary")
        )
        print(f"done {label}", flush=True)
        _write_json_atomically(results, out / "checkpoint_study.json")
    results["interaction"] = bootstrap_interaction(results["designs"], builds, n_boot_ratio, args.seed)
    results["verdicts"] = judge(results)
    _write_json_atomically(results, out / "checkpoint_study.json")
    print(json.dumps(results["verdicts"], indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
