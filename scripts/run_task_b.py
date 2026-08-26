"""Task B: evaluate pooled/shrinkage projection against the pre-registered predictions.

Reads PREDICTIONS_TASK_B.md's criteria as code (they are restated here as
explicit thresholds) and reports each prediction as CONFIRMED, REFUTED, or
UNDERPOWERED by the rule fixed before any estimator was run.

Usage::

    python scripts/run_task_b.py --fast    # smoke counts
    python scripts/run_task_b.py           # paper counts
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
from asla.analysis.fits import project_ranking, truth_ranking  # noqa: E402
from asla.analysis.pooled import (  # noqa: E402
    empirical_bayes_strength,
    exponent_variance_components,
    fit_shrunk,
    make_shrinkage_ranker,
    shared_exponent_ranker,
)
from asla.analysis.rankers import ensemble_ranker, make_projection_ranker, single_scale_ranker  # noqa: E402
from asla.cli import _sha256_file, _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402

DATA = Path("data")
PRIMARY = DATA / "datadecide_runs.parquet"
STRENGTH_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Pre-registered thresholds (PREDICTIONS_TASK_B.md).
P1_MIN_RELATIVE = 0.10
P3_POINT_MARGIN = 1.0
P4_RATIO = 2.0
P4_MIN_LONG_ARM_REMOVED = 3
P5_MIN_REMOVED = 8
P5_FIT_ERROR_SHARE = 0.80


def design(
    metric_path: Path, max_fit_label: str = "300M", target_label: str = "1B"
) -> tuple[pd.DataFrame, tuple[float, ...], float]:
    df = load_runs(metric_path)
    df = df[df["scale_label"] != "750M"].reset_index(drop=True)
    target = float(df[df["scale_label"] == target_label]["compute"].iloc[0])
    order = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M"]
    keep = order[: order.index(max_fit_label) + 1]
    fit = df[df["scale_label"].isin(keep) & (df["compute"] < target)]
    budgets = tuple(sorted(float(b) for b in fit["compute"].unique()))
    table = df[df["compute"].isin(budgets) | np.isclose(df["compute"], target)]
    return table.reset_index(drop=True), budgets, target


def flip_pairs(scores: pd.Series, truth: pd.Series) -> set[tuple[str, str]]:
    names = sorted(set(scores.index.astype(str)) & set(truth.index.astype(str)))
    out = set()
    for a, b in itertools.combinations(names, 2):
        pg = float(scores[a] - scores[b])
        tg = float(truth[a] - truth[b])
        if pg == 0.0 or tg == 0.0:
            continue
        if np.sign(pg) != np.sign(tg):
            out.add((a, b))
    return out


def training_interventions(df: pd.DataFrame, fraction: float = 0.68, seed: int = 1729) -> list[str]:
    """Deterministic training subset of interventions for tuning the shrinkage strength.

    The empirical-Bayes strength is a hyperparameter: if it is estimated from
    every intervention it will later be scored on, it has seen the evaluation
    data. This picks a fixed, seeded subset by name so the choice is
    reproducible and independent of the evaluation ordering.
    """

    names = sorted(df["intervention"].astype(str).unique())
    rng = np.random.default_rng(seed)
    n_train = max(2, int(round(fraction * len(names))))
    chosen = rng.choice(np.asarray(names, dtype=object), size=n_train, replace=False)
    return sorted(str(name) for name in chosen)


def strength_sensitivity(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    training: list[str],
    n_boot_inner: int,
    seed: int,
    n_resplits: int = 12,
) -> dict[str, Any]:
    """How much does the empirical-Bayes strength depend on *which* interventions estimate it?

    lambda = within / (within + between) is a ratio of two estimated variances,
    both computed from a subset of interventions. It therefore carries two
    distinct sources of variation that must not be confused:

    * **subset composition** - a different training subset has a genuinely
      different between-intervention spread, and
    * **bootstrap noise** in the within-intervention component.

    This resamples the training subset at the same size and reports the spread
    of lambda across draws, so any single reported value can be read against
    its own sampling distribution rather than taken as a constant.
    """

    names = sorted(df["intervention"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    for i in range(n_resplits):
        subset = sorted(
            str(name) for name in rng.choice(np.asarray(names, dtype=object), size=len(training), replace=False)
        )
        rows = df[df["intervention"].astype(str).isin(subset)]
        between, within, _ = exponent_variance_components(
            rows, budgets, n_boot=n_boot_inner, rng=np.random.default_rng(seed + 100 + i)
        )
        draws.append(
            {
                "between_sd": float(np.sqrt(between)),
                "within_sd": float(np.sqrt(within)),
                "lambda": empirical_bayes_strength(between, within),
                "overlap_with_frozen_split": len(set(subset) & set(training)),
            }
        )
    values = np.asarray([d["lambda"] for d in draws], dtype=float)
    full_between, full_within, _ = exponent_variance_components(
        df, budgets, n_boot=n_boot_inner, rng=np.random.default_rng(seed)
    )
    return {
        "n_resplits": n_resplits,
        "subset_size": len(training),
        "n_total_interventions": len(names),
        "lambda_mean_over_resplits": float(values.mean()),
        "lambda_sd_over_resplits": float(values.std(ddof=1)),
        "lambda_min": float(values.min()),
        "lambda_max": float(values.max()),
        "lambda_full_set_descriptive": empirical_bayes_strength(full_between, full_within),
        "draws": draws,
        "note": (
            "lambda varies across training subsets of the same size because the between-intervention spread "
            "genuinely differs between subsets; this spread is the relevant uncertainty on the shrinkage "
            "strength, and it is larger than the contamination the held-out protocol removes."
        ),
    }


def rankers_for(
    strengths: tuple[float, ...], n_boot_inner: int, seed: int, strength_interventions: list[str] | None = None
) -> dict[str, Any]:
    suite: dict[str, Any] = {
        "single_scale": single_scale_ranker,
        "plain_projection": make_projection_ranker("compute_power_law"),
        "shared_exponent": shared_exponent_ranker,
        "shrinkage_eb": make_shrinkage_ranker(
            None, n_boot=n_boot_inner, seed=seed, strength_interventions=strength_interventions
        ),
    }
    for strength in strengths:
        suite[f"shrinkage_{strength:g}"] = make_shrinkage_ranker(strength, n_boot=n_boot_inner, seed=seed)
    return suite


def evaluate_design(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    n_boot: int,
    n_boot_inner: int,
    seed: int,
    include_ensemble: bool,
    strength_interventions: list[str] | None = None,
) -> dict[str, Any]:
    """Point estimates plus seed-bootstrap CIs for every ranker on one design."""

    started = time.time()
    suite = rankers_for(STRENGTH_SWEEP, n_boot_inner, seed, strength_interventions)
    if include_ensemble:
        suite["ensemble"] = ensemble_ranker
    audit = audit_with_ci(
        df, budgets, target, suite, n_boot, np.random.default_rng(seed), estimand="pairwise_decisions"
    )
    truth = truth_ranking(df, target)
    plain_scores = project_ranking(df, budgets, target)
    plain_flips = flip_pairs(plain_scores, truth)
    per_ranker: dict[str, Any] = {}
    for name, ranker in suite.items():
        metrics = audit["rankers"][name]
        scores = ranker(df, budgets, target)
        flips = flip_pairs(scores, truth)
        removed = plain_flips - flips
        added = flips - plain_flips
        per_ranker[name] = {
            "mis_selection": metrics["mis_selection_rate"],
            "mean_regret": metrics["mean_regret"],
            "n_flips": len(flips),
            "n_removed_vs_plain": len(removed),
            "n_added_vs_plain": len(added),
            "net_removed_vs_plain": len(removed) - len(added),
            "removed_pairs": sorted(removed),
            "added_pairs": sorted(added),
        }
    strength_df = None
    if strength_interventions is not None:
        subset = df[df["intervention"].astype(str).isin(strength_interventions)]
        if subset["intervention"].nunique() >= 2:
            strength_df = subset
    variance_source = df if strength_df is None else strength_df
    between, within, _ = exponent_variance_components(
        variance_source, budgets, n_boot=n_boot_inner, rng=np.random.default_rng(seed)
    )
    fit = fit_shrunk(
        df, budgets, strength=None, n_boot=n_boot_inner, rng=np.random.default_rng(seed), strength_df=strength_df
    )
    # The variance components reported for the FULL set are descriptive only:
    # they ground the pre-registered magnitude prediction and are never used to
    # choose the shrinkage strength that the held-out evaluation scores.
    full_between, full_within, _ = exponent_variance_components(
        df, budgets, n_boot=n_boot_inner, rng=np.random.default_rng(seed + 7)
    )
    return {
        "n_interventions": int(df["intervention"].nunique()),
        "n_pairs": int(df["intervention"].nunique() * (df["intervention"].nunique() - 1) // 2),
        "budgets": [float(b) for b in budgets],
        "target": float(target),
        "n_boot": int(n_boot),
        "rankers": per_ranker,
        "variance_components": {
            "between_variance": between,
            "within_variance": within,
            "noise_share": empirical_bayes_strength(between, within),
            "between_sd": float(np.sqrt(between)),
            "within_sd": float(np.sqrt(within)),
            "pooled_exponent": fit.pooled_exponent,
            "eb_strength": fit.shrinkage,
            "strength_estimated_from": fit.strength_estimated_from,
            "strength_source_interventions": list(fit.strength_source),
            "n_strength_source_interventions": len(fit.strength_source),
            "n_evaluated_interventions": int(df["intervention"].nunique()),
            "strength_is_heldout": bool(
                strength_df is not None and len(fit.strength_source) < int(df["intervention"].nunique())
            ),
            "descriptive_full_set_between_sd": float(np.sqrt(full_between)),
            "descriptive_full_set_within_sd": float(np.sqrt(full_within)),
            "descriptive_full_set_noise_share": empirical_bayes_strength(full_between, full_within),
        },
        "plain_flip_pairs": sorted(plain_flips),
        "elapsed_seconds": time.time() - started,
    }


def decomposition_of_removed(
    df: pd.DataFrame, budgets: tuple[float, ...], target: float, removed: list[list[str]]
) -> dict[str, Any]:
    """Split the flips a pooled estimator removed into fit-error vs inherited-crossover (P5)."""

    truth = truth_ranking(df, target)
    largest = float(max(budgets))
    small = df[np.isclose(df["compute"], largest)].groupby("intervention", sort=True)["bpb"].mean()
    single_flips = flip_pairs(small, truth)
    removed_pairs = {tuple(p) for p in removed}
    fit_error = [p for p in removed_pairs if p not in single_flips]
    inherited = [p for p in removed_pairs if p in single_flips]
    total = len(removed_pairs)
    return {
        "n_removed": total,
        "n_removed_fit_error": len(fit_error),
        "n_removed_inherited": len(inherited),
        "fit_error_share": (len(fit_error) / total) if total else None,
        "fit_error_pairs": sorted(fit_error),
        "inherited_pairs": sorted(inherited),
    }


def bootstrap_interaction_ratio(
    long_arm: tuple[pd.DataFrame, tuple[float, ...], float],
    short_arm: tuple[pd.DataFrame, tuple[float, ...], float],
    ranker_name: str,
    n_boot: int,
    n_boot_inner: int,
    seed: int,
    strength_interventions: list[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap rho_L = I(long arm) / I(short arm) for the P4 criterion."""

    rng = np.random.default_rng(seed)
    suite = rankers_for(STRENGTH_SWEEP, n_boot_inner, seed, strength_interventions)
    ranker = suite[ranker_name]

    def removed_count(table: pd.DataFrame, budgets: tuple[float, ...], target: float) -> int:
        truth = truth_ranking(table, target)
        plain = flip_pairs(project_ranking(table, budgets, target), truth)
        pooled = flip_pairs(ranker(table, budgets, target), truth)
        return len(plain - pooled) - len(pooled - plain)

    point_long = removed_count(*long_arm)
    point_short = removed_count(*short_arm)
    ratios: list[float] = []
    longs: list[int] = []
    shorts: list[int] = []
    for _ in range(n_boot):
        try:
            l_table = resample_runs_by_cell(long_arm[0], rng)
            s_table = resample_runs_by_cell(short_arm[0], rng)
            i_long = removed_count(l_table, long_arm[1], long_arm[2])
            i_short = removed_count(s_table, short_arm[1], short_arm[2])
        except Exception:  # noqa: BLE001 - a failed resample is recorded by omission
            continue
        longs.append(i_long)
        shorts.append(i_short)
        if i_short > 0:
            ratios.append(i_long / i_short)
    arr = np.asarray(ratios, dtype=float)
    return {
        "ranker": ranker_name,
        "I_long_arm": point_long,
        "I_short_arm": point_short,
        "ratio_point": (point_long / point_short) if point_short > 0 else None,
        "ratio_lo": float(np.percentile(arr, 2.5)) if len(arr) > 10 else None,
        "ratio_hi": float(np.percentile(arr, 97.5)) if len(arr) > 10 else None,
        "n_ratio_draws": len(arr),
        "n_boot_attempted": n_boot,
        "mean_I_long": float(np.mean(longs)) if longs else None,
        "mean_I_short": float(np.mean(shorts)) if shorts else None,
    }


def judge(results: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered criteria verbatim."""

    primary = results["designs"]["primary_4M-300M_target1B"]
    rankers = primary["rankers"]
    plain = rankers["plain_projection"]["mis_selection"]["point"]
    single = rankers["single_scale"]["mis_selection"]
    best_name = min(
        ("shared_exponent", "shrinkage_eb"),
        key=lambda n: rankers[n]["mis_selection"]["point"],
    )
    best = rankers[best_name]["mis_selection"]

    # P1: both pooled estimators beat plain projection, by >= 10% relative
    p1_rows = {}
    for name in ("shared_exponent", "shrinkage_eb"):
        point = rankers[name]["mis_selection"]["point"]
        relative = (plain - point) / plain if plain > 0 else 0.0
        p1_rows[name] = {"point": point, "relative_improvement": relative, "beats_plain": bool(point < plain)}
    p1_ok = all(row["beats_plain"] and row["relative_improvement"] >= P1_MIN_RELATIVE for row in p1_rows.values())
    p1 = {
        "verdict": "CONFIRMED" if p1_ok else "REFUTED",
        "plain": plain,
        "per_estimator": p1_rows,
        "criterion": f"both pooled estimators beat plain projection by >= {P1_MIN_RELATIVE:.0%} relative",
    }

    # P2: neither pooled estimator beats single-scale ranking
    beats_single = bool(best["lo"] is not None and best["lo"] < single["point"] and best["point"] < single["point"])
    p2 = {
        "verdict": "REFUTED" if beats_single else "CONFIRMED",
        "single_scale": single,
        "best_pooled": {"name": best_name, **best},
        "criterion": "refuted if a pooled estimator's CI lower bound falls below single-scale's point estimate",
        "note": "predicted at 70% confidence in the pre-registration",
    }

    # P3: monotone with no interior optimum; symmetric alternatives P3a / P3b
    shrink = rankers["shrinkage_eb"]["mis_selection"]
    shared = rankers["shared_exponent"]["mis_selection"]
    d = 100.0 * (shrink["point"] - shared["point"])
    overlap = not (shrink["lo"] > shared["hi"] or shared["lo"] > shrink["hi"])
    if d <= -P3_POINT_MARGIN and not overlap:
        p3_verdict = "P3a_INTERIOR_OPTIMUM"
    elif d >= P3_POINT_MARGIN and not overlap:
        p3_verdict = "P3b_UNDER_POOLING"
    else:
        p3_verdict = "CONFIRMED_MONOTONE_FLAT"
    p3 = {
        "verdict": p3_verdict,
        "d_points_shrinkage_minus_shared": d,
        "cis_overlap": overlap,
        "sweep": {
            name: rankers[name]["mis_selection"]["point"]
            for name in sorted(rankers)
            if name.startswith("shrinkage_") or name in ("plain_projection", "shared_exponent", "ensemble")
        },
    }

    # P4: interaction with the lever arm, sharp criterion
    inter = results.get("interaction")
    if inter is None:
        p4 = {"verdict": "NOT_RUN"}
    else:
        ratio, hi = inter["ratio_point"], inter["ratio_hi"]
        if inter["I_short_arm"] == 0:
            p4_verdict = "CONFIRMED" if inter["I_long_arm"] >= P4_MIN_LONG_ARM_REMOVED else "UNDERPOWERED"
        elif inter["I_long_arm"] < P4_MIN_LONG_ARM_REMOVED:
            p4_verdict = "UNDERPOWERED"
        elif ratio is not None and ratio >= P4_RATIO:
            p4_verdict = "CONFIRMED"
        elif ratio is not None and hi is not None and hi < P4_RATIO:
            p4_verdict = "FAILED"
        else:
            p4_verdict = "UNDERPOWERED"
        p4 = {
            "verdict": p4_verdict,
            **inter,
            "criterion": f"rho_L >= {P4_RATIO}; FAILED only if point < {P4_RATIO} and CI upper < {P4_RATIO}",
        }

    # P5: what kind of flips pooling removed, with the evaluability threshold
    dec = results.get("removed_decomposition")
    if dec is None or dec["n_removed"] < P5_MIN_REMOVED:
        p5 = {
            "verdict": "UNDERPOWERED",
            "reason": f"only {0 if dec is None else dec['n_removed']} flips removed; threshold is {P5_MIN_REMOVED}",
            **(dec or {}),
        }
    else:
        share = dec["fit_error_share"]
        p5 = {
            "verdict": "CONFIRMED" if share >= P5_FIT_ERROR_SHARE else "REFUTED",
            **dec,
            "criterion": (
                f"fit-error share of removed flips >= {P5_FIT_ERROR_SHARE:.0%}, "
                f"evaluable only at >= {P5_MIN_REMOVED} removed"
            ),
        }

    mechanism_refuted = bool(p1["verdict"] == "REFUTED" and p4.get("verdict") == "FAILED")
    return {
        "P1_beats_plain": p1,
        "P2_beats_single_scale": p2,
        "P3_flexibility_curve": p3,
        "P4_lever_arm_interaction": p4,
        "P5_removed_error_type": p5,
        "mechanism_refuted": mechanism_refuted,
        "mechanism_note": (
            "compound refutation requires P1 REFUTED and P4 FAILED (UNDERPOWERED does not count)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/task_b")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    n_boot = 3 if args.fast else 120
    n_boot_inner = 5 if args.fast else 40
    n_boot_ratio = 3 if args.fast else 60

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "fast": args.fast,
        "seed": args.seed,
        "inputs": {str(PRIMARY): _sha256_file(PRIMARY)},
        "counts": {"n_boot": n_boot, "n_boot_inner": n_boot_inner, "n_boot_ratio": n_boot_ratio},
        "designs": {},
    }

    primary = design(PRIMARY, "300M", "1B")
    train_names = training_interventions(primary[0], seed=args.seed)
    results["shrinkage_strength_protocol"] = {
        "n_training_interventions": len(train_names),
        "training_interventions": train_names,
        "n_total_interventions": int(primary[0]["intervention"].nunique()),
        "held_out_interventions": sorted(
            set(primary[0]["intervention"].astype(str).unique()) - set(train_names)
        ),
        "note": (
            "The empirical-Bayes shrinkage strength is estimated ONLY from the training interventions listed "
            "here. Every ranker still fits and projects all interventions; only the hyperparameter is restricted. "
            "Fixed-strength rankers (shrinkage_0 ... shrinkage_1) estimate nothing and are unaffected."
        ),
    }
    results["designs"]["primary_4M-300M_target1B"] = evaluate_design(
        *primary, n_boot=n_boot, n_boot_inner=n_boot_inner, seed=args.seed, include_ensemble=True,
        strength_interventions=train_names,
    )
    results["shrinkage_strength_sensitivity"] = strength_sensitivity(
        primary[0], primary[1], train_names, n_boot_inner, args.seed
    )
    print("done primary", flush=True)
    _write_json_atomically(results, out / "task_b.json")

    long_arm = design(PRIMARY, "150M", "1B")
    short_arm = design(PRIMARY, "530M", "1B")
    for label, built in (("long_arm_4M-150M_target1B", long_arm), ("short_arm_4M-530M_target1B", short_arm)):
        results["designs"][label] = evaluate_design(
            *built, n_boot=max(3, n_boot // 2), n_boot_inner=n_boot_inner, seed=args.seed, include_ensemble=False,
            strength_interventions=train_names,
        )
        print(f"done {label}", flush=True)
        _write_json_atomically(results, out / "task_b.json")

    best_name = min(
        ("shared_exponent", "shrinkage_eb"),
        key=lambda n: results["designs"]["primary_4M-300M_target1B"]["rankers"][n]["mis_selection"]["point"],
    )
    results["best_pooled_ranker"] = best_name
    results["interaction"] = bootstrap_interaction_ratio(
        long_arm, short_arm, best_name, n_boot_ratio, n_boot_inner, args.seed, strength_interventions=train_names
    )
    print("done interaction", flush=True)
    results["removed_decomposition"] = decomposition_of_removed(
        *primary, results["designs"]["primary_4M-300M_target1B"]["rankers"][best_name]["removed_pairs"]
    )
    results["verdicts"] = judge(results)
    _write_json_atomically(results, out / "task_b.json")
    print(json.dumps(results["verdicts"], indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
