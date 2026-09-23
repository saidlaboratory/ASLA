"""Candidate-level tests of published decision-accuracy comparisons.

Pre-registered in PREDICTIONS_TASK_PUBLISHED_COMPARISONS.md. Neither DataDecide
nor Signal and Noise reports an interval on a difference between methods. This
script reconstructs the comparisons the released data supports and asks whether
each published conclusion holds when the 25 recipes, not the 300 pairs, are the
unit of inference.

* D1 --- DataDecide's scaling-law variants against single-scale ranking, from the
  released per-recipe predictions (``scaling_law_fit``), gated on reproducing
  the released ``decision_acc`` exactly.
* D2 --- Correct Prob against accuracy for single-scale ranking, from the
  released per-seed macro-average evaluations.
* The pair-resampling correction curve R(n) = sqrt(1 + 2(n-2) zeta1/zeta2) at
  every comparison's measured components, with a subsampling check on C4.

Writes ``results/external/published_comparisons.json``.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.target_scoring import (
    expected_charges,
    target_evidence,
    u_statistic_components,
    u_statistic_test,
    u_statistic_variance,
)
from asla.data.sources.datadecide import _filter_seed_replicates, select_common_checkpoints

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "datadecide"
TARGET = "1B"
MACRO = "olmes_10_macro_avg"
TASKS = (
    MACRO,
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
# The eight baselines of DataDecide section 3.2 / Appendix C, fitted on the default size set.
SL_VARIANTS = (
    "3_param-default",
    "2_param-default",
    "5_param-ai2",
    "3_param-1_step",
    "5_param-1_step-ai2",
    "3_param-default-helper_points",
    "3_param-default-step2=0.5",
    "3_param-default-helper_points-step2=0.5",
)
SMALL_SCALES = ("4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M")
SL_COMPARATORS = ("750M", "150M")
FIELDS = ("primary_metric", "correct_prob_per_char")
CURVE_N = (5, 10, 25, 50, 100, 200)
SUBSAMPLE_SIZES = (8, 12, 16, 20)
N_SUBSAMPLES = 2000
SEED = 20260922


def load_task(task: str) -> pd.DataFrame:
    """Final common checkpoint per (scale, recipe, seed) for one task, with the needed metrics."""

    raw = pd.read_parquet(RAW / "eval_macro_avg.parquet", columns=["params", "data", "task", "step", "seed", "metrics"])
    raw = _filter_seed_replicates(raw[raw["task"] == task].drop(columns=["task"]))
    chosen = select_common_checkpoints(raw)
    raw = raw.merge(chosen, on=["params", "step"])
    parsed = [json.loads(text) for text in raw["metrics"].tolist()]
    for field in FIELDS:
        raw[field] = [float(entry[field]) for entry in parsed]
    return raw.drop(columns=["metrics"]).reset_index(drop=True)


def _agreement(values: pd.Series, target: pd.Series) -> dict[tuple[str, str], float]:
    """1 if the predicted order matches the target order, 0 if not, 0.5 on a tie. Higher is better."""

    out = {}
    for a, b in combinations(sorted(target.index), 2):
        predicted = float(np.sign(values[a] - values[b]))
        truth = float(np.sign(target[a] - target[b]))
        out[(a, b)] = 0.5 if predicted == 0 or truth == 0 else float(predicted == truth)
    return out


def targets(table: pd.DataFrame) -> dict[str, pd.Series]:
    big = table[table["params"] == TARGET]
    return {
        "three_seed_mean": big.groupby("data")["primary_metric"].mean(),
        "default_seed": big[big["seed"] == "default"].set_index("data")["primary_metric"],
    }


def single_scale(table: pd.DataFrame, scale: str, field: str, target: pd.Series) -> dict[tuple[str, str], float]:
    """DataDecide's per-seed protocol: rank by each small-scale seed, average agreement over seeds."""

    rows = table[table["params"] == scale]
    per_seed = [_agreement(group.set_index("data")[field], target) for _, group in rows.groupby("seed")]
    return {pair: float(np.mean([s[pair] for s in per_seed])) for pair in per_seed[0]}


def _mapping(fits: pd.DataFrame, default_target: pd.Series) -> dict[str, str]:
    """Released-fit recipe names to eval-table names, by exact equality of the 1B default-seed target."""

    anchor = fits[(fits["task"] == MACRO) & (fits["metric"] == "primary_metric") & (fits["setup"] == SL_VARIANTS[0])]
    mapping = {}
    for mix, y in anchor.set_index("mix")["stacked_y"].items():
        distance = (default_target - y).abs()
        if distance.min() > 1e-12 or (distance < 1e-12).sum() != 1:
            raise ValueError(f"no unique exact target match for {mix}")
        mapping[str(mix)] = str(distance.idxmin())
    return mapping


def summarise(kernel: dict[tuple[str, str], float], baseline_rate: float) -> dict[str, Any]:
    test = u_statistic_test({pair: 100 * value for pair, value in kernel.items()})
    values = np.array([100 * v for v in kernel.values()])
    pair_se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "difference_pp": float(test["mean"]),
        "baseline_decision_accuracy": baseline_rate,
        "ci_low_pp": float(test["ci_low"]),
        "ci_high_pp": float(test["ci_high"]),
        "p_two_sided": float(test["p_two_sided"]),
        "excludes_zero": bool(test["excludes_zero"]),
        "candidate_se_pp": float(test["standard_error"]),
        "pair_resampled_se_pp": pair_se,
        "zeta1_over_zeta2": float(test["zeta1"] / test["zeta2"]) if test["zeta2"] > 0 else float("nan"),
    }


def d1(table_by_task: dict[str, pd.DataFrame]) -> dict[str, Any]:
    fits = pd.read_parquet(RAW / "eval_scaling_law_fit.parquet")
    mapping = _mapping(fits, targets(table_by_task[MACRO])["default_seed"])
    gate_rows, comparisons = [], {}
    for task in TASKS:
        tgt = targets(table_by_task[task])
        sub = fits[(fits["task"] == task) & (fits["metric"] == "primary_metric")]
        for setup in SL_VARIANTS:
            rows = sub[sub["setup"] == setup].assign(data=lambda f: f["mix"].map(mapping)).set_index("data")
            released = float(rows["decision_acc"].iloc[0]) / 100
            own_target = _agreement(rows["stacked_pred"], rows["stacked_y"])
            gate_rows.append(
                {
                    "task": task,
                    "setup": setup,
                    "released": released,
                    "recomputed_with_released_target": float(np.mean(list(own_target.values()))),
                    "recomputed_with_three_seed_mean": float(
                        np.mean(list(_agreement(rows["stacked_pred"], tgt["three_seed_mean"]).values()))
                    ),
                }
            )
            for target_name, target in tgt.items():
                law = _agreement(rows["stacked_pred"], target)
                for scale in SL_COMPARATORS:
                    base = single_scale(table_by_task[task], scale, "primary_metric", target)
                    kernel = {pair: law[pair] - base[pair] for pair in law}
                    comparisons[f"{task}|{setup}|{scale}|{target_name}"] = summarise(
                        kernel, float(np.mean(list(base.values())))
                    )
    gate_error = max(abs(r["recomputed_with_released_target"] - r["released"]) for r in gate_rows)
    return {
        "recipe_mapping": mapping,
        "gate": {
            "max_abs_error_released_target": gate_error,
            "n_rows_exact": sum(abs(r["recomputed_with_released_target"] - r["released"]) < 1e-6 for r in gate_rows),
            "passes": gate_error < 1e-6,
            "max_abs_error_three_seed_mean_target": max(
                abs(r["recomputed_with_three_seed_mean"] - r["released"]) for r in gate_rows
            ),
            "rows": gate_rows,
        },
        "comparisons": comparisons,
    }


def d2(table_by_task: dict[str, pd.DataFrame]) -> dict[str, Any]:
    comparisons = {}
    for task in TASKS:
        tgt = targets(table_by_task[task])["three_seed_mean"]
        for scale in SMALL_SCALES:
            prob = single_scale(table_by_task[task], scale, "correct_prob_per_char", tgt)
            acc = single_scale(table_by_task[task], scale, "primary_metric", tgt)
            kernel = {pair: prob[pair] - acc[pair] for pair in prob}
            comparisons[f"{task}|{scale}"] = summarise(kernel, float(np.mean(list(acc.values()))))
    return {"comparisons": comparisons}


def c4_kernel() -> dict[tuple[str, str], float]:
    """The C4 projection-vs-single-scale expected-error kernel, as in run_expected_error.py."""

    import importlib
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    ee = importlib.import_module("scripts.run_expected_error")
    frame = ee.load("c4_en_bits_per_token")
    target = float(frame[frame["scale_label"] == "530M"]["compute"].iloc[0])
    max_compute = float(frame[frame["scale_label"] == "300M"]["compute"].iloc[0])
    budgets = ee.normalize_budgets(sorted(float(c) for c in frame["compute"].unique() if c <= max_compute), target=target)
    cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
    evidence = target_evidence(
        {str(k): float(v) for k, v in cells.mean().items()},
        {str(k): float(v) for k, v in cells.std(ddof=1).items()},
        {str(k): int(v) for k, v in cells.count().items()},
    )
    values = ee.predictions_for(frame, budgets, target)
    names = sorted(str(k) for k in cells.mean().index)
    left = expected_charges(ee._gaps(values["projection"], names), evidence)
    right = expected_charges(ee._gaps(values["single_scale"], names), evidence)
    return {pair: 100 * (left[pair] - right[pair]) for pair in left if pair in right}


def ratio_curve(zeta_ratio: float) -> dict[str, float]:
    return {str(n): float(np.sqrt(1 + 2 * (n - 2) * zeta_ratio)) for n in CURVE_N}


def correction_curve(kernel: dict[tuple[str, str], float], rng: np.random.Generator) -> dict[str, Any]:
    """R(n) at the C4 components, and P5: does subsampling the real candidates reproduce R(m)?"""

    components = u_statistic_components(kernel)
    n = int(components["n_candidates"])
    ratio = components["zeta1"] / components["zeta2"]
    full_variance = u_statistic_variance(components["zeta1"], components["zeta2"], n)
    names = sorted({name for pair in kernel for name in pair})
    rows = []
    for m in SUBSAMPLE_SIZES:
        statistics, pair_ses = [], []
        for _ in range(N_SUBSAMPLES):
            chosen = sorted(rng.choice(names, size=m, replace=False))
            values = np.array([kernel[pair] for pair in combinations(chosen, 2)])
            statistics.append(values.mean())
            pair_ses.append(values.std(ddof=1) / np.sqrt(values.size))
        # Subsampling from the 25 observed candidates has variance V(m) - V(25);
        # adding back V(25) estimates the superpopulation V(m).
        candidate_sd = float(np.sqrt(np.var(statistics, ddof=1) + full_variance))
        empirical = candidate_sd / float(np.mean(pair_ses))
        predicted = float(np.sqrt(1 + 2 * (m - 2) * ratio))
        rows.append(
            {"m": m, "empirical_ratio": empirical, "formula_ratio": predicted, "relative_error": empirical / predicted - 1}
        )
    return {
        "formula": "R(n) = sqrt(1 + 2 (n - 2) zeta1 / zeta2)",
        "zeta1_over_zeta2": ratio,
        "curve": ratio_curve(ratio),
        "at_observed_n": float(np.sqrt(1 + 2 * (n - 2) * ratio)),
        "subsampling_check": rows,
        "P5_confirmed": all(abs(r["relative_error"]) <= 0.15 for r in rows),
    }


def predictions(d1_out: dict[str, Any], d2_out: dict[str, Any], curve: dict[str, Any]) -> dict[str, Any]:
    primary = {
        key: row
        for key, row in d1_out["comparisons"].items()
        if key.startswith(f"{MACRO}|") and key.endswith("|750M|three_seed_mean")
    }
    macro_d2 = {key: row for key, row in d2_out["comparisons"].items() if key.startswith(f"{MACRO}|")}
    cp_better = [key for key, row in macro_d2.items() if row["excludes_zero"] and row["difference_pp"] > 0]
    return {
        "P1_gate": {"confirmed": d1_out["gate"]["passes"], "max_abs_error": d1_out["gate"]["max_abs_error_released_target"]},
        "P2_no_variant_differs": {
            "n_excluding_zero": sum(row["excludes_zero"] for row in primary.values()),
            "n_variants": len(primary),
            "confirmed": not any(row["excludes_zero"] for row in primary.values()),
        },
        "P3_correct_prob_rarely_established": {
            "scales_where_correct_prob_reliably_better": [key.split("|")[1] for key in cp_better],
            "confirmed": len(cp_better) <= 2,
        },
        "P4_signal_and_noise_bpb": {
            "status": "not reconstructed",
            "reason": (
                "the Figure 2 table names 15 of the 30 tasks plus group averages; the 30-task membership is stated "
                "neither in the paper nor in the released data, so the published 77.0 / 83.7 cannot be rebuilt"
            ),
            "confirmed": None,
        },
        "P5_correction_curve": {"confirmed": curve["P5_confirmed"]},
    }


def _curve_range(ratios: list[float]) -> dict[str, Any]:
    """R(n) at the 10th, 50th and 90th percentile of zeta1/zeta2 across comparisons.

    The unbiased zeta1 estimate can fall below zero on a single comparison; the
    true zeta1 is a variance, so it is clipped at zero before the curve is drawn.
    """

    clipped = np.clip(np.asarray(ratios, dtype=float), 0.0, None)
    quantiles = {q: float(np.percentile(clipped, q)) for q in (10, 50, 90)}
    return {
        "n_comparisons": len(ratios),
        "n_negative_zeta1_estimates": int((np.asarray(ratios) < 0).sum()),
        "zeta1_over_zeta2_percentiles": {str(q): v for q, v in quantiles.items()},
        "curve_by_percentile": {str(q): ratio_curve(v) for q, v in quantiles.items()},
    }


def _signs(comparisons: dict[str, Any], target_name: str) -> dict[str, int]:
    rows = [row for key, row in comparisons.items() if key.endswith(f"|{target_name}")]
    return {
        "n": len(rows),
        "law_reliably_better": sum(r["excludes_zero"] and r["difference_pp"] > 0 for r in rows),
        "law_reliably_worse": sum(r["excludes_zero"] and r["difference_pp"] < 0 for r in rows),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    table_by_task = {task: load_task(task) for task in TASKS}
    d1_out = d1(table_by_task)
    d2_out = d2(table_by_task)
    curve = correction_curve(c4_kernel(), rng)
    zetas = [row["zeta1_over_zeta2"] for block in (d1_out, d2_out) for row in block["comparisons"].values()]
    finite = [z for z in zetas if np.isfinite(z)]
    out = {
        "sources": {
            "datadecide": "arXiv:2504.11393v2; released allenai/DataDecide-eval-results (macro_avg, scaling_law_fit)",
            "datadecide_uncertainty_as_published": (
                "SD over 3 small-model seed prediction attempts for single-scale ranking; one attempt, no interval, "
                "for scaling laws; no bootstrap, confidence interval or test on any method difference"
            ),
            "signal_and_noise_uncertainty_as_published": (
                "point estimates for method comparisons; resampling only over final checkpoints"
            ),
        },
        "n_candidates": 25,
        "d1_scaling_laws_vs_single_scale": d1_out,
        "d2_correct_prob_vs_accuracy": d2_out,
        "correction_curve_c4": curve,
        "correction_curve_range": _curve_range(finite),
        "uncheckable": {
            "signal_and_noise_table_1_checkpoint_averaging": "source-size aggregation not stated",
            "signal_and_noise_bpb_30_task_average": "30-task membership not stated in the paper or released data",
            "signal_and_noise_high_snr_subsets": "subset selections not released",
            "datadecide_intermediate_checkpoints": "figure-only, no released per-method outputs",
        },
        "d1_gate_note": (
            "Open discrepancy under investigation with the authors: no convention tried (prediction column, target "
            "column or seed mean, tie rule) matches every released decision_acc. D1 is exploratory until it is "
            "resolved and makes no claim about the paper's numbers. Reproduction package: repro/datadecide_decision_acc."
        ),
        "d1_sign_breakdown_three_seed_target": _signs(d1_out["comparisons"], "three_seed_mean"),
    }
    out["predictions"] = predictions(d1_out, d2_out, curve)
    path = REPO / "results" / "external" / "published_comparisons.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["predictions"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
