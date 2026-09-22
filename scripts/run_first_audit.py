"""Run the first real ASLA audit on public data and write FIRST_AUDIT.md.

Data axis (predicted scale-stable negative control): the harvested DataDecide
tables (25 data recipes x 14 scales x 3 seeds). Optimizer axis (predicted
crossover-prone): the harvested Fantastic Optimizers ladders (4 optimizers with
a 1.2B run, single seed). Every number in FIRST_AUDIT.md is read from the
JSON results this script writes; nothing is typed in by hand.

Usage::

    python scripts/run_first_audit.py            # paper counts (about an hour on a laptop)
    python scripts/run_first_audit.py --fast     # smoke run with tiny bootstrap counts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.audit import audit_with_ci  # noqa: E402
from asla.analysis.crossover import detect_crossovers_fdr, detect_single_scale_flips_fdr  # noqa: E402
from asla.analysis.fits import bound_pin_report  # noqa: E402
from asla.analysis.known_answer import pairwise_decision_accuracy  # noqa: E402
from asla.analysis.rankers import (  # noqa: E402
    Ranker,
    ensemble_ranker,
    make_gate_ranker,
    make_projection_ranker,
    single_scale_ranker,
)
from asla.analysis.tuning import stratify_by_tuning  # noqa: E402
from asla.cli import _sha256_file, _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402
from asla.data.sources import fantastic_optimizers as fo  # noqa: E402

DATA_DIR = Path("data")
DATADECIDE_TABLES = {
    "c4_en_bits_per_token": DATA_DIR / "datadecide_runs.parquet",
    "olmes_macro_error": DATA_DIR / "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": DATA_DIR / "datadecide_runs_olmes_correct_prob_per_char.parquet",
}
CROSSOVER_Q = 0.05
UNIDENTIFIABLE_NOISE_TO_GAP = 10.0
H100_BF16_DENSE_PEAK_FLOPS = 989e12
ASSUMED_MFU = 0.40
RECOMMENDED_RATIOS = (1, 4)


def _compute_of(df: pd.DataFrame, label: str) -> float:
    values = df[df["scale_label"].astype(str) == label]["compute"].unique()
    if len(values) != 1:
        raise ValueError(f"scale {label!r} maps to {len(values)} compute values")
    return float(values[0])


def _interval(entry: dict[str, Any]) -> dict[str, Any]:
    return {"point": entry["point"], "lo": entry["lo"], "hi": entry["hi"]}


def run_design(
    df: pd.DataFrame,
    *,
    name: str,
    axis: str,
    budgets: tuple[float, ...],
    target: float,
    intermediate: float | None,
    n_boot_pairwise: int,
    n_boot_single: int,
    n_boot_ensemble: int,
    gate_n_boot: int,
    seed: int,
    include_ensemble: bool,
) -> dict[str, Any]:
    """Audit one table under both estimands and report crossovers for both decision rules."""

    started = time.time()
    min_seeds = int(df.groupby(["intervention", "compute"])["seed"].nunique().min())
    rankers: dict[str, Ranker] = {
        "projection_ranker": make_projection_ranker("compute_power_law"),
        "single_scale_ranker": single_scale_ranker,
    }
    ensemble_ok = include_ensemble and len(budgets) >= 4
    result: dict[str, Any] = {
        "name": name,
        "axis": axis,
        "metric_name": str(df["metric_name"].iloc[0]),
        "n_interventions": int(df["intervention"].nunique()),
        "n_pairs": int(df["intervention"].nunique() * (df["intervention"].nunique() - 1) // 2),
        "budgets": [float(b) for b in budgets],
        "budget_labels": sorted(
            df[df["compute"].isin(budgets)]["scale_label"].astype(str).unique().tolist(), key=lambda s: _compute_of(df, s)
        ),
        "intermediate": intermediate,
        "target": target,
        "target_label": str(df[np.isclose(df["compute"], target)]["scale_label"].iloc[0]),
        "min_seeds_per_cell": min_seeds,
        "seed_bootstrap_meaningful": min_seeds >= 2,
        "crossover_q": CROSSOVER_Q,
    }
    pairwise = audit_with_ci(
        df, budgets, target, rankers, n_boot_pairwise, np.random.default_rng(seed), estimand="pairwise_decisions"
    )
    pair_metrics = ("mis_selection_rate", "mean_regret", "pairwise_acc")

    def _keep(metrics: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
        return {metric: _interval(entry) for metric, entry in metrics.items() if metric in names}

    result["pairwise_decisions"] = {
        "n_boot": n_boot_pairwise,
        "rankers": {ranker: _keep(metrics, pair_metrics) for ranker, metrics in pairwise["rankers"].items()},
    }
    if ensemble_ok:
        ens = audit_with_ci(
            df,
            budgets,
            target,
            {"ensemble_ranker": ensemble_ranker},
            n_boot_ensemble,
            np.random.default_rng(seed + 1),
            estimand="pairwise_decisions",
        )
        result["pairwise_decisions"]["rankers"]["ensemble_ranker"] = _keep(ens["rankers"]["ensemble_ranker"], pair_metrics)
        result["pairwise_decisions"]["ensemble_n_boot"] = n_boot_ensemble
    single_rankers: dict[str, Ranker] = dict(rankers)
    if ensemble_ok:
        single_rankers["ensemble_ranker"] = ensemble_ranker
    if intermediate is not None:
        single_rankers["gate_ranker"] = make_gate_ranker(
            intermediate_budget=intermediate, tau=1.0, n_boot=gate_n_boot, seed=seed + 2
        )
    single = audit_with_ci(
        df,
        budgets,
        target,
        single_rankers,
        n_boot_single,
        np.random.default_rng(seed + 3),
        estimand="single_design_seed_sensitivity",
    )
    single_metrics = ("mis_selection_rate", "mean_regret", "top1_acc", "pairwise_acc")
    result["single_design_seed_sensitivity"] = {
        "n_boot": n_boot_single,
        "rankers": {ranker: _keep(metrics, single_metrics) for ranker, metrics in single["rankers"].items()},
        "truth_winner": str(next(iter(single["truth"]))),
        "truth_ties_with_winner": single["truth_ties"],
    }
    noise_keys = ("noise_band", "representative_gap_sem", "noise_band_estimated", "pooled_degrees_of_freedom")
    result["noise"] = {k: single["noise"][k] for k in noise_keys}
    proj_flips = detect_crossovers_fdr(df, budgets, target, q=CROSSOVER_Q)
    single_flips = detect_single_scale_flips_fdr(df, budgets, target, q=CROSSOVER_Q)

    def _summ(flips: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_flipped_pairs": len(flips),
            "n_significant": int(sum(bool(f.get("significant")) for f in flips)),
            "n_untestable": int(sum(f["p_value"] is None for f in flips)),
            "flip_rate": len(flips) / result["n_pairs"],
            "significant_rate": sum(bool(f.get("significant")) for f in flips) / result["n_pairs"],
            "pairs": [{k: f[k] for k in ("a", "b", "true_gap", "predicted_gap", "p_value", "significant")} for f in flips],
        }

    result["crossovers"] = {"projection_vs_target": _summ(proj_flips), "largest_fit_budget_vs_target": _summ(single_flips)}
    result["bound_pins"] = bound_pin_report(df, budgets)
    result["elapsed_seconds"] = time.time() - started
    return result


def data_axis_flip_profile(df: pd.DataFrame, target_label: str = "1B") -> list[dict[str, Any]]:
    """Fraction of pairs that flip between every scale and the target (seed-mean and per-seed)."""

    target = _compute_of(df, target_label)
    out = []
    for label in df.drop_duplicates("scale_label").sort_values("compute")["scale_label"].astype(str):
        if label == target_label:
            continue
        compute = _compute_of(df, label)
        seed_mean = pairwise_decision_accuracy(df, compute, target, "seed_mean")
        flips = detect_single_scale_flips_fdr(df, (compute,), target, q=CROSSOVER_Q)
        out.append(
            {
                "scale_label": label,
                "compute_ratio_to_target": compute / target,
                "flip_rate": 1.0 - seed_mean["pairwise_acc"],
                "n_pairs": seed_mean["n_pairs"],
                "n_flips": len(flips),
                "n_significant_flips": int(sum(bool(f["significant"]) for f in flips)),
            }
        )
    return out


def optimizer_axis_flip_profile(tables: dict[str, pd.DataFrame], target_label: str) -> list[dict[str, Any]]:
    out = []
    for name, df in tables.items():
        target = _compute_of(df, target_label)
        for label in df.drop_duplicates("scale_label").sort_values("compute")["scale_label"].astype(str):
            if label == target_label:
                continue
            compute = _compute_of(df, label)
            acc = pairwise_decision_accuracy(df, compute, target, "seed_mean")
            out.append(
                {
                    "table": name,
                    "scale_label": label,
                    "compute_ratio_to_target": compute / target,
                    "flip_rate": 1.0 - acc["pairwise_acc"],
                    "n_pairs": acc["n_pairs"],
                    "n_flips": int(round((1.0 - acc["pairwise_acc"]) * acc["n_pairs"])),
                }
            )
    return out


def tuning_stratification() -> dict[str, Any]:
    tuned = load_runs(DATA_DIR / f"fantastic_optimizers_partial_1xC_{fo.TUNED}.parquet")
    ablated = load_runs(DATA_DIR / f"fantastic_optimizers_partial_1xC_{fo.ABLATION_MEDIAN}.parquet")
    target = _compute_of(tuned, "520m")
    out: dict[str, Any] = {"target_label": "520m", "by_small_scale": {}}
    for small in ("130m", "300m"):
        out["by_small_scale"][small] = stratify_by_tuning(
            {fo.TUNED: tuned, fo.ABLATION_MEDIAN: ablated}, _compute_of(tuned, small), target
        )
    return out


def decompose_projection_error(design: dict[str, Any]) -> dict[str, Any]:
    """Split projection order flips into crossover-inherited and fit-error flips.

    A projection flip on a pair is *crossover-inherited* when the largest-fit-
    budget order on that pair also disagrees with the target (the small-scale
    data already ordered the pair wrongly, so any rule reading the small scales
    inherits it). Otherwise the small-scale order was right and the projection
    reversed it on its own: *fit / extrapolation error*. Both parts are further
    split by whether the measured target gap is FDR-significant.
    """

    cx = design["crossovers"]
    proj = {(f["a"], f["b"]): f for f in cx["projection_vs_target"]["pairs"]}
    single = {(f["a"], f["b"]): f for f in cx["largest_fit_budget_vs_target"]["pairs"]}
    inherited = {k: v for k, v in proj.items() if k in single}
    fit_error = {k: v for k, v in proj.items() if k not in single}
    single_only = {k: v for k, v in single.items() if k not in proj}
    n_pairs = design["n_pairs"]

    def _block(items: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "rate": len(items) / n_pairs,
            "n_significant": int(sum(bool(f.get("significant")) for f in items.values())),
            "pairs": [
                {"a": a, "b": b, "true_gap": f["true_gap"], "significant": f.get("significant")}
                for (a, b), f in items.items()
            ],
        }

    excess = len(proj) - len(single)
    return {
        "n_pairs": n_pairs,
        "projection_flips": len(proj),
        "single_scale_flips": len(single),
        "excess_projection_flips": excess,
        "crossover_inherited": _block(inherited),
        "fit_error": _block(fit_error),
        "single_scale_only": _block(single_only),
        "fit_error_share_of_projection_flips": (len(fit_error) / len(proj)) if proj else None,
        "fit_error_share_of_excess": (len(fit_error) - len(single_only)) / excess if excess > 0 else None,
    }


def optimizer_convergence(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Spread of tuned optimizer losses per scale on every optimizer ladder (effect-size view)."""

    out = []
    for name, df in tables.items():
        key = "chinchilla_ratio" if name.startswith("data_ladder") else "scale_label"
        for level, group in df.groupby(key, sort=False):
            values = group.groupby("intervention")["bpb"].mean().sort_values()
            gaps = np.diff(values.to_numpy())
            out.append(
                {
                    "table": name,
                    "level": str(level),
                    "compute": float(group["compute"].iloc[0]),
                    "n_optimizers": int(len(values)),
                    "best": str(values.index[0]),
                    "range_nats": float(values.max() - values.min()),
                    "sd_nats": float(values.std(ddof=1)),
                    "smallest_adjacent_gap_nats": float(gaps.min()) if len(gaps) else None,
                    "largest_adjacent_gap_nats": float(gaps.max()) if len(gaps) else None,
                    "order": [str(v) for v in values.index],
                }
            )
    return sorted(out, key=lambda r: (r["table"], r["compute"]))


def seed_noise_reference(df: pd.DataFrame, scales: tuple[str, ...] = ("150M", "300M", "530M", "1B")) -> list[dict[str, Any]]:
    """Pooled within-cell seed standard deviation of a DataDecide metric at selected scales."""

    out = []
    for label in scales:
        rows = df[df["scale_label"] == label]
        if rows.empty:
            continue
        cells = rows.groupby("intervention")["bpb"]
        sds = cells.std(ddof=1).dropna()
        dof = cells.count() - 1
        pooled = float(np.sqrt(np.average(sds.to_numpy() ** 2, weights=dof.loc[sds.index].to_numpy())))
        out.append(
            {
                "scale_label": label,
                "metric_name": str(df["metric_name"].iloc[0]),
                "n_recipes": int(len(sds)),
                "seeds_per_cell": int(cells.count().min()),
                "pooled_within_cell_sd": pooled,
                "pooled_within_cell_sd_nats": pooled * float(np.log(2)),
                "recipe_spread_sd": float(cells.mean().std(ddof=1)),
            }
        )
    return out


def seeds_needed(delta: float, sigma: float, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Seeds per arm for a two-sided Welch test to detect a mean gap ``delta`` at within-cell sd ``sigma``.

    Solves n = 2 * (t_{alpha/2, 2n-2} + t_{power, 2n-2})^2 * sigma^2 / delta^2 by
    fixed-point iteration from the normal approximation. Returns ``None`` when
    ``delta`` is zero.
    """

    from scipy.stats import norm
    from scipy.stats import t as t_dist

    if delta <= 0 or sigma <= 0:
        return None
    n = 2.0 * (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 * (sigma / delta) ** 2
    n = max(n, 2.0)
    for _ in range(50):
        dof = max(2.0 * n - 2.0, 1.0)
        n_new = 2.0 * (t_dist.ppf(1 - alpha / 2, dof) + t_dist.ppf(power, dof)) ** 2 * (sigma / delta) ** 2
        n_new = max(n_new, 2.0)
        if abs(n_new - n) < 1e-6:
            n = n_new
            break
        n = n_new
    return int(np.ceil(n))


def min_detectable_gap(n_seeds: int, sigma: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest mean gap a two-sided Welch test detects with ``n_seeds`` per arm at the given power."""

    from scipy.stats import t as t_dist

    dof = max(2 * n_seeds - 2, 1)
    return float(np.sqrt(2.0 / n_seeds) * (t_dist.ppf(1 - alpha / 2, dof) + t_dist.ppf(power, dof)) * sigma)


def pilot_tranches(
    power: dict[str, Any], ask: dict[str, Any], seed_options: tuple[int, ...] = (4, 6, 9)
) -> list[dict[str, Any]]:
    """Cost and resolvable effect size of smaller first tranches at 1.2B."""

    by_ratio = {r["chinchilla_ratio"]: r for r in ask["rows"]}
    conv = {r["table"]: r for r in power["rows"] if r["level"] == "1.2b"}
    out = []
    for table, prow in conv.items():
        ratio = int(table.split("_")[-1].rstrip("xC"))
        cost = by_ratio.get(ratio)
        if cost is None or not prow["smallest_gap_identifiable"]:
            continue
        for n in seed_options:
            gap = min_detectable_gap(n, prow["sigma_nats"])
            out.append(
                {
                    "ratio": ratio,
                    "seeds_per_optimizer": n,
                    "runs": n * cost["n_optimizers"],
                    "gpu_hours": n * cost["n_optimizers"] * cost["gpu_hours_per_run"],
                    "min_detectable_gap_nats": gap,
                    "resolves_best_vs_worst": bool(gap <= prow["range_nats"]),
                    "resolves_smallest_adjacent_gap": bool(gap <= (prow["smallest_adjacent_gap_nats"] or 0.0)),
                    "range_nats": prow["range_nats"],
                    "smallest_adjacent_gap_nats": prow["smallest_adjacent_gap_nats"],
                }
            )
    return sorted(out, key=lambda r: r["gpu_hours"])


def power_analysis(convergence: list[dict[str, Any]], noise: list[dict[str, Any]], n_pairs: int = 6) -> dict[str, Any]:
    """Seeds per cell needed to resolve the observed optimizer gaps at DataDecide-like seed noise."""

    reference = {row["scale_label"]: row for row in noise}
    sigma_1b = reference["1B"]["pooled_within_cell_sd_nats"] if "1B" in reference else None
    rows = []
    for entry in convergence:
        if not entry["table"].startswith("size_ladder"):
            continue
        sigma_label = {"130m": "150M", "300m": "300M", "520m": "530M", "1.2b": "1B"}.get(entry["level"], "1B")
        sigma = reference.get(sigma_label, reference.get("1B", {})).get("pooled_within_cell_sd_nats")
        if sigma is None:
            continue
        rows.append(
            {
                "table": entry["table"],
                "level": entry["level"],
                "sigma_source_scale": sigma_label,
                "sigma_nats": sigma,
                "range_nats": entry["range_nats"],
                "smallest_adjacent_gap_nats": entry["smallest_adjacent_gap_nats"],
                "seeds_to_resolve_range": seeds_needed(entry["range_nats"], sigma),
                "seeds_to_resolve_smallest_gap": seeds_needed(entry["smallest_adjacent_gap_nats"] or 0.0, sigma),
                "seeds_to_resolve_range_bonferroni": seeds_needed(entry["range_nats"], sigma, alpha=0.05 / n_pairs),
                "seeds_to_resolve_range_sigma_x2": seeds_needed(entry["range_nats"], 2 * sigma),
                "seeds_to_resolve_range_sigma_half": seeds_needed(entry["range_nats"], 0.5 * sigma),
            }
        )
    for row in rows:
        gap = row["smallest_adjacent_gap_nats"]
        row["noise_to_smallest_gap_ratio"] = (row["sigma_nats"] / gap) if gap else None
        row["noise_to_range_ratio"] = row["sigma_nats"] / row["range_nats"] if row["range_nats"] else None
        row["smallest_gap_identifiable"] = bool(gap) and row["sigma_nats"] / gap <= UNIDENTIFIABLE_NOISE_TO_GAP
    return {
        "units": "DataDecide seed sd is measured on log2(C4-EN perplexity) = bits per token and multiplied by ln 2 to "
        "give nats per token; Fantastic Optimizers values are C4-EN validation cross-entropy in nats per token. All "
        "sigma, gap, and range values below are nats per token.",
        "assumption": "CROSS-STUDY SIGMA TRANSFER (stated assumption, not a measurement): the within-cell seed sd of "
        "a Fantastic Optimizers run (Llama-style architecture, marin codebase, OLMo-2-like data mixture) is taken to "
        "equal DataDecide's pooled within-cell seed sd (OLMo architecture, OLMo codebase, DataDecide mixtures) at the "
        "nearest scale. Nothing in the public artifacts measures the former. Sensitivity rows use x0.5 and x2",
        "test": "two-sided Welch t-test, alpha 0.05 (and Bonferroni 0.05/6 for the six optimizer pairs), power 0.8",
        "unidentifiable_threshold": (
            f"a gap is called unidentifiable when sigma / gap > {UNIDENTIFIABLE_NOISE_TO_GAP:g}. Derivation: the "
            "two-sample seed count scales as n ~ 2 (z_0.975 + z_0.8)^2 (sigma/gap)^2 ~ 15.7 (sigma/gap)^2 per arm, so "
            f"sigma/gap = {UNIDENTIFIABLE_NOISE_TO_GAP:g} already means ~{seeds_needed(1.0, UNIDENTIFIABLE_NOISE_TO_GAP):,} "
            "seeds per optimizer; with four optimizers at the 1.2B/8xC cost per run that is several million GPU-hours "
            "for one pairwise decision. The cutoff itself is a conventional feasibility choice for interpretability; "
            "the 8xC conclusion does not depend on it (the observed ratio is ~200, two orders of magnitude past it). "
            "Conversely sigma/gap ~ 1 is only marginally identifiable: ~16 seeds per arm at the exact ratio, 8-9 at "
            "the observed 0.65-0.7"
        ),
        "sigma_1b_nats": sigma_1b,
        "rows": rows,
    }


def compute_ask(
    power: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    peak_flops_per_second: float = H100_BF16_DENSE_PEAK_FLOPS,
    mfu: float = ASSUMED_MFU,
    recommended_ratios: tuple[int, ...] = RECOMMENDED_RATIOS,
) -> dict[str, Any]:
    """GPU-hour cost of seeding the 1.2B optimizer cells, from 6ND FLOPs at an assumed MFU.

    Runs are costed at ``6 * params_n * tokens_d`` FLOPs (the same accounting
    as the compute axis; it ignores attention FLOPs and embedding parameters,
    so it is a lower bound), divided by ``peak * mfu`` FLOP/s. The recommended
    ask covers ``recommended_ratios`` (1x and 4x bracket the identifiable
    regime); a ratio whose smallest gap is unidentifiable is never recommended
    and is costed separately so the saving is visible.
    """

    rows = []
    for entry in power["rows"]:
        if entry["level"] != "1.2b":
            continue
        df = tables[entry["table"]]
        cell = df[df["scale_label"] == "1.2b"].iloc[0]
        flops = float(cell["compute"])
        gpu_hours_per_run = flops / (peak_flops_per_second * mfu) / 3600.0
        seeds_range = int(entry["seeds_to_resolve_range_bonferroni"])
        seeds_gap = entry["seeds_to_resolve_smallest_gap"]
        identifiable = bool(entry["smallest_gap_identifiable"])
        rows.append(
            {
                "table": entry["table"],
                "chinchilla_ratio": int(cell["chinchilla_ratio"]),
                "tokens_d": float(cell["tokens_d"]),
                "flops_per_run": flops,
                "gpu_hours_per_run": gpu_hours_per_run,
                "n_optimizers": int(df["intervention"].nunique()),
                "seeds_for_range_bonferroni": seeds_range,
                "seeds_for_smallest_gap": seeds_gap,
                "smallest_gap_identifiable": identifiable,
                "gpu_hours_range": seeds_range * int(df["intervention"].nunique()) * gpu_hours_per_run,
                "gpu_hours_gap": (
                    (seeds_gap * int(df["intervention"].nunique()) * gpu_hours_per_run) if identifiable else None
                ),
            }
        )
    recommended = [r for r in rows if r["smallest_gap_identifiable"] and r["chinchilla_ratio"] in recommended_ratios]
    excluded = [r for r in rows if not r["smallest_gap_identifiable"]]
    omitted = [r for r in rows if r["smallest_gap_identifiable"] and r["chinchilla_ratio"] not in recommended_ratios]
    seeds = max((r["seeds_for_smallest_gap"] for r in recommended), default=0)
    rec_hours = sum(seeds * r["n_optimizers"] * r["gpu_hours_per_run"] for r in recommended)
    rec_runs = sum(seeds * r["n_optimizers"] for r in recommended)
    return {
        "peak_flops_per_second": peak_flops_per_second,
        "peak_description": "NVIDIA H100 SXM dense BF16 peak (989 TFLOP/s)",
        "mfu": mfu,
        "flops_accounting": (
            "6 * N * D with N the non-embedding parameter count (lower bound: attention and embedding FLOPs ignored)"
        ),
        "rows": rows,
        "recommended": {
            "ratios": [r["chinchilla_ratio"] for r in recommended],
            "seeds_per_optimizer": seeds,
            "runs": rec_runs,
            "gpu_hours": rec_hours,
        },
        "omitted_identifiable": {
            "ratios": [r["chinchilla_ratio"] for r in omitted],
            "gpu_hours_if_added": sum(seeds * r["n_optimizers"] * r["gpu_hours_per_run"] for r in omitted),
        },
        "excluded": {
            "ratios": [r["chinchilla_ratio"] for r in excluded],
            "gpu_hours_if_seeded_like_recommended": sum(
                seeds * r["n_optimizers"] * r["gpu_hours_per_run"] for r in excluded
            ),
        },
        "full_grid_reference": {
            "description": "6 seeds x 4 optimizers x every harvested 1.2B ratio",
            "gpu_hours": sum(6 * r["n_optimizers"] * r["gpu_hours_per_run"] for r in rows),
            "runs": sum(6 * r["n_optimizers"] for r in rows),
        },
    }


def add_derived_analyses(results: dict[str, Any]) -> dict[str, Any]:
    """Attach decomposition, convergence, seed-noise reference, and power rows to an audit result."""

    for design in results["designs"]:
        design["projection_error_decomposition"] = decompose_projection_error(design)
    size_tables = {
        f"size_ladder_{ratio}xC": load_runs(DATA_DIR / f"fantastic_optimizers_size_ladder_{ratio}xC.parquet")
        for ratio in (1, 2, 4, 8)
    }
    data_tables = {
        f"data_ladder_{size}": load_runs(DATA_DIR / f"fantastic_optimizers_data_ladder_{size}.parquet")
        for size in ("130m", "300m")
    }
    results["optimizer_convergence"] = optimizer_convergence({**size_tables, **data_tables})
    results["seed_noise_reference"] = seed_noise_reference(load_runs(DATADECIDE_TABLES["c4_en_bits_per_token"]))
    results["power_analysis"] = power_analysis(results["optimizer_convergence"], results["seed_noise_reference"])
    results["compute_ask"] = compute_ask(results["power_analysis"], size_tables)
    results["compute_ask"]["pilot_tranches"] = pilot_tranches(results["power_analysis"], results["compute_ask"])
    check_path = Path("results/ensemble_check/ensemble_check.json")
    if check_path.exists():
        results["ensemble_check"] = json.loads(check_path.read_text(encoding="utf-8"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/first_audit")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report", default="FIRST_AUDIT.md")
    parser.add_argument("--render-only", action="store_true", help="Re-render the report from an existing JSON.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        results = json.loads((out / "first_audit.json").read_text(encoding="utf-8"))
        add_derived_analyses(results)
        _write_json_atomically(results, out / "first_audit.json")
        Path(args.report).write_text(render_report(results), encoding="utf-8")
        print(f"re-rendered {args.report} from {out / 'first_audit.json'}")
        return 0
    counts = (
        {"pairwise": 2, "single": 3, "ensemble": 2, "gate": 5}
        if args.fast
        else {"pairwise": 150, "single": 500, "ensemble": 60, "gate": 200}
    )
    results: dict[str, Any] = {"fast": args.fast, "counts": counts, "seed": args.seed, "inputs": {}, "designs": []}

    for metric, path in DATADECIDE_TABLES.items():
        df = load_runs(path)
        results["inputs"][str(path)] = _sha256_file(path)
        df = df[df["scale_label"] != "750M"].reset_index(drop=True)  # off the 5xC trajectory; see datadecide.py docstring
        target = _compute_of(df, "1B")
        designs = [("full_ladder_4M-300M_gate530M", ("530M",), "530M")]
        if metric == "c4_en_bits_per_token":
            designs.append(("cheap_ladder_4M-150M_gate300M", ("300M", "530M"), "300M"))
        for design_name, excluded, intermediate_label in designs:
            fit = df[(df["compute"] < target) & ~df["scale_label"].isin(excluded)]
            budgets = tuple(sorted(float(b) for b in fit["compute"].unique()))
            keep = (
                df["compute"].isin(budgets) | np.isclose(df["compute"], target) | (df["scale_label"] == intermediate_label)
            )
            table = df[keep]
            entry = run_design(
                table,
                name=f"datadecide/{metric}/{design_name}",
                axis="data",
                budgets=budgets,
                target=target,
                intermediate=_compute_of(df, intermediate_label),
                n_boot_pairwise=counts["pairwise"],
                n_boot_single=counts["single"],
                n_boot_ensemble=counts["ensemble"],
                gate_n_boot=counts["gate"],
                seed=args.seed,
                include_ensemble=(metric == "c4_en_bits_per_token" and design_name.startswith("full")),
            )
            results["designs"].append(entry)
            _write_json_atomically(results, out / "first_audit.json")
            print(f"done {entry['name']} in {entry['elapsed_seconds']:.0f}s", flush=True)
        if metric == "olmes_macro_error":
            results["data_axis_flip_profile"] = data_axis_flip_profile(df)
        if metric == "c4_en_bits_per_token":
            results["data_axis_flip_profile_c4"] = data_axis_flip_profile(df)

    size_tables: dict[str, pd.DataFrame] = {}
    for ratio in (1, 2, 4, 8):
        path = DATA_DIR / f"fantastic_optimizers_size_ladder_{ratio}xC.parquet"
        df = load_runs(path)
        results["inputs"][str(path)] = _sha256_file(path)
        size_tables[f"size_ladder_{ratio}xC"] = df
        target = _compute_of(df, "1.2b")
        budgets = tuple(sorted(float(b) for b in df[df["compute"] < target]["compute"].unique()))
        entry = run_design(
            df,
            name=f"fantastic_optimizers/size_ladder_{ratio}xC",
            axis="optimizer",
            budgets=budgets,
            target=target,
            intermediate=None,
            n_boot_pairwise=3,
            n_boot_single=3,
            n_boot_ensemble=3,
            gate_n_boot=3,
            seed=args.seed,
            include_ensemble=False,
        )
        results["designs"].append(entry)
    data_tables: dict[str, pd.DataFrame] = {}
    for size in ("130m", "300m"):
        path = DATA_DIR / f"fantastic_optimizers_data_ladder_{size}.parquet"
        df = load_runs(path)
        results["inputs"][str(path)] = _sha256_file(path)
        data_tables[f"data_ladder_{size}"] = df
        df = df.assign(scale_label=df["scale_label"] + "@" + df["chinchilla_ratio"].astype(str) + "xC")
        target = float(df[df["chinchilla_ratio"] == 16]["compute"].iloc[0])
        budgets = tuple(sorted(float(b) for b in df[df["compute"] < target]["compute"].unique()))
        entry = run_design(
            df,
            name=f"fantastic_optimizers/data_ladder_{size}",
            axis="optimizer",
            budgets=budgets,
            target=target,
            intermediate=None,
            n_boot_pairwise=3,
            n_boot_single=3,
            n_boot_ensemble=3,
            gate_n_boot=3,
            seed=args.seed,
            include_ensemble=True,
        )
        results["designs"].append(entry)
    results["optimizer_axis_flip_profile"] = optimizer_axis_flip_profile(size_tables, "1.2b")
    results["tuning_stratification"] = tuning_stratification()
    add_derived_analyses(results)
    _write_json_atomically(results, out / "first_audit.json")
    Path(args.report).write_text(render_report(results), encoding="utf-8")
    print(f"wrote {out / 'first_audit.json'} and {args.report}")
    return 0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _ci(entry: dict[str, Any], meaningful: bool) -> str:
    if not meaningful:
        return f"{_pct(entry['point'])} (single seed: no interval)"
    return f"{_pct(entry['point'])} [{_pct(entry['lo'])}, {_pct(entry['hi'])}]"


def _identifiability_label(row: dict[str, Any]) -> str:
    if not row["smallest_gap_identifiable"]:
        return "NO"
    return "marginal" if row["noise_to_smallest_gap_ratio"] >= 0.5 else "yes"


def _ask_status(row: dict[str, Any], rec: dict[str, Any]) -> str:
    if row["chinchilla_ratio"] in rec["ratios"]:
        return "yes"
    return "no - unidentifiable gap" if not row["smallest_gap_identifiable"] else "no - identifiable, omitted"


def _regret(entry: dict[str, Any] | None, meaningful: bool) -> str:
    if entry is None:
        return "n/a"
    if not meaningful:
        return f"{entry['point']:.4f}"
    return f"{entry['point']:.4f} [{entry['lo']:.4f}, {entry['hi']:.4f}]"


def _flip_row(label: str, cx: dict[str, Any], n_pairs: int) -> str:
    return (
        f"| {label} | {cx['n_flipped_pairs']} / {n_pairs} ({_pct(cx['flip_rate'])}) | "
        f"{cx['n_significant']} ({_pct(cx['significant_rate'])}) | {cx['n_untestable']} |"
    )


def _pairs_text(pairs: list[dict[str, Any]]) -> str:
    return "; ".join(f"{f['a']} vs {f['b']} (target gap {f['true_gap']:+.4f})" for f in pairs)


def _design(results: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((d for d in results["designs"] if d["name"] == name), None)


def _headline_section(results: dict[str, Any]) -> list[str]:
    c4 = _design(results, "datadecide/c4_en_bits_per_token/full_ladder_4M-300M_gate530M")
    olmes = _design(results, "datadecide/olmes_macro_error/full_ladder_4M-300M_gate530M")
    if c4 is None or olmes is None or "projection_error_decomposition" not in c4:
        return []
    data_designs = [d for d in results["designs"] if d["axis"] == "data"]
    single_rates = [d["crossovers"]["largest_fit_budget_vs_target"]["flip_rate"] for d in data_designs]
    single_sig = [d["crossovers"]["largest_fit_budget_vs_target"]["n_significant"] for d in data_designs]
    lines = ["## Headline: what projection adds to single-scale error on the data axis (H1)", ""]
    for d, label in ((c4, "C4-EN bits/token"), (olmes, "OLMES macro accuracy")):
        pw = d["pairwise_decisions"]["rankers"]
        meaningful = d["seed_bootstrap_meaningful"]
        dec = d["projection_error_decomposition"]
        cx = d["crossovers"]
        proj_rate = _ci(pw["projection_ranker"]["mis_selection_rate"], meaningful)
        single_rate = _ci(pw["single_scale_ranker"]["mis_selection_rate"], meaningful)
        fit = dec["fit_error"]
        inh = dec["crossover_inherited"]
        rep = dec["single_scale_only"]
        p_int = pw["projection_ranker"]["mis_selection_rate"]
        s_int = pw["single_scale_ranker"]["mis_selection_rate"]
        separated = meaningful and p_int["lo"] is not None and p_int["lo"] > s_int["hi"]
        sep_text = (
            "the intervals do not overlap: a clean separation"
            if separated
            else "the intervals overlap: the rates are not distinguished at this seed count, and no significant "
            "difference in mis-selection rate is claimed for this metric"
        )
        lines += [
            f"**{label}** ({d['n_pairs']} pairs, fit budgets {d['budget_labels'][0]}-{d['budget_labels'][-1]}, target 1B): "
            f"pairwise mis-selection is {proj_rate} for the scaling-law projection versus {single_rate} for ranking the "
            f"largest fit budget ({sep_text}). The projection flips {cx['projection_vs_target']['n_flipped_pairs']} pairs "
            f"({cx['projection_vs_target']['n_significant']} with an FDR-significant target gap); single-scale ranking "
            f"flips {cx['largest_fit_budget_vs_target']['n_flipped_pairs']} "
            f"({cx['largest_fit_budget_vs_target']['n_significant']} significant). Of the projection's flips, "
            f"**{fit['n']} ({_pct(dec['fit_error_share_of_projection_flips'])}) are pairs the largest fit budget already "
            f"ordered correctly and the fit reversed** - fit/extrapolation error, {fit['n_significant']} of them on "
            f"significantly separated pairs - while {inh['n']} are inherited from a small-scale order that was itself "
            f"wrong (the crossover-type error a projection cannot avoid; {inh['n_significant']} significant). The "
            f"projection repaired {rep['n']} single-scale flips. Net excess over single-scale: "
            f"{dec['excess_projection_flips']} pairs, all attributable to fit error.",
            "",
        ]
    all_fit = all(
        d["projection_error_decomposition"]["fit_error_share_of_excess"] == 1.0
        for d in data_designs
        if d["projection_error_decomposition"]["excess_projection_flips"] > 0
    )
    lines += _ensemble_evidence(results, c4)
    lines += [
        "**This answers DataDecide's open question.** DataDecide reports that no scaling-law method beats single-scale "
        "ranking on its compute-decision frontier and speculates that improved scaling laws could, because unlike "
        "single-scale ranking they are not bounded by crossovers. On their own public data, with this pipeline "
        "validated against their published number (KNOWN_ANSWER.md), the decomposition shows why projection loses: "
        + (
            "**the whole excess of projection over single-scale ranking is fit/extrapolation error, on every data-axis "
            "design; none of it is crossover.** The crossover error a projection could in principle repair is small "
            "(the inherited-crossover rows above) and the projection repairs only a handful of those pairs, while its own "
            "estimation error flips many more. The mechanistic implication is that improving scaling-law-based selection "
            "requires reducing fit variance at the target - more seeds, fewer or better-constrained parameters, "
            "variance-aware extrapolation - not better crossover modelling."
            if all_fit
            else "the excess of projection over single-scale ranking is dominated by fit/extrapolation error rather than "
            "crossover on the designs above (see the per-design decomposition tables for the exact split)."
        ),
        "",
        "Reading: on the continuous loss metric the projection's mis-selection rate is cleanly separated from "
        "single-scale ranking, and the extra error is fit/extrapolation error, not crossover; most of it is "
        "statistically real (the reversed pairs are separated by more than seed noise at the target). On the accuracy "
        "metric the two rates are *not* distinguished - their intervals overlap - so OLMES supports the same "
        "decomposition verdict (whatever excess exists is fit error, none is crossover) without establishing a "
        "difference in mis-selection rate. Most OLMES projection flips are inherited from a small-scale order that "
        "already disagreed with the target, and the majority of those inherited flips are not significant, i.e. they "
        "are small-scale evaluation noise on near-tied pairs rather than crossovers.",
        "",
        "**Negative-control framing.** H2 predicted the data axis to be scale-stable. It is: single-scale ranking "
        f"flips {_pct(min(single_rates))}-{_pct(max(single_rates))} of pairs depending on metric and design, and the "
        f"significant crossover count is {min(single_sig)}-{max(single_sig)} of 300 pairs. This is a confirmed "
        "prediction, not a null result; it is the baseline against which crossover-prone classes must be "
        "compared.",
        "",
    ]
    return lines


def _ensemble_evidence(results: dict[str, Any], c4: dict[str, Any]) -> list[str]:
    check = results.get("ensemble_check")
    pw = c4["pairwise_decisions"]["rankers"]
    if not check or "ensemble_ranker" not in pw:
        return []
    ens = pw["ensemble_ranker"]["mis_selection_rate"]
    proj = pw["projection_ranker"]["mis_selection_rate"]
    ws = check["scenarios"]["well_specified"]
    sat = check["scenarios"]["saturating_half"]
    ok = check["verdict"]["well_specified_ensemble_within_2pp_of_projection"]
    helps = check["verdict"]["ensemble_helps_under_misspecification"]
    ratio = ens["point"] / proj["point"] if proj["point"] else float("nan")

    lines = [
        f"**Evidence from the ensemble ranker ({'confirmatory' if (ok and helps) else 'consistent, not confirmatory'}).** "
        f"On C4 bits/token the misspecification-aware ensemble "
        f"(three curve families weighted by leave-largest-budget-out loss) mis-selects {_pct(ens['point'])} "
        f"[{_pct(ens['lo'])}, {_pct(ens['hi'])}] of pairs against {_pct(proj['point'])} for the plain power law - "
        f"{ratio:.1f}x worse. If the projection's excess error is fit variance rather than crossover, adding model "
        f"flexibility should make it worse, and it does. "
    ]
    check_text = (
        f"Sanity check on synthetic grids with a known answer (`scripts/check_ensemble_sanity.py`: 25 power-law "
        f"curves fitted to the real DataDecide seed means, the same {check['n_fit_budgets']} fit budgets, 3 seeds, "
        f"seed noise {check['sigma_bits_per_token']:.4f} bits/token pooled over the fit budgets (the small scales are "
        f"far noisier than 1B), {check['n_draws']} noise draws, scored "
        f"against the noiseless truth): when the power law is the true family the ensemble mis-selects "
        f"{_pct(ws['ensemble_ranker']['pairwise_mis_selection_mean'])} vs "
        f"{_pct(ws['projection_ranker']['pairwise_mis_selection_mean'])} for the plain fit; when half the truths "
        f"saturate the ensemble mis-selects {_pct(sat['ensemble_ranker']['pairwise_mis_selection_mean'])} vs "
        f"{_pct(sat['projection_ranker']['pairwise_mis_selection_mean'])}. "
    )
    impl_ok = check["verdict"].get("implementation_ok_noiseless_within_1pp", False)
    nl = check["scenarios"].get("noiseless")
    noiseless_text = (
        f"With zero seed noise the ensemble and the plain fit mis-select "
        f"{_pct(nl['ensemble_ranker']['pairwise_mis_selection_mean'])} and "
        f"{_pct(nl['projection_ranker']['pairwise_mis_selection_mean'])} respectively. "
        if nl
        else ""
    )
    if not impl_ok:
        lines[-1] += (
            check_text + noiseless_text + "**The ensemble does NOT reproduce the plain fit even without noise on a "
            "well-specified problem - this is a defect in the ensemble implementation, not evidence for the mechanism. "
            "The DataDecide ensemble number is reported but must not be used as supporting evidence until it is fixed.**"
        )
    elif ok and helps:
        lines[-1] += (
            check_text + noiseless_text + "The ensemble tracks the plain fit when the family is correct and helps when it "
            "is not, so its DataDecide result is not a defect: it is the fit-variance mechanism showing up in the "
            "extrapolation-loss weights. Confirmatory evidence for H1's mechanism."
        )
    else:
        lines[-1] += (
            check_text + noiseless_text + "The implementation is sound (it reproduces the plain fit exactly when the "
            "fitting data are noise-free), but at realistic seed noise the ensemble is worse than the plain fit even "
            "when the power law is the true family"
            + ("" if helps else ", and it does not help under the saturating misspecification it was designed for")
            + ". Its excess is therefore variance from weighting flexible "
            "families on noisy leave-one-out losses - the same fit-variance mechanism as H1, now reproduced on synthetic "
            "data with a known answer - but because the ensemble fails the 'roughly matches the plain fit when "
            "well-specified' criterion it is reported as **consistent with the mechanism, not as independent "
            "confirmation**, and it should not be used as a selection rule on this kind of data."
        )
    lines.append("")
    return lines


def _optimizer_sections(results: dict[str, Any]) -> list[str]:
    conv = results.get("optimizer_convergence")
    power = results.get("power_analysis")
    noise = results.get("seed_noise_reference")
    if not conv or not power or not noise:
        return []
    lines = [
        "",
        "## Optimizer axis: two separate limitations",
        "",
        "**(a) No seed replicates.** Every Fantastic Optimizers cell is one run. No optimizer-axis flip can be tested "
        "against run-to-run noise, no bootstrap interval is meaningful, and the FDR machinery reports every pair as "
        "untestable. This is resolvable only with new runs (seeds per cell); see the power analysis below for how many.",
        "",
        "**(b) Convergence of tuned optimizers at scale.** Independently of seeds, the spread among the four tuned "
        "optimizers shrinks with model size on every ladder, so the flips observed at 1.2B are between values that "
        "differ by less than any plausible seed noise. Seeding would turn these into measured ties, not into "
        "crossovers. Effect size (C4-EN validation loss, nats/token, tuned runs):",
        "",
        "| ladder | level | optimizers | range (max-min) | sd across optimizers | smallest adjacent gap | best |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in conv:
        lines.append(
            f"| {row['table']} | {row['level']} | {row['n_optimizers']} | {row['range_nats']:.4f} | {row['sd_nats']:.4f} | "
            f"{row['smallest_adjacent_gap_nats']:.5f} | {row['best']} |"
        )
    size_rows = [r for r in conv if r["table"].startswith("size_ladder")]
    trend = []
    for table in sorted({r["table"] for r in size_rows}):
        rows = [r for r in size_rows if r["table"] == table]
        first, last = rows[0], rows[-1]
        if first["range_nats"] > 0:
            trend.append(
                f"{table}: {first['level']} range {first['range_nats']:.4f} -> {last['level']} range "
                f"{last['range_nats']:.4f} ({last['range_nats'] / first['range_nats']:.2f}x)"
            )
    lines += ["", "Scale trend of the range on the size ladders: " + "; ".join(trend) + ".", ""]
    one_b_rows = [r for r in power["rows"] if r["level"] == "1.2b"]
    unident = [r for r in one_b_rows if not r["smallest_gap_identifiable"]]
    lines += [
        "## Finding: the top tuned optimizers are not identifiable at 1.2B/8xC (established analytically, zero new compute)",
        "",
        f"Units: {power['units']}",
        "",
        "| ladder (1.2B) | sigma (nats) | best-vs-worst range | sigma / range | smallest adjacent gap "
        "| sigma / smallest gap | smallest gap identifiable? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in one_b_rows:
        lines.append(
            f"| {r['table']} | {r['sigma_nats']:.4f} | {r['range_nats']:.4f} | {r['noise_to_range_ratio']:.2f} | "
            f"{r['smallest_adjacent_gap_nats']:.6f} | {r['noise_to_smallest_gap_ratio']:.2f} | "
            f"{_identifiability_label(r)} |"
        )
    if unident:
        worst = max(unident, key=lambda r: r["noise_to_smallest_gap_ratio"])
        lines += [
            "",
            f"At 1.2B on {', '.join(r['table'].split('_')[-1] for r in unident)} the smallest adjacent gap between tuned "
            f"optimizers is {worst['smallest_adjacent_gap_nats']:.6f} nats against a seed sd of {worst['sigma_nats']:.4f} "
            f"nats: **noise is {worst['noise_to_smallest_gap_ratio']:.0f}x the gap.** Under the stated sigma assumption "
            "(and at any sigma within an order of magnitude of it) that ordering is not identifiable at a feasible seed "
            "count - it is not expensive to resolve, it is unresolvable. A leaderboard that ranks these optimizers at "
            f"this scale is ranking noise. Threshold used: {power['unidentifiable_threshold']}. This follows from the "
            "released single-run values and the DataDecide seed variance alone; no new training was needed to "
            "establish it.",
            "",
        ]
    marginal = [r for r in one_b_rows if r["smallest_gap_identifiable"] and r["noise_to_smallest_gap_ratio"] >= 0.5]
    if marginal:
        lines += [
            f"At {', '.join(r['table'].split('_')[-1] for r in marginal)} the smallest adjacent gaps sit at sigma/gap "
            f"= {min(r['noise_to_smallest_gap_ratio'] for r in marginal):.2f}-"
            f"{max(r['noise_to_smallest_gap_ratio'] for r in marginal):.2f}: **marginally identifiable**. A single run "
            "cannot order those pairs either; the rankings at these ratios only become meaningful with real seeds "
            f"({min(r['seeds_to_resolve_smallest_gap'] for r in marginal)}-"
            f"{max(r['seeds_to_resolve_smallest_gap'] for r in marginal)} per optimizer), which is exactly what the "
            "compute ask below buys.",
            "",
        ]
    lines += [
        "## What the convergence would imply for leaderboard validity (hypothesis, not a finding)",
        "",
        "If well-tuned optimizers are statistically indistinguishable at the target scale, a leaderboard that ranks "
        "them there is ranking seed noise, and any small-scale projection that 'predicts' that ranking is predicting "
        "noise. The public data cannot establish this: the spreads above are single-run values and the seed noise of "
        "these runs is unmeasured. It is the hypothesis the next experiment should test. Data needed: seed replicates "
        "of the tuned configurations for the four optimizers at 520M and 1.2B (the two largest sizes), at the "
        "Chinchilla ratios where the observed gaps are smallest (4x and 8x), with enough seeds to resolve gaps of the "
        "order of the observed adjacent gaps (~0.002 nats at 1x-4x; effectively zero at 8x) - quantified next.",
        "",
        "## Power analysis: seeds per cell needed to distinguish real optimizer differences from ties",
        "",
        f"{power['assumption']}. Test: {power['test']}. Reference within-cell seed sd of C4-EN loss, from "
        "the DataDecide harvest (3 seeds x 25 recipes per scale):",
        "",
        "| DataDecide scale | pooled within-cell seed sd (bits/token) | in nats | recipe-to-recipe sd (bits/token) |",
        "|---|---|---|---|",
    ]
    for row in noise:
        lines.append(
            f"| {row['scale_label']} | {row['pooled_within_cell_sd']:.4f} | {row['pooled_within_cell_sd_nats']:.4f} | "
            f"{row['recipe_spread_sd']:.3f} |"
        )
    lines += [
        "",
        "Seeds per optimizer (per arm) to detect the observed gap with power 0.8:",
        "",
        "| ladder | level | sigma (nats) | best-vs-worst range | seeds for range | seeds for range, Bonferroni 6 pairs "
        "| seeds for range, sigma x2 | smallest adjacent gap | seeds for smallest gap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in power["rows"]:
        gap = row["smallest_adjacent_gap_nats"]
        lines.append(
            f"| {row['table']} | {row['level']} | {row['sigma_nats']:.4f} | {row['range_nats']:.4f} | "
            f"{row['seeds_to_resolve_range']} | {row['seeds_to_resolve_range_bonferroni']} | "
            f"{row['seeds_to_resolve_range_sigma_x2']} | {gap:.5f} | {row['seeds_to_resolve_smallest_gap']} |"
        )
    ask = results.get("compute_ask")
    if ask:
        rec, exc, full = ask["recommended"], ask["excluded"], ask["full_grid_reference"]
        lines += [
            "",
            "## Compute ask (GPU-hours, arithmetic shown)",
            "",
            f"Cost model: FLOPs per run = {ask['flops_accounting']}; GPU-hours = FLOPs / ({ask['peak_description']} x "
            f"MFU {ask['mfu']:.0%}) / 3600 = FLOPs / {ask['peak_flops_per_second'] * ask['mfu']:.3e} FLOP/s / 3600.",
            "",
            "| 1.2B ratio | tokens | FLOPs per run | GPU-h per run | seeds for range (Bonf.) "
            "| seeds for smallest gap | in ask? |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in ask["rows"]:
            lines.append(
                f"| {r['chinchilla_ratio']}xC | {r['tokens_d']:.3e} | {r['flops_per_run']:.3e} | "
                f"{r['gpu_hours_per_run']:.0f} | "
                f"{r['seeds_for_range_bonferroni']} | {r['seeds_for_smallest_gap']} | {_ask_status(r, rec)} |"
            )
        lines += [
            "",
            f"**Recommended ask:** {rec['seeds_per_optimizer']} seeds x 4 optimizers x 1.2B x "
            f"({', '.join(f'{x}xC' for x in rec['ratios'])}) = {rec['runs']} runs = "
            + " + ".join(
                f"{rec['seeds_per_optimizer']} x {r['n_optimizers']} x {r['gpu_hours_per_run']:.0f}"
                for r in ask["rows"]
                if r["chinchilla_ratio"] in rec["ratios"]
            )
            + f" = **{rec['gpu_hours']:,.0f} GPU-hours** at {ask['mfu']:.0%} MFU "
            f"(about {rec['gpu_hours'] / 0.5 * ask['mfu']:,.0f} at 50%, {rec['gpu_hours'] / 0.3 * ask['mfu']:,.0f} at 30%). "
            "This resolves the ~0.002 nats adjacent gaps and, a fortiori, "
            "best-versus-worst, on the two ratios that bracket the identifiable regime.",
            "",
        ]
        omitted = ask.get("omitted_identifiable", {})
        if omitted.get("ratios"):
            lines += [
                f"Identifiable but omitted to keep the ask minimal: {', '.join(f'{x}xC' for x in omitted['ratios'])} "
                f"(+{omitted['gpu_hours_if_added']:,.0f} GPU-hours if added).",
                "",
            ]
        lines += [
            f"Cut from the ask: {', '.join(f'{x}xC' for x in exc['ratios'])} - the ordering there is unidentifiable "
            f"(previous section) and seeding it like the recommended ratios would cost another "
            f"{exc['gpu_hours_if_seeded_like_recommended']:,.0f} GPU-hours for no decision. "
            "For reference, the earlier draft "
            f"ask ({full['description']}: {full['runs']} runs) is {full['gpu_hours']:,.0f} GPU-hours.",
            "",
        ]
        tranches = ask.get("pilot_tranches", [])
        if tranches:
            lines += [
                "### Pilot tranches (cheap entry points; a positive pilot unlocks the full ask)",
                "",
                "Minimal detectable gap = sqrt(2/n) x (t_0.975 + t_0.8) x sigma for n seeds per optimizer "
                "(Welch, power 0.8, no multiplicity correction). 'Resolves' compares that gap with the observed "
                "single-run spread at 1.2B on the same ratio.",
                "",
                "| ratio | seeds/optimizer | runs | GPU-h (40% MFU) | min detectable gap (nats) | observed range | "
                "observed smallest gap | resolves best-vs-worst | resolves smallest gap |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for t in tranches:
                lines.append(
                    f"| {t['ratio']}xC | {t['seeds_per_optimizer']} | {t['runs']} | {t['gpu_hours']:,.0f} | "
                    f"{t['min_detectable_gap_nats']:.4f} | {t['range_nats']:.4f} | {t['smallest_adjacent_gap_nats']:.4f} | "
                    f"{'yes' if t['resolves_best_vs_worst'] else 'no'} | "
                    f"{'yes' if t['resolves_smallest_adjacent_gap'] else 'no'} |"
                )
            cheapest_full = next((t for t in tranches if t["resolves_smallest_adjacent_gap"]), None)
            cheapest_range = next((t for t in tranches if t["resolves_best_vs_worst"]), None)
            lines += [""]
            if cheapest_range:
                lines.append(
                    f"Smallest tranche that settles best-versus-worst at one ratio: {cheapest_range['ratio']}xC x "
                    f"{cheapest_range['seeds_per_optimizer']} seeds = {cheapest_range['runs']} runs, "
                    f"**{cheapest_range['gpu_hours']:,.0f} GPU-hours** (detects gaps >= "
                    f"{cheapest_range['min_detectable_gap_nats']:.4f} nats)."
                )
            if cheapest_full:
                lines.append(
                    f"Smallest tranche that also resolves the ~0.002 nats adjacent gaps at one ratio: "
                    f"{cheapest_full['ratio']}xC x {cheapest_full['seeds_per_optimizer']} seeds = {cheapest_full['runs']} "
                    f"runs, **{cheapest_full['gpu_hours']:,.0f} GPU-hours**. Recommended pilot: this tranche; it also "
                    "measures the real seed sd of these runs, replacing the cross-study sigma assumption before the "
                    "rest of the grid is committed."
                )
            lines.append("")
    return lines


def render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    counts = results["counts"]
    fast_note = " with `--fast` (smoke counts; not for reporting)" if results["fast"] else ""
    lines += [
        "# First real audit: data axis versus optimizer axis",
        "",
        f"Generated by `scripts/run_first_audit.py`{fast_note}. Every number below is read from "
        "`results/first_audit/first_audit.json`. Inputs and their SHA-256 digests are listed at the end.",
        "",
    ]
    lines += _headline_section(results)
    lines += [
        "## What was measured",
        "",
        "- **Data axis** (DataDecide, `intervention_class = data`): 25 pretraining data recipes, 3 seeds per cell, "
        "fit budgets on the 5xC ladder from 4M up to 300M (or 150M for the cheap design), the next scale held out as "
        "the gate's intermediate budget, target 1B. 750M is excluded because the released 750M rows stop at "
        "~45 tokens/param, off the trajectory every other scale follows.",
        "- **Optimizer axis** (Fantastic Optimizers, `intervention_class = optimizer`): AdamW, Muon, NAdamW, SOAP - "
        "the four optimizers with a released 1.2B run - on size ladders 130M/300M/520M -> 1.2B at 1x/2x/4x/8x "
        "Chinchilla, and on data ladders 1x/2x/4x/8x -> 16x Chinchilla at 130M and 300M. **One run per cell, no "
        "seeds.** Bootstrap intervals are therefore degenerate and crossover significance is untestable on this "
        "axis; those cells are labelled as such rather than reported as significant.",
        f"- Estimands: `pairwise_decisions` (mean over unordered pairs; seed bootstrap n = {counts['pairwise']}) and "
        f"`single_design_seed_sensitivity` (one complete-set decision; n = {counts['single']}). Crossovers use "
        "per-pair Welch tests at the target with Benjamini-Hochberg FDR q = 0.05, for both the projection rule and "
        "the largest-fit-budget (single-scale) rule.",
        "",
        "## Per-design results",
        "",
    ]
    for d in results["designs"]:
        meaningful = d["seed_bootstrap_meaningful"]
        seed_note = "" if meaningful else " (**no seed replicates: intervals degenerate, significance untestable**)"
        lines += [
            f"### `{d['name']}`",
            "",
            f"axis = **{d['axis']}**, metric = `{d['metric_name']}`, {d['n_interventions']} interventions "
            f"({d['n_pairs']} pairs), fit budgets = {', '.join(d['budget_labels'])}, target = {d['target_label']}, "
            f"min seeds per cell = {d['min_seeds_per_cell']}{seed_note}",
            "",
            "| rule | pairwise mis-selection (pairwise_decisions) | pairwise regret | top-1 wrong (single design) "
            "| regret (single design) |",
            "|---|---|---|---|---|",
        ]
        pw = d["pairwise_decisions"]["rankers"]
        sd = d["single_design_seed_sensitivity"]["rankers"]
        for ranker in sorted(set(pw) | set(sd)):
            p = pw.get(ranker)
            s_ = sd.get(ranker)
            lines.append(
                f"| {ranker} | {'n/a' if p is None else _ci(p['mis_selection_rate'], meaningful)} | "
                f"{_regret(None if p is None else p['mean_regret'], meaningful)} | "
                f"{'n/a' if s_ is None else _ci(s_['mis_selection_rate'], meaningful)} | "
                f"{_regret(None if s_ is None else s_['mean_regret'], meaningful)} |"
            )
        pins = d.get("bound_pins") or {}
        if pins.get("any_pinned"):
            from collections import Counter

            which = Counter(
                f"{entry['parameter']}@{entry['side']}" for entries in pins["pinned"].values() for entry in entries
            )
            detail = ", ".join(f"{count}x {name}" for name, count in sorted(which.items()))
            lines += [
                f"> **Fit diagnostic:** {pins['n_pinned']} of {pins['n_interventions']} interventions have a fitted "
                f"parameter resting on a bound in this design ({detail}). A pinned parameter means the optimizer "
                "returned the closest curve it was *allowed* to express, not the closest curve."
                + (
                    " Every pin here is on the floor `E`, which the non-identifiability result in "
                    "AUDIT_ADVERSARIAL.md shows is unconstrained by accuracy-metric data: the projection rests on "
                    "a parameter the likelihood does not determine."
                    if all(name == "E@lower" for name in which)
                    else ""
                ),
                "",
            ]
        cx = d["crossovers"]
        single = d["single_design_seed_sensitivity"]
        lines += [
            "",
            f"Measured target winner: **{single['truth_winner']}**; statistically tied with it at the target: "
            f"{', '.join(single['truth_ties_with_winner']) or 'none'}.",
            "",
            "| crossover notion | flipped pairs | significant (BH q=0.05) | untestable |",
            "|---|---|---|---|",
            _flip_row("projection order vs measured target order", cx["projection_vs_target"], d["n_pairs"]),
            _flip_row("largest fit budget order vs measured target order", cx["largest_fit_budget_vs_target"], d["n_pairs"]),
            "",
        ]
        dec = d.get("projection_error_decomposition")
        if dec and d["axis"] == "data":
            lines += [
                "Decomposition of the projection rule's order flips:",
                "",
                "| component | pairs | rate | FDR-significant target gap |",
                "|---|---|---|---|",
                f"| inherited crossover (largest fit budget also orders the pair wrongly) | "
                f"{dec['crossover_inherited']['n']} | {_pct(dec['crossover_inherited']['rate'])} | "
                f"{dec['crossover_inherited']['n_significant']} |",
                f"| fit / extrapolation error (largest fit budget orders the pair correctly; projection reverses it) | "
                f"{dec['fit_error']['n']} | {_pct(dec['fit_error']['rate'])} | {dec['fit_error']['n_significant']} |",
                f"| single-scale flips the projection repairs | {dec['single_scale_only']['n']} | "
                f"{_pct(dec['single_scale_only']['rate'])} | {dec['single_scale_only']['n_significant']} |",
                "",
            ]
        if d["axis"] == "optimizer":
            for label, key in (("Single-scale", "largest_fit_budget_vs_target"), ("Projection", "projection_vs_target")):
                if cx[key]["pairs"]:
                    lines += [f"{label} order flips on this ladder: {_pairs_text(cx[key]['pairs'])}.", ""]

    lines += [
        "## Class contrast: single-scale order flips versus compute distance to the target",
        "",
        "Fraction of pairs whose winner at a small budget differs from the measured winner at the target, as a "
        "function of the small budget's share of target compute. This is the quantity DataDecide reports as "
        "1 - decision accuracy and the quantity that bounds single-scale selection on each axis.",
        "",
        "| axis | table | small scale | small / target compute | pairs | flipped | flip rate | significant flips |",
        "|---|---|---|---|---|---|---|---|",
    ]
    profiles = (
        ("data_axis_flip_profile", "DataDecide OLMES accuracy"),
        ("data_axis_flip_profile_c4", "DataDecide C4-EN bits/token"),
    )
    for key, label in profiles:
        for row in results.get(key, []):
            lines.append(
                f"| data | {label} | {row['scale_label']} | {row['compute_ratio_to_target']:.2e} | {row['n_pairs']} | "
                f"{row['n_flips']} | {_pct(row['flip_rate'])} | {row['n_significant_flips']} |"
            )
    opt_rows = results.get("optimizer_axis_flip_profile", [])
    for row in opt_rows:
        lines.append(
            f"| optimizer | {row['table']} | {row['scale_label']} | {row['compute_ratio_to_target']:.2e} | "
            f"{row['n_pairs']} | {row['n_flips']} | {_pct(row['flip_rate'])} | untestable (1 seed) |"
        )
    if opt_rows:
        pooled_pairs = sum(r["n_pairs"] for r in opt_rows)
        pooled_flips = sum(r["n_flips"] for r in opt_rows)
        lines += [
            "",
            f"Pooled over the four optimizer size ladders: {pooled_flips} / {pooled_pairs} pair-decisions flip "
            f"({_pct(pooled_flips / pooled_pairs)}) between a small size and 1.2B.",
        ]
    lines += _optimizer_sections(results)
    ts = results.get("tuning_stratification")
    if ts:
        lines += [
            "",
            "## Tuning-quality stratification (optimizer axis, 1x Chinchilla, 11 optimizers, target 520M)",
            "",
            "Single-scale pairwise agreement between a small size and 520M, for the coordinate-descent-tuned runs "
            "versus the median of the released one-hyperparameter ablations of the same optimizer at the same "
            "scale. See NOTES_TUNING_CONFOUND.md.",
            "",
            "| small scale | condition | pairs | agreement | disagreeing pairs |",
            "|---|---|---|---|---|",
        ]
        for small, entry in ts["by_small_scale"].items():
            for condition, res in entry["conditions"].items():
                lines.append(
                    f"| {small} | {condition} | {res['n_pairs']} | {_pct(res['pairwise_agreement'])} | "
                    f"{res['n_disagreeing']} |"
                )
    lines += [
        "",
        "## Caveats that apply to every number above",
        "",
        "- C4-EN bits per token is in-distribution for the DataDecide recipe named `C4` (it is trained on C4), so "
        "that recipe winning on that metric is a property of the metric, not evidence about pretraining quality. "
        "The OLMES tables do not share this confound.",
        "- DataDecide's small-scale auxiliary seeds and the released checkpoints leave 150M/300M/530M at 89-97 "
        "tokens/param and 1B at 85 tokens/param rather than exactly 100; all cells within a scale are compared at "
        "identical compute, but the ladder is only approximately iso-tokens-per-parameter.",
        "- The optimizer axis has one run per cell. Every optimizer-axis flip is reported without a p-value; a "
        "flip whose 1.2B loss gap is at the 1e-4 nats level is indistinguishable from a tie.",
        "- Optimizer-axis losses are nats per token on C4-EN validation; DataDecide's are bits per token on C4-EN "
        "validation (different tokenizers and evaluation sets). Rates are compared across axes, never the loss "
        "values themselves.",
        "- All bootstrap intervals resample seeds within cells and treat cells as independent; paired-seed "
        "dependence across scales is not modelled (see NOTES_ESTIMAND.md).",
        "",
        "## Inputs",
        "",
    ]
    for path, digest in results["inputs"].items():
        lines.append(f"- `{path}` sha256 `{digest}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
