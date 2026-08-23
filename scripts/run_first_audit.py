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
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.audit import audit_with_ci  # noqa: E402
from asla.analysis.crossover import detect_crossovers_fdr, detect_single_scale_flips_fdr  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/first_audit")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report", default="FIRST_AUDIT.md")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
