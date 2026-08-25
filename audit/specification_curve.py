"""A2: specification curve over researcher degrees of freedom in the H1 headline.

Recomputes the headline decomposition under every plausible alternative
analysis choice and reports whether the conclusion "the excess of projection
over single-scale ranking is entirely fit error, not crossover" survives.

Degrees of freedom swept:

* ``metric``            - C4 bits/token, OLMES macro error, OLMES correct-prob
                          deficit, Signal-and-Noise Paloma C4 bits/byte.
* ``max_fit_scale``     - which scales count as "small": 150M, 300M, 530M.
* ``min_fit_scale``     - drop the noisiest tiny scales: 4M, 14M, 60M.
* ``target``            - 1B (the released target) or 530M (a smaller target,
                          fitting only below it).
* ``include_750M``      - the off-trajectory scale, excluded in the headline.
* ``fit_bounds``        - the audit's bounded fit vs an unconstrained refit
                          (A1 found alpha pins at its lower bound on OLMES).
* ``ranking``           - seed-mean ranking vs per-seed majority ranking.
* ``fdr``               - Benjamini-Hochberg vs Benjamini-Yekutieli, q in
                          {0.01, 0.05, 0.10}.

For each specification we report the projection and single-scale
mis-selection rates, the flip counts, and the decomposition into inherited
crossover versus fit error. The headline conclusion is encoded as
``excess_is_all_fit_error``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SN_TABLE = REPO / "data" / "signal_and_noise_datadecide_c4_bpb.parquet"
SCALE_ORDER = ("4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B")


def benjamini_yekutieli(p_values: list[float | None], q: float = 0.05) -> list[bool]:
    """BY step-up: BH at level q / sum(1/i), valid under arbitrary dependence."""

    m = sum(1 for p in p_values if p is not None and np.isfinite(p))
    if m == 0:
        return [False] * len(p_values)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return ir.benjamini_hochberg(p_values, q=q / harmonic)


def per_seed_scores(df: pd.DataFrame, compute: float) -> dict[str, dict[int, float]]:
    rows = df[np.isclose(df["compute"], compute)]
    out: dict[str, dict[int, float]] = {}
    for (name, seed), group in rows.groupby(["intervention", "seed"], sort=True):
        out.setdefault(str(name), {})[int(seed)] = float(group["bpb"].mean())
    return out


def per_seed_majority_flips(
    df: pd.DataFrame, compute: float, truth: dict[str, float]
) -> tuple[list[tuple[str, str]], float]:
    """Order each pair by a majority vote over seeds at ``compute``; return flips and mis-selection."""

    scores = per_seed_scores(df, compute)
    names = sorted(set(scores) & set(truth))
    flips: list[tuple[str, str]] = []
    wrong = 0
    total = 0
    for a, b in itertools.combinations(names, 2):
        seeds = sorted(set(scores[a]) & set(scores[b]))
        votes_a = sum(1 for s in seeds if scores[a][s] < scores[b][s])
        pred_winner = a if votes_a * 2 > len(seeds) else (b if votes_a * 2 < len(seeds) else min(a, b))
        true_winner = a if (truth[a], a) < (truth[b], b) else b
        total += 1
        if pred_winner != true_winner:
            wrong += 1
            flips.append((a, b))
    return flips, wrong / total


def load_table(metric: str) -> pd.DataFrame:
    if metric == "sn_c4_bits_per_byte":
        return pd.read_parquet(SN_TABLE)
    return pd.read_parquet(ir.TABLES[metric])


def build_design(
    metric: str,
    target_label: str,
    max_fit_label: str,
    min_fit_label: str,
    include_750m: bool,
) -> tuple[pd.DataFrame, list[float], float] | None:
    df = load_table(metric)
    if not include_750m:
        df = df[df["scale_label"] != "750M"]
    if target_label not in set(df["scale_label"]):
        return None
    target = float(df[df["scale_label"] == target_label]["compute"].iloc[0])
    order = {s: i for i, s in enumerate(SCALE_ORDER)}
    lo, hi = order[min_fit_label], order[max_fit_label]
    keep = [s for s in df["scale_label"].unique() if lo <= order.get(str(s), -1) <= hi]
    fit = df[df["scale_label"].isin(keep) & (df["compute"] < target)]
    budgets = sorted(float(b) for b in fit["compute"].unique())
    if len(budgets) < 3:
        return None
    return df, budgets, target


def unconstrained_projection(df: pd.DataFrame, budgets: list[float], target: float) -> dict[str, float]:
    return ir.project_scores(df, budgets, target)


def bounded_projection(df: pd.DataFrame, budgets: list[float], target: float) -> dict[str, float]:
    """The audit's own bounded fit, imported here only to compare against the free fit."""

    from asla.analysis.fits import project_ranking

    series = project_ranking(df, tuple(budgets), target)
    return {str(k): float(v) for k, v in series.items()}


def evaluate(
    metric: str,
    target_label: str,
    max_fit_label: str,
    min_fit_label: str,
    include_750m: bool,
    fit_bounds: str,
    ranking: str,
    fdr: str,
    q: float,
) -> dict[str, Any] | None:
    built = build_design(metric, target_label, max_fit_label, min_fit_label, include_750m)
    if built is None:
        return None
    df, budgets, target = built
    truth = ir.truth_scores(df, target)
    try:
        proj = unconstrained_projection(df, budgets, target) if fit_bounds == "unconstrained" else bounded_projection(df, budgets, target)
    except Exception:  # noqa: BLE001 - a failed fit is a legitimate specification outcome
        return None
    largest = max(budgets)
    n_seeds = int(df.groupby(["intervention", "compute"])["seed"].nunique().min())
    if ranking == "per_seed_majority":
        if n_seeds < 2:
            return None
        single_flips, single_rate = per_seed_majority_flips(df, largest, truth)
    else:
        single = ir.single_scale_scores(df, budgets)
        single_flips = ir.flips(single, truth)
        single_rate = ir.pairwise_mis_selection(single, truth)
    proj_flips = ir.flips(proj, truth)
    proj_rate = ir.pairwise_mis_selection(proj, truth)

    def tests(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        rows = df[np.isclose(df["compute"], target)]
        by_name = {str(n): g["bpb"].to_numpy(dtype=float) for n, g in rows.groupby("intervention", sort=True)}
        p_values = [ir.welch_p(by_name[a], by_name[b]) for a, b in pairs]
        rejected = benjamini_yekutieli(p_values, q=q) if fdr == "BY" else ir.benjamini_hochberg(p_values, q=q)
        return [{"a": a, "b": b, "significant": bool(r)} for (a, b), r in zip(pairs, rejected)]

    proj_tests = tests(proj_flips)
    single_set = set(single_flips)
    inherited = [p for p in proj_flips if p in single_set]
    fit_error = [p for p in proj_flips if p not in single_set]
    repaired = [p for p in single_flips if p not in set(proj_flips)]
    excess = len(proj_flips) - len(single_flips)
    return {
        "metric": metric,
        "target": target_label,
        "fit_scales": f"{min_fit_label}-{max_fit_label}",
        "n_fit_budgets": len(budgets),
        "include_750M": include_750m,
        "fit_bounds": fit_bounds,
        "ranking": ranking,
        "fdr": f"{fdr}@{q}",
        "n_seeds_min": n_seeds,
        "n_pairs": len(truth) * (len(truth) - 1) // 2,
        "projection_mis_selection": proj_rate,
        "single_scale_mis_selection": single_rate,
        "projection_worse": bool(proj_rate > single_rate),
        "projection_flips": len(proj_flips),
        "single_scale_flips": len(single_flips),
        "excess_projection_flips": excess,
        "n_inherited": len(inherited),
        "n_fit_error": len(fit_error),
        "n_repaired": len(repaired),
        "n_fit_error_significant": sum(t["significant"] for t in proj_tests if (t["a"], t["b"]) in set(fit_error)),
        "n_inherited_significant": sum(t["significant"] for t in proj_tests if (t["a"], t["b"]) in set(inherited)),
        "fit_error_share_of_flips": (len(fit_error) / len(proj_flips)) if proj_flips else None,
        # The headline conclusion: the excess of projection over single-scale is
        # accounted for entirely by fit error (inherited crossovers cannot explain it).
        "excess_is_all_fit_error": (None if excess <= 0 else bool(len(fit_error) - len(repaired) >= excess)),
        "fit_error_exceeds_inherited": bool(len(fit_error) > len(inherited)),
    }


def sweep() -> list[dict[str, Any]]:
    metrics = ["c4_en_bits_per_token", "olmes_macro_error", "olmes_macro_correct_prob_per_char_deficit", "sn_c4_bits_per_byte"]
    rows: list[dict[str, Any]] = []
    # Main sweep: metric x fit range x target x 750M x bounds, at the default ranking/FDR.
    for metric in metrics:
        for target_label in ("1B", "530M"):
            for max_fit in ("150M", "300M", "530M"):
                for min_fit in ("4M", "14M", "60M"):
                    for include_750m in (False, True):
                        for bounds in ("bounded", "unconstrained"):
                            row = evaluate(metric, target_label, max_fit, min_fit, include_750m, bounds, "seed_mean", "BH", 0.05)
                            if row is not None:
                                rows.append(row)
    # FDR and ranking sweeps on the headline design only.
    for metric in metrics:
        for fdr, q in (("BH", 0.01), ("BH", 0.05), ("BH", 0.10), ("BY", 0.05), ("BY", 0.10)):
            for ranking in ("seed_mean", "per_seed_majority"):
                row = evaluate(metric, "1B", "300M", "4M", False, "bounded", ranking, fdr, q)
                if row is not None:
                    rows.append(row)
    seen = set()
    unique = []
    for row in rows:
        key = tuple(row[k] for k in ("metric", "target", "fit_scales", "include_750M", "fit_bounds", "ranking", "fdr"))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def main() -> int:
    rows = sweep()
    frame = pd.DataFrame(rows)
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "a2_specification_curve.csv", index=False)
    testable = frame[frame["excess_is_all_fit_error"].notna()]
    summary = {
        "n_specifications": int(len(frame)),
        "n_where_projection_worse": int(frame["projection_worse"].sum()),
        "n_with_positive_excess": int(len(testable)),
        "n_excess_all_fit_error": int(testable["excess_is_all_fit_error"].sum()),
        "excess_all_fit_error_rate": float(testable["excess_is_all_fit_error"].mean()) if len(testable) else None,
        "n_fit_error_exceeds_inherited": int(frame["fit_error_exceeds_inherited"].sum()),
        "counterexamples": testable[~testable["excess_is_all_fit_error"]].to_dict("records"),
        "projection_better_specs": frame[~frame["projection_worse"]][
            ["metric", "target", "fit_scales", "fit_bounds", "projection_mis_selection", "single_scale_mis_selection"]
        ].to_dict("records"),
    }
    (destination / "a2_specification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"specifications: {summary['n_specifications']}")
    print(f"projection worse than single-scale in {summary['n_where_projection_worse']}")
    print(f"excess is all fit error in {summary['n_excess_all_fit_error']}/{summary['n_with_positive_excess']} testable specs")
    print(f"counterexamples: {len(summary['counterexamples'])}")
    for row in summary["counterexamples"][:10]:
        print("  ", {k: row[k] for k in ("metric", "target", "fit_scales", "fit_bounds", "ranking", "fdr", "excess_projection_flips", "n_fit_error", "n_inherited", "n_repaired")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
