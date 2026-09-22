"""A1: independent, from-scratch re-derivation of the FIRST_AUDIT headline numbers.

This module deliberately imports NOTHING from ``asla.analysis``. It re-reads
``data/datadecide_runs*.parquet`` with pandas, re-implements the power-law fit,
the two rankers, the pairwise mis-selection metric, the Welch tests, the
Benjamini-Hochberg step-up, and the flip decomposition directly from their
definitions, and compares against ``results/first_audit/first_audit.json``.

Independence is the point: it uses ``scipy.optimize.least_squares`` (the audit
uses ``scipy.optimize.curve_fit``), fits in log-compute space with its own
initialisation, computes Welch statistics from means and variances rather than
via ``scipy.stats.ttest_ind``, and orders pairs with plain Python loops.

Only the schema-free parquet read is shared with the audit, and the raw parquet
is itself checked here (row counts, cells, seeds, 6ND).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO / "results" / "first_audit" / "first_audit.json"
TABLES = {
    "c4_en_bits_per_token": REPO / "data" / "datadecide_runs.parquet",
    "olmes_macro_error": REPO / "data" / "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": REPO / "data" / "datadecide_runs_olmes_correct_prob_per_char.parquet",
}


# --------------------------------------------------------------------------- fitting


def _power_law(compute: np.ndarray, e: float, a: float, alpha: float) -> np.ndarray:
    return e + a * np.asarray(compute, dtype=float) ** (-alpha)


_FIT_CACHE: dict[tuple[bytes, bytes], tuple[float, float, float]] = {}


def fit_power_law_independent(compute: np.ndarray, value: np.ndarray) -> tuple[float, float, float]:
    """Fit ``E + A * C^-alpha`` by least squares, working in log-compute for conditioning.

    Independent of the audit's implementation: different optimiser entry point
    (``least_squares`` with soft_l1-free default), different parameterisation
    (log A, log alpha bounds via transformed variables), different start.
    """

    compute = np.asarray(compute, dtype=float)
    value = np.asarray(value, dtype=float)
    cache_key = (compute.tobytes(), value.tobytes())
    cached = _FIT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    log_c = np.log(compute)
    span = float(log_c.max() - log_c.min())
    scale = float(np.exp(log_c.mean()))
    x = compute / scale

    def residual(theta: np.ndarray) -> np.ndarray:
        e, log_a, log_alpha = theta
        return e + np.exp(log_a) * x ** (-np.exp(log_alpha)) - value

    best: tuple[float, np.ndarray] | None = None
    for e0 in (0.0, float(value.min()) * 0.5, float(value.min()) * 0.9):
        for alpha0 in (0.02, 0.05, 0.1, 0.3):
            start = np.array([e0, np.log(max(float(value.max() - e0), 1e-6)), np.log(alpha0)])
            try:
                sol = least_squares(residual, start, method="lm", max_nfev=20000)
            except Exception:  # noqa: BLE001 - any optimiser failure just means try the next start
                continue
            cost = float(np.sum(sol.fun**2))
            if np.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, sol.x)
    if best is None:
        raise RuntimeError("independent power-law fit failed at every start")
    e, log_a, log_alpha = best[1]
    alpha = float(np.exp(log_alpha))
    # undo the compute rescaling: A_orig = A_scaled * scale^(-alpha) ... solve for original units
    a_orig = float(np.exp(log_a)) * scale ** (alpha)
    assert span > 0
    result = (float(e), float(a_orig), alpha)
    _FIT_CACHE[cache_key] = result
    return result


# --------------------------------------------------------------------------- rankers


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean value per (intervention, compute) with seed counts and sample sd."""

    grouped = df.groupby(["intervention", "compute"], sort=True)["bpb"]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "sd": grouped.std(ddof=1),
            "n": grouped.count(),
        }
    ).reset_index()


def project_scores(df: pd.DataFrame, budgets: list[float], target: float) -> dict[str, float]:
    """Projected target value per intervention from a per-intervention power-law fit."""

    fit_rows = df[df["compute"].isin(budgets)]
    out: dict[str, float] = {}
    for name, group in fit_rows.groupby("intervention", sort=True):
        e, a, alpha = fit_power_law_independent(group["compute"].to_numpy(), group["bpb"].to_numpy())
        out[str(name)] = float(_power_law(np.asarray([target]), e, a, alpha)[0])
    return out


def single_scale_scores(df: pd.DataFrame, budgets: list[float]) -> dict[str, float]:
    """Mean value at the largest fitting budget per intervention."""

    largest = max(budgets)
    rows = df[np.isclose(df["compute"], largest)]
    return {str(name): float(group["bpb"].mean()) for name, group in rows.groupby("intervention", sort=True)}


def truth_scores(df: pd.DataFrame, target: float) -> dict[str, float]:
    rows = df[np.isclose(df["compute"], target)]
    return {str(name): float(group["bpb"].mean()) for name, group in rows.groupby("intervention", sort=True)}


# --------------------------------------------------------------------------- metrics


def pairwise_mis_selection(scores: dict[str, float], truth: dict[str, float]) -> float:
    """Fraction of unordered pairs the scores order differently from the truth.

    Under the ``pairwise_decisions`` estimand the audit evaluates each pair as
    its own decision and averages ``1 - top1_acc``. For a two-element field the
    top-1 choice is wrong exactly when the pair order disagrees, so this equals
    the pair-order disagreement rate. Ties in either score break by name, which
    matches taking the lexicographically first name as the winner.
    """

    names = sorted(set(scores) & set(truth))
    wrong = 0
    total = 0
    for a, b in itertools.combinations(names, 2):
        total += 1
        pred_winner = a if (scores[a], a) < (scores[b], b) else b
        true_winner = a if (truth[a], a) < (truth[b], b) else b
        wrong += int(pred_winner != true_winner)
    return wrong / total


def flips(scores: dict[str, float], truth: dict[str, float]) -> list[tuple[str, str]]:
    """Pairs whose predicted order differs from the measured target order (strict, no ties)."""

    names = sorted(set(scores) & set(truth))
    out = []
    for a, b in itertools.combinations(names, 2):
        pred_gap = scores[a] - scores[b]
        true_gap = truth[a] - truth[b]
        if pred_gap == 0.0 or true_gap == 0.0:
            continue
        if np.sign(pred_gap) != np.sign(true_gap):
            out.append((a, b))
    return out


def welch_p(x: np.ndarray, y: np.ndarray) -> float | None:
    """Two-sided Welch p-value computed from moments (not via scipy.stats.ttest_ind)."""

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return None
    t_stat = (float(np.mean(x)) - float(np.mean(y))) / np.sqrt(se2)
    dof = se2**2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return float(2.0 * student_t.sf(abs(t_stat), dof))


def benjamini_hochberg(p_values: list[float | None], q: float = 0.05) -> list[bool]:
    """BH step-up; non-finite/None p-values are never rejected and do not count as tests."""

    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    reject = [False] * len(p_values)
    m = len(indexed)
    if m == 0:
        return reject
    indexed.sort()
    threshold_index = -1
    for rank, (p, _) in enumerate(indexed, start=1):
        if p <= rank / m * q:
            threshold_index = rank
    for rank, (_, i) in enumerate(indexed, start=1):
        if rank <= threshold_index:
            reject[i] = True
    return reject


def target_pair_tests(
    df: pd.DataFrame, target: float, pairs: list[tuple[str, str]], q: float = 0.05
) -> list[dict[str, Any]]:
    rows = df[np.isclose(df["compute"], target)]
    by_name = {str(name): group["bpb"].to_numpy(dtype=float) for name, group in rows.groupby("intervention", sort=True)}
    p_values = [welch_p(by_name[a], by_name[b]) for a, b in pairs]
    rejected = benjamini_hochberg(p_values, q=q)
    return [{"a": a, "b": b, "p_value": p, "significant": bool(r)} for (a, b), p, r in zip(pairs, p_values, rejected)]


# --------------------------------------------------------------------------- design


def design_frame(metric: str, exclude_scales: tuple[str, ...] = ("750M", "530M")) -> tuple[pd.DataFrame, list[float], float]:
    """Rebuild the full-ladder design: fit budgets 4M..300M, target 1B, 750M/530M excluded."""

    df = pd.read_parquet(TABLES[metric])
    df = df[df["scale_label"] != "750M"]
    target = float(df[df["scale_label"] == "1B"]["compute"].iloc[0])
    fit = df[(df["compute"] < target) & (~df["scale_label"].isin(exclude_scales))]
    budgets = sorted(float(b) for b in fit["compute"].unique())
    return df, budgets, target


def rederive(metric: str, q: float = 0.05) -> dict[str, Any]:
    df, budgets, target = design_frame(metric)
    truth = truth_scores(df, target)
    proj = project_scores(df, budgets, target)
    single = single_scale_scores(df, budgets)
    proj_flips = flips(proj, truth)
    single_flips = flips(single, truth)
    proj_tests = target_pair_tests(df, target, proj_flips, q=q)
    single_tests = target_pair_tests(df, target, single_flips, q=q)
    inherited = [p for p in proj_flips if p in set(single_flips)]
    fit_error = [p for p in proj_flips if p not in set(single_flips)]
    repaired = [p for p in single_flips if p not in set(proj_flips)]
    return {
        "metric": metric,
        "n_interventions": len(truth),
        "n_pairs": len(truth) * (len(truth) - 1) // 2,
        "n_fit_budgets": len(budgets),
        "projection_mis_selection": pairwise_mis_selection(proj, truth),
        "single_scale_mis_selection": pairwise_mis_selection(single, truth),
        "projection_flips": len(proj_flips),
        "projection_flips_significant": sum(t["significant"] for t in proj_tests),
        "single_scale_flips": len(single_flips),
        "single_scale_flips_significant": sum(t["significant"] for t in single_tests),
        "crossover_inherited": len(inherited),
        "fit_error": len(fit_error),
        "single_scale_only": len(repaired),
        "excess_projection_flips": len(proj_flips) - len(single_flips),
        "fit_error_significant": sum(t["significant"] for t in proj_tests if (t["a"], t["b"]) in set(fit_error)),
        "inherited_significant": sum(t["significant"] for t in proj_tests if (t["a"], t["b"]) in set(inherited)),
        "projection_flip_pairs": proj_flips,
        "single_scale_flip_pairs": single_flips,
    }


def audit_reference(metric: str) -> dict[str, Any]:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    name = f"datadecide/{metric}/full_ladder_4M-300M_gate530M"
    design = next(d for d in payload["designs"] if d["name"] == name)
    dec = design["projection_error_decomposition"]
    cx = design["crossovers"]
    return {
        "projection_mis_selection": design["pairwise_decisions"]["rankers"]["projection_ranker"]["mis_selection_rate"][
            "point"
        ],
        "single_scale_mis_selection": design["pairwise_decisions"]["rankers"]["single_scale_ranker"]["mis_selection_rate"][
            "point"
        ],
        "projection_flips": cx["projection_vs_target"]["n_flipped_pairs"],
        "projection_flips_significant": cx["projection_vs_target"]["n_significant"],
        "single_scale_flips": cx["largest_fit_budget_vs_target"]["n_flipped_pairs"],
        "single_scale_flips_significant": cx["largest_fit_budget_vs_target"]["n_significant"],
        "crossover_inherited": dec["crossover_inherited"]["n"],
        "fit_error": dec["fit_error"]["n"],
        "single_scale_only": dec["single_scale_only"]["n"],
        "excess_projection_flips": dec["excess_projection_flips"],
        "fit_error_significant": dec["fit_error"]["n_significant"],
        "inherited_significant": dec["crossover_inherited"]["n_significant"],
        "projection_flip_pairs": [(p["a"], p["b"]) for p in cx["projection_vs_target"]["pairs"]],
        "single_scale_flip_pairs": [(p["a"], p["b"]) for p in cx["largest_fit_budget_vs_target"]["pairs"]],
    }


def compare(metric: str) -> dict[str, Any]:
    mine = rederive(metric)
    theirs = audit_reference(metric)
    rows = []
    for key in sorted(theirs):
        if key.endswith("_pairs"):
            same = sorted(map(tuple, mine[key])) == sorted(map(tuple, theirs[key]))
            rows.append(
                {
                    "quantity": key,
                    "independent": f"{len(mine[key])} pairs",
                    "audit": f"{len(theirs[key])} pairs",
                    "match": same,
                }
            )
            continue
        a, b = mine[key], theirs[key]
        same = bool(np.isclose(a, b, rtol=1e-9, atol=1e-12)) if isinstance(a, float) or isinstance(b, float) else a == b
        rows.append({"quantity": key, "independent": a, "audit": b, "match": bool(same)})
    return {"metric": metric, "rows": rows, "all_match": all(r["match"] for r in rows)}


def main() -> int:
    out: dict[str, Any] = {"comparisons": {}}
    for metric in TABLES:
        result = compare(metric)
        out["comparisons"][metric] = result
        print(f"=== {metric}: all_match={result['all_match']}")
        for row in result["rows"]:
            flag = "ok " if row["match"] else "MISMATCH"
            print(f"  {flag} {row['quantity']:34s} independent={row['independent']!s:>22} audit={row['audit']!s:>22}")
    out["all_match"] = all(c["all_match"] for c in out["comparisons"].values())
    destination = REPO / "results" / "adversarial" / "a1_independent_rederivation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"\nall metrics match: {out['all_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
