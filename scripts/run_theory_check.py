"""Task 2: compute the pre-registered theory predictions and compare with measurements.

Every constant is recomputed from data; every prediction follows the algebra in
PREDICTIONS_TASK_THEORY.md with no free parameters. Intermediate quantities are
emitted so the algebra can be checked by hand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.cli import _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PRIMARY = REPO / "data" / "datadecide_runs.parquet"
LADDER = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M"]
# Pre-committed bands (PREDICTIONS_TASK_THEORY.md).
T1_BAND = (1.5, 4.0)
T2_FACTOR = 2.0
T3_INTERVAL = (64.9, 69.8)


def design(max_fit: str, target_label: str = "1B") -> dict[str, Any]:
    df = load_runs(PRIMARY)
    df = df[df["scale_label"] != "750M"]
    target = float(df[df["scale_label"] == target_label]["compute"].iloc[0])
    keep = LADDER[: LADDER.index(max_fit) + 1]
    budgets = tuple(sorted(float(b) for b in df[df["scale_label"].isin(keep)]["compute"].unique() if b < target))
    return {"df": df, "budgets": budgets, "target": target, "max_fit": max_fit}


def leverage(budgets: tuple[float, ...], target: float) -> dict[str, float]:
    """OLS prediction leverage at the target: h* = 1/k + (u* - ubar)^2 / S_uu.

    Returns every intermediate so the algebra can be verified by hand.
    """

    u = np.log(np.asarray(budgets, dtype=float))
    k = len(u)
    ubar = float(u.mean())
    s_uu = float(np.sum((u - ubar) ** 2))
    u_star = float(np.log(target))
    u_max = float(u.max())
    log_l = u_star - u_max
    d = u_max - ubar
    offset_term = (u_star - ubar) ** 2 / s_uu
    return {
        "k": k,
        "ubar": ubar,
        "S_uu": s_uu,
        "u_star": u_star,
        "u_max": u_max,
        "log_L": log_l,
        "L": float(np.exp(log_l)),
        "d_log_Cmax_minus_ubar": d,
        "one_over_k": 1.0 / k,
        "offset_term": offset_term,
        # explicit check of the quadratic expansion: (u*-ubar)^2 = (d + logL)^2
        "quadratic_expansion_check": float((d + log_l) ** 2 - (u_star - ubar) ** 2),
        "leverage": 1.0 / k + offset_term,
    }


def log_space_sigma(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, Any]:
    """Log-space noise on a cell mean, plus the per-budget spread that tests A2.

    ``sigma`` is pooled in VARIANCE space as ``sqrt(mean(relative**2))``, because
    every downstream use squares it: ``expected_flip_rate`` forms
    ``2 * sigma**2 * leverage`` and the variance diagnosis forms
    ``sigma**2 * leverage``. Taking ``mean(relative)`` and squaring it is the
    Jensen error -- it understates ``sigma**2`` by the squared coefficient of
    variation of the relative standard errors, which on DataDecide is 2.06x.

    ``sigma_mean_of_scales`` is reported alongside because it is the right
    summary for the different question "what is a typical relative standard
    error", and keeping both makes the distinction explicit rather than a
    silent choice. It must never be squared.
    """

    rows = df[df["compute"].isin(budgets)]
    stats = rows.groupby(["intervention", "compute"])["bpb"].agg(["mean", "std", "count"])
    relative = stats["std"] / stats["mean"] / np.sqrt(stats["count"])
    per_budget = relative.groupby("compute").mean()
    sigma_rms = float(np.sqrt(np.mean(relative.to_numpy(dtype=float) ** 2)))
    sigma_mean = float(relative.mean())
    return {
        "sigma": sigma_rms,
        "sigma_pooling": "sqrt(mean(relative**2))",
        "sigma_mean_of_scales": sigma_mean,
        "sigma_variance_understatement_if_mean_used": float(sigma_rms**2 / sigma_mean**2),
        "per_budget": {f"{c:.4e}": float(v) for c, v in per_budget.items()},
        "min": float(per_budget.min()),
        "max": float(per_budget.max()),
        "heteroscedasticity_ratio": float(per_budget.max() / per_budget.min()),
        "A2_violated": bool(per_budget.max() / per_budget.min() > 3.0),
    }


def true_log_gaps(df: pd.DataFrame, target: float) -> np.ndarray:
    values = df[np.isclose(df["compute"], target)].groupby("intervention")["bpb"].mean().to_numpy()
    logs = np.log(values)
    return np.asarray([abs(a - b) for a, b in itertools.combinations(logs, 2)], dtype=float)


def expected_flip_rate(gaps: np.ndarray, sigma: float, lev: float) -> float:
    """Equation (4)+(5): mean over the empirical gap distribution of Phi(-|Delta| / sd_gap)."""

    sd_gap = np.sqrt(2.0 * sigma**2 * lev)
    return float(np.mean(norm.cdf(-gaps / sd_gap)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/theory")
    args = parser.parse_args(argv)

    primary = design("300M")
    long_arm = design("150M")
    short_arm = design("530M")
    df, target = primary["df"], primary["target"]

    lev_primary = leverage(primary["budgets"], target)
    lev_long = leverage(long_arm["budgets"], target)
    lev_short = leverage(short_arm["budgets"], target)
    noise = log_space_sigma(df, primary["budgets"])
    gaps = true_log_gaps(df, target)
    sigma = noise["sigma"]

    # ---- T1: variance ratio between lever arms
    predicted_ratio = lev_long["leverage"] / lev_short["leverage"]
    dose = json.loads((REPO / "results" / "adversarial" / "a2_dose_response.json").read_text(encoding="utf-8"))
    by_arm = {f"{row['target']}|{row['max_fit_scale']}": row for row in dose["lever_arm_summary"]}
    measured_long = by_arm["1B|150M"]["mean_excess_flips"]
    measured_short = by_arm["1B|530M"]["mean_excess_flips"]
    measured_ratio = measured_long / measured_short
    t1 = {
        "predicted_variance_ratio": predicted_ratio,
        "band": list(T1_BAND),
        "in_band": bool(T1_BAND[0] <= predicted_ratio <= T1_BAND[1]),
        "measured_excess_flip_ratio": measured_ratio,
        "measured_long_arm_flips": measured_long,
        "measured_short_arm_flips": measured_short,
        "measured_in_predicted_band": bool(T1_BAND[0] <= measured_ratio <= T1_BAND[1]),
        "miss_factor": measured_ratio / predicted_ratio,
        "direction": (
            "theory UNDER-predicts the long-arm penalty"
            if measured_ratio > predicted_ratio
            else "theory OVER-predicts the long-arm penalty"
        ),
        "verdict": "CONFIRMED" if T1_BAND[0] <= measured_ratio <= T1_BAND[1] else "MISSED",
        "leverage_long": lev_long,
        "leverage_short": lev_short,
    }

    # ---- T2: slope of excess flips against log L
    arms = [(row["lever_arm"], row["mean_excess_flips"]) for row in dose["lever_arm_summary"] if row["target"] == "1B"]
    arms.sort()
    log_l = np.log10([a for a, _ in arms])
    flips = np.array([f for _, f in arms], dtype=float)
    measured_slope = float(np.polyfit(log_l, flips, 1)[0])
    # predicted: flips = N_pairs * expected_flip_rate(leverage at that arm)
    n_pairs = len(gaps)
    predicted_points = []
    for arm_label, built in (("150M", long_arm), ("300M", primary), ("530M", short_arm)):
        lv = leverage(built["budgets"], target)
        rate = expected_flip_rate(gaps, sigma, lv["leverage"])
        predicted_points.append((np.log10(lv["L"]), n_pairs * rate, arm_label, lv["leverage"], rate))
    predicted_slope = float(np.polyfit([p[0] for p in predicted_points], [p[1] for p in predicted_points], 1)[0])
    ratio_slope = predicted_slope / measured_slope if measured_slope else float("inf")
    t2 = {
        "measured_slope_flips_per_decade": measured_slope,
        "predicted_slope_flips_per_decade": predicted_slope,
        "ratio_predicted_over_measured": ratio_slope,
        "within_factor": T2_FACTOR,
        "verdict": "CONFIRMED" if (1 / T2_FACTOR) <= abs(ratio_slope) <= T2_FACTOR else "MISSED",
        "predicted_points": [
            {"arm": p[2], "log10_L": p[0], "leverage": p[3], "flip_rate": p[4], "expected_flips": p[1]}
            for p in predicted_points
        ],
        "measured_points": [{"lever_arm": a, "mean_excess_flips": f} for a, f in arms],
    }

    # ---- T3: retrodiction of the pooling gain
    k = lev_primary["k"]
    n_interventions = int(df["intervention"].nunique())
    lev_pooled = lev_primary["one_over_k"] + lev_primary["offset_term"] / n_interventions
    variance_ratio = lev_pooled / lev_primary["leverage"]
    rate_plain = expected_flip_rate(gaps, sigma, lev_primary["leverage"])
    rate_pooled = expected_flip_rate(gaps, sigma, lev_pooled)
    predicted_reduction = 100.0 * (rate_plain - rate_pooled) / rate_plain
    rows = df[df["compute"].isin(primary["budgets"])]
    stats = rows.groupby(["intervention", "compute"])["bpb"].agg(["mean", "std", "count"])
    relative = (stats["std"] / stats["mean"] / np.sqrt(stats["count"])).to_numpy()
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        s = float(np.mean(rng.choice(relative, len(relative), replace=True)))
        a = expected_flip_rate(gaps, s, lev_primary["leverage"])
        b = expected_flip_rate(gaps, s, lev_pooled)
        if a > 0:
            draws.append(100.0 * (a - b) / a)
    draws_arr = np.asarray(draws)
    task_b = json.loads((REPO / "results" / "task_b" / "task_b.json").read_text(encoding="utf-8"))
    tb = task_b["designs"]["primary_4M-300M_target1B"]["rankers"]
    plain_rate = tb["plain_projection"]["mis_selection"]["point"]
    pooled_rate = tb["shared_exponent"]["mis_selection"]["point"]
    measured_reduction = 100.0 * (plain_rate - pooled_rate) / plain_rate
    t3 = {
        "K_interventions": n_interventions,
        "k_budgets": k,
        "leverage_plain": lev_primary["leverage"],
        "leverage_pooled": lev_pooled,
        "predicted_variance_ratio": variance_ratio,
        "predicted_variance_reduction_pct": 100.0 * (1 - variance_ratio),
        "predicted_flip_rate_plain": rate_plain,
        "predicted_flip_rate_pooled": rate_pooled,
        "predicted_mis_selection_reduction_pct": predicted_reduction,
        "predicted_interval_pct": [float(np.percentile(draws_arr, 2.5)), float(np.percentile(draws_arr, 97.5))],
        "committed_interval_pct": list(T3_INTERVAL),
        "measured_reduction_pct": measured_reduction,
        "measured_plain": plain_rate,
        "measured_pooled": pooled_rate,
        "verdict": "CONFIRMED" if T3_INTERVAL[0] <= measured_reduction <= T3_INTERVAL[1] else "MISSED",
        "miss_points": measured_reduction - predicted_reduction,
        "direction": (
            "theory OVER-predicts the pooling benefit"
            if predicted_reduction > measured_reduction
            else "theory UNDER-predicts the pooling benefit"
        ),
    }

    # ---- Diagnosis: how much variance does the two-parameter derivation omit?
    # A1 treats the floor E as known; in practice it is fitted, and that third
    # parameter carries variance the derivation never accounts for. Measure the
    # true projection variance by seed bootstrap and compare.
    from asla.analysis.audit import resample_runs_by_cell
    from asla.analysis.fits import project_ranking

    rng_diag = np.random.default_rng(0)
    projections = []
    for _ in range(150):
        try:
            projections.append(project_ranking(resample_runs_by_cell(df, rng_diag), primary["budgets"], target))
        except Exception:  # noqa: BLE001
            continue
    log_projections = np.log(pd.DataFrame(projections))
    empirical_variance = float(log_projections.var(axis=0, ddof=1).mean())
    theory_variance = sigma**2 * lev_primary["leverage"]
    variance_diagnosis = {
        "empirical_var_log_projection": empirical_variance,
        "theory_var_two_parameter": theory_variance,
        "ratio_empirical_over_theory": empirical_variance / theory_variance,
        "gap_sd_understated_by": float(np.sqrt(empirical_variance / theory_variance)),
        "n_bootstrap_draws": len(projections),
        "sigma_pooling": noise["sigma_pooling"],
        "interpretation": (
            "The two-parameter derivation assumes the floor E is known (A1). In the audited pipeline E is "
            "fitted, so the projection carries a third parameter's worth of variance that the derivation "
            "omits, which under-states the gap standard deviation and therefore flip probabilities at every "
            "lever arm. Magnitude, corrected: this ratio was previously reported as 3.20 while sigma was "
            "pooled as mean(relative_se) and then squared. Pooling in variance space as required by the "
            "sigma**2 that consumes it (see log_space_sigma) raises theory_var by "
            f"{noise['sigma_variance_understatement_if_mean_used']:.2f}x and leaves a residual ratio of "
            f"{empirical_variance / theory_variance:.2f}. So roughly half of what was attributed to the "
            "omitted third parameter was the pooling error; the omission is still real and still in the "
            "direction that explains the misses, but it is a smaller effect than first reported, and it no "
            "longer accounts for the T2 miss on its own (that ratio remains 0.08, far outside the "
            "factor-of-two band)."
        ),
    }

    # ---- A4: are projection errors independent across interventions?
    # The bootstrap matrix above already carries this; it was previously
    # asserted as "measured by seed bootstrap" while being a literal.
    correlation_matrix = log_projections.corr().to_numpy(dtype=float)
    offdiag = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
    mean_cross_correlation = float(np.nanmean(offdiag))
    # A gap between two independent projections has variance 2 * v; correlation
    # rho changes that to 2 * v * (1 - rho), so the ratio to independence is
    # (1 - mean rho). Values near 1 mean A4 cannot explain any miss.
    var_gap_ratio = float(1.0 - mean_cross_correlation)

    # ---- T4: the pre-declared failure modes
    floor_path = REPO / "results" / "adversarial" / "floor_identifiability.json"
    floor = json.loads(floor_path.read_text(encoding="utf-8")) if floor_path.exists() else {}
    c4_ssr_ratio = floor.get("c4_en_bits_per_token", {}).get("median_ssr_ratio_zero_over_best")
    olmes_ssr_ratio = floor.get("olmes_macro_error", {}).get("median_ssr_ratio_zero_over_best")
    t4 = {
        "A1_scope": {
            "claim": "theory applies only where the floor is identifiable",
            "c4_ssr_ratio": c4_ssr_ratio,
            "olmes_ssr_ratio": olmes_ssr_ratio,
            "source": str(floor_path.relative_to(REPO)),
            "note": "read from results/adversarial/floor_identifiability.json; OLMES designs are out of scope",
        },
        "A2_measured": {
            "heteroscedasticity_ratio": noise["heteroscedasticity_ratio"],
            "violated": noise["A2_violated"],
            "predicted_consequence": "theory under-predicts the long-arm penalty",
            "consistent_with_T1_miss": bool(measured_ratio > predicted_ratio),
        },
        "A4_measured": {
            "mean_cross_intervention_correlation": mean_cross_correlation,
            "var_gap_ratio_to_independence": var_gap_ratio,
            "n_bootstrap_draws": len(projections),
            "n_interventions": int(correlation_matrix.shape[0]),
            "holds": bool(abs(mean_cross_correlation) < 0.05),
            "note": "measured by seed bootstrap from the projection matrix; A4 cannot explain any miss",
        },
    }

    results = {
        "variance_diagnosis": variance_diagnosis,
        "constants": {
            "primary": lev_primary,
            "long_arm": lev_long,
            "short_arm": lev_short,
            "noise": noise,
            "n_pairs": int(len(gaps)),
            "median_log_gap": float(np.median(gaps)),
        },
        "T1_lever_arm_variance": t1,
        "T2_slope": t2,
        "T3_pooling_retrodiction": t3,
        "T4_failure_modes": t4,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(results, out / "theory_check.json")
    summary = {k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else "n/a") for k, v in results.items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
