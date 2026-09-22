"""Round 3: expected-error significance, the published figure, and re-scoring.

Scored against ``PREDICTIONS_TASK_EXPECTED_ERROR.md``, committed before this ran.

Expected-error is the primary rule here. Determined-subset scoring is retained
as a diagnostic but is not the headline, because on the primary design both
rankers make zero errors on the determined subset and a comparison with no
errors cannot distinguish anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from asla.analysis.fits import cell_means_and_sigma, normalize_budgets
from asla.analysis.target_scoring import (
    bootstrap_two_sided_p,
    candidate_bootstrap,
    candidates_for_power,
    expected_charges,
    jackknife_standard_error,
    score_ranker,
    target_evidence,
    u_statistic_components,
    u_statistic_test,
    u_statistic_variance,
)
from asla.models import FitError, bpb_power_law, fit_power_law

REPO = Path(__file__).resolve().parents[1]
OFF_TRAJECTORY = ("750M",)
METRICS = {
    "c4_en_bits_per_token": "datadecide_runs.parquet",
    "olmes_macro_error": "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": "datadecide_runs_olmes_correct_prob_per_char.parquet",
}
DESIGNS = (
    ("primary_4M-300M_gate530M", "300M", "530M"),
    ("long_4M-150M_gate300M", "150M", "300M"),
    ("short_4M-530M_gate1B", "530M", "1B"),
)
# A metric is "low noise" when most target pairs are resolvable at three seeds.
LOW_NOISE_DETERMINED_FRACTION = 0.50
N_RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(metric: str) -> pd.DataFrame:
    frame = pd.read_parquet(REPO / "data" / METRICS[metric])
    return frame[~frame["scale_label"].isin(OFF_TRAJECTORY)].reset_index(drop=True)


def predictions_for(frame: pd.DataFrame, fit_budgets: tuple[float, ...], target: float) -> dict[str, dict[str, float]]:
    """Per-candidate predicted target value for each ranker."""

    means = cell_means_and_sigma(frame)
    names = sorted(frame["intervention"].unique())
    top = max(fit_budgets)

    projected: dict[str, float] = {}
    single: dict[str, float] = {}
    params_by_name: dict[str, tuple[float, float, float]] = {}
    for name in names:
        rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))].sort_values("compute")
        compute = rows["compute"].to_numpy(dtype=float)
        values = rows["bpb_mean"].to_numpy(dtype=float)
        if len(rows) >= 3:
            try:
                params = fit_power_law(compute, values)
                params_by_name[name] = params
                projected[name] = float(bpb_power_law(target, *params))
            except FitError:
                projected[name] = float("nan")
        cell = rows[np.isclose(rows["compute"], top)]
        single[name] = float(cell["bpb_mean"].iloc[0]) if len(cell) else float("nan")

    shared: dict[str, float] = {}
    if params_by_name:
        alpha = float(np.median([p[2] for p in params_by_name.values()]))
        for name in names:
            rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))].sort_values("compute")
            if len(rows) < 3:
                shared[name] = float("nan")
                continue
            basis = rows["compute"].to_numpy(dtype=float) ** (-alpha)
            design = np.column_stack([np.ones_like(basis), basis])
            coefficients, *_ = np.linalg.lstsq(design, rows["bpb_mean"].to_numpy(dtype=float), rcond=None)
            shared[name] = float(coefficients[0] + coefficients[1] * target ** (-alpha))

    out = {"single_scale": single, "projection": projected}
    if shared:
        out["shared_exponent"] = shared
    return out


def _gaps(values: dict[str, float], names: list[str]) -> dict[tuple[str, str], float]:
    usable = [n for n in names if np.isfinite(values.get(n, float("nan")))]
    return {(a, b): values[a] - values[b] for a, b in combinations(sorted(usable), 2)}


def candidate_resampled_difference(
    values: dict[str, dict[str, float]],
    evidence_by_pair: dict[tuple[str, str], Any],
    names: list[str],
    rng: np.random.Generator,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap the expected-error difference by resampling CANDIDATES.

    Pairs share candidates, so the 300 pair outcomes are not independent and a
    bootstrap over pairs understates the standard error. The candidate is the
    independent experimental unit; resampling it and rebuilding the induced pair
    set respects that dependence. Both are reported so the size of the
    correction is visible.
    """

    def unique_set_difference(subset: list[str]) -> float | None:
        pairs = [(a, b) for a, b in combinations(sorted(set(subset)), 2) if (a, b) in evidence_by_pair]
        if len(pairs) < 3:
            return None
        chosen = [evidence_by_pair[p] for p in pairs]
        left = score_ranker(_gaps(values["projection"], names), chosen)
        right = score_ranker(_gaps(values["single_scale"], names), chosen)
        if not np.isfinite(left["expected_rate"]) or not np.isfinite(right["expected_rate"]):
            return None
        return 100 * (left["expected_rate"] - right["expected_rate"])

    evidence = list(evidence_by_pair.values())
    left_charges = expected_charges(_gaps(values["projection"], names), evidence)
    right_charges = expected_charges(_gaps(values["single_scale"], names), evidence)
    kernel = {pair: 100 * (left_charges[pair] - right_charges[pair]) for pair in left_charges if pair in right_charges}
    point = float(np.mean(list(kernel.values())))
    candidate_draws = list(candidate_bootstrap(kernel, rng, n_resamples))

    # The superseded bootstrap, kept so the size of its error is on record
    # (PREDICTIONS_TASK_POWER.md, Q1): it scored only the unique candidates drawn.
    unique_draws = [
        d
        for d in (unique_set_difference(list(rng.choice(names, size=len(names)))) for _ in range(n_resamples))
        if d is not None
    ]

    all_pairs = sorted(kernel)
    pair_values = np.array([kernel[p] for p in all_pairs])
    pair_draws = [
        float(pair_values[rng.integers(0, len(all_pairs), size=len(all_pairs))].mean()) for _ in range(n_resamples)
    ]

    def summarise(draws: list[float], label: str) -> dict[str, Any]:
        array = np.asarray(draws, dtype=float)
        low, high = (float(np.percentile(array, q)) for q in (2.5, 97.5))
        return {
            "unit": label,
            "n_draws": int(array.size),
            "mean": float(array.mean()),
            "standard_error": float(array.std(ddof=1)),
            "ci_low_pp": low,
            "ci_high_pp": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "p_two_sided": bootstrap_two_sided_p(array),
        }

    candidate = summarise(candidate_draws, "candidate")
    pairs = summarise(pair_draws, "pair (anti-conservative)")
    unique = summarise(unique_draws, "candidate, unique-set (superseded: evaluates ~63% of candidates)")
    components = u_statistic_components(kernel)
    n = int(components["n_candidates"])
    formula_se = float(np.sqrt(u_statistic_variance(components["zeta1"], components["zeta2"], n)))
    return {
        "point_difference_pp": point,
        "candidate_resampling": candidate,
        "pair_resampling_naive": pairs,
        "unique_set_candidate_resampling_superseded": unique,
        "dependence_correction_factor": candidate["standard_error"] / pairs["standard_error"],
        "unique_set_over_corrected_se": unique["standard_error"] / candidate["standard_error"],
        # Headline (a declared deviation from the pre-registered primary): the
        # unbiased U-statistic test. The weighted bootstrap is conservative here;
        # see "estimator_calibration".
        "u_statistic": {**u_statistic_test(kernel), "formula_standard_error": formula_se},
        "jackknife_standard_error": jackknife_standard_error(kernel),
        "estimator_invariance": _invariance(
            point, candidate, unique, u_statistic_test(kernel), jackknife_standard_error(kernel)
        ),
        "power": power_analysis(components, point, rng),
        "_kernel": kernel,
    }


def _invariance(
    point: float, boot: dict[str, Any], unique: dict[str, Any], u_test: dict[str, Any], jack_se: float
) -> dict[str, Any]:
    """Two-sided p from every candidate-level estimator, so the conclusion's dependence on the choice is visible."""

    from scipy import stats

    p_values = {
        "u_statistic": float(u_test["p_two_sided"]),
        "weighted_bootstrap": float(boot["p_two_sided"]),
        "unique_set_bootstrap": float(unique["p_two_sided"]),
        "jackknife_normal": float(2 * stats.norm.sf(abs(point) / jack_se)),
    }
    return {"p_values": p_values, "p_min": min(p_values.values()), "p_max": max(p_values.values())}


POWER_TARGET = 0.8
POWER_DELTAS_PP = (0.5, 1.0, 2.0, 5.0)
SUBSAMPLE_SIZES = (8, 12, 16, 20)
N_SUBSAMPLES = 2000


def power_analysis(components: dict[str, float], point: float, rng: np.random.Generator) -> dict[str, Any]:
    """Candidates needed to detect a difference, from the Hoeffding components."""

    zeta1, zeta2 = components["zeta1"], components["zeta2"]
    return {
        "power": POWER_TARGET,
        "alpha": 0.05,
        "test": "two-sided z on the pair-mean U-statistic, zeta1 and zeta2 held at their estimates",
        "candidates_at_point_estimate": candidates_for_power(zeta1, zeta2, point, POWER_TARGET),
        "candidates_by_delta_pp": {
            f"{delta:g}": candidates_for_power(zeta1, zeta2, delta, POWER_TARGET) for delta in POWER_DELTAS_PP
        },
        "power_at_observed_count": _power_at(zeta1, zeta2, point, int(components["n_candidates"])),
    }


def _power_at(zeta1: float, zeta2: float, delta: float, n: int) -> float:
    from scipy import stats

    se = float(np.sqrt(u_statistic_variance(zeta1, zeta2, n)))
    z = float(stats.norm.ppf(0.975))
    return float(stats.norm.cdf(abs(delta) / se - z) + stats.norm.cdf(-abs(delta) / se - z))


def subsampling_check(kernel: dict[tuple[str, str], float], rng: np.random.Generator) -> list[dict[str, float]]:
    """Q4: does the formula's variance match subsampling m of the n candidates?

    Subsampling without replacement from the observed candidates has variance
    V(m) - V(n), since the full-sample statistic is the conditional mean of the
    subsample one.
    """

    components = u_statistic_components(kernel)
    n = int(components["n_candidates"])
    names = sorted({name for pair in kernel for name in pair})
    rows = []
    for m in SUBSAMPLE_SIZES:
        values = []
        for _ in range(N_SUBSAMPLES):
            chosen = sorted(rng.choice(names, size=m, replace=False))
            values.append(np.mean([kernel[(a, b)] for a, b in combinations(chosen, 2)]))
        predicted = u_statistic_variance(components["zeta1"], components["zeta2"], m) - u_statistic_variance(
            components["zeta1"], components["zeta2"], n
        )
        empirical = float(np.std(values, ddof=1))
        rows.append(
            {
                "m": m,
                "empirical_sd": empirical,
                "formula_sd": float(np.sqrt(max(predicted, 0.0))),
                "ratio": empirical / float(np.sqrt(predicted)) if predicted > 0 else float("nan"),
            }
        )
    return rows


def published_estimators(frame: pd.DataFrame, focal: str, target_label: str) -> dict[str, Any]:
    """Two principled corrections to a published decision-accuracy figure.

    The previous round's "accuracy on determined pairs" is withdrawn: determined
    pairs are the well-separated ones, so that number describes an easier
    population rather than correcting the same quantity.
    """

    target = float(frame[frame["scale_label"] == target_label]["compute"].iloc[0])
    focal_compute = float(frame[frame["scale_label"] == focal]["compute"].iloc[0])
    cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
    means = {str(k): float(v) for k, v in cells.mean().items()}
    sds = {str(k): float(v) for k, v in cells.std(ddof=1).items()}
    counts = {str(k): int(v) for k, v in cells.count().items()}
    focal_means = frame[np.isclose(frame["compute"], focal_compute)].groupby("intervention")["bpb"].mean()

    evidence = target_evidence(means, sds, counts, alpha=0.05, multiplicity="bonferroni")
    predictions = {(a, b): float(focal_means[a] - focal_means[b]) for a, b in combinations(sorted(focal_means.index), 2)}
    scored = score_ranker(predictions, evidence)

    observed_accuracy = 1 - scored["observed_rate"]
    posterior_accuracy = 1 - scored["expected_rate"]

    # Disattenuation: a_obs = a*r + (1-a)*(1-r) => a = (a_obs + r - 1) / (2r - 1).
    # r is the reference's probability of matching the true order, averaged.
    reliabilities = np.asarray([e.probability_observed_order_correct for e in evidence], dtype=float)
    r_mean = float(reliabilities.mean())
    disattenuated = float((observed_accuracy + r_mean - 1) / (2 * r_mean - 1)) if r_mean > 0.5 else float("nan")

    # Stratify by true-gap size, because the independence assumption behind
    # disattenuation is least plausible where both ranker and reference err:
    # on small gaps. Reporting per stratum shows how much the pooled number
    # leans on that assumption.
    gaps = np.asarray([abs(e.gap) for e in evidence], dtype=float)
    edges = np.percentile(gaps, [0, 33, 67, 100])
    strata = []
    for index in range(3):
        low, high = edges[index], edges[index + 1]
        subset = [e for e in evidence if (low <= abs(e.gap) <= high if index == 2 else low <= abs(e.gap) < high)]
        if len(subset) < 5:
            continue
        sub_scored = score_ranker(predictions, subset)
        sub_r = float(np.mean([e.probability_observed_order_correct for e in subset]))
        sub_obs = 1 - sub_scored["observed_rate"]
        strata.append(
            {
                "stratum": ["small gaps", "medium gaps", "large gaps"][index],
                "n_pairs": len(subset),
                "mean_reference_reliability": sub_r,
                "observed_accuracy": sub_obs,
                "posterior_accuracy": 1 - sub_scored["expected_rate"],
                "disattenuated_accuracy": (float((sub_obs + sub_r - 1) / (2 * sub_r - 1)) if sub_r > 0.5 else float("nan")),
            }
        )

    return {
        "focal_scale": focal,
        "target_scale": target_label,
        "n_pairs": scored["n_scored"],
        "determined_fraction_bonferroni": sum(1 for e in evidence if e.determined) / len(evidence),
        "observed_accuracy": observed_accuracy,
        "posterior_expected_accuracy": posterior_accuracy,
        "disattenuated_accuracy": disattenuated,
        "mean_reference_reliability": r_mean,
        "by_true_gap_stratum": strata,
        "withdrawn": (
            "The previous 'accuracy on determined pairs' figure is withdrawn as "
            "selection-biased: determined pairs are the well-separated ones, where any ranker "
            "scores near 1, so it measures an easier population rather than correcting the same "
            "quantity. The determined fraction itself is kept as a finding."
        ),
        "assumptions": {
            "posterior_expected_accuracy": (
                "Pair-conditional. Charges each pair by the probability its target order is "
                "correct. Makes no claim about unresolved pairs, which contribute near 0.5 and "
                "therefore pull the average toward chance."
            ),
            "disattenuated_accuracy": (
                "Extrapolates to unresolved pairs assuming ranker errors and reference errors "
                "are INDEPENDENT. That is doubtful here, because both err preferentially on "
                "small true gaps, so the pooled value is reported as a bound and the per-stratum "
                "values show how much it leans on the assumption."
            ),
        },
    }


def dose_response(frame: pd.DataFrame) -> dict[str, Any]:
    """Priority 4: does the lever-arm dose-response survive expected-error scoring?

    The relationship between excess projection error and lever arm was measured
    against observed targets. If it lived in reference noise it would weaken
    under expected-error scoring, which discounts pairs the reference does not
    resolve.
    """

    order = frame.groupby("scale_label")["compute"].first().sort_values()
    labels = list(order.index)
    means_table = cell_means_and_sigma(frame)

    rows = []
    for target_index in range(3, len(labels)):
        target = float(order.iloc[target_index])
        for fit_index in range(2, target_index):
            max_compute = float(order.iloc[fit_index])
            fit_budgets = normalize_budgets(
                sorted(float(c) for c in frame["compute"].unique() if c <= max_compute), target=target
            )
            if len(fit_budgets) < 3:
                continue
            cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
            means = {str(k): float(v) for k, v in cells.mean().items()}
            sds = {str(k): float(v) for k, v in cells.std(ddof=1).items()}
            counts = {str(k): int(v) for k, v in cells.count().items()}
            evidence = target_evidence(means, sds, counts, multiplicity="bonferroni")

            top = max(fit_budgets)
            projected: dict[str, float] = {}
            single: dict[str, float] = {}
            for name in sorted(means):
                subset = means_table[
                    (means_table["intervention"] == name) & (means_table["compute"].isin(fit_budgets))
                ].sort_values("compute")
                try:
                    params = fit_power_law(subset["compute"].to_numpy(dtype=float), subset["bpb_mean"].to_numpy(dtype=float))
                    projected[name] = float(bpb_power_law(target, *params))
                except (FitError, RuntimeError, ValueError):
                    projected[name] = float("nan")
                cell = subset[np.isclose(subset["compute"], top)]
                single[name] = float(cell["bpb_mean"].iloc[0]) if len(cell) else float("nan")

            names = sorted(means)
            left = score_ranker(_gaps(projected, names), evidence)
            right = score_ranker(_gaps(single, names), evidence)
            rows.append(
                {
                    "target": labels[target_index],
                    "max_fit": labels[fit_index],
                    "lever_arm": target / max_compute,
                    "observed_excess_pp": 100 * (left["observed_rate"] - right["observed_rate"]),
                    "expected_excess_pp": 100 * (left["expected_rate"] - right["expected_rate"]),
                }
            )

    log_lever = np.log([r["lever_arm"] for r in rows])
    out: dict[str, Any] = {"n_designs": len(rows), "designs": rows}
    for key in ("observed_excess_pp", "expected_excess_pp"):
        values = np.asarray([r[key] for r in rows], dtype=float)
        result = stats.spearmanr(log_lever, values)
        out[key] = {"spearman_vs_log_lever_arm": float(result.statistic), "p_value": float(result.pvalue)}
    out["verdict"] = (
        "SURVIVES"
        if out["expected_excess_pp"]["spearman_vs_log_lever_arm"] > 0.35 and out["expected_excess_pp"]["p_value"] < 0.01
        else "WEAKENS"
    )
    return out


CALIBRATION_REPLICATES = 60
CALIBRATION_RESAMPLES = 800


def estimator_calibration(zeta1: float, zeta2: float, n: int, rng: np.random.Generator) -> dict[str, Any]:
    """How each candidate-level standard error compares with a known truth.

    Simulates the additive kernel h(a, b) = u_a + u_b + e_ab at the measured
    Hoeffding components, where zeta1 = Var(u) and zeta2 = 2 Var(u) + Var(e),
    so the true standard error at n candidates is known exactly.
    """

    sd_u = float(np.sqrt(max(zeta1, 0.0)))
    sd_e = float(np.sqrt(max(zeta2 - 2 * zeta1, 0.0)))
    truth = float(np.sqrt(u_statistic_variance(zeta1, zeta2, n)))
    names = [f"c{i:02d}" for i in range(n)]
    ratios: dict[str, list[float]] = {
        "weighted_bootstrap": [],
        "jackknife": [],
        "u_statistic": [],
        "unique_set_bootstrap": [],
    }
    for _ in range(CALIBRATION_REPLICATES):
        u = rng.normal(0.0, sd_u, size=n)
        kernel = {(names[i], names[j]): u[i] + u[j] + rng.normal(0.0, sd_e) for i, j in combinations(range(n), 2)}
        ratios["weighted_bootstrap"].append(float(np.std(candidate_bootstrap(kernel, rng, CALIBRATION_RESAMPLES))) / truth)
        ratios["jackknife"].append(jackknife_standard_error(kernel) / truth)
        ratios["u_statistic"].append(float(u_statistic_test(kernel)["standard_error"]) / truth)
        unique = []
        for _ in range(CALIBRATION_RESAMPLES):
            chosen = sorted(set(rng.choice(names, size=n)))
            unique.append(np.mean([kernel[pair] for pair in combinations(chosen, 2)]))
        ratios["unique_set_bootstrap"].append(float(np.std(unique)) / truth)
    return {
        "zeta1": zeta1,
        "zeta2": zeta2,
        "n_candidates": n,
        "true_standard_error": truth,
        "replicates": CALIBRATION_REPLICATES,
        "mean_se_over_truth": {name: float(np.mean(values)) for name, values in ratios.items()},
    }


def power_predictions(significance: dict[str, Any]) -> dict[str, Any]:
    """Q1-Q4 of PREDICTIONS_TASK_POWER.md, scored on the C4 primary design."""

    factor = significance["unique_set_over_corrected_se"]
    needed = significance["power"]["candidates_at_point_estimate"]
    ratios = [row["ratio"] for row in significance["subsampling_check"]]
    return {
        "Q1_unique_set_overstates_se": {
            "measured": factor,
            "band": [1.1, 1.4],
            "confirmed": 1.1 <= factor <= 1.4,
            "note": (
                "Refuted in the opposite direction. The weighted bootstrap is itself conservative when "
                "pair-level variance dominates; see estimator_calibration."
            ),
        },
        "Q2_interval_includes_zero": {
            "measured": [
                significance["candidate_resampling"]["ci_low_pp"],
                significance["candidate_resampling"]["ci_high_pp"],
            ],
            "confirmed": not significance["candidate_resampling"]["excludes_zero"],
        },
        "Q3_candidates_for_power": {
            "measured": needed,
            "band": [40, 120],
            "confirmed": needed is not None and 40 <= needed <= 120,
        },
        "Q4_formula_matches_subsampling": {
            "measured_ratios": ratios,
            "confirmed": all(abs(r - 1) <= 0.2 for r in ratios),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "target_scoring" / "expected_error.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    payload: dict[str, Any] = {
        "inputs": {name: _sha(REPO / "data" / path) for name, path in METRICS.items()},
        "primary_rule": "expected_error",
        "why": (
            "Determined-subset scoring is vacuous on the primary design: both rankers make zero "
            "errors there, so it cannot distinguish them. Expected-error charges every pair in "
            "expectation and retains a measurable difference."
        ),
        "n_resamples": N_RESAMPLES,
        "regimes": {},
    }

    for metric in METRICS:
        frame = load(metric)
        for design_name, max_fit, target_label in DESIGNS:
            labels = set(frame["scale_label"])
            if target_label not in labels or max_fit not in labels:
                continue
            target = float(frame[frame["scale_label"] == target_label]["compute"].iloc[0])
            max_compute = float(frame[frame["scale_label"] == max_fit]["compute"].iloc[0])
            fit_budgets = normalize_budgets(
                sorted(float(c) for c in frame["compute"].unique() if c <= max_compute), target=target
            )
            if len(fit_budgets) < 3:
                continue

            cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
            means = {str(k): float(v) for k, v in cells.mean().items()}
            sds = {str(k): float(v) for k, v in cells.std(ddof=1).items()}
            counts = {str(k): int(v) for k, v in cells.count().items()}
            evidence = target_evidence(means, sds, counts, alpha=0.05, multiplicity="bonferroni")
            determined_fraction = sum(1 for e in evidence if e.determined) / len(evidence)

            values = predictions_for(frame, fit_budgets, target)
            names = sorted(means)
            evidence_by_pair = {(e.left, e.right): e for e in evidence}

            entry: dict[str, Any] = {
                "metric": metric,
                "design": design_name,
                "determined_fraction_bonferroni": determined_fraction,
                "regime": "low_noise" if determined_fraction >= LOW_NOISE_DETERMINED_FRACTION else "high_noise",
                "rankers": {},
            }
            for name, per_candidate in values.items():
                entry["rankers"][name] = score_ranker(_gaps(per_candidate, names), evidence)

            if design_name == "primary_4M-300M_gate530M":
                significance = candidate_resampled_difference(values, evidence_by_pair, names, rng)
                significance["subsampling_check"] = subsampling_check(significance.pop("_kernel"), rng)
                significance["estimator_calibration"] = estimator_calibration(
                    significance["u_statistic"]["zeta1"],
                    significance["u_statistic"]["zeta2"],
                    int(significance["u_statistic"]["n_candidates"]),
                    rng,
                )
                entry["significance"] = significance
            payload["regimes"][f"{metric}/{design_name}"] = entry

    payload["power_predictions"] = power_predictions(
        payload["regimes"]["c4_en_bits_per_token/primary_4M-300M_gate530M"]["significance"]
    )
    payload["published_figure"] = published_estimators(load("olmes_macro_error"), "150M", "1B")
    payload["dose_response"] = dose_response(load("c4_en_bits_per_token"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    key = "c4_en_bits_per_token/primary_4M-300M_gate530M"
    print(json.dumps(payload["regimes"][key]["significance"], indent=2))
    print(json.dumps({k: v for k, v in payload["published_figure"].items() if isinstance(v, float)}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
