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
from asla.analysis.target_scoring import score_ranker, target_evidence
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

    def difference(subset: list[str]) -> float | None:
        pairs = [(a, b) for a, b in combinations(sorted(set(subset)), 2) if (a, b) in evidence_by_pair]
        if len(pairs) < 3:
            return None
        chosen = [evidence_by_pair[p] for p in pairs]
        left = score_ranker(_gaps(values["projection"], names), chosen)
        right = score_ranker(_gaps(values["single_scale"], names), chosen)
        if not np.isfinite(left["expected_rate"]) or not np.isfinite(right["expected_rate"]):
            return None
        return 100 * (left["expected_rate"] - right["expected_rate"])

    point = difference(names)
    candidate_draws = []
    for _ in range(n_resamples):
        drawn = list(rng.choice(names, size=len(names), replace=True))
        value = difference(drawn)
        if value is not None:
            candidate_draws.append(value)

    all_pairs = sorted(evidence_by_pair)
    pair_draws = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(all_pairs), size=len(all_pairs))
        chosen = [evidence_by_pair[all_pairs[i]] for i in indices]
        left = score_ranker(_gaps(values["projection"], names), chosen)
        right = score_ranker(_gaps(values["single_scale"], names), chosen)
        pair_draws.append(100 * (left["expected_rate"] - right["expected_rate"]))

    def summarise(draws: list[float], label: str) -> dict[str, Any]:
        array = np.asarray(draws, dtype=float)
        low, high = (float(np.percentile(array, q)) for q in (2.5, 97.5))
        return {
            "unit": label,
            "n_draws": int(array.size),
            "mean": float(array.mean()),
            "ci_low_pp": low,
            "ci_high_pp": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "p_two_sided": float(2 * min((array <= 0).mean(), (array >= 0).mean())),
        }

    return {
        "point_difference_pp": point,
        "candidate_resampling": summarise(candidate_draws, "candidate"),
        "pair_resampling_naive": summarise(pair_draws, "pair (anti-conservative)"),
        "dependence_correction_factor": (
            (summarise(candidate_draws, "c")["ci_high_pp"] - summarise(candidate_draws, "c")["ci_low_pp"])
            / (summarise(pair_draws, "p")["ci_high_pp"] - summarise(pair_draws, "p")["ci_low_pp"])
        ),
    }


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
                entry["significance"] = candidate_resampled_difference(values, evidence_by_pair, names, rng)
            payload["regimes"][f"{metric}/{design_name}"] = entry

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
