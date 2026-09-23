"""Task 4 of PREDICTIONS_TASK_CALIBRATED_INFERENCE.md: every inferential claim, restated and corrected.

Every claim the paper presents as inferential is recomputed under the
calibrated procedure: raw Welch determination, moderated-SE expected-error
weights, and the candidate-level interval that the calibration benchmark
supports for each ranker. The claims are then corrected across the paper-level
family, with Holm and with Benjamini-Yekutieli. BY is the appropriate one here:
every claim rests on the same 25 DataDecide recipes, so the tests are
dependent.

The primary family is the random-candidate estimand, since the claims concern
selection methods in general. The fixed-candidate estimand is a second family,
reported for the ranker comparisons whose data support it. Its claims are
scoped to these 25 recipes.

Writes ``results/target_scoring/restated_claims.json``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.provenance import Source  # noqa: E402

ALPHA = 0.05
METRICS = ("c4_en_bits_per_token", "olmes_macro_error")
RANKERS = ("projection", "shared_exponent", "eb_shrinkage", "ensemble", "checkpoint_augmented")
SEED = 20260923


def holm(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (m - rank):
            reject[index] = True
        else:
            break
    return reject


def benjamini_yekutieli(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    m = len(p_values)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p_values)
    passing = [k for k, index in enumerate(order, start=1) if p_values[index] <= alpha * k / (m * harmonic)]
    cutoff = max(passing) if passing else 0
    reject = [False] * m
    for k, index in enumerate(order, start=1):
        reject[index] = k <= cutoff
    return reject


def _p(estimate: float, se: float) -> float:
    """Two-sided normal p; a zero standard error means certainty: p = 1 at a zero estimate, else 0."""

    if se > 0:
        return float(2 * stats.norm.sf(abs(estimate) / se))
    return 1.0 if estimate == 0 else 0.0


def ranker_claims(calibrated: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Both estimands for every ranker comparison, on both metrics, with calibrated critical values.

    C4 is the metric the calibration benchmark was built on. For OLMES macro error
    the calibrated critical values are transported, which is stated on each
    claim.
    """

    calibration = importlib.import_module("scripts.run_calibration")
    random_claims, fixed_claims, detail = [], [], {}
    for metric in METRICS:
        calibration.METRIC = metric
        calibration._CONTEXT.clear()
        real = calibration.apply_calibration(
            calibration.real_data(np.random.default_rng([SEED, METRICS.index(metric)])), calibrated
        )
        detail[metric] = real
        scope = "" if metric == calibration.CALIBRATED_METRIC else " (critical value transported from the C4 benchmark)"
        for name in RANKERS:
            row = real["comparisons"][name]
            r, f = row["calibrated_random_candidate"], row["calibrated_fixed_candidate"]
            random_claims.append(
                {
                    "claim": f"{name} vs single-scale, {metric}",
                    "family": "ranker comparison",
                    "estimate_pp": row["estimate_pp"],
                    "ci_pp": [r["ci_low"], r["ci_high"]],
                    "method": r["se_method"] + scope,
                    "p": r["p_two_sided"] if r["p_two_sided"] is not None else 1.0,
                }
            )
            fixed_claims.append(
                {
                    "claim": f"{name} vs single-scale, {metric} (these 25 recipes)",
                    "family": "ranker comparison, fixed-candidate",
                    "estimate_pp": row["estimate_pp"],
                    "ci_pp": [f["ci_low"], f["ci_high"]],
                    "method": "parametric resampling, calibrated critical value" + scope,
                    "p": f["p_two_sided"] if f["p_two_sided"] is not None else 1.0,
                }
            )
    return random_claims, fixed_claims, detail


def _calibrated(estimate: float, u_se: float, jk_se: float, calibrated: dict[str, Any]) -> tuple[float, list[float], str]:
    calibration = importlib.import_module("scripts.run_calibration")
    label = calibrated["random_adopted"]
    se = jk_se if label == "random_jackknife" else u_se
    block = calibrated[label]
    k = block["critical_value"]
    p = (
        calibration.calibrated_p(abs(estimate) / se, block["null_quantiles_of_abs_t"], block["n_statistics"])
        if se > 0
        else 1.0
    )
    return p, [estimate - k * se, estimate + k * se], f"{label} (critical value transported from the C4 benchmark)"


def other_claims(calibrated: dict[str, Any]) -> list[dict[str, Any]]:
    rescoring = Source.load("target_scoring/rescoring.json")
    expected = Source.load("target_scoring/expected_error.json")
    published = Source.load("external/published_comparisons.json")
    claims = []
    for row in rescoring.get("optimal_allocation.by_budget_fraction"):
        test = row["candidate_test"]
        p, ci, method = _calibrated(test["point_pp"], test["standard_error_pp"], test["jackknife_se_pp"], calibrated)
        claims.append(
            {
                "claim": f"optimised allocation at {row['budget_fraction']:g} of budget vs single-scale, c4",
                "family": "allocation",
                "estimate_pp": test["point_pp"],
                "ci_pp": ci,
                "method": method,
                "p": p,
            }
        )
    claims.append(
        {
            "claim": "projection excess rises with log lever arm (expected-error scoring)",
            "family": "dose-response",
            "estimate_pp": None,
            "ci_pp": None,
            "method": "delete-one-recipe jackknife of the Spearman correlation's Fisher z",
            "p": expected.number("dose_response.candidate_jackknife.p_two_sided"),
        }
    )
    d1 = published.get("d1_scaling_laws_vs_single_scale.comparisons")
    for key, row in sorted(d1.items()):
        task, setup, scale, target = key.split("|")
        if task == "olmes_10_macro_avg" and scale == "750M" and target == "three_seed_mean":
            p, ci, method = _calibrated(
                row["difference_pp"], row["candidate_se_pp"], row["candidate_jackknife_se_pp"], calibrated
            )
            claims.append(
                {
                    "claim": f"DataDecide {setup} vs single-scale at 750M (exploratory: open discrepancy)",
                    "family": "published re-test, scaling laws",
                    "estimate_pp": row["difference_pp"],
                    "ci_pp": ci,
                    "method": method,
                    "p": p,
                }
            )
    d2 = published.get("d2_correct_prob_vs_accuracy.comparisons")
    for key, row in sorted(d2.items()):
        task, scale = key.split("|")
        if task == "olmes_10_macro_avg":
            p, ci, method = _calibrated(
                row["difference_pp"], row["candidate_se_pp"], row["candidate_jackknife_se_pp"], calibrated
            )
            claims.append(
                {
                    "claim": f"Correct Prob vs accuracy at {scale}, OLMES macro",
                    "family": "published re-test, proxy metric",
                    "estimate_pp": row["difference_pp"],
                    "ci_pp": ci,
                    "method": method,
                    "p": p,
                }
            )
    return claims


def corrected(claims: list[dict[str, Any]]) -> dict[str, Any]:
    p_values = [c["p"] for c in claims]
    h, by = holm(p_values), benjamini_yekutieli(p_values)
    for claim, holm_reject, by_reject in zip(claims, h, by):
        claim["nominal_reject"] = bool(claim["p"] <= ALPHA)
        claim["holm_reject"] = bool(holm_reject)
        claim["by_reject"] = bool(by_reject)
    return {
        "m": len(claims),
        "n_nominal": sum(c["nominal_reject"] for c in claims),
        "n_holm": sum(c["holm_reject"] for c in claims),
        "n_by": sum(c["by_reject"] for c in claims),
        "claims": claims,
    }


def main() -> None:
    calibrated = Source.load("target_scoring/calibration.json").get("calibrated")
    random_claims, fixed_claims, detail = ranker_claims(calibrated)
    primary = corrected(random_claims + other_claims(calibrated))
    fixed = corrected(fixed_claims)
    out = {
        "alpha": ALPHA,
        "random_candidate_se": calibrated["random_adopted"],
        "primary_family_random_candidate": primary,
        "fixed_candidate_family": fixed,
        "ranker_detail": detail,
        "correction_note": (
            "Holm controls the family-wise error rate under any dependence; Benjamini-Yekutieli controls the "
            "false discovery rate under arbitrary dependence. Every claim shares the 25 DataDecide recipes."
        ),
    }
    path = REPO / "results" / "target_scoring" / "restated_claims.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for family in ("primary_family_random_candidate", "fixed_candidate_family"):
        block = out[family]
        print(family, {k: block[k] for k in ("m", "n_nominal", "n_holm", "n_by")})
        for c in block["claims"]:
            if c["nominal_reject"]:
                print(f"   {c['claim'][:70]:70s} p={c['p']:.2g} holm={c['holm_reject']} by={c['by_reject']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
