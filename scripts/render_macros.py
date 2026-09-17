"""Generate ``paper/macros.tex`` from committed JSON.

The paper quotes numbers; retyping them is how ``4.22`` survived into five
documents and a draft. Every macro here is read from a results file by key path
via :class:`asla.provenance.Source`, so a correction in the JSON propagates to
the paper on the next render and a stale draft is detectable.

The file opens with a generation timestamp and the source commit, so a draft can
be checked against the repository state it came from.

Usage::

    python scripts/render_macros.py
    \\input{macros}   % in the paper
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from asla.provenance import Source

REPO = Path(__file__).resolve().parents[1]


def _source_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def _macro(name: str, value: str, origin: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}  % {origin}"


def build() -> str:
    theory_v2 = Source.load("theory_v2/theory_v2.json")
    theory = Source.load("theory/theory_check.json")
    adversarial = Source.load("adversarial/a1_independent_rederivation.json")
    dose = Source.load("adversarial/a2_dose_response.json")
    resolution = Source.load("resolution/resolution_study.json")
    abstention = Source.load("abstention/abstention_1B.json")
    allocation = Source.load("allocation/allocation_1B.json")
    curvature = Source.load("curvature/curvature_gate.json")

    real = "FINDING_misspecification_is_structured.evidence.1_misspecification_is_real"

    single_rate = next(
        row["audit"]
        for row in adversarial.get("comparisons.c4_en_bits_per_token.rows")
        if row["quantity"] == "single_scale_mis_selection"
    )
    projection_rate = next(
        row["audit"]
        for row in adversarial.get("comparisons.c4_en_bits_per_token.rows")
        if row["quantity"] == "projection_mis_selection"
    )

    excesses = [
        entry["scored"]["decision_optimal"]["excess_over_single_scale"]
        for entry in allocation.get("sweep").values()
        if entry.get("scored", {}).get("decision_optimal", {}).get("excess_over_single_scale") is not None
    ]
    share_gaps = [
        entry["max_cost_share_gap_decision_vs_estimation"]
        for entry in allocation.get("sweep").values()
        if "max_cost_share_gap_decision_vs_estimation" in entry
    ]
    cost = next(row for row in abstention.get("cost_of_correctness") if row.get("seed_budget") == 3)
    unresolved = resolution.integer("summary.total_unresolved_at_3_seeds")
    adjacent = resolution.integer("summary.total_adjacent_pairs")

    entries: list[tuple[str, str, str]] = [
        # --- Model error (section: structure) ---
        (
            "residRatio",
            f"{theory_v2.number(real + '.residual_scatter_over_seed_se'):.2f}",
            "theory_v2.json",
        ),
        (
            "varInflation",
            f"{theory_v2.number(real + '.variance_inflation'):.2f}",
            "theory_v2.json",
        ),
        (
            "residRatioSuperseded",
            f"{theory_v2.number(real + '.superseded.residual_scatter_over_seed_se'):.2f}",
            "theory_v2.json (WRONG, kept for the correction note)",
        ),
        (
            "varInflationSuperseded",
            f"{theory_v2.number(real + '.superseded.variance_inflation'):.1f}",
            "theory_v2.json (WRONG, kept for the correction note)",
        ),
        # --- Selection rates ---
        ("singleRate", f"{single_rate * 100:.2f}", "a1_independent_rederivation.json"),
        ("projRate", f"{projection_rate * 100:.2f}", "a1_independent_rederivation.json"),
        (
            "maxExcessWinning",
            f"{dose.number('max_excess_among_winning'):.1f}",
            "a2_dose_response.json",
        ),
        ("nMatchedLadders", f"{dose.integer('n_matched_ladders')}", "a2_dose_response.json"),
        (
            "leverArmRho",
            f"{dose.number('rank_correlations.spearman_excess_vs_log_lever_arm'):+.3f}",
            "a2_dose_response.json",
        ),
        (
            "leverArmP",
            f"{dose.number('rank_correlations.spearman_excess_vs_log_lever_arm_p'):.1e}",
            "a2_dose_response.json",
        ),
        (
            "budgetCountRho",
            f"{dose.number('rank_correlations.spearman_excess_vs_budgets'):+.3f}",
            "a2_dose_response.json",
        ),
        # --- Allocation ---
        ("allocExcessMin", f"{min(excesses):+.3f}", "allocation_1B.json"),
        ("allocExcessMax", f"{max(excesses):+.3f}", "allocation_1B.json"),
        ("maxShareGap", f"{max(share_gaps):.5f}", "allocation_1B.json"),
        # --- Confidence ---
        (
            "bootCoverage",
            f"{abstention.number('coverage.bootstrap.coverage'):.2f}",
            "abstention_1B.json",
        ),
        (
            "confCoverage",
            f"{abstention.number('coverage.conformal.coverage'):.2f}",
            "abstention_1B.json",
        ),
        (
            "bootErrorToWidth",
            f"{abstention.number('coverage.bootstrap_centring.error_to_width_ratio'):.2f}",
            "abstention_1B.json",
        ),
        (
            "centringInSeedSE",
            f"{abstention.number('coverage.bootstrap_centring.centring_error_in_seed_standard_errors'):.0f}",
            "abstention_1B.json",
        ),
        (
            "bootEffectiveDelta",
            f"{abstention.number('coverage.bootstrap.effective_error_at_nominal_delta_0_05'):.3f}",
            "abstention_1B.json",
        ),
        (
            "confEffectiveDelta",
            f"{abstention.number('coverage.conformal.effective_error_at_nominal_delta_0_05'):.3f}",
            "abstention_1B.json",
        ),
        # --- Certification and resolution ---
        ("certifiedPairs", f"{cost['n_certified']}", "abstention_1B.json"),
        ("totalPairs", f"{cost['n_pairs']}", "abstention_1B.json"),
        ("certifiedErrors", f"{cost['certified_wrong']}", "abstention_1B.json"),
        (
            "correctCallsAbstained",
            f"{cost['fraction_of_correct_calls_abstained'] * 100:.1f}",
            "abstention_1B.json",
        ),
        ("unresolvedPairs", f"{unresolved}", "resolution_study.json"),
        ("adjacentPairs", f"{adjacent}", "resolution_study.json"),
        ("unresolvedPct", f"{100 * unresolved / adjacent:.1f}", "resolution_study.json"),
        (
            "unidentifiablePairs",
            f"{resolution.integer('summary.total_unidentifiable_at_any_budget')}",
            "resolution_study.json",
        ),
        # --- Curvature gate and the ladder confound ---
        (
            "overshoot",
            f"{curvature.number('gate.C3_quantitative_recovery.mean_observed_overshoot'):+.4f}",
            "curvature_gate.json",
        ),
        (
            "curvatureRho",
            f"{curvature.number('gate.C2_curvature_predicts_overshoot.spearman_rho'):+.3f}",
            "curvature_gate.json",
        ),
        (
            "curvatureReduction",
            f"{100 * curvature.number('gate_adjudication.curvature_absolute_error_reduction'):.0f}",
            "curvature_gate.json",
        ),
        (
            "tokensPerParamReduction",
            f"{100 * curvature.number('gate_adjudication.tokens_per_param_absolute_error_reduction'):.0f}",
            "curvature_gate.json",
        ),
        (
            "targetTokensPerParam",
            f"{curvature.number('alternative_explanations.tokens_per_param.target'):.1f}",
            "curvature_gate.json",
        ),
        (
            "tokensPerParamCorr",
            f"{curvature.number('alternative_explanations.tokens_per_param.corr_log_tpp_with_log_compute'):+.2f}",
            "curvature_gate.json",
        ),
        (
            "confoundSpearman",
            f"{curvature.number('confound_scope.spearman_between_projections'):.4f}",
            "curvature_gate.json",
        ),
        # --- Two-parameter theory diagnosis (corrected pooling) ---
        (
            "varDiagnosisRatio",
            f"{theory.number('variance_diagnosis.ratio_empirical_over_theory'):.2f}",
            "theory_check.json",
        ),
        (
            "gapSdUnderstated",
            f"{theory.number('variance_diagnosis.gap_sd_understated_by'):.2f}",
            "theory_check.json",
        ),
        (
            "sigmaJensenFactor",
            f"{theory.number('constants.noise.sigma_variance_understatement_if_mean_used'):.2f}",
            "theory_check.json",
        ),
    ]

    lines = [
        "% ASLA paper macros --- GENERATED FILE, DO NOT EDIT BY HAND.",
        "%",
        "% Every value is read from a committed results JSON by key path via",
        "% scripts/render_macros.py. Editing a number here breaks the provenance",
        "% chain that exists because a retyped 4.22 survived into five documents",
        "% and a draft before it was found to be wrong.",
        "%",
        "% Regenerate with:  python scripts/render_macros.py",
        "%",
        f"% generated:      {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"% source commit:  {_source_commit()}",
        "",
    ]
    lines.extend(_macro(name, value, origin) for name, value, origin in entries)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "paper" / "macros.tex")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
