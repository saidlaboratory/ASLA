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

import numpy as np

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


RANKER_LABELS = {
    "projection": "Projection",
    "shared_exponent": "Shared exponent",
    "eb_shrinkage": "Empirical-Bayes shrinkage",
    "ensemble": "Ensemble",
    "checkpoint_augmented": "Checkpoint-augmented",
}


def _row(label: str, observed: float, expected: float, test: dict[str, float], verdict: str) -> str:
    return (
        f"{label} & ${observed:+.2f}$ & ${expected:+.2f}$ & "
        f"$[{test['ci_low_pp']:+.2f},\\ {test['ci_high_pp']:+.2f}]$ & {verdict.lower()} \\\\"
    )


def _rescoring_entries() -> list[tuple[str, str, str]]:
    """Table bodies and scalars for the re-scoring section."""

    rescoring = Source.load("target_scoring/rescoring.json")
    entries: list[tuple[str, str, str]] = []
    for macro, metric in (("rescoreRowsLow", "c4_en_bits_per_token"), ("rescoreRowsHigh", "olmes_macro_error")):
        rankers = rescoring.get(f"cells.{metric}.rankers")
        rows = [
            _row(
                RANKER_LABELS[name],
                float(row["excess_observed_pp"]),
                float(row["excess_expected_pp"]),
                row["candidate_test"],
                str(row["verdict"]),
            )
            for name, row in sorted(rankers.items(), key=lambda kv: RANKER_LABELS[kv[0]])
        ]
        entries.append((macro, " ".join(rows), "rescoring.json"))
    entries.append(
        (
            "rescoreDetLow",
            f"{100 * rescoring.number('cells.c4_en_bits_per_token.determined_fraction_bonferroni'):.0f}",
            "rescoring.json",
        )
    )
    entries.append(
        (
            "rescoreDetHigh",
            f"{100 * rescoring.number('cells.olmes_macro_error.determined_fraction_bonferroni'):.0f}",
            "rescoring.json",
        )
    )
    allocation = [
        _row(
            f"{float(row['budget_fraction']):g} of budget",
            float(row["excess_observed_pp"]),
            float(row["excess_expected_pp"]),
            row["candidate_test"],
            str(row["verdict"]),
        )
        for row in rescoring.get("optimal_allocation.by_budget_fraction")
    ]
    entries.append(("rescoreRowsAllocation", " ".join(allocation), "rescoring.json"))
    for name, rule in (("Normal", "normal"), ("StudentT", "student_t")):
        base = f"certification.rules.{rule}"
        entries.append((f"cert{name}N", f"{rescoring.integer(base + '.n_certified')}", "rescoring.json"))
        entries.append(
            (f"cert{name}Expected", f"{rescoring.number(base + '.certified_errors_expected'):.2f}", "rescoring.json")
        )
    entries.append(
        (
            "certNormalMinP",
            f"{rescoring.number('certification.rules.normal.min_target_order_probability_on_certified'):.2f}",
            "rescoring.json",
        )
    )
    return entries


def _power_entries(expected: Source, base: str) -> list[tuple[str, str, str]]:
    """Candidate-level inference on the C4 comparison, and what a comparison needs."""

    origin = "expected_error.json"
    calibration = base + ".estimator_calibration.mean_se_over_truth"
    by_delta = base + ".power.candidates_by_delta_pp"
    return [
        ("nCandidates", f"{expected.integer(base + '.u_statistic.n_candidates')}", origin),
        ("expectedSE", f"{expected.number(base + '.u_statistic.standard_error'):.2f}", origin),
        ("bootCandLow", f"{expected.number(base + '.candidate_resampling.ci_low_pp'):+.2f}", origin),
        ("bootCandHigh", f"{expected.number(base + '.candidate_resampling.ci_high_pp'):+.2f}", origin),
        ("bootCandP", f"{expected.number(base + '.candidate_resampling.p_two_sided'):.2f}", origin),
        ("uniqueSetP", f"{expected.number(base + '.unique_set_candidate_resampling_superseded.p_two_sided'):.2f}", origin),
        (
            "zetaRatio",
            f"{expected.number(base + '.u_statistic.zeta2') / expected.number(base + '.u_statistic.zeta1'):.0f}",
            origin,
        ),
        ("calibU", f"{expected.number(calibration + '.u_statistic'):.2f}", origin),
        ("calibBoot", f"{expected.number(calibration + '.weighted_bootstrap'):.2f}", origin),
        ("calibJack", f"{expected.number(calibration + '.jackknife'):.2f}", origin),
        ("calibUnique", f"{expected.number(calibration + '.unique_set_bootstrap'):.2f}", origin),
        ("powerNeeded", f"{expected.integer(base + '.power.candidates_at_point_estimate')}", origin),
        ("powerNeededOne", f"{expected.integer(by_delta + '.1')}", origin),
        ("powerNeededTwo", f"{expected.integer(by_delta + '.2')}", origin),
        ("powerNeededFive", f"{expected.integer(by_delta + '.5')}", origin),
        ("powerAtObserved", f"{100 * expected.number(base + '.power.power_at_observed_count'):.0f}", origin),
    ]


def _published_entries() -> list[tuple[str, str, str]]:
    """Candidate-level re-tests of published comparisons, and the correction curve."""

    out = Source.load("external/published_comparisons.json")
    expected = Source.load("target_scoring/expected_error.json")
    origin = "published_comparisons.json"
    gate_rows = out.get("d1_scaling_laws_vs_single_scale.gate.rows")
    exact = out.integer("d1_scaling_laws_vs_single_scale.gate.n_rows_exact")
    d1 = out.get("d1_scaling_laws_vs_single_scale.comparisons")
    macro_750 = {k: v for k, v in d1.items() if k.startswith("olmes_10_macro_avg|") and k.endswith("|750M|three_seed_mean")}
    worst_key = min(macro_750, key=lambda k: macro_750[k]["difference_pp"])
    worst = macro_750[worst_key]
    others = [v for k, v in macro_750.items() if k != worst_key]
    d2 = out.get("d2_correct_prob_vs_accuracy.comparisons")
    d2_macro = [v for k, v in d2.items() if k.startswith("olmes_10_macro_avg|")]
    curve = out.get("correction_curve_range.curve_by_percentile")
    checks = out.get("correction_curve_c4.subsampling_check")
    invariance = "regimes.c4_en_bits_per_token/primary_4M-300M_gate530M.significance.estimator_invariance"
    signs = "d1_sign_breakdown_three_seed_target"
    return [
        ("gateExact", f"{exact}", origin),
        ("gateRows", f"{len(gate_rows)}", origin),
        ("gateMaxError", f"{out.number('d1_scaling_laws_vs_single_scale.gate.max_abs_error_released_target'):.2f}", origin),
        ("dOneVariants", f"{len(macro_750)}", origin),
        ("dOneMacroExcl", f"{sum(v['excludes_zero'] for v in macro_750.values())}", origin),
        ("dOneWorstDiff", f"{worst['difference_pp']:+.1f}", origin),
        ("dOneWorstLow", f"{worst['ci_low_pp']:+.1f}", origin),
        ("dOneWorstHigh", f"{worst['ci_high_pp']:+.1f}", origin),
        ("dOneTypicalHalfWidth", f"{np.median([(v['ci_high_pp'] - v['ci_low_pp']) / 2 for v in others]):.0f}", origin),
        ("dOneAll", f"{out.integer(signs + '.n')}", origin),
        ("dOneAllBetter", f"{out.integer(signs + '.law_reliably_better')}", origin),
        ("dOneAllWorse", f"{out.integer(signs + '.law_reliably_worse')}", origin),
        ("dTwoScales", f"{len(d2_macro)}", origin),
        ("dTwoMacroBetter", f"{sum(v['excludes_zero'] and v['difference_pp'] > 0 for v in d2_macro)}", origin),
        ("dTwoMacroWorse", f"{sum(v['excludes_zero'] and v['difference_pp'] < 0 for v in d2_macro)}", origin),
        ("dTwoMacroPointWorse", f"{sum(v['difference_pp'] < 0 for v in d2_macro)}", origin),
        ("dTwoAll", f"{len(d2)}", origin),
        ("dTwoAllBetter", f"{sum(v['excludes_zero'] and v['difference_pp'] > 0 for v in d2.values())}", origin),
        ("dTwoAllWorse", f"{sum(v['excludes_zero'] and v['difference_pp'] < 0 for v in d2.values())}", origin),
        ("curveComparisons", f"{out.integer('correction_curve_range.n_comparisons')}", origin),
        ("curveZetaMedian", f"{out.number('correction_curve_range.zeta1_over_zeta2_percentiles.50'):.3f}", origin),
        ("curveMedTen", f"{curve['50']['10']:.2f}", origin),
        ("curveMedTwentyFive", f"{curve['50']['25']:.2f}", origin),
        ("curveMedHundred", f"{curve['50']['100']:.2f}", origin),
        ("curveLowTwentyFive", f"{curve['10']['25']:.2f}", origin),
        ("curveHighTwentyFive", f"{curve['90']['25']:.2f}", origin),
        ("curveCFour", f"{out.number('correction_curve_c4.at_observed_n'):.2f}", origin),
        ("curveCheckWorstLarge", f"{100 * max(abs(r['relative_error']) for r in checks[1:]):.0f}", origin),
        ("curveCheckSmallest", f"{100 * abs(checks[0]['relative_error']):.0f}", origin),
        ("curveCheckSmallestM", f"{checks[0]['m']}", origin),
        ("pInvariantMin", f"{expected.number(invariance + '.p_min'):.2f}", "expected_error.json"),
        ("pInvariantMax", f"{expected.number(invariance + '.p_max'):.2f}", "expected_error.json"),
    ]


def _noise_model_entries() -> list[tuple[str, str, str]]:
    """Seed coupling, moderated variance and its known-answer tests."""

    coupling = Source.load("target_scoring/seed_coupling.json")
    moderated = Source.load("target_scoring/moderated_variance.json")
    c4 = "metrics.c4_en_bits_per_token.all_labels_below_1B.summary"
    err = "metrics.olmes_macro_error.all_labels_below_1B.summary"
    focus = "metrics.c4_en_bits_per_token.focus_pairs.300M|530M"
    null = "polypythias_null_test.rates_over_nominal"
    rates = moderated.get(null)
    bonferroni_level = min(rates["moderated_t"], key=float)
    brier = moderated.get("polypythias_weight_calibration.brier")
    det = "metrics.{}.determination.530M.bonferroni.{}"
    return [
        ("couplingPairs", f"{coupling.integer(c4 + '.n_scale_pairs')}", "seed_coupling.json"),
        ("couplingMedianCFour", f"{coupling.number(c4 + '.median_rho'):+.3f}", "seed_coupling.json"),
        ("couplingMedianErr", f"{coupling.number(err + '.median_rho'):+.3f}", "seed_coupling.json"),
        ("couplingRejectCFour", f"{100 * coupling.number(c4 + '.fraction_p_below_0_05'):.1f}", "seed_coupling.json"),
        ("couplingRejectErr", f"{100 * coupling.number(err + '.fraction_p_below_0_05'):.1f}", "seed_coupling.json"),
        ("couplingFocusRho", f"{coupling.number(focus + '.rho'):.2f}", "seed_coupling.json"),
        ("couplingFocusP", f"{coupling.number(focus + '.p_two_sided'):.2f}", "seed_coupling.json"),
        ("polyCells", f"{moderated.integer('polypythias.n_cells')}", "moderated_variance.json"),
        ("polyBetter", f"{moderated.integer('polypythias.n_cells_moderation_better')}", "moderated_variance.json"),
        ("polyRatioRaw", f"{moderated.number('polypythias.median_ratio_raw'):.3f}", "moderated_variance.json"),
        ("polyRatioMod", f"{moderated.number('polypythias.median_ratio_moderated'):.3f}", "moderated_variance.json"),
        ("polyRmseRaw", f"{moderated.number('polypythias.median_rmse_log_raw'):.2f}", "moderated_variance.json"),
        ("polyRmseMod", f"{moderated.number('polypythias.median_rmse_log_moderated'):.2f}", "moderated_variance.json"),
        ("nullTests", f"{moderated.integer('polypythias_null_test.n_tests')}", "moderated_variance.json"),
        ("nullModFive", f"{rates['moderated_t']['0.05']:.2f}", "moderated_variance.json"),
        ("nullModBonf", f"{rates['moderated_t'][bonferroni_level]:.1f}", "moderated_variance.json"),
        ("nullRawFive", f"{rates['raw_welch']['0.05']:.2f}", "moderated_variance.json"),
        ("nullRawBonf", f"{rates['raw_welch'][bonferroni_level]:.2f}", "moderated_variance.json"),
        ("brierRaw", f"{brier['a_normal_raw_se']:.4f}", "moderated_variance.json"),
        ("brierModT", f"{brier['b_t_moderated_se_prior_df']:.4f}", "moderated_variance.json"),
        ("brierAdopted", f"{brier['c_t_moderated_se_raw_df']:.4f}", "moderated_variance.json"),
        (
            "detModCFour",
            f"{100 * moderated.number(det.format('c4_en_bits_per_token', 'moderated_determined_fraction')):.0f}",
            "moderated_variance.json",
        ),
        (
            "detModErr",
            f"{100 * moderated.number(det.format('olmes_macro_error', 'moderated_determined_fraction')):.0f}",
            "moderated_variance.json",
        ),
        (
            "detRawErrFiveThirty",
            f"{100 * moderated.number(det.format('olmes_macro_error', 'raw_determined_fraction')):.0f}",
            "moderated_variance.json",
        ),
    ]


COMPARATOR_LABELS = {
    "projection": "Projection",
    "shared_exponent": "Shared exponent",
    "eb_shrinkage": "EB shrinkage",
    "ensemble": "Ensemble",
    "checkpoint_augmented": "Checkpoint-aug.",
    "single_scale_150M": "Top rung 150M (planted)",
    "single_scale_90M": "Top rung 90M (planted)",
    "single_scale_60M": "Top rung 60M (planted)",
    "single_scale_20M": "Top rung 20M (planted)",
}


def _tex(text: str) -> str:
    return text.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")


def _calibration_entries() -> list[tuple[str, str, str]]:
    """The end-to-end calibration benchmark and the two estimands on the real C4 data."""

    cal = Source.load("target_scoring/calibration.json")
    by = cal.get("calibration.by_comparator")
    procedures = cal.get("calibrated")
    fixed, jack, u_stat = procedures["fixed_bootstrap"], procedures["random_jackknife"], procedures["random_u"]
    rows = []
    for name, label in COMPARATOR_LABELS.items():
        row = by[name]
        rows.append(
            f"{label} & ${row['theta_fixed']:+.2f}$ & ${row['theta_random']:+.2f}$ & ${row['bias_fixed']:+.2f}$ & "
            f"{row['fixed_candidate_coverage']:.2f} & {row['random_candidate_coverage_u']:.2f} & "
            f"{fixed['cross_fit_coverage'][name]:.2f} & {jack['cross_fit_coverage'][name]:.2f} & "
            f"{fixed['power_at_critical_value'][name]:.2f} & {jack['power_at_critical_value'][name]:.2f} \\\\"
        )
    real = cal.get("real_data.comparisons")
    head = real["projection"]
    fixed_ci, random_ci = head["calibrated_fixed_candidate"], head["calibrated_random_candidate"]
    origin = "calibration.json"
    return [
        ("calibrationRows", " ".join(rows), origin),
        ("calFixedWorlds", f"{cal.integer('design.worlds.fixed')}", origin),
        ("calRandomWorlds", f"{cal.integer('design.worlds.random')}", origin),
        ("calBootDraws", f"{cal.integer('design.bootstrap_draws_per_fixed_world')}", origin),
        ("calFixedK", f"{fixed['critical_value']:.2f}", origin),
        ("calJackK", f"{jack['critical_value']:.2f}", origin),
        ("calUK", f"{u_stat['critical_value']:.2f}", origin),
        ("calJackHalf", f"{jack['median_half_width_pp']:.2f}", origin),
        ("calUHalf", f"{u_stat['median_half_width_pp']:.2f}", origin),
        ("calFixedCovMin", f"{min(fixed['cross_fit_coverage'].values()):.2f}", origin),
        ("calFixedCovMax", f"{max(fixed['cross_fit_coverage'].values()):.2f}", origin),
        ("calJackCovMin", f"{min(jack['cross_fit_coverage'].values()):.2f}", origin),
        ("calJackCovMax", f"{max(jack['cross_fit_coverage'].values()):.2f}", origin),
        ("calUCovMin", f"{min(by[n]['random_candidate_coverage_u'] for n in COMPARATOR_LABELS):.2f}", origin),
        ("calUCovMax", f"{max(by[n]['random_candidate_coverage_u'] for n in COMPARATOR_LABELS):.2f}", origin),
        ("calProjFixedCov", f"{fixed['cross_fit_coverage']['projection']:.2f}", origin),
        ("calMaxBias", f"{max(abs(by[n]['bias_fixed']) for n in COMPARATOR_LABELS):.2f}", origin),
        (
            "calCouplingShift",
            f"{max(abs(v) for v in cal.get('calibration.coupling_sensitivity_shift_pp').values()):.2f}",
            origin,
        ),
        ("headEstimate", f"{head['estimate_pp']:.2f}", origin),
        ("headFixedLow", f"{fixed_ci['ci_low']:+.2f}", origin),
        ("headFixedHigh", f"{fixed_ci['ci_high']:+.2f}", origin),
        ("headFixedP", f"{fixed_ci['p_two_sided']:.3f}", origin),
        ("headRandomLow", f"{random_ci['ci_low']:+.2f}", origin),
        ("headRandomHigh", f"{random_ci['ci_high']:+.2f}", origin),
        ("headRandomP", f"{random_ci['p_two_sided']:.2f}", origin),
        ("ensFixedP", f"{real['ensemble']['calibrated_fixed_candidate']['p_two_sided']:.4f}", origin),
        ("ensRandomP", f"{real['ensemble']['calibrated_random_candidate']['p_two_sided']:.3f}", origin),
    ]


def _dose_jackknife_entries() -> list[tuple[str, str, str]]:
    expected = Source.load("target_scoring/expected_error.json")
    base = "dose_response.candidate_jackknife"
    ci = expected.get(base + ".rho_ci")
    return [
        ("doseJackP", f"{expected.number(base + '.p_two_sided'):.2g}", "expected_error.json"),
        ("doseJackLow", f"{ci[0]:+.2f}", "expected_error.json"),
        ("doseJackHigh", f"{ci[1]:+.2f}", "expected_error.json"),
    ]


def _restated_entries() -> list[tuple[str, str, str]]:
    """Every inferential claim under the calibrated procedure, with Holm and BY."""

    restated = Source.load("target_scoring/restated_claims.json")
    origin = "restated_claims.json"
    out = []
    for key, macro in (("primary_family_random_candidate", "Primary"), ("fixed_candidate_family", "Fixed")):
        block = restated.get(key)
        rows = []
        for claim in block["claims"]:
            ci = claim["ci_pp"]
            ci_text = f"$[{ci[0]:+.1f},\\ {ci[1]:+.1f}]$" if ci else "---"
            est = f"${claim['estimate_pp']:+.2f}$" if claim["estimate_pp"] is not None else "---"
            verdict = (
                "Holm"
                if claim["holm_reject"]
                else ("BY" if claim["by_reject"] else ("nominal" if claim["nominal_reject"] else "---"))
            )
            rows.append(f"{_tex(claim['claim'])} & {est} & {ci_text} & {claim['p']:.2g} & {verdict} \\\\")
        out += [
            (f"restated{macro}Rows", " ".join(rows), origin),
            (f"restated{macro}M", f"{block['m']}", origin),
            (f"restated{macro}Nominal", f"{block['n_nominal']}", origin),
            (f"restated{macro}Holm", f"{block['n_holm']}", origin),
            (f"restated{macro}BY", f"{block['n_by']}", origin),
        ]
    return out


def _fit_structure_entries() -> list[tuple[str, str, str]]:
    """Floor profile likelihood and the design theorem."""

    floor = Source.load("adversarial/floor_profile.json")
    old_floor = Source.load("adversarial/floor_identifiability.json")
    fit = Source.load("target_scoring/fit_structure.json")
    theorem = "design_theorem"
    three_decades = theorem + ".region_optimal_designs.3_decades"
    return [
        ("floorN", f"{floor.integer('c4_en_bits_per_token.n_interventions')}", "floor_profile.json"),
        ("floorCFourExcl", f"{floor.integer('c4_en_bits_per_token.n_interval_excludes_zero')}", "floor_profile.json"),
        ("floorErrExcl", f"{floor.integer('olmes_macro_error.n_interval_excludes_zero')}", "floor_profile.json"),
        (
            "floorErrWidthRel",
            f"{100 * floor.number('olmes_macro_error.median_interval_width_relative_to_ymin'):.0f}",
            "floor_profile.json",
        ),
        (
            "floorCFourWidthRel",
            f"{100 * floor.number('c4_en_bits_per_token.median_interval_width_relative_to_ymin'):.0f}",
            "floor_profile.json",
        ),
        ("floorCondRatio", f"{floor.number('predictions.F3_conditioning.measured_ratio'):.0f}", "floor_profile.json"),
        (
            "floorMisspecCFour",
            f"{floor.number('c4_en_bits_per_token.median_misspecification_ratio'):.1f}",
            "floor_profile.json",
        ),
        ("floorNewWidth", f"{floor.number('c4_en_bits_per_token.median_interval_width'):.2f}", "floor_profile.json"),
        (
            "floorOldWidth",
            f"{old_floor.number('c4_en_bits_per_token.median_admissible_floor_width'):.2f}",
            "floor_identifiability.json",
        ),
        ("floorFCrit", f"{floor.number('c4_en_bits_per_token.per_intervention.0.f_critical'):.2f}", "floor_profile.json"),
        ("thmDesigns", f"{fit.integer(theorem + '.n_random_designs')}", "fit_structure.json"),
        ("thmCorr", f"{fit.number(theorem + '.region_vs_point_correlation'):.6f}", "fit_structure.json"),
        (
            "thmExcessThree",
            f"{100 * fit.number(three_decades + '.excess_point_variance_of_region_optimal_design'):.4f}",
            "fit_structure.json",
        ),
        (
            "thmSolverExcess",
            f"{100 * fit.number(theorem + '.project_solver_excess_point_variance'):.1f}",
            "fit_structure.json",
        ),
    ]


def build() -> str:
    theory_v2 = Source.load("theory_v2/theory_v2.json")
    theory = Source.load("theory/theory_check.json")
    adversarial = Source.load("adversarial/a1_independent_rederivation.json")
    dose = Source.load("adversarial/a2_dose_response.json")
    resolution = Source.load("resolution/resolution_study.json")
    abstention = Source.load("abstention/abstention_1B.json")
    allocation = Source.load("allocation/allocation_1B.json")
    curvature = Source.load("curvature/curvature_gate.json")
    scoring = Source.load("target_scoring/target_scoring.json")
    known = Source.load("known_answer/datadecide_known_answer.json")
    robust = Source.load("target_scoring/robustness.json")
    expected = Source.load("target_scoring/expected_error.json")
    c4_primary = "regimes.c4_en_bits_per_token/primary_4M-300M_gate530M"
    c4_sig = c4_primary + ".significance"
    c4_pair_se = c4_sig + ".pair_resampling_naive.standard_error"

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
        # --- Uncertainty-aware decision accuracy (the new headline) ---
        (
            "determinedFracBonf",
            f"{100 * scoring.number('by_multiplicity.bonferroni.determined_fraction'):.0f}",
            "target_scoring.json",
        ),
        (
            "determinedFracBH",
            f"{100 * scoring.number('by_multiplicity.bh.determined_fraction'):.0f}",
            "target_scoring.json",
        ),
        (
            "excessObserved",
            f"{scoring.number('by_multiplicity.bonferroni.excess_observed_pp'):.2f}",
            "target_scoring.json",
        ),
        (
            "excessDetermined",
            f"{scoring.number('by_multiplicity.bonferroni.excess_determined_pp'):.2f}",
            "target_scoring.json",
        ),
        (
            "excessExpected",
            f"{scoring.number('by_multiplicity.bonferroni.excess_expected_pp'):.2f}",
            "target_scoring.json",
        ),
        (
            "welchDfMedian",
            f"{scoring.number('by_multiplicity.bonferroni.welch_df.median'):.2f}",
            "target_scoring.json",
        ),
        (
            "welchDfMin",
            f"{scoring.number('by_multiplicity.bonferroni.welch_df.min'):.2f}",
            "target_scoring.json",
        ),
        (
            "welchDfMax",
            f"{scoring.number('by_multiplicity.bonferroni.welch_df.max'):.2f}",
            "target_scoring.json",
        ),
        (
            "tRatioAssumed",
            f"{scoring.number('student_t_recomputed.ratio_at_assumed_df'):.2f}",
            "target_scoring.json (superseded: assumed 4 df)",
        ),
        (
            "tRatioMedian",
            f"{scoring.number('student_t_recomputed.ratio_at_measured_df.median'):.2f}",
            "target_scoring.json",
        ),
        (
            "tRatioPTen",
            f"{scoring.number('student_t_recomputed.ratio_at_measured_df.p10'):.2f}",
            "target_scoring.json",
        ),
        (
            "tRatioPNinety",
            f"{scoring.number('student_t_recomputed.ratio_at_measured_df.p90'):.2f}",
            "target_scoring.json",
        ),
        (
            "knownAnswerPerSeed",
            f"{known.number('checks.per_seed.computed'):.4f}",
            "datadecide_known_answer.json",
        ),
        (
            "knownAnswerSeedMean",
            f"{known.number('checks.seed_mean.computed'):.2f}",
            "datadecide_known_answer.json",
        ),
        (
            "knownAnswerPublished",
            f"{known.number('published.value'):.2f}",
            "datadecide_known_answer.json",
        ),
        (
            "focalScale",
            known.text("focal_scale"),
            "datadecide_known_answer.json",
        ),
        (
            "targetScaleLabel",
            known.text("target_label"),
            "datadecide_known_answer.json",
        ),
        (
            "maxDfAtThreeSeeds",
            f"{scoring.number('student_t_recomputed.assumed_df'):.0f}",
            "target_scoring.json",
        ),
        # --- Expected-error significance (the C4 headline) ---
        (
            "expectedGap",
            f"{expected.number(c4_primary + '.significance.point_difference_pp'):.2f}",
            "expected_error.json",
        ),
        (
            "expectedCandLow",
            f"{expected.number(c4_primary + '.significance.u_statistic.ci_low'):+.2f}",
            "expected_error.json",
        ),
        (
            "expectedCandHigh",
            f"{expected.number(c4_primary + '.significance.u_statistic.ci_high'):+.2f}",
            "expected_error.json",
        ),
        (
            "expectedCandP",
            f"{expected.number(c4_primary + '.significance.u_statistic.p_two_sided'):.2f}",
            "expected_error.json",
        ),
        (
            "expectedPairLow",
            f"{expected.number(c4_primary + '.significance.pair_resampling_naive.ci_low_pp'):+.2f}",
            "expected_error.json",
        ),
        (
            "expectedPairHigh",
            f"{expected.number(c4_primary + '.significance.pair_resampling_naive.ci_high_pp'):+.2f}",
            "expected_error.json",
        ),
        (
            "expectedPairP",
            f"{expected.number(c4_primary + '.significance.pair_resampling_naive.p_two_sided'):.3f}",
            "expected_error.json",
        ),
        (
            "dependenceFactor",
            f"{expected.number(c4_sig + '.u_statistic.standard_error') / expected.number(c4_pair_se):.2f}",
            "expected_error.json",
        ),
        # --- Published figure: two estimators that bracket it ---
        (
            "pubObserved",
            f"{expected.number('published_figure.observed_accuracy'):.3f}",
            "expected_error.json",
        ),
        (
            "pubPosterior",
            f"{expected.number('published_figure.posterior_expected_accuracy'):.3f}",
            "expected_error.json",
        ),
        (
            "pubDisattenuated",
            f"{expected.number('published_figure.disattenuated_accuracy'):.3f}",
            "expected_error.json",
        ),
        (
            "pubReliability",
            f"{expected.number('published_figure.mean_reference_reliability'):.3f}",
            "expected_error.json",
        ),
        (
            "pubDetFrac",
            f"{100 * expected.number('published_figure.determined_fraction_bonferroni'):.0f}",
            "expected_error.json",
        ),
        (
            "relSmallGaps",
            f"{expected.number('published_figure.by_true_gap_stratum.0.mean_reference_reliability'):.3f}",
            "expected_error.json",
        ),
        (
            "relLargeGaps",
            f"{expected.number('published_figure.by_true_gap_stratum.2.mean_reference_reliability'):.3f}",
            "expected_error.json",
        ),
        (
            "doseRhoObserved",
            f"{expected.number('dose_response.observed_excess_pp.spearman_vs_log_lever_arm'):+.3f}",
            "expected_error.json",
        ),
        (
            "doseRhoExpected",
            f"{expected.number('dose_response.expected_excess_pp.spearman_vs_log_lever_arm'):+.3f}",
            "expected_error.json",
        ),
        (
            "doseDesigns",
            f"{expected.integer('dose_response.n_designs')}",
            "expected_error.json",
        ),
        # --- Provenance ---
        (
            "shippedProjFlips",
            f"{scoring.integer('reproduction_check.committed_first_audit.projection_flips')}",
            "target_scoring.json (superseded; does not reproduce)",
        ),
        (
            "recomputedProjFlips",
            f"{scoring.integer('reproduction_check.recomputed_with_audit_detectors.projection_flips')}",
            "target_scoring.json",
        ),
        (
            "recomputedSingleFlips",
            f"{scoring.integer('reproduction_check.recomputed_with_audit_detectors.single_scale_flips')}",
            "target_scoring.json",
        ),
        # --- Equivalence bound and the published figure ---
        (
            "equivBoundPP",
            f"{robust.number('cells.c4_en_bits_per_token/primary_4M-300M_gate530M.equivalence_bonferroni.largest_difference_ruled_out_pp'):.2f}",
            "robustness.json",
        ),
        (
            "publishedOverall",
            f"{robust.number('published_figure.bonferroni.accuracy_overall'):.3f}",
            "robustness.json",
        ),
        (
            "publishedDeterminedBonf",
            f"{robust.number('published_figure.bonferroni.accuracy_on_determined'):.3f}",
            "robustness.json",
        ),
        (
            "publishedShiftBonf",
            f"{robust.number('published_figure.bonferroni.shift_pp'):.1f}",
            "robustness.json",
        ),
        (
            "publishedDeterminedBH",
            f"{robust.number('published_figure.bh.accuracy_on_determined'):.3f}",
            "robustness.json",
        ),
        (
            "publishedShiftBH",
            f"{robust.number('published_figure.bh.shift_pp'):.1f}",
            "robustness.json",
        ),
        (
            "publishedDetFracBonf",
            f"{100 * robust.number('published_figure.bonferroni.determined_fraction'):.0f}",
            "robustness.json",
        ),
        (
            "publishedDetFracBH",
            f"{100 * robust.number('published_figure.bh.determined_fraction'):.0f}",
            "robustness.json",
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
    entries.extend(_power_entries(expected, c4_primary + ".significance"))
    entries.extend(_rescoring_entries())
    entries.extend(_published_entries())
    entries.extend(_fit_structure_entries())
    entries.extend(_noise_model_entries())
    entries.extend(_calibration_entries())
    entries.extend(_restated_entries())
    entries.extend(_dose_jackknife_entries())
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
