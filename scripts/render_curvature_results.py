"""Score the pre-registered curvature predictions C1-C7 from committed JSON.

C4-C7 were never evaluated: the gate rule in ``PREDICTIONS_TASK_CURVATURE.md``
makes them conditional on 4a, and 4a's mechanism was refuted. Recording them as
NOT EVALUATED, with the reason, is the honest alternative to leaving a reader to
wonder whether they were run and quietly dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NOT_EVALUATED = "NOT EVALUATED"


def render(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    adjudication = payload["gate_adjudication"]
    alternatives = payload["alternative_explanations"]
    scope = payload["confound_scope"]
    fits = alternatives["fits"]
    tpp = alternatives["tokens_per_param"]

    c1 = gate["C1_sign_consistency"]
    c2 = gate["C2_curvature_predicts_overshoot"]
    c3 = gate["C3_quantitative_recovery"]

    blocked = f"gate stopped after 4a ({adjudication['verdict']}); 4b-4d were pre-registered as conditional on it"

    lines: list[str] = []
    lines.append(f"# Curvature gate: {payload['target_scale']} target")
    lines.append("")
    lines.append(
        f"Target {payload['target_compute']:.3e} FLOPs, "
        f"{len(payload['fit_budgets'])} fitting rungs, "
        f"{gate['n_recipes']} recipes. Scored against `PREDICTIONS_TASK_CURVATURE.md`."
    )
    lines.append("")
    lines.append("## Pre-registered predictions")
    lines.append("")
    lines.append("| prediction | verdict | evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| C1 (quadratic consistently signed) | **{c1['verdict']}** | "
        f"{c1['n_majority']}/{gate['n_recipes']} share a sign "
        f"(sign test p = {c1['sign_test_p']:.1e}); mean coefficient "
        f"{c1['mean_quadratic_coef']:+.2e} |"
    )
    lines.append(
        f"| C2 (curvature predicts overshoot) | **{c2['verdict']}** | "
        f"Spearman {c2['spearman_rho']:+.3f} (p = {c2['spearman_p']:.1e}) |"
    )
    lines.append(
        f"| C3 (quantitative recovery) | **{c3['verdict']}** | "
        f"observed overshoot {c3['mean_observed_overshoot']:+.4f}, quadratic implies "
        f"{c3['mean_quadratic_implied_overshoot']:+.4f}; "
        f"{c3['fraction_recovered']:.0%} recovered, i.e. the wrong sign |"
    )
    for name in (
        "C4 (gap centring error)",
        "C5 (gap intervals valid and narrower)",
        "C6 (abstention falls)",
        "C7 (loading CV predicts scope)",
    ):
        lines.append(f"| {name} | **{NOT_EVALUATED}** | {blocked} |")
    lines.append("")

    lines.append("## Why the gate stopped despite C1 and C2 passing")
    lines.append("")
    lines.append(
        "C1 and C2 establish that residual curvature is systematic and tracks the "
        "overshoot. C3 refutes it as the *mechanism*: the fitted quadratic "
        "extrapolates to an undershoot where an overshoot is observed. Comparing "
        "rival explanations on how much absolute projection error each removes "
        "settles it."
    )
    lines.append("")
    lines.append("| fit | mean signed error | mean absolute error | overshooting |")
    lines.append("| --- | --- | --- | --- |")
    labels = {
        "power_law": "power law (compute only)",
        "log_log_quadratic": "quadratic in log-log",
        "with_tokens_per_param": "+ log(tokens/param)",
    }
    for key, label in labels.items():
        entry = fits[key]
        lines.append(
            f"| {label} | {entry['mean_signed_error']:+.4f} | "
            f"{entry['mean_absolute_error']:.4f} | "
            f"{entry['n_overshooting']}/{entry['n']} |"
        )
    lines.append("")
    lines.append(
        f"Taking the curvature hypothesis at face value removes "
        f"**{adjudication['curvature_absolute_error_reduction']:.0%}** of the absolute "
        f"error; adding tokens-per-parameter removes "
        f"**{adjudication['tokens_per_param_absolute_error_reduction']:.0%}**."
    )
    lines.append("")

    lines.append("## The actual source: a ladder confound")
    lines.append("")
    lines.append(
        f"DataDecide's tokens-per-parameter is not constant along the ladder. It holds "
        f"near 100 across {tpp['n_rungs_at_nominal_100']} rungs and then drifts over "
        f"the remaining {tpp['n_rungs_drifted']}, reaching **{tpp['target']:.1f}** at "
        f"the target. On the fitting ladder it correlates with log compute at "
        f"**{tpp['corr_log_tpp_with_log_compute']:+.2f}**, so a compute-only fit "
        f"attributes one effect to the other and mis-extrapolates."
    )
    lines.append("")
    lines.append(
        f"The confound is common-mode. The compute-only and tokens-aware projections "
        f"agree at Spearman **{scope['spearman_between_projections']:.4f}** over "
        f"{scope['n_pairs']} pairs, with mis-selection "
        f"{scope['mis_selection_compute_only']:.2%} against "
        f"{scope['mis_selection_with_tokens_per_param']:.2%}. It shifts the projected "
        f"level without reordering recipes, so selection results are unaffected and "
        f"the coverage and centring results are what it explains."
    )
    lines.append("")
    lines.append(
        "The log-log quadratic coefficient is positive in "
        f"{alternatives['log_log_quadratic_coefficient']['n_positive']} of "
        f"{alternatives['log_log_quadratic_coefficient']['n']} recipes, so the curves "
        "genuinely are convex. Convexity is simply not what drives the error."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/curvature/curvature_gate.json"))
    parser.add_argument("--out", type=Path, default=Path("results/curvature/RESULTS.md"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.results.read_text())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
