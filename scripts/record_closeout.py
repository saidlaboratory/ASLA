"""Close-out record for the mechanism search and correct-prob's long design (no new statistics).

Every number is read from committed results:

* the mechanism statement, from ``pattern.json`` (alignment sweep, measurability)
  and ``structure_swap.json`` (the residual swaps);
* correct-prob's long design as large-scale crossover, from ``pattern.json``.

Writes ``results/target_scoring/mechanism_record.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "target_scoring"
OUT = RESULTS / "mechanism_record.json"


def main() -> None:
    pattern = json.loads((RESULTS / "pattern.json").read_text())
    robust = json.loads((RESULTS / "mechanism_robustness.json").read_text())
    swap = json.loads((RESULTS / "structure_swap.json").read_text())["configurations"]
    sweep = pattern["task_2"]["sweep"]
    points = [(d, c, p) for d, g in sweep.items() for c, p in g["points"].items()]
    physical = [p for _, _, p in points if p["physical"]]
    task3 = pattern["task_3"]
    t4 = pattern["task_4"]
    record = {
        "mechanism": {
            "statement": (
                "Reassigning C4's recipe-specific residuals among recipes cannot flip C4 at any alignment or design; "
                "OLMES's residuals can. The driver is the shape of the recipe-specific residuals, which no tested "
                "summary captures and which cannot be measured without the target."
            ),
            "primary_evidence": {
                "noiseless_counts_on_real_curves": {
                    "projection_wrong_pairs": robust["noiseless"]["projection_wrong_total"],
                    "single_scale_wrong_pairs": robust["noiseless"]["single_wrong_total"],
                    "designs": len(robust["noiseless"]["designs"]),
                    "designs_where_projection_beats_single": robust["noiseless"]["designs_where_projection_beats_single"],
                    "source": "mechanism_robustness.json: fixed-candidate real shrunk C4 curves, no noise",
                },
                "reading": (
                    "with the noise removed, projection is wrong on more pairs than single-scale at every one of "
                    "the 55 designs; this is the primary mechanism evidence"
                ),
            },
            "attribution_depends_on_base_form": {
                "power_law_base": robust["variants"]["baseline"]["attribution"],
                "chinchilla_base": robust["variants"]["chinchilla"]["attribution"],
                "reading": (
                    "the variance / misspecification split of the lever-arm slope depends on which functional form "
                    "counts as correctly specified; a Chinchilla-form truth is itself misspecified relative to the "
                    "compute power law projection fits, so its arm A no longer isolates variance"
                ),
            },
            "support": {
                "c4_reassignment_sweep": {
                    "points": len(points),
                    "physical_points": len(physical),
                    "points_where_projection_wins": sum(p["sign"] == "wins" for _, _, p in points),
                    "smallest_excess_pp": min(p["excess_pp"] for _, _, p in points),
                    "sign_changes": {d: g["sign_changes_at_achieved_alignment"] for d, g in sweep.items()},
                },
                "olmes_residuals_in_c4": {
                    "c4_with_olmes_residual_pp": swap["c4+iv"]["excess_pp"],
                    "c4_with_olmes_residual_ci": swap["c4+iv"]["ci"],
                    "c4_with_olmes_recipe_specific_residual_pp": swap["c4+iv_specific"]["excess_pp"],
                    "c4_with_olmes_recipe_specific_residual_ci": swap["c4+iv_specific"]["ci"],
                    "c4_baseline_pp": swap["c4"]["excess_pp"],
                },
                "no_tested_summary_captures_it": (
                    "15 summaries in the last two rounds (19 over three); see pattern_posthoc.json statistics_tried"
                ),
                "not_measurable_without_target": {
                    "target_free_proxy_sign_agreement": f"{task3['c_observed_sign_agrees']} of 12",
                    "reason": "the residual at the target comes from a fit that includes the target",
                },
            },
        },
        "correct_prob_long_design": {
            "statement": (
                "Large-scale crossover that extrapolation does not recover, not an extrapolation win: the 8M ordering "
                "reverses at 1B, so single-scale ranking is worse than chance even without noise, and projection is "
                "only slightly better than chance."
            ),
            "design": t4["design"],
            "noiseless_single_scale_error": t4["noiseless_single_scale_rate"],
            "single_scale_error": t4["single_scale_rate"],
            "projection_error": t4["projection_rate"],
            "chance": t4["chance_rate"],
            "projection_better_than_chance_pp": 100 * (t4["chance_rate"] - t4["projection_rate"]),
        },
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["mechanism"]["support"]["c4_reassignment_sweep"], indent=1))
    print(record["correct_prob_long_design"]["projection_better_than_chance_pp"])


if __name__ == "__main__":
    main()
