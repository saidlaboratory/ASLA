"""Is the parameter-optimal shrinkage strength also the decision-optimal one?

Empirical-Bayes shrinkage chooses `lambda = within / (within + between)` to
minimise expected squared error in the *parameter* (the fitted exponent). Our
objective is decision accuracy, which depends only on the *ordering* of the
projected target values. These are different losses, so there is no reason the
same lambda should optimise both - and the Task B sweep suggests it does not:
the EB value lands near 0.22 while fixed strengths of 0.6-0.8 rank better.

If that gap is systematic and in a consistent direction, it is a direct
extension of the project's headline reframing (fit quality and decision quality
come apart), not a defect in the estimator. This module measures it across every
available design and reports the direction, size, and consistency.

Method: for each design, read the mis-selection rate at every fixed strength in
the sweep, locate the decision-optimal strength, and compare it with the
empirical-Bayes strength actually estimated for that design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASK_B = REPO / "results" / "task_b" / "task_b.json"


def sweep_curve(design: dict[str, Any]) -> list[dict[str, float]]:
    """Mis-selection at each fixed shrinkage strength, ascending in strength."""

    rows = []
    for name, entry in design["rankers"].items():
        if not name.startswith("shrinkage_") or name == "shrinkage_eb":
            continue
        strength = float(name.removeprefix("shrinkage_"))
        rows.append(
            {
                "strength": strength,
                "mis_selection": float(entry["mis_selection"]["point"]),
                "lo": float(entry["mis_selection"]["lo"]),
                "hi": float(entry["mis_selection"]["hi"]),
            }
        )
    return sorted(rows, key=lambda r: r["strength"])


def analyse_design(name: str, design: dict[str, Any]) -> dict[str, Any]:
    curve = sweep_curve(design)
    if not curve:
        return {"design": name, "available": False}
    best = min(curve, key=lambda r: r["mis_selection"])
    # All strengths whose CI overlaps the best strength's CI: the plateau of
    # strengths the data cannot distinguish from optimal.
    plateau = [r for r in curve if r["lo"] <= best["hi"] and r["hi"] >= best["lo"]]
    vc = design["variance_components"]
    eb = float(vc["eb_strength"])
    eb_entry = design["rankers"].get("shrinkage_eb", {}).get("mis_selection", {})
    return {
        "design": name,
        "available": True,
        "curve": curve,
        "decision_optimal_strength": best["strength"],
        "decision_optimal_mis_selection": best["mis_selection"],
        "plateau_strengths": [r["strength"] for r in plateau],
        "eb_strength": eb,
        "eb_mis_selection": float(eb_entry.get("point")) if eb_entry.get("point") is not None else None,
        "eb_below_decision_optimum": bool(eb < best["strength"]),
        "strength_gap": float(best["strength"] - eb),
        "eb_inside_plateau": bool(any(abs(eb - r["strength"]) <= 0.1 for r in plateau)),
        "cost_of_eb_points": (
            None
            if eb_entry.get("point") is None
            else 100.0 * (float(eb_entry["point"]) - best["mis_selection"])
        ),
        "between_sd": vc["between_sd"],
        "within_sd": vc["within_sd"],
        "noise_share": vc["noise_share"],
    }


def main() -> int:
    payload = json.loads(TASK_B.read_text(encoding="utf-8"))
    rows = [analyse_design(name, design) for name, design in sorted(payload["designs"].items())]
    usable = [r for r in rows if r["available"]]
    directions = [r["eb_below_decision_optimum"] for r in usable]
    gaps = [r["strength_gap"] for r in usable]
    costs = [r["cost_of_eb_points"] for r in usable if r["cost_of_eb_points"] is not None]
    summary = {
        "n_designs": len(usable),
        "n_eb_below_decision_optimum": int(sum(directions)),
        "consistent_direction": bool(all(directions) or not any(directions)),
        "mean_strength_gap": float(np.mean(gaps)) if gaps else None,
        "mean_cost_of_eb_points": float(np.mean(costs)) if costs else None,
        "designs": rows,
        "all_eb_inside_plateau": bool(all(r["eb_inside_plateau"] for r in usable)),
        "plateau_spans_whole_sweep": bool(
            all(len(r["plateau_strengths"]) == len(r["curve"]) for r in usable)
        ),
        "interpretation": (
            "Empirical Bayes minimises squared error in the exponent; decision accuracy depends only on the "
            "ordering of projections. A systematic gap between the parameter-optimal and decision-optimal "
            "shrinkage strength would be the same fit-quality-versus-decision-quality dissociation the project's "
            "headline identifies, appearing inside the estimator we built to exploit it."
        ),
        "statistical_caveat": (
            "The direction is consistent but the evidence is weak: the seed-bootstrap CIs of every strength in the "
            "sweep overlap the optimum's, so no strength is distinguishable from any other at this seed count. The "
            "point estimates suggest EB under-pools for the decision objective; the intervals do not establish it. "
            "Reported as a suggested direction requiring more seeds, not as a finding."
        ),
    }
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pooling_objective_mismatch.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"designs analysed: {summary['n_designs']}")
    print(f"EB below the decision optimum in {summary['n_eb_below_decision_optimum']}/{summary['n_designs']} "
          f"(consistent direction: {summary['consistent_direction']})")
    for row in usable:
        print(
            f"  {row['design']:32s} EB={row['eb_strength']:.3f} decision-optimal={row['decision_optimal_strength']:.1f} "
            f"gap={row['strength_gap']:+.3f} cost={row['cost_of_eb_points']:+.2f} pts "
            f"plateau={row['plateau_strengths']} eb_in_plateau={row['eb_inside_plateau']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
