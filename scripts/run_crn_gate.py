"""The gate of PREDICTIONS_TASK_CRN.md: apply the pre-registered decision rule to Tasks 0-2.

Rule (fixed before measurement): recommend the controlled experiment if the
Task 1 point estimate of the cross-recipe seed correlation r is at least 0.2;
if Task 1 is unavailable, if rho_D is at least 0.2; do not recommend if every
available point estimate is below 0.2.

Writes ``results/crn/gate.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "crn"
THRESHOLD = 0.2


def decide(r_point: float | None, rho_d_point: float | None) -> str:
    if r_point is not None and r_point >= THRESHOLD:
        return "recommend"
    if r_point is None:
        if rho_d_point is not None and rho_d_point >= THRESHOLD:
            return "recommend"
        return "do not recommend"
    available = [v for v in (r_point, rho_d_point) if v is not None]
    if all(v < THRESHOLD for v in available):
        return "do not recommend"
    return "not decided by the rule"


def main() -> None:
    novelty = json.loads((RESULTS / "novelty.json").read_text())
    task1 = json.loads((RESULTS / "compute_multipliers_crn.json").read_text())
    task2 = json.loads((RESULTS / "polypythias_rho_d.json").read_text())
    icc = task1["metrics"][task1["primary_metric"]]["icc"]
    primary2 = task2["primary"]
    out: dict[str, Any] = {
        "threshold_gap_variance_reduction": THRESHOLD,
        "task_0_novelty": novelty["verdict"],
        "task_1": {
            "design_status": task1["protocol"]["verdict"],
            "metric": task1["primary_metric"],
            "r": icc["r"],
            "r_ci": icc["r_ci"],
            "reduction_factor": icc["reduction_factor"],
            "secondary_r": {m: b["icc"]["r"] for m, b in task1["metrics"].items() if m != task1["primary_metric"]},
            "note": "three seeds per recipe; the interval is wide",
        },
        "task_2": {
            "rho_d": primary2["rho_d"],
            "rho_d_ci": primary2["rho_d_ci"],
            "note": "three runs per arm, F(2, 2); an upper bound on data-order-only CRN",
        },
        "decision": decide(icc["r"], primary2["rho_d"]),
        "not_tested_by_tasks_1_2": (
            "cross-recipe correlation when recipes share the architecture and the initialisation is copied "
            "explicitly (the regime Task 3 would use); Task 2's point estimate puts most seed variance in "
            "initialisation, which only such a design could share"
        ),
    }
    (RESULTS / "gate.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("decision",)}), out["task_1"]["r"], out["task_2"]["rho_d"])


if __name__ == "__main__":
    main()
