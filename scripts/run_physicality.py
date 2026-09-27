"""Task 1 of PREDICTIONS_TASK_TRANSFER.md: are the extended-grid truths physical?

Scans lambda from 0 to 8 in steps of 0.01 along C4's map axis (mu_P + common +
lambda x specific), C4's ensemble axis (mu_P + lambda x R) and OLMES's map
axis, and reports where strict monotonicity, tolerant monotonicity (no rise
larger than one per-run seed sd) and the valid range first fail. The map of
``regime_map.json`` is then restricted to the physical region and the
extended-grid findings are re-read on it.

Writes ``results/target_scoring/physicality.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import transfer_worlds as tw  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "physicality.json"
GRID = np.round(np.arange(0.0, 8.0 + 1e-9, 0.01), 2)
CONDITIONS = ("strict_monotone", "tolerant_monotone", "in_range")


def axis_truth(metric: str, axis: str, lam: float):  # noqa: ANN201
    ctx = tw.context(metric)
    if axis == "map":
        return ctx["parametric"] + ctx["common_table"] + lam * ctx["specific_table"]
    return ctx["parametric"] + lam * (ctx["common_table"] + ctx["specific_table"])


def first_failure(metric: str, axis: str) -> dict[str, Any]:
    ctx = tw.context(metric)
    first: dict[str, float | None] = {c: None for c in CONDITIONS}
    for lam in GRID:
        checks = tw.physical(axis_truth(metric, axis, float(lam)), ctx["sigma_table"], metric)
        for c in CONDITIONS:
            if first[c] is None and not checks[c]:
                first[c] = float(lam)
    operative = [v for v in (first["tolerant_monotone"], first["in_range"]) if v is not None]
    return {
        "first_failure": first,
        "operative_first_failure": min(operative) if operative else None,
        "at_lambda_1": tw.physical(axis_truth(metric, axis, 1.0), ctx["sigma_table"], metric),
    }


def main() -> None:
    axes = {
        "c4_map": first_failure("c4", "map"),
        "c4_ensemble": first_failure("c4", "ensemble"),
        "olmes_map": first_failure("olmes", "map"),
    }
    own = {m: tw.physical(tw.context(m)["mu"], tw.context(m)["sigma_table"], m) for m in tw.METRICS}
    regime = json.loads((REPO / "results" / "target_scoring" / "regime_map.json").read_text())
    limit = axes["c4_map"]["operative_first_failure"]
    limit = float("inf") if limit is None else limit
    cells = regime["task_1b"]["cells"]
    physical_lams = [lam for lam in regime["grid"]["lambda"] if lam < limit]
    short_wins = [key for key, c in cells.items() if c["short"]["sign"] == "wins"]
    candidates = regime["task_3"]["at_measured_noise_by_lambda"]
    olmes_lambda = regime["task_1c"]["olmes"]["map_position"][1]
    restricted = {
        "c4_map_physical_below_lambda": limit,
        "physical_grid_lambdas": physical_lams,
        "short_design_winning_cells": short_wins,
        "short_design_winning_cells_physical": [k for k in short_wins if cells[k]["lambda"] < limit],
        "candidates_by_lambda_physical": {k: v["candidates_for_2pp"] for k, v in candidates.items() if float(k) < limit},
        "candidates_fall_cells_physical": [k for k in ("4", "8") if float(k) < limit],
        "olmes_lambda_on_c4_axis": olmes_lambda,
        "olmes_inside_c4_physical_region": olmes_lambda < limit,
        "primary_winning_cells": [
            key
            for key, c in cells.items()
            if c["primary"]["sign"] == "wins" or (c["s"] == 0 and c["primary"]["excess_pp"] < 0)
        ],
        "long_winning_cells": [key for key, c in cells.items() if c["long"]["sign"] == "wins"],
    }
    out = {
        "grid": {"from": 0.0, "to": 8.0, "step": 0.01},
        "tolerance": "a rise between adjacent rungs larger than the recipe's per-run seed sd at the higher rung",
        "axes": axes,
        "own_truth": own,
        "restricted_map": restricted,
    }
    out["predictions"] = {
        "P1_1_c4_fails_below_4_and_olmes_outside": limit < 4 and not restricted["olmes_inside_c4_physical_region"],
        "P1_2_c4_strict_olmes_tolerant_only": own["c4"]["strict_monotone"]
        and not own["olmes"]["strict_monotone"]
        and own["olmes"]["tolerant_monotone"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v["first_failure"] | {"operative": v["operative_first_failure"]} for k, v in axes.items()}, indent=1)
    )
    print(json.dumps(restricted, indent=1))
    print(out["predictions"])


if __name__ == "__main__":
    main()
