"""Post hoc, exploratory (not pre-registered): what the swap data suggests after the alignment sweep refuted P2.

The alignment of the recipe-specific residual with the target ordering did not
flip C4's sign at any design, and OLMES's residual flips C4 while opposing its
gaps. A candidate the swaps point to is *transient* misspecification: a
recipe-specific deviation at the top fitted rung that does not persist to the
target, which misleads single-scale ranking while a multi-rung fit averages it
out. For every primary-design cell (4 metrics' own worlds and 18 swap
configurations) this reports

* persistence: Pearson correlation of the recipe-specific residual at the top
  fitted rung with that at the target;
* transient size: RMS over recipes of (residual at f - residual at t), in units
  of the median true target gap;

against projection's true excess. No simulation; read from committed results.

Writes ``results/target_scoring/pattern_posthoc.json``.
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

from scripts import transfer_worlds as tw  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "pattern_posthoc.json"
F, T = tw.PRIMARY


def quantities(specific_f: np.ndarray, specific_t: np.ndarray, gap: float) -> dict[str, float]:
    return {
        "persistence": float(np.corrcoef(specific_f, specific_t)[0, 1]),
        "transient_size_in_gaps": float(np.sqrt(np.mean((specific_f - specific_t) ** 2)) / gap),
        "top_rung_size_in_gaps": float(np.sqrt(np.mean(specific_f**2)) / gap),
    }


def main() -> None:
    transfer = json.loads((REPO / "results" / "target_scoring" / "dimensionless_transfer.json").read_text())
    swap_json = json.loads((REPO / "results" / "target_scoring" / "structure_swap.json").read_text())
    swap = importlib.import_module("scripts.run_structure_swap")
    cells: dict[str, Any] = {}
    for m in tw.METRICS:
        ctx = tw.context(m)
        spec = ctx["specific_table"]
        q = quantities(spec[F].to_numpy(), spec[T].to_numpy(), tw.median_gap(ctx["mu"][T]))
        truth = transfer["ground_truth"][m]["primary"]
        cells[f"own:{m}"] = {**q, "excess_pp": truth["excess_pp"], "sign": truth["sign"]}
    for name, (base, sources) in swap.configurations().items():
        truth_table, _ = tw.hybrid(base, sources)
        src = sources.get("iv_specific", sources.get("iv", base))
        s_ctx, b = tw.context(src), tw.context(base)
        spec = (s_ctx["specific_table"] / s_ctx["g"] * b["g"]).loc[b["recipes"]]
        q = quantities(spec[F].to_numpy(), spec[T].to_numpy(), tw.median_gap(truth_table[T]))
        row = swap_json["configurations"][name]
        cells[f"swap:{name}"] = {**q, "excess_pp": row["excess_pp"], "sign": row["sign"].replace("projection ", "")}
    excess = np.array([c["excess_pp"] for c in cells.values()])
    summary = {
        key: {
            "spearman_with_excess": float(stats.spearmanr([c[key] for c in cells.values()], excess).statistic),
            "mean_where_projection_wins": float(np.mean([c[key] for c in cells.values() if c["sign"] == "wins"])),
            "mean_where_projection_loses": float(np.mean([c[key] for c in cells.values() if c["sign"] == "loses"])),
        }
        for key in ("persistence", "transient_size_in_gaps", "top_rung_size_in_gaps")
    }
    pattern = json.loads((REPO / "results" / "target_scoring" / "pattern.json").read_text())
    wins_sizes = [c["top_rung_size_in_gaps"] for c in cells.values() if c["sign"] == "wins"]
    lose_sizes = [c["top_rung_size_in_gaps"] for c in cells.values() if c["sign"] == "loses"]
    counterexamples = {
        "loses_despite_size_at_or_above_smallest_winning_size": sorted(
            k for k, c in cells.items() if c["sign"] == "loses" and c["top_rung_size_in_gaps"] >= min(wins_sizes)
        ),
        "wins_despite_size_at_or_below_largest_losing_size": sorted(
            k for k, c in cells.items() if c["sign"] == "wins" and c["top_rung_size_in_gaps"] <= max(lose_sizes)
        ),
        "distinct_residual_values_among_cells": len({round(c["top_rung_size_in_gaps"], 6) for c in cells.values()}),
    }
    out = {
        "label": "post hoc and exploratory: suggested by the swap data after P2 was refuted; not a pre-registered test",
        "status": (
            "EXPLORATORY, NOT A FINDING. The top-rung-magnitude statistic tracks the excess across these cells, but "
            "the cells carry only a few distinct residuals, the counterexamples below contradict it, and it is one "
            "of the statistics listed under statistics_tried, chosen after seeing the data. The mechanism search is "
            "closed; no further pattern statistics are run."
        ),
        "counterexamples": counterexamples,
        "statistics_tried": {
            "this_round (PREDICTIONS_TASK_PATTERN.md)": [
                "a: alignment of target residual with true target ordering",
                "b: alignment with top-fitted-rung ordering (truth)",
                "b: alignment with top-fitted-rung ordering (observed)",
                "c: in-sample residual proxy (truth)",
                "c: in-sample residual proxy (observed)",
                "crossover rate between f and t",
                "persistence of the residual from f to t (post hoc)",
                "transient residual size in target gaps (post hoc)",
                "top-rung residual size in target gaps (post hoc)",
            ],
            "previous_round (PREDICTIONS_TASK_TRANSFER.md)": [
                "q1: top-rung cell-mean noise / median target gap",
                "q2: recipe-specific misspecification / median target gap",
                "q3: smallest-rung / top-rung noise ratio",
                "q4: lever arm",
                "q5: floor level of the mean curve",
                "q6: decay of the mean curve",
            ],
            "count_this_and_previous_round": 15,
            "also_two_rounds_earlier (regime map coordinates)": [
                "seed noise / between-recipe spread",
                "loading sd x shape rms / spread",
                "loading CV x residual RMS / spread",
                "recipe-specific residual RMS / spread",
            ],
            "count_including_two_rounds_earlier": 19,
        },
        "task_2_construction_deviation": {
            "registered": "vary the rank-one recipe-specific loadings b with the remainder E fixed, sd(b) at measured",
            "why_changed": (
                "made after the registered construction failed to span the alignment range: with E fixed it reaches "
                "only the reported Pearson ranges, because the remainder dominates the recipe-specific residual at "
                "the target"
            ),
            "registered_construction_reach": pattern["task_2"]["preregistered_construction_reach"],
            "used_instead": (
                "reassign the measured recipe-specific residual rows among recipes by a score correlated c with the "
                "target values; amplitude held exactly"
            ),
        },
        "design": "primary 300M -> 530M; 22 cells (4 own worlds, 18 swap configurations)",
        "cells": cells,
        "summary": summary,
    }
    assert (
        len(out["statistics_tried"]["this_round (PREDICTIONS_TASK_PATTERN.md)"])
        + len(out["statistics_tried"]["previous_round (PREDICTIONS_TASK_TRANSFER.md)"])
        == out["statistics_tried"]["count_this_and_previous_round"]
    )
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for k, c in cells.items():
        print(
            f"{k:22s} pers {c['persistence']:+.2f} trans {c['transient_size_in_gaps']:.2f} "
            f"top {c['top_rung_size_in_gaps']:.2f} excess {c['excess_pp']:+.2f} {c['sign']}"
        )


if __name__ == "__main__":
    main()
