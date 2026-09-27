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
    out = {
        "label": "post hoc and exploratory: suggested by the swap data after P2 was refuted; not a pre-registered test",
        "design": "primary 300M -> 530M; 22 cells (4 own worlds, 18 swap configurations)",
        "cells": cells,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for k, c in cells.items():
        print(
            f"{k:22s} pers {c['persistence']:+.2f} trans {c['transient_size_in_gaps']:.2f} "
            f"top {c['top_rung_size_in_gaps']:.2f} excess {c['excess_pp']:+.2f} {c['sign']}"
        )


if __name__ == "__main__":
    main()
