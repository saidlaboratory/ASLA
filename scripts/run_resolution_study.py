"""Task 4: apply the resolution diagnostic to every public leaderboard we hold.

Fantastic Optimizers at each (model size, Chinchilla ratio) cell, plus the
DataDecide recipe leaderboard at each scale as a contrast where sigma is
measured rather than transferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.resolution import resolution_report, sensitivity  # noqa: E402
from asla.cli import _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
# DataDecide's measured 1B within-cell seed sd, converted to nats for the
# optimizer grid (whose losses are nats/token). This is a CROSS-STUDY TRANSFER.
DATADECIDE_1B_SD_BITS = 0.0020648
SIGMA_NATS = DATADECIDE_1B_SD_BITS * float(np.log(2))
TRANSFER_NOTE = (
    "DataDecide's measured within-cell seed sd at 1B (0.00206 bits/token) converted to nats (x ln 2). "
    "Fantastic Optimizers reports one run per cell, so its own noise cannot be measured; this is a "
    "cross-study transfer and an assumption, not a measurement. Sensitivity rows span x0.5 to x2."
)


def optimizer_leaderboards() -> list[dict[str, Any]]:
    rows = []
    cells = pd.read_csv(DATA / "fantastic_optimizers_cells.csv")
    cells = cells.dropna(subset=["loss_min"])
    for (size, ratio), group in cells.groupby(["size", "chinchilla"]):
        entries = dict(zip(group["optimizer"].astype(str), group["loss_min"].astype(float)))
        if len(entries) < 2:
            continue
        report = resolution_report(entries, sigma=SIGMA_NATS, seed_budget=3, top_k=3)
        report["sensitivity"] = sensitivity(entries, sigma=SIGMA_NATS, top_k=3)
        rows.append(
            {
                "leaderboard": f"fantastic_optimizers/{size}/{ratio}xC",
                "suite": "fantastic_optimizers",
                "size": str(size),
                "chinchilla_ratio": int(ratio),
                "sigma_source": "transferred",
                "sigma_note": TRANSFER_NOTE,
                **report,
            }
        )
    return rows


def datadecide_leaderboards() -> list[dict[str, Any]]:
    """Contrast case: sigma is measured from the suite's own seeds."""

    df = load_runs(DATA / "datadecide_runs.parquet")
    rows = []
    for scale, group in df.groupby("scale_label"):
        means = group.groupby("intervention")["bpb"].mean()
        sds = group.groupby("intervention")["bpb"].std(ddof=1)
        sigma = float(np.sqrt((sds**2).mean()))
        if not np.isfinite(sigma) or sigma <= 0 or len(means) < 2:
            continue
        report = resolution_report(dict(means.astype(float)), sigma=sigma, seed_budget=3, top_k=3)
        report["sensitivity"] = sensitivity(dict(means.astype(float)), sigma=sigma, top_k=3)
        rows.append(
            {
                "leaderboard": f"datadecide/{scale}",
                "suite": "datadecide",
                "size": str(scale),
                "sigma_source": "measured",
                "sigma_note": "measured within-cell seed sd at this scale, from the suite's own 3 seeds",
                **report,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/resolution")
    args = parser.parse_args(argv)
    boards = optimizer_leaderboards() + datadecide_leaderboards()
    summary = {
        "n_leaderboards": len(boards),
        "total_adjacent_pairs": int(sum(b["n_adjacent_pairs"] for b in boards)),
        "total_unresolved_at_3_seeds": int(sum(b["n_unresolved_at_budget"] for b in boards)),
        "total_unidentifiable_at_any_budget": int(sum(b["n_unidentifiable_at_any_budget"] for b in boards)),
        "boards_with_unresolved_top3": [b["leaderboard"] for b in boards if not b["top_k_fully_resolved"]],
        "by_suite": {},
    }
    for suite in sorted({b["suite"] for b in boards}):
        sub = [b for b in boards if b["suite"] == suite]
        pairs = sum(b["n_adjacent_pairs"] for b in sub)
        unresolved = sum(b["n_unresolved_at_budget"] for b in sub)
        summary["by_suite"][suite] = {
            "n_leaderboards": len(sub),
            "adjacent_pairs": pairs,
            "unresolved_at_3_seeds": unresolved,
            "fraction_unresolved": unresolved / pairs if pairs else None,
            "unidentifiable_at_any_budget": sum(b["n_unidentifiable_at_any_budget"] for b in sub),
            "top3_unresolved_boards": sum(1 for b in sub if not b["top_k_fully_resolved"]),
            "sigma_source": sub[0]["sigma_source"],
        }
    results = {"summary": summary, "leaderboards": boards}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(results, out / "resolution_study.json")
    print(json.dumps(summary, indent=2, default=str)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
