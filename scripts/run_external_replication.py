"""Task 3b/3c: does the H1 decomposition replicate outside the DataDecide evaluation?

The headline test: is the excess mis-selection of projection over single-scale
ranking still entirely fit error, on a table our audit has never been run on?

Suites tested and what each does and does not establish:

* **signal_and_noise** - Paloma C4-EN *bits per byte* for the DataDecide models,
  produced by a different evaluation pipeline (Heineman et al.). This replicates
  the **metric and evaluation axis**: same underlying models, independently
  measured, different units. It is NOT an independent model suite, and saying
  otherwise would overstate it. One evaluated run per cell, so no seed-noise
  band and no crossover significance tests: flip counts only.
* **datadecide reference** - the committed primary result, recomputed here so the
  comparison is like-for-like rather than quoted.

ColPret was assessed and found structurally unusable (see
`scripts/assess_colpret.py` and `results/external/colpret_assessment.json`).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.fits import project_ranking, truth_ranking  # noqa: E402
from asla.analysis.metrics import decision_metrics  # noqa: E402
from asla.analysis.rankers import single_scale_ranker  # noqa: E402
from asla.cli import _sha256_file, _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUITES = {
    "signal_and_noise": REPO / "data" / "signal_and_noise_datadecide_c4_bpb.parquet",
    "datadecide_reference": REPO / "data" / "datadecide_runs.parquet",
}


def flips(scores: pd.Series, truth: pd.Series) -> set[tuple[str, str]]:
    names = sorted(set(scores.index.astype(str)) & set(truth.index.astype(str)))
    out = set()
    for a, b in itertools.combinations(names, 2):
        pg, tg = float(scores[a] - scores[b]), float(truth[a] - truth[b])
        if pg == 0.0 or tg == 0.0:
            continue
        if np.sign(pg) != np.sign(tg):
            out.add((a, b))
    return out


def decompose(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> dict[str, Any]:
    truth = truth_ranking(df, target)
    projected = project_ranking(df, budgets, target)
    single = single_scale_ranker(df, budgets, target)
    proj_flips = flips(projected, truth)
    single_flips = flips(single, truth)
    inherited = proj_flips & single_flips
    fit_error = proj_flips - single_flips
    repaired = single_flips - proj_flips
    excess = len(proj_flips) - len(single_flips)
    n_pairs = len(truth) * (len(truth) - 1) // 2
    return {
        "n_interventions": int(len(truth)),
        "n_pairs": n_pairs,
        "projection_mis_selection": 1.0 - decision_metrics(projected, truth, k=2)["pairwise_acc"],
        "single_scale_mis_selection": 1.0 - decision_metrics(single, truth, k=2)["pairwise_acc"],
        "projection_flips": len(proj_flips),
        "single_scale_flips": len(single_flips),
        "n_inherited_crossover": len(inherited),
        "n_fit_error": len(fit_error),
        "n_repaired": len(repaired),
        "excess_projection_flips": excess,
        "fit_error_share_of_flips": (len(fit_error) / len(proj_flips)) if proj_flips else None,
        # The headline claim: the excess over single-scale is accounted for by fit error.
        "excess_is_all_fit_error": (None if excess <= 0 else bool(len(fit_error) - len(repaired) >= excess)),
        "fit_error_exceeds_inherited": bool(len(fit_error) > len(inherited)),
        "projection_worse_than_single_scale": bool(len(proj_flips) > len(single_flips)),
    }


def build(name: str, path: Path, max_fit: str, target_label: str) -> dict[str, Any]:
    df = load_runs(path)
    df = df[df["scale_label"] != "750M"].reset_index(drop=True)
    order = [s for s in df.drop_duplicates("scale_label").sort_values("compute")["scale_label"]]
    target = float(df[df["scale_label"] == target_label]["compute"].iloc[0])
    keep = order[: order.index(max_fit) + 1]
    budgets = tuple(sorted(float(b) for b in df[df["scale_label"].isin(keep)]["compute"].unique() if b < target))
    table = df[df["compute"].isin(budgets) | np.isclose(df["compute"], target)]
    seeds = int(df.groupby(["intervention", "compute"])["seed"].nunique().min())
    return {
        "suite": name,
        "path": str(path),
        "sha256": _sha256_file(path),
        "metric": str(df["metric_name"].iloc[0]),
        "scales_available": order,
        "fit_scales": keep,
        "target_scale": target_label,
        "n_fit_budgets": len(budgets),
        "min_seeds_per_cell": seeds,
        "significance_testable": bool(seeds >= 2),
        **decompose(table, budgets, target),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/external")
    args = parser.parse_args(argv)

    designs = [
        build("signal_and_noise", SUITES["signal_and_noise"], "300M", "1B"),
        build("signal_and_noise", SUITES["signal_and_noise"], "150M", "1B"),
        build("datadecide_reference", SUITES["datadecide_reference"], "300M", "1B"),
    ]
    replicated = [d for d in designs if d["suite"] != "datadecide_reference" and d["excess_is_all_fit_error"]]
    testable = [d for d in designs if d["suite"] != "datadecide_reference" and d["excess_is_all_fit_error"] is not None]
    results = {
        "designs": designs,
        "replication": {
            "n_external_designs": len(testable),
            "n_replicating": len(replicated),
            "all_replicate": bool(testable and len(replicated) == len(testable)),
            "headline": ("the excess of projection over single-scale ranking is entirely fit error, not crossover"),
            "scope_note": (
                "Signal-and-Noise re-evaluates the SAME DataDecide models through a different pipeline and "
                "metric (Paloma C4-EN bits per byte). It therefore replicates across the evaluation and metric "
                "axis, not across an independent model suite. No fully independent suite with a shared compute "
                "ladder and seed replicates was found in public data; see colpret_assessment.json."
            ),
            "limitation": (
                "One evaluated run per cell in Signal-and-Noise, so flip counts are exact but no crossover "
                "significance test or seed-bootstrap interval is available on that suite."
            ),
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(results, out / "external_replication.json")
    for d in designs:
        print(
            f"{d['suite']:22s} fit<={d['fit_scales'][-1]:>4s} k={d['n_fit_budgets']:2d} seeds={d['min_seeds_per_cell']} "
            f"| proj {d['projection_mis_selection'] * 100:5.2f}% vs single {d['single_scale_mis_selection'] * 100:5.2f}% "
            f"| flips {d['projection_flips']:3d}/{d['single_scale_flips']:3d} "
            f"| fit_err={d['n_fit_error']:3d} inherited={d['n_inherited_crossover']:3d} "
            f"excess={d['excess_projection_flips']:+3d} -> all_fit_error={d['excess_is_all_fit_error']}"
        )
    rep = results["replication"]
    print(f"\nreplication: {rep['n_replicating']}/{rep['n_external_designs']} external designs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
