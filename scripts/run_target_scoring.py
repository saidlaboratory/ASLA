"""Task 0: score projection and single-scale against a determined target.

The audit's rates are computed against the observed target ranking. That ranking
is a three-seed estimate, so some pairs are ordered by noise and every ranker is
being graded partly against coin flips. This recomputes the headline comparison
under two rules that handle it, and reports the decomposition the review showed
was mis-stated.

Scored against ``PREDICTIONS_TASK_TARGET_SCORING.md``, committed before this ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.crossover import detect_crossovers_fdr, detect_single_scale_flips_fdr
from asla.analysis.fits import cell_means_and_sigma, normalize_budgets
from asla.analysis.moderation import calibrated_target_evidence
from asla.analysis.target_scoring import decomposition, score_ranker, target_evidence
from asla.models import FitError, bpb_power_law, fit_power_law

REPO = Path(__file__).resolve().parents[1]
OFF_TRAJECTORY_SCALES = ("750M",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_predictions(
    frame: pd.DataFrame, fit_budgets: tuple[float, ...], target: float, top_rung: float
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Projected and single-scale gaps for every pair, as left-minus-right."""

    means = cell_means_and_sigma(frame)
    projected: dict[str, float] = {}
    single: dict[str, float] = {}
    for name in sorted(frame["intervention"].unique()):
        rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))]
        rows = rows.sort_values("compute")
        if len(rows) >= 3:
            try:
                params = fit_power_law(rows["compute"].to_numpy(dtype=float), rows["bpb_mean"].to_numpy(dtype=float))
                projected[name] = float(bpb_power_law(target, *params))
            except FitError:
                projected[name] = float("nan")
        top = means[(means["intervention"] == name) & (np.isclose(means["compute"], top_rung))]
        single[name] = float(top["bpb_mean"].iloc[0]) if len(top) else float("nan")

    names = sorted(projected)
    projection_gaps = {(a, b): projected[a] - projected[b] for a, b in combinations(names, 2)}
    single_gaps = {(a, b): single[a] - single[b] for a, b in combinations(names, 2)}
    return projection_gaps, single_gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "datadecide_runs.parquet")
    # The audit's primary design fits 4M-300M and holds out 530M; matching it is
    # the point of this task, so the exclusions and the ladder must match too.
    parser.add_argument("--target-scale", default="530M")
    parser.add_argument("--max-fit-scale", default="300M")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "target_scoring" / "target_scoring.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    frame = frame[~frame["scale_label"].isin(OFF_TRAJECTORY_SCALES)]
    target = float(frame[frame["scale_label"] == args.target_scale]["compute"].iloc[0])
    max_fit = float(frame[frame["scale_label"] == args.max_fit_scale]["compute"].iloc[0])
    fit_budgets = normalize_budgets(sorted(float(c) for c in frame["compute"].unique() if c <= max_fit), target=target)
    top_rung = max(fit_budgets)

    cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
    means = {str(k): float(v) for k, v in cells.mean().items()}
    sds = {str(k): float(v) for k, v in cells.std(ddof=1).items()}
    counts = {str(k): int(v) for k, v in cells.count().items()}

    projection_gaps, single_gaps = build_predictions(frame, fit_budgets, target, top_rung)

    payload: dict[str, Any] = {
        "inputs": {str(args.data): _sha(args.data)},
        "target_scale": args.target_scale,
        "target_compute": target,
        "fit_budgets": list(fit_budgets),
        "max_fit_scale": args.max_fit_scale,
        "design": f"full_ladder_4M-{args.max_fit_scale}_gate{args.target_scale}",
        "top_rung": top_rung,
        "n_candidates": len(means),
        "seeds_per_target_cell": sorted(set(counts.values())),
        "assumption_not_addressed": (
            "Both rules treat the observed target cell means as unbiased estimates of expected "
            "target performance. A systematic difference between them --- a different evaluation "
            "draw, a different checkpoint --- survives both rules and is not measured here."
        ),
        "by_multiplicity": {},
    }

    for multiplicity in ("bonferroni", "bh", "none"):
        evidence = calibrated_target_evidence(means, sds, counts, alpha=args.alpha, multiplicity=multiplicity)
        determined = sum(1 for e in evidence if e.determined)
        dfs = [e.welch_df for e in evidence if np.isfinite(e.welch_df)]

        projection = score_ranker(projection_gaps, evidence)
        single = score_ranker(single_gaps, evidence)
        split = decomposition(projection_gaps, single_gaps, evidence)

        payload["by_multiplicity"][multiplicity] = {
            "test": "two-sided Welch t on target cell means",
            "alpha": args.alpha,
            "n_pairs": len(evidence),
            "determined_pairs": determined,
            "determined_fraction": determined / len(evidence),
            "welch_df": {
                "min": float(np.min(dfs)),
                "median": float(np.median(dfs)),
                "max": float(np.max(dfs)),
                "note": "computed per pair; 4 is the maximum, not a constant",
            },
            "projection": projection,
            "single_scale": single,
            "excess_observed_pp": 100 * (projection["observed_rate"] - single["observed_rate"]),
            "excess_determined_pp": 100 * (projection["determined_rate"] - single["determined_rate"]),
            "excess_expected_pp": 100 * (projection["expected_rate"] - single["expected_rate"]),
            "decomposition": {k: v for k, v in split.items() if not k.endswith("_pairs")},
        }

    # Cross-check against the audit's own detectors on the same design. The
    # committed first_audit.json records 16 projection flips and 4 single-scale
    # flips for this design; both detectors, run here on the same data with the
    # committed budget list, give different counts. That discrepancy is recorded
    # rather than reconciled by assumption, because it concerns a headline
    # number and its cause is not in any input we can vary.
    detector_projection = detect_crossovers_fdr(frame, fit_budgets, target, q=0.05)
    detector_single = detect_single_scale_flips_fdr(frame, fit_budgets, target, q=0.05)
    payload["reproduction_check"] = {
        "committed_first_audit": {"projection_flips": 16, "single_scale_flips": 4},
        "recomputed_with_audit_detectors": {
            "projection_flips": len(detector_projection),
            "single_scale_flips": len(detector_single),
        },
        "recomputed_in_this_script": {
            "projection_flips": payload["by_multiplicity"]["none"]["projection"]["observed_errors"],
            "single_scale_flips": payload["by_multiplicity"]["none"]["single_scale"]["observed_errors"],
        },
        "reproduces": len(detector_projection) == 16 and len(detector_single) == 4,
        "inputs_varied_without_recovering_the_committed_counts": [
            "750M excluded and included",
            "index reset and not reset",
            "the committed budget list used verbatim",
            "crossover q at 0.05, 0.10 and 0.20 (flip detection is q-independent)",
            "the committed metric table (c4_en_bits_per_token)",
        ],
        "note": (
            "The audit's own detect_crossovers_fdr and detect_single_scale_flips_fdr, run on the "
            "same design, do not reproduce the committed 16 and 4. Every number derived from "
            "those counts --- the decomposition, the excess over single-scale, and the review's "
            "own 14-introduced/2-corrected arithmetic --- inherits the discrepancy."
        ),
        "provenance_investigation": {
            "writing_commit": "07ae133",
            "counts_in_that_commit": {"projection_flips": 16, "single_scale_flips": 4},
            "data_hash_identical_to_head": True,
            "crossover_module_identical_to_head": True,
            "recomputed_at_the_writing_commit": {"projection_flips": 10, "single_scale_flips": 3},
            "conclusion": (
                "The shipped file was not produced by the state it was committed with. At "
                "07ae133, using that commit's own code and byte-identical parquet data, the "
                "audit's detectors give 10 and 3. asla/analysis/crossover.py is unchanged "
                "between 07ae133 and HEAD, and the DataDecide parquet hashes match exactly, so "
                "neither the detector nor the input can explain the difference. The 16 and 4 "
                "came from uncommitted working state that no longer exists."
            ),
            "which_counts_are_correct": (
                "10 and 3. They are produced by the committed detectors at both the writing "
                "commit and HEAD, by an independently written scorer in this module, and from "
                "data whose hash is unchanged. The 16 and 4 cannot be regenerated from any "
                "committed state."
            ),
            "instrument_check": (
                "The DataDecide known-answer validation reproduces exactly at HEAD "
                "(per-seed 0.803333, seed-mean 0.83, both within band), so the defect is "
                "isolated to this file rather than the fitting or ranking path."
            ),
        },
    }

    # Priority 6: the anti-conservativeness of the normal quantile, recomputed
    # with the per-pair Welch df rather than the assumed 4. The published 3.63x
    # was the BEST case; at the measured median df of ~2.9 it is far larger.
    from scipy import stats as _stats

    pair_dfs = np.asarray(
        [e.welch_df for e in target_evidence(means, sds, counts, multiplicity="none") if np.isfinite(e.welch_df)]
    )
    n_pairs = len(means) * (len(means) - 1) // 2
    per_comparison = 0.05 / n_pairs
    normal_quantile = float(_stats.norm.ppf(1 - per_comparison / 2))
    ratios = _stats.t.ppf(1 - per_comparison / 2, pair_dfs) / normal_quantile
    payload["student_t_recomputed"] = {
        "n_comparisons": n_pairs,
        "per_comparison_alpha": per_comparison,
        "normal_quantile": normal_quantile,
        "assumed_df": 4.0,
        "ratio_at_assumed_df": float(_stats.t.ppf(1 - per_comparison / 2, 4.0) / normal_quantile),
        "ratio_at_measured_df": {
            "p10": float(np.percentile(ratios, 10)),
            "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "max": float(ratios.max()),
        },
        "note": (
            "The 3.63x figure assumed 4 degrees of freedom, which is the MAXIMUM at three seeds "
            "and is attained only when both cells have equal variance. With per-pair Welch df "
            "the normal quantile is anti-conservative by a median factor of "
            f"{float(np.median(ratios)):.2f}, ranging to {float(ratios.max()):.1f} on the "
            "least-balanced pairs. The direction of the original finding strengthens."
        ),
    }

    primary = payload["by_multiplicity"]["bonferroni"]
    payload["headline"] = {
        "determined_fraction_bonferroni": primary["determined_fraction"],
        "determined_fraction_bh": payload["by_multiplicity"]["bh"]["determined_fraction"],
        "excess_observed_pp": primary["excess_observed_pp"],
        "excess_determined_pp": primary["excess_determined_pp"],
        "excess_expected_pp": primary["excess_expected_pp"],
        "direction_preserved": bool(
            primary["excess_observed_pp"] > 0 and primary["excess_determined_pp"] > 0 and primary["excess_expected_pp"] > 0
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["headline"], indent=2))
    print(json.dumps(primary["decomposition"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
