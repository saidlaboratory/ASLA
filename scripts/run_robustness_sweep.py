"""Priorities 3 and 5: is the determined-subset result robust, and what does it do to a published figure?

Scored against ``PREDICTIONS_TASK_ROBUSTNESS.md``, committed before this ran.

Three things:

* the projection-minus-single-scale gap as a continuous function of the
  determination threshold, rather than at one cut;
* an equivalence bound on the determined-subset difference, because a difference
  of zero is only informative if the comparison could have detected one;
* the same three scoring rules across designs, metrics and rankers, and applied
  to the published DataDecide decision-accuracy figure.
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
from scipy import stats

from asla.analysis.fits import cell_means_and_sigma, normalize_budgets
from asla.analysis.target_scoring import score_ranker, target_evidence
from asla.models import FitError, bpb_power_law, fit_power_law

REPO = Path(__file__).resolve().parents[1]
OFF_TRAJECTORY = ("750M",)

# Designs as the audit defines them: (name, max fitting scale, held-out target).
DESIGNS = (
    ("primary_4M-300M_gate530M", "300M", "530M"),
    ("long_4M-150M_gate300M", "150M", "300M"),
    ("short_4M-530M_gate1B", "530M", "1B"),
)
METRICS = {
    "c4_en_bits_per_token": "datadecide_runs.parquet",
    "olmes_macro_error": "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": "datadecide_runs_olmes_correct_prob_per_char.parquet",
}
# Determination thresholds spanning strict to permissive, as per-comparison
# alpha values. Bonferroni over 300 pairs at family alpha 0.05 is 1.67e-4.
THRESHOLDS = (1e-6, 1e-5, 0.05 / 300, 1e-3, 0.01, 0.05, 0.2, 0.5, 1.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(metric: str) -> pd.DataFrame:
    frame = pd.read_parquet(REPO / "data" / METRICS[metric])
    return frame[~frame["scale_label"].isin(OFF_TRAJECTORY)].reset_index(drop=True)


def rankers_for(
    frame: pd.DataFrame, fit_budgets: tuple[float, ...], target: float
) -> dict[str, dict[tuple[str, str], float]]:
    """Predicted target gaps for each ranker, as left-minus-right per pair."""

    means = cell_means_and_sigma(frame)
    names = sorted(frame["intervention"].unique())
    top = max(fit_budgets)

    projected: dict[str, float] = {}
    single: dict[str, float] = {}
    exponents: dict[str, tuple[float, float, float]] = {}
    for name in names:
        rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))]
        rows = rows.sort_values("compute")
        compute = rows["compute"].to_numpy(dtype=float)
        values = rows["bpb_mean"].to_numpy(dtype=float)
        if len(rows) >= 3:
            try:
                params = fit_power_law(compute, values)
                exponents[name] = params
                projected[name] = float(bpb_power_law(target, *params))
            except FitError:
                projected[name] = float("nan")
        cell = rows[np.isclose(rows["compute"], top)]
        single[name] = float(cell["bpb_mean"].iloc[0]) if len(cell) else float("nan")

    # Shared-exponent pooling: one alpha for all candidates, refit amplitude.
    shared: dict[str, float] = {}
    if exponents:
        alpha = float(np.median([p[2] for p in exponents.values()]))
        for name in names:
            rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))]
            rows = rows.sort_values("compute")
            if len(rows) < 3:
                shared[name] = float("nan")
                continue
            compute = rows["compute"].to_numpy(dtype=float)
            values = rows["bpb_mean"].to_numpy(dtype=float)
            basis = compute ** (-alpha)
            design = np.column_stack([np.ones_like(basis), basis])
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            shared[name] = float(coefficients[0] + coefficients[1] * target ** (-alpha))

    def gaps(values: dict[str, float]) -> dict[tuple[str, str], float]:
        usable = sorted(k for k in values if np.isfinite(values[k]))
        return {(a, b): values[a] - values[b] for a, b in combinations(usable, 2)}

    out = {"single_scale": gaps(single), "projection": gaps(projected)}
    if shared:
        out["shared_exponent"] = gaps(shared)
    return out


def target_cells(frame: pd.DataFrame, target: float) -> tuple[dict, dict, dict]:
    cells = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"]
    return (
        {str(k): float(v) for k, v in cells.mean().items()},
        {str(k): float(v) for k, v in cells.std(ddof=1).items()},
        {str(k): int(v) for k, v in cells.count().items()},
    )


def threshold_curve(
    means: dict, sds: dict, counts: dict, rankers: dict, reference: str, comparator: str
) -> list[dict[str, Any]]:
    """The gap between two rankers as the determination threshold varies.

    One cut cannot show whether a vanishing difference is real or an artifact of
    a subset too small to contain any errors, so the whole curve is reported
    along with the error counts that make it interpretable.
    """

    rows = []
    for alpha in THRESHOLDS:
        evidence = target_evidence(means, sds, counts, alpha=alpha, multiplicity="none")
        determined = [e for e in evidence if e.determined]
        left = score_ranker(rankers[comparator], determined)
        right = score_ranker(rankers[reference], determined)
        total_errors = left["determined_errors"] + right["determined_errors"]
        rows.append(
            {
                "per_comparison_alpha": alpha,
                "determined_pairs": len(determined),
                "determined_fraction": len(determined) / len(evidence),
                f"{comparator}_errors": left["determined_errors"],
                f"{reference}_errors": right["determined_errors"],
                "gap_pp": 100 * (left["determined_rate"] - right["determined_rate"]) if determined else float("nan"),
                "total_errors_in_subset": total_errors,
                "comparison_is_vacuous": bool(total_errors == 0),
            }
        )
    return rows


def equivalence_bound(left_errors: int, right_errors: int, n_pairs: int, confidence: float = 0.90) -> dict[str, Any]:
    """Two-sided CI on the difference in error rates, and what it rules out.

    Uses the Wilson-style normal approximation on the paired difference. With
    zero errors on both sides the interval is driven entirely by ``n_pairs``,
    which is the point: an empty comparison bounds nothing tightly.
    """

    if n_pairs == 0:
        return {"n_pairs": 0, "note": "no determined pairs"}
    p_left = left_errors / n_pairs
    p_right = right_errors / n_pairs
    difference = p_left - p_right
    # Conservative: treat the two counts as independent binomials.
    se = float(np.sqrt(p_left * (1 - p_left) / n_pairs + p_right * (1 - p_right) / n_pairs))
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    # With zero events the normal SE collapses; use the rule-of-three upper
    # bound on a zero-count rate instead, which is the honest width.
    if left_errors == 0 and right_errors == 0:
        rule_of_three = 3.0 / n_pairs
        return {
            "n_pairs": n_pairs,
            "difference_pp": 0.0,
            "ci_low_pp": -100 * rule_of_three,
            "ci_high_pp": 100 * rule_of_three,
            "largest_difference_ruled_out_pp": 100 * rule_of_three,
            "method": "rule of three (zero events on both sides)",
            "vacuous": True,
            "note": (
                "Neither method errs on this subset, so the data cannot distinguish them. "
                "The bound is set by the subset size alone, not by any observed difference."
            ),
        }
    return {
        "n_pairs": n_pairs,
        "difference_pp": 100 * difference,
        "ci_low_pp": 100 * (difference - z * se),
        "ci_high_pp": 100 * (difference + z * se),
        "largest_difference_ruled_out_pp": 100 * (abs(difference) + z * se),
        "method": f"normal approximation, {int(confidence * 100)}% CI",
        "vacuous": False,
    }


def published_figure_decomposition(frame: pd.DataFrame, focal: str, target_label: str) -> dict[str, Any]:
    """Priority 5: what does the protocol do to a published decision-accuracy number?

    DataDecide reports ~0.80 pairwise decision accuracy for ranking at a single
    small scale. That figure is computed against observed target means, so it
    mixes decisions the reference resolves with coin flips it does not.
    """

    target = float(frame[frame["scale_label"] == target_label]["compute"].iloc[0])
    focal_compute = float(frame[frame["scale_label"] == focal]["compute"].iloc[0])
    means, sds, counts = target_cells(frame, target)
    focal_means = frame[np.isclose(frame["compute"], focal_compute)].groupby("intervention")["bpb"].mean()
    predictions = {(a, b): float(focal_means[a] - focal_means[b]) for a, b in combinations(sorted(focal_means.index), 2)}

    out: dict[str, Any] = {"focal_scale": focal, "target_scale": target_label}
    for rule in ("bonferroni", "bh"):
        evidence = target_evidence(means, sds, counts, alpha=0.05, multiplicity=rule)
        scored = score_ranker(predictions, evidence)
        determined = [e for e in evidence if e.determined]
        out[rule] = {
            "n_pairs": scored["n_scored"],
            "determined_pairs": len(determined),
            "determined_fraction": len(determined) / len(evidence),
            "accuracy_overall": 1 - scored["observed_rate"],
            "accuracy_on_determined": 1 - scored["determined_rate"],
            "accuracy_expected": 1 - scored["expected_rate"],
            "shift_pp": 100 * ((1 - scored["determined_rate"]) - (1 - scored["observed_rate"])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "target_scoring" / "robustness.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "inputs": {name: _sha(REPO / "data" / path) for name, path in METRICS.items() if (REPO / "data" / path).exists()},
        "thresholds_swept": list(THRESHOLDS),
        "note_on_df": "per-pair Welch-Satterthwaite throughout; 4 df is the maximum, not a constant",
        "cells": {},
    }

    for metric in METRICS:
        frame = load(metric)
        for design_name, max_fit, target_label in DESIGNS:
            if target_label not in set(frame["scale_label"]) or max_fit not in set(frame["scale_label"]):
                continue
            target = float(frame[frame["scale_label"] == target_label]["compute"].iloc[0])
            max_compute = float(frame[frame["scale_label"] == max_fit]["compute"].iloc[0])
            fit_budgets = normalize_budgets(
                sorted(float(c) for c in frame["compute"].unique() if c <= max_compute), target=target
            )
            if len(fit_budgets) < 3:
                continue

            means, sds, counts = target_cells(frame, target)
            rankers = rankers_for(frame, fit_budgets, target)
            key = f"{metric}/{design_name}"
            cell: dict[str, Any] = {
                "metric": metric,
                "design": design_name,
                "n_fit_budgets": len(fit_budgets),
                "rankers": {},
            }

            for rule in ("bonferroni", "bh", "none"):
                evidence = target_evidence(means, sds, counts, alpha=0.05, multiplicity=rule)
                determined = [e for e in evidence if e.determined]
                cell.setdefault("determined", {})[rule] = {
                    "pairs": len(determined),
                    "fraction": len(determined) / len(evidence),
                }
                for name, predictions in rankers.items():
                    scored = score_ranker(predictions, evidence)
                    cell["rankers"].setdefault(name, {})[rule] = scored

            baseline = "single_scale"
            if "projection" in rankers:
                cell["threshold_curve"] = threshold_curve(means, sds, counts, rankers, baseline, "projection")
                evidence = target_evidence(means, sds, counts, alpha=0.05, multiplicity="bonferroni")
                determined = [e for e in evidence if e.determined]
                left = score_ranker(rankers["projection"], determined)
                right = score_ranker(rankers[baseline], determined)
                cell["equivalence_bonferroni"] = equivalence_bound(
                    left["determined_errors"], right["determined_errors"], len(determined)
                )
            payload["cells"][key] = cell

    # Priority 5: the published DataDecide figure.
    olmes = load("olmes_macro_error")
    payload["published_figure"] = published_figure_decomposition(olmes, "150M", "1B")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["published_figure"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
