"""Task 3: reconcile our extrapolation null with Open Athena's 300x success.

Open Athena's Delphi suite (openathena.ai/blog/delphi) fits seven IsoFLOP optima
from 3e18 to 3e20 FLOPs and predicts a 1e23 FLOP run's loss within 0.2%, with
held-out runs at 1e21 and 1e22 within 0.5% and three seeds at each landing inside
the bootstrap 95% interval. A paper claiming extrapolation-based selection fails
has to engage that.

The reconciliation is not that one of us is wrong. They solve a *level* problem
for one recipe; we solve a *sign* problem between recipes. This script quantifies
the gap between those two tasks on our own committed data, so the argument rests
on numbers rather than on a distinction drawn in prose.

Verified against the source before writing (see results/external/ for the record):
  - the fit is seven IsoFLOP optima over 3e18-3e20, target 1e23;
  - the enabler is a token-horizon correction eta ~ (T0/T)^0.3 plus AdamH, which
    removes weight decay from the search;
  - the metric is pretraining loss, not a downstream benchmark;
  - **they never rank candidates**: each law predicts one configuration's loss,
    and Attempt 1 vs Attempt 2 is a diagnosis of why a recipe failed, not a
    selection between candidates. Absolute losses are reported, not gaps.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.provenance import Source

REPO = Path(__file__).resolve().parents[1]
OFF_TRAJECTORY_SCALES = ("750M",)

# Accuracy levels Open Athena reports, as fractions of the loss level.
REPORTED_ACCURACIES = (0.002, 0.005, 0.010)


def load_target(data: Path, target_scale: str) -> tuple[pd.Series, float]:
    frame = pd.read_parquet(data)
    frame = frame[~frame["scale_label"].isin(OFF_TRAJECTORY_SCALES)]
    target = float(frame[frame["scale_label"] == target_scale]["compute"].iloc[0])
    means = frame[np.isclose(frame["compute"], target)].groupby("intervention")["bpb"].mean()
    return means, target


def level_accuracy_vs_decision(means: pd.Series) -> dict[str, Any]:
    """How many decisions would a given level accuracy fail to resolve?

    A level claim bounds each candidate's predicted value. A decision needs the
    *sign of a difference*. If two candidates' true gap is smaller than the level
    error, an independent per-candidate guarantee at that accuracy says nothing
    about their ordering.
    """

    names = sorted(means.index)
    gaps = np.asarray([abs(float(means[a]) - float(means[b])) for a, b in itertools.combinations(names, 2)])
    level = float(means.mean())

    rows = []
    for accuracy in REPORTED_ACCURACIES:
        tolerance = level * accuracy
        unresolved = int(np.sum(gaps < tolerance))
        rows.append(
            {
                "level_accuracy": accuracy,
                "tolerance_in_metric_units": float(tolerance),
                "pairs_with_gap_below_tolerance": unresolved,
                "fraction_unresolved": float(unresolved / len(gaps)),
            }
        )
    return {
        "n_candidates": len(names),
        "n_pairs": int(len(gaps)),
        "mean_level": level,
        "gap_median": float(np.median(gaps)),
        "gap_mean": float(gaps.mean()),
        "gap_min": float(gaps.min()),
        "by_accuracy": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "datadecide_runs.parquet")
    parser.add_argument("--target-scale", default="1B")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "external" / "counterexample_delphi.json")
    args = parser.parse_args()

    means, target = load_target(args.data, args.target_scale)
    transfer = level_accuracy_vs_decision(means)

    # Loaded by key path rather than retyped, per the provenance module.
    audit = Source.load("first_audit/first_audit.json")
    tuning = {
        scale: {
            condition: audit.number(f"tuning_stratification.by_small_scale.{scale}.agreement_by_condition.{condition}")
            for condition in ("coordinate_descent_tuned", "single_hp_ablation_median")
        }
        for scale in ("130m", "300m")
    }

    payload: dict[str, Any] = {
        "source": {
            "name": "Open Athena, Scaling Laws That Extrapolate 300x Past the Fit (Delphi)",
            "url": "https://openathena.ai/blog/delphi/",
            "verified_against_source": True,
            "fit_range_flops": [3e18, 3e20],
            "n_isoflop_optima_in_fit": 7,
            "target_flops": 1e23,
            "extrapolation_factor": 300,
            "reported_accuracy_at_target": 0.002,
            "reported_accuracy_heldout": 0.005,
            "heldout_budgets_flops": [1e21, 1e22],
            "seeds_per_heldout_budget": 3,
            "metric": "pretraining loss",
            "enablers": [
                "token-horizon learning-rate correction eta ~ (T0/T)^0.3",
                "AdamH optimizer, which removes weight decay from the search",
            ],
        },
        "what_they_did_not_do": {
            "ranked_candidates": False,
            "reported_gaps_between_recipes": False,
            "note": (
                "Every fit predicts one configuration's loss. Delphi Attempt 1 versus Attempt 2 "
                "is a diagnosis of why a recipe mis-extrapolated, then a rewrite, not a selection "
                "between candidates at a target budget. Absolute losses are reported; the "
                "difference between two recipes at the target never is."
            ),
        },
        "why_the_results_are_compatible": (
            "They solve a level problem for one recipe; we solve a sign problem between recipes. "
            "A bound on each candidate's predicted value does not bound the sign of a difference "
            "unless the residual error is common-mode across candidates --- which is exactly the "
            "cancellation question this project measures, and which is metric-dependent."
        ),
        "level_accuracy_does_not_imply_decision_accuracy": transfer,
        "target_scale": args.target_scale,
        "target_compute": target,
        "moderators_this_suggests": {
            "hyperparameter_transfer_regime": (
                "Their stated enabler is a token-horizon correction to the learning rate. That is "
                "a claim about hyperparameter transfer, and it is testable as a moderator: "
                "extrapolation should work better for recipes whose hyperparameters transfer "
                "correctly across scale. We cannot test it here -- DataDecide varies corpora "
                "under one recipe, so there is no hyperparameter-transfer contrast in it, and the "
                "suite that would provide one (compute-multipliers, seven recipes differing in "
                "optimizer, schedule and initialisation) releases only its top budget."
            ),
            "related_evidence_in_this_project": {
                "measure": "pairwise agreement between small-scale and target ordering",
                "by_scale": tuning,
                "note": (
                    "The tuning-confound study already measures a version of this: agreement "
                    "falls when a single hyperparameter is mis-tuned relative to coordinate-"
                    "descent tuning, at both small scales. That is the same mechanism, in the "
                    "direction their token-horizon correction is designed to remove."
                ),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(transfer["by_accuracy"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
