"""Known-answer validation against DataDecide's published decision accuracy.

DataDecide (Magnusson et al., 2025) defines decision accuracy as the fraction
of unordered data-recipe pairs whose predicted order at the target scale (1B,
3-seed mean OLMES ACCURACY) matches the observed order. Its headline single-
scale baseline is "~80% of comparisons correct" when ranking fully trained
150M models (abstract and Figure 1). Figure 1 averages the decision accuracy
of three prediction attempts, one per small-model seed.

Two protocols are computed here:

* ``seed_mean`` - ASLA's ``single_scale_ranker``: rank recipes by their
  3-seed mean at the small scale, compare with the 3-seed-mean order at the
  target. This is what ``asla audit`` evaluates under ``pairwise_decisions``.
* ``per_seed`` - DataDecide's Figure 1 protocol: rank by each small-scale seed
  separately, score against the 3-seed-mean target order, and average the
  three accuracies (their standard deviation is reported alongside).

The published value is approximate ("~80%"); the tolerance band is documented
in ``KNOWN_ANSWER.md`` and encoded in :data:`PUBLISHED_SINGLE_SCALE_150M`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.fits import truth_ranking
from asla.analysis.metrics import decision_metrics
from asla.analysis.rankers import single_scale_ranker
from asla.data.schema import validate

PUBLISHED_SINGLE_SCALE_150M: dict[str, Any] = {
    "value": 0.80,
    "band": (0.75, 0.85),
    "metric": "olmes_macro_error",
    "scale_label": "150M",
    "source": "Magnusson et al. 2025 (arXiv:2504.11393), abstract and Figure 1: '~80% of comparisons correct' "
    "for ranking 150M models to predict 1B pairwise winners on OLMES ACCURACY",
    "precision": "approximate; the paper reports the value rounded and graphically, not as a table entry",
}


def _compute_for_scale(df: pd.DataFrame, scale_label: str) -> float:
    rows = df[df["scale_label"].astype(str) == scale_label]
    if rows.empty:
        raise ValueError(f"scale {scale_label!r} not present in the runs table")
    values = rows["compute"].astype(float).unique()
    if len(values) != 1:
        raise ValueError(f"scale {scale_label!r} maps to several compute values: {values.tolist()}")
    return float(values[0])


def pairwise_decision_accuracy(
    df: pd.DataFrame,
    small_compute: float,
    target_compute: float,
    protocol: str = "seed_mean",
) -> dict[str, Any]:
    """Return pairwise decision accuracy of single-scale ranking for one protocol."""

    validate(df)
    truth = truth_ranking(df, target_compute)
    if protocol == "seed_mean":
        predicted = single_scale_ranker(df, (small_compute,), target_compute)
        metrics = decision_metrics(predicted, truth, k=1)
        return {
            "protocol": protocol,
            "pairwise_acc": float(metrics["pairwise_acc"]),
            "n_pairs": int(len(truth) * (len(truth) - 1) // 2),
            "n_interventions": int(len(truth)),
        }
    if protocol != "per_seed":
        raise ValueError(f"unknown protocol {protocol!r}; choose 'seed_mean' or 'per_seed'")
    small = df[np.isclose(df["compute"].astype(float), small_compute)]
    if small.empty:
        raise ValueError(f"no rows at small compute {small_compute}")
    per_seed: dict[int, float] = {}
    for seed, group in small.groupby("seed", sort=True):
        predicted = group.groupby("intervention", sort=True)["bpb"].mean().sort_values(kind="mergesort")
        if set(predicted.index.astype(str)) != set(truth.index.astype(str)):
            raise ValueError(f"seed {seed} does not cover every intervention at the small scale")
        per_seed[int(seed)] = float(decision_metrics(predicted, truth, k=1)["pairwise_acc"])
    values = np.asarray(list(per_seed.values()), dtype=float)
    return {
        "protocol": protocol,
        "pairwise_acc": float(values.mean()),
        "pairwise_acc_sd_over_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
        "per_seed": per_seed,
        "n_pairs": int(len(truth) * (len(truth) - 1) // 2),
        "n_interventions": int(len(truth)),
    }


def known_answer_report(
    df: pd.DataFrame,
    target_label: str = "1B",
    published: dict[str, Any] = PUBLISHED_SINGLE_SCALE_150M,
) -> dict[str, Any]:
    """Compute single-scale decision accuracy at every scale and compare with the published value."""

    validate(df)
    if "scale_label" not in df.columns:
        raise ValueError("known-answer validation needs the DataDecide 'scale_label' column")
    metric_name = str(df["metric_name"].iloc[0]) if "metric_name" in df.columns else None
    if metric_name != published["metric"]:
        raise ValueError(f"known answer is defined on metric {published['metric']!r}; table carries {metric_name!r}")
    target = _compute_for_scale(df, target_label)
    scales = [
        label
        for label in df.drop_duplicates("scale_label").sort_values("compute")["scale_label"].astype(str)
        if label != target_label
    ]
    per_scale: list[dict[str, Any]] = []
    for label in scales:
        compute = _compute_for_scale(df, label)
        entry: dict[str, Any] = {
            "scale_label": label,
            "compute": compute,
            "percent_of_target_compute": 100.0 * compute / target,
        }
        for protocol in ("seed_mean", "per_seed"):
            result = pairwise_decision_accuracy(df, compute, target, protocol=protocol)
            entry[protocol] = result["pairwise_acc"]
            if protocol == "per_seed":
                entry["per_seed_sd"] = result["pairwise_acc_sd_over_seeds"]
        per_scale.append(entry)
    focal = next(entry for entry in per_scale if entry["scale_label"] == published["scale_label"])
    lo, hi = published["band"]
    checks = {
        protocol: {
            "computed": float(focal[protocol]),
            "published": float(published["value"]),
            "gap": float(focal[protocol] - published["value"]),
            "within_band": bool(lo <= focal[protocol] <= hi),
        }
        for protocol in ("seed_mean", "per_seed")
    }
    return {
        "metric_name": metric_name,
        "target_label": target_label,
        "target_compute": target,
        "published": dict(published),
        "focal_scale": published["scale_label"],
        "checks": checks,
        "reproduced": bool(all(check["within_band"] for check in checks.values())),
        "per_scale": per_scale,
    }
