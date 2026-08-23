"""Stratify decision stability by hyperparameter tuning quality.

Lourie et al. (2026, arXiv:2608.11859) argue that apparent inconsistencies
between small- and large-scale results are largely under-tuning artifacts at
small scale. This module lets an audit report whether single-scale decision
agreement and projection crossovers persist among runs whose hyperparameters
were tuned at their own scale, using the nullable ``tuning_quality`` column.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asla.data.schema import validate


def _means_at(df: pd.DataFrame, compute: float) -> pd.Series:
    rows = df[np.isclose(df["compute"].astype(float), float(compute))]
    if rows.empty:
        raise ValueError(f"no rows at compute {compute}")
    return rows.groupby("intervention", sort=True)["bpb"].mean()


def single_scale_pairwise_agreement(df: pd.DataFrame, small_compute: float, target_compute: float) -> dict[str, Any]:
    """Fraction of intervention pairs whose order at ``small_compute`` matches ``target_compute``.

    Both orders use seed means. Pairs tied at either budget are counted as
    disagreements only when the other budget separates them; exact ties on
    both sides count as agreements. Single-seed tables are accepted; the
    result then carries no uncertainty statement.
    """

    validate(df)
    small = _means_at(df, small_compute)
    target = _means_at(df, target_compute)
    names = sorted(set(small.index.astype(str)) & set(target.index.astype(str)))
    if len(names) < 2:
        raise ValueError("need at least two interventions present at both budgets")
    disagreeing: list[dict[str, Any]] = []
    total = 0
    for a, b in itertools.combinations(names, 2):
        total += 1
        small_gap = float(small.loc[a] - small.loc[b])
        target_gap = float(target.loc[a] - target.loc[b])
        if np.sign(small_gap) != np.sign(target_gap):
            disagreeing.append({"a": a, "b": b, "small_gap": small_gap, "target_gap": target_gap})
    return {
        "small_compute": float(small_compute),
        "target_compute": float(target_compute),
        "n_interventions": len(names),
        "n_pairs": total,
        "pairwise_agreement": float((total - len(disagreeing)) / total),
        "n_disagreeing": len(disagreeing),
        "disagreeing_pairs": disagreeing,
        "min_seeds_per_cell": int(df.groupby(["intervention", "compute"])["seed"].nunique().min()),
    }


def tuning_quality_levels(df: pd.DataFrame) -> list[str | None]:
    """Return the distinct ``tuning_quality`` labels in a table (``None`` when absent)."""

    if "tuning_quality" not in df.columns:
        return [None]
    values = df["tuning_quality"]
    labels: list[str | None] = [str(value) for value in sorted(values.dropna().unique())]
    if values.isna().any():
        labels.append(None)
    return labels


def stratify_by_tuning(
    tables: Mapping[str, pd.DataFrame],
    small_compute: float,
    target_compute: float,
) -> dict[str, Any]:
    """Compute single-scale pairwise agreement separately per tuning condition.

    ``tables`` maps a tuning-condition label to a runs table containing the
    same interventions measured under that condition (for example the
    coordinate-descent optimum versus single-hyperparameter ablations of it).
    Each table must carry one ``tuning_quality`` value equal to its label.
    """

    if not tables:
        raise ValueError("at least one tuning-condition table is required")
    out: dict[str, Any] = {"small_compute": float(small_compute), "target_compute": float(target_compute), "conditions": {}}
    for label, table in tables.items():
        levels = tuning_quality_levels(table)
        if levels != [label]:
            raise ValueError(f"table for condition {label!r} carries tuning_quality levels {levels}")
        out["conditions"][label] = single_scale_pairwise_agreement(table, small_compute, target_compute)
    agreements = {label: entry["pairwise_agreement"] for label, entry in out["conditions"].items()}
    out["agreement_by_condition"] = agreements
    return out
