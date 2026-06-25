"""Deterministic persisted held-out splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from asla.config import Paths
from asla.data.schema import validate

SPLIT_COLUMN = "__asla_split"
Split = dict[str, object]


def make_split(
    df: pd.DataFrame,
    by: Iterable[str] = ("class", "scale"),
    seed: int = 1729,
    path: str | Path = Paths().split_path,
) -> Split:
    """Create or load a persisted deterministic train/test split."""

    validate(df)
    split_path = Path(path)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    rng = np.random.default_rng(seed)
    test_idx: set[int] = set()
    by_set = set(by)
    if "class" in by_set:
        classes = sorted(df["intervention_class"].astype(str).unique())
        n_holdout = max(1, len(classes) // 4)
        heldout = set(rng.choice(classes, size=n_holdout, replace=False).tolist())
        test_idx.update(df.index[df["intervention_class"].astype(str).isin(heldout)].astype(int).tolist())
    if "scale" in by_set:
        computes = np.asarray(sorted(df["compute"].astype(float).unique()), dtype=float)
        if len(computes) > 1:
            scale = float(rng.choice(computes[1:], size=1)[0])
            test_idx.update(df.index[np.isclose(df["compute"].astype(float), scale)].astype(int).tolist())

    all_idx = set(int(i) for i in df.index.tolist())
    split = {
        "train": sorted(all_idx - test_idx),
        "test": sorted(test_idx),
        "by": sorted(by_set),
        "seed": int(seed),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2, sort_keys=True)
    return split


def train_rows(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Return train rows marked for guard checks."""

    rows = df.loc[list(split["train"])].copy()
    rows[SPLIT_COLUMN] = "train"
    rows.attrs["split_role"] = "train"
    return rows


def test_rows(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Return test rows marked for guard checks."""

    rows = df.loc[list(split["test"])].copy()
    rows[SPLIT_COLUMN] = "test"
    rows.attrs["split_role"] = "test"
    return rows


test_rows.__test__ = False


def assert_not_test(rows: pd.DataFrame) -> None:
    """Raise when rows marked as held-out test data are passed to fitting code."""

    role = rows.attrs.get("split_role")
    if role == "test":
        raise AssertionError("held-out test rows cannot be used for fitting")
    if SPLIT_COLUMN in rows.columns and (rows[SPLIT_COLUMN] == "test").any():
        raise AssertionError("held-out test rows cannot be used for fitting")
