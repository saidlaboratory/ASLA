"""Deterministic persisted held-out splits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, TypedDict, cast

import numpy as np
import pandas as pd

from asla.config import Paths
from asla.data.schema import validate

SPLIT_COLUMN = "__asla_split"

class Split(TypedDict):
    """Serialized train/test split with dataset identity and held-out groups."""

    train: list[int]
    test: list[int]
    by: list[str]
    seed: int
    heldout_classes: list[str]
    heldout_scales: list[float]
    table_fingerprint: str


def _table_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint of table content, order, indices, and dtypes."""

    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in df.columns).encode("utf-8"))
    digest.update("\x1f".join(str(dtype) for dtype in df.dtypes).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def _check_split_matches(df: pd.DataFrame, split: Split, path: Path) -> None:
    """Raise when a persisted split does not partition the current table's rows."""

    train = set(int(i) for i in split.get("train", []))
    test = set(int(i) for i in split.get("test", []))
    current = set(int(i) for i in df.index.tolist())
    if train & test:
        raise ValueError(f"persisted split at {path} has overlapping train and test indices")
    if (train | test) != current:
        raise ValueError(
            f"persisted split at {path} does not match the current runs table; "
            "delete it or point at the split created for this table"
        )
    expected_fingerprint = split.get("table_fingerprint")
    if expected_fingerprint is None:
        raise ValueError(
            f"persisted split at {path} predates dataset fingerprinting; "
            "delete it and create a split for the current table"
        )
    if expected_fingerprint != _table_fingerprint(df):
        raise ValueError(
            f"persisted split at {path} does not match the current runs table contents; "
            "delete it or point at the split created for this table"
        )


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
            split = cast(Split, json.load(fh))
        _check_split_matches(df, split, split_path)
        return split

    rng = np.random.default_rng(seed)
    test_idx: set[int] = set()
    by_set = set(by)
    unknown = sorted(by_set - {"class", "scale"})
    if unknown:
        raise ValueError(f"unknown split dimensions: {unknown}; expected 'class' and/or 'scale'")
    if not by_set:
        raise ValueError("at least one split dimension is required")
    heldout_classes: list[str] = []
    heldout_scales: list[float] = []
    if "class" in by_set:
        classes = sorted(df["intervention_class"].astype(str).unique())
        if len(classes) > 1:
            n_holdout = min(len(classes) - 1, max(1, len(classes) // 4))
            heldout = set(rng.choice(classes, size=n_holdout, replace=False).tolist())
            heldout_classes = sorted(str(value) for value in heldout)
            test_idx.update(df.index[df["intervention_class"].astype(str).isin(heldout)].astype(int).tolist())
    if "scale" in by_set:
        computes = np.asarray(sorted(df["compute"].astype(float).unique()), dtype=float)
        if len(computes) > 1:
            scale = float(rng.choice(computes[1:], size=1)[0])
            heldout_scales = [scale]
            test_idx.update(df.index[np.isclose(df["compute"].astype(float), scale)].astype(int).tolist())

    all_idx = set(int(i) for i in df.index.tolist())
    if not test_idx:
        raise ValueError("split dimensions did not produce any held-out test rows")
    if test_idx == all_idx:
        raise ValueError("split dimensions held out every row; at least one training row is required")
    split: Split = {
        "train": sorted(all_idx - test_idx),
        "test": sorted(test_idx),
        "by": sorted(by_set),
        "seed": int(seed),
        "heldout_classes": heldout_classes,
        "heldout_scales": heldout_scales,
        "table_fingerprint": _table_fingerprint(df),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=split_path.parent,
            prefix=f".{split_path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(split, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(split_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
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
