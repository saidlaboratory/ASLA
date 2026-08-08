"""Parquet I/O with schema validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from asla.data.schema import validate


def load_runs(path: str | Path) -> pd.DataFrame:
    """Load a parquet runs table and validate it."""

    df = pd.read_parquet(path)
    validate(df)
    return df


def save_runs(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and atomically save a runs table as parquet."""

    validate(df)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = df.copy()
    serializable.attrs = {}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".parquet",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        serializable.to_parquet(temporary_path, index=False)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_runs_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and atomically save a canonical runs table as CSV."""

    validate(df)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            df.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
