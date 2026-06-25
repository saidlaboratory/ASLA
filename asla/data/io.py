"""Parquet I/O with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from asla.data.schema import validate


def load_runs(path: str | Path) -> pd.DataFrame:
    """Load a parquet runs table and validate it."""

    df = pd.read_parquet(path)
    validate(df)
    return df


def save_runs(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and save a runs table as parquet."""

    validate(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = df.copy()
    serializable.attrs = {}
    serializable.to_parquet(path, index=False)
