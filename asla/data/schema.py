"""Canonical runs-table schema."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

REQUIRED_COLUMNS = (
    "intervention",
    "intervention_class",
    "compute",
    "seed",
    "bpb",
)
OPTIONAL_COLUMNS = ("downstream", "params_n", "tokens_d")


class SchemaError(ValueError):
    """Raised when a runs table does not satisfy the canonical schema."""


def _is_string_like(series: pd.Series) -> bool:
    return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series)


def _format_errors(errors: Iterable[str]) -> str:
    return "Runs table schema validation failed: " + "; ".join(errors)


def validate(df: pd.DataFrame) -> None:
    """Validate the canonical runs-table schema.

    Required columns are ``intervention`` and ``intervention_class`` as strings,
    ``compute`` and ``bpb`` as numeric values, and ``seed`` as integers.
    ``downstream`` is optional and, when present, must be numeric or nullable
    numeric.

    Raises:
        SchemaError: with all missing or invalid columns listed.
    """

    errors: list[str] = []
    if df.empty:
        errors.append("runs table must contain at least one row")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")

    if "intervention" in df.columns and not _is_string_like(df["intervention"]):
        errors.append("column 'intervention' must be string-like")
    if "intervention_class" in df.columns and not _is_string_like(df["intervention_class"]):
        errors.append("column 'intervention_class' must be string-like")
    if "compute" in df.columns and not ptypes.is_numeric_dtype(df["compute"]):
        errors.append("column 'compute' must be numeric")
    if "seed" in df.columns and not ptypes.is_integer_dtype(df["seed"]):
        errors.append("column 'seed' must be integer")
    if "bpb" in df.columns and not ptypes.is_numeric_dtype(df["bpb"]):
        errors.append("column 'bpb' must be numeric")
    if "downstream" in df.columns and not ptypes.is_numeric_dtype(df["downstream"]):
        errors.append("optional column 'downstream' must be numeric when present")
    if "params_n" in df.columns and not ptypes.is_numeric_dtype(df["params_n"]):
        errors.append("optional column 'params_n' must be numeric when present")
    if "tokens_d" in df.columns and not ptypes.is_numeric_dtype(df["tokens_d"]):
        errors.append("optional column 'tokens_d' must be numeric when present")

    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            errors.append(f"column '{col}' must not contain null values")
    for col in ("intervention", "intervention_class"):
        if col in df.columns and _is_string_like(df[col]):
            bad = df[col].dropna().map(lambda value: not isinstance(value, str))
            if bool(bad.any()):
                errors.append(f"column '{col}' must contain only strings")
            blank = df[col].dropna().astype(str).str.strip().eq("")
            if bool(blank.any()):
                errors.append(f"column '{col}' must not contain blank strings")
    if "compute" in df.columns and ptypes.is_numeric_dtype(df["compute"]):
        values = pd.to_numeric(df["compute"], errors="coerce")
        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if (not np.isfinite(numeric).all()) or (numeric <= 0).any():
            errors.append("column 'compute' must contain finite positive numbers")
    if "bpb" in df.columns and ptypes.is_numeric_dtype(df["bpb"]):
        values = pd.to_numeric(df["bpb"], errors="coerce")
        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if (not np.isfinite(numeric).all()) or (numeric <= 0).any():
            errors.append("column 'bpb' must contain finite positive numbers")
    if "seed" in df.columns and ptypes.is_integer_dtype(df["seed"]):
        if (df["seed"].to_numpy(dtype=np.int64) < 0).any():
            errors.append("column 'seed' must contain non-negative integers")
    if "downstream" in df.columns and ptypes.is_numeric_dtype(df["downstream"]):
        values = pd.to_numeric(df["downstream"], errors="coerce").dropna()
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            errors.append("optional column 'downstream' must contain finite numbers when present")
    for col in ("params_n", "tokens_d"):
        if col in df.columns and ptypes.is_numeric_dtype(df[col]):
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            numeric = values.to_numpy(dtype=float, na_value=np.nan)
            if (not np.isfinite(numeric).all()) or (numeric <= 0).any():
                errors.append(f"optional column '{col}' must contain finite positive numbers when present")

    identity_columns = ["intervention", "compute", "seed"]
    if all(col in df.columns for col in identity_columns):
        duplicates = df.duplicated(identity_columns, keep=False)
        if bool(duplicates.any()):
            examples = df.loc[duplicates, identity_columns].head(5).to_dict("records")
            errors.append(
                "columns ('intervention', 'compute', 'seed') must uniquely identify runs; "
                f"duplicate examples: {examples}"
            )
    if all(col in df.columns for col in ("intervention", "intervention_class")):
        class_counts = df.groupby("intervention", dropna=False)["intervention_class"].nunique(dropna=False)
        inconsistent = sorted(str(name) for name in class_counts[class_counts > 1].index)
        if inconsistent:
            errors.append(f"each intervention must have exactly one intervention_class; inconsistent: {inconsistent}")

    if errors:
        raise SchemaError(_format_errors(errors))
