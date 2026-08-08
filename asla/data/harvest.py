"""Weights & Biases discovery and harvesting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import REQUIRED_COLUMNS, validate

LOGGER = logging.getLogger(__name__)
_MISSING = object()


def _mapping_value(mapping: object, key: str) -> object:
    """Resolve an exact or dotted nested key from a W&B mapping-like store."""

    if not isinstance(mapping, Mapping):
        return _MISSING
    if key in mapping:
        return mapping[key]
    current: object = mapping
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _run_value(run: object, key: str) -> object:
    """Read an explicitly sourced W&B field without guessing between stores."""

    config = getattr(run, "config", {}) or {}
    summary = getattr(run, "summary", {}) or {}
    if key.startswith("config."):
        value = _mapping_value(config, key.removeprefix("config."))
        return None if value is _MISSING else value
    if key.startswith("summary."):
        value = _mapping_value(summary, key.removeprefix("summary."))
        return None if value is _MISSING else value
    config_value = _mapping_value(config, key)
    summary_value = _mapping_value(summary, key)
    if config_value is not _MISSING and summary_value is not _MISSING:
        raise ValueError(
            f"W&B field {key!r} exists in both config and summary; "
            "use an explicit 'config.' or 'summary.' prefix in the field map"
        )
    if config_value is not _MISSING:
        return config_value
    if summary_value is not _MISSING:
        return summary_value
    return None


def _missing(value: object) -> bool:
    """Return whether a harvested scalar should count as missing."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except ValueError:
        return False


def _coerce_harvested_rows(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, int]:
    """Coerce harvested rows to schema dtypes and drop missing required fields."""

    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0

    df["intervention"] = df["intervention"].astype("string")
    df["intervention_class"] = df["intervention_class"].astype("string")
    df["compute"] = pd.to_numeric(df["compute"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["bpb"] = pd.to_numeric(df["bpb"], errors="coerce")
    for optional in ("downstream", "params_n", "tokens_d"):
        if optional in df.columns:
            df[optional] = pd.to_numeric(df[optional], errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    dropped = before - len(df)
    if df.empty:
        return df, dropped
    df["intervention"] = df["intervention"].astype(str)
    df["intervention_class"] = df["intervention_class"].astype(str)
    df["compute"] = df["compute"].astype(float)
    if not (df["seed"] % 1 == 0).all():
        bad = df.loc[df["seed"] % 1 != 0, "seed"].tolist()
        raise ValueError(f"seed values must be integer-like after coercion; bad values: {bad[:5]}")
    df["seed"] = df["seed"].astype(int)
    df["bpb"] = df["bpb"].astype(float)
    return df, dropped


def discover(entity_project: str) -> None:
    """Print config and summary keys for one finished W&B run."""

    import wandb

    api = wandb.Api()
    runs = api.runs(entity_project, filters={"state": "finished"}, per_page=1)
    for run in runs:
        print("config keys:")
        print(json.dumps(sorted((getattr(run, "config", {}) or {}).keys()), indent=2))
        print("summary keys:")
        print(json.dumps(sorted((getattr(run, "summary", {}) or {}).keys()), indent=2))
        return
    print(f"No finished runs found for {entity_project}")


def harvest(entity_project: str, field_map: Mapping[str, str], out_path: str | Path) -> pd.DataFrame:
    """Harvest finished W&B runs using an explicit schema field map."""

    import wandb

    missing_map = [col for col in REQUIRED_COLUMNS if col not in field_map]
    if missing_map:
        raise ValueError(f"field_map is missing required schema keys: {missing_map}")

    api = wandb.Api()
    rows: list[dict[str, object]] = []
    dropped = 0
    for run in api.runs(entity_project, filters={"state": "finished"}):
        row = {schema_key: _run_value(run, source_key) for schema_key, source_key in field_map.items()}
        if any(_missing(row.get(col)) for col in REQUIRED_COLUMNS):
            dropped += 1
            continue
        rows.append(row)

    df, coerced_dropped = _coerce_harvested_rows(rows)
    dropped += coerced_dropped
    if df.empty:
        LOGGER.warning(
            "zero rows survived harvesting; this is a real finding that runs do not log the required fields"
        )
        print("Zero rows survived harvesting; the runs do not log what is needed.")
        return df
    validate(df)
    save_runs(df, out_path)
    summary = df.groupby(["intervention", "compute"], sort=True)["seed"].nunique()
    print(f"Harvested {len(df)} rows, dropped {dropped} incomplete runs.")
    print(f"Interventions: {sorted(df['intervention'].unique().tolist())}")
    print(f"Budgets: {sorted(df['compute'].unique().tolist())}")
    print("Seeds per cell:")
    print(summary.to_string())
    return df
