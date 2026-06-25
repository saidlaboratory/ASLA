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


def _run_value(run: object, key: str) -> object:
    config = getattr(run, "config", {}) or {}
    summary = getattr(run, "summary", {}) or {}
    if key in config:
        return config[key]
    if key in summary:
        return summary[key]
    return None


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
        if any(row.get(col) is None for col in REQUIRED_COLUMNS):
            dropped += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        LOGGER.warning(
            "zero rows survived harvesting; this is a real finding that runs do not log the required fields"
        )
        print("Zero rows survived harvesting; the runs do not log what is needed.")
        return df
    df["intervention"] = df["intervention"].astype(str)
    df["intervention_class"] = df["intervention_class"].astype(str)
    df["compute"] = df["compute"].astype(float)
    df["seed"] = df["seed"].astype(int)
    df["bpb"] = df["bpb"].astype(float)
    if "downstream" in df.columns:
        df["downstream"] = pd.to_numeric(df["downstream"], errors="coerce")
    validate(df)
    save_runs(df, out_path)
    summary = df.groupby(["intervention", "compute"], sort=True)["seed"].nunique()
    print(f"Harvested {len(df)} rows, dropped {dropped} incomplete runs.")
    print(f"Interventions: {sorted(df['intervention'].unique().tolist())}")
    print(f"Budgets: {sorted(df['compute'].unique().tolist())}")
    print("Seeds per cell:")
    print(summary.to_string())
    return df

