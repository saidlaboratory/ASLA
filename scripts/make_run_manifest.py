"""Create a no-W&B run manifest for ASLA data collection."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_INTERVENTION_COLUMNS = ("intervention", "intervention_class")
REQUIRED_BUDGET_COLUMNS = ("compute", "role")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def _read_interventions(path: str) -> pd.DataFrame:
    """Read and validate the intervention plan."""

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_INTERVENTION_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"intervention file missing required columns: {missing}")
    if df.empty:
        raise SystemExit("intervention file has no rows")
    if df[list(REQUIRED_INTERVENTION_COLUMNS)].isna().any().any():
        raise SystemExit("intervention file contains missing intervention or intervention_class values")
    if df["intervention"].astype(str).str.strip().eq("").any():
        raise SystemExit("intervention names must be non-empty")
    unsafe = sorted(name for name in df["intervention"].astype(str).unique() if SAFE_IDENTIFIER.fullmatch(name) is None)
    if unsafe:
        raise SystemExit(
            "intervention names must be path-safe identifiers containing only letters, numbers, '.', '_', '+', or '-'; "
            f"invalid: {unsafe}"
        )
    if df["intervention_class"].astype(str).str.strip().eq("").any():
        raise SystemExit("intervention classes must be non-empty")
    duplicates = df["intervention"].astype(str).duplicated(keep=False)
    if bool(duplicates.any()):
        names = sorted(df.loc[duplicates, "intervention"].astype(str).unique().tolist())
        raise SystemExit(f"intervention names must be unique; duplicates: {names}")
    return df


def _read_budgets(path: str) -> pd.DataFrame:
    """Read and validate the budget plan."""

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_BUDGET_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"budget file missing required columns: {missing}")
    if df.empty:
        raise SystemExit("budget file has no rows")
    df = df.copy()
    df["compute"] = pd.to_numeric(df["compute"], errors="raise").astype(float)
    df["role"] = df["role"].astype(str)
    if not set(df["role"]).issubset({"fit", "intermediate", "target"}):
        raise SystemExit("budget role must be 'fit', 'intermediate', or 'target'")
    if (not np.isfinite(df["compute"].to_numpy(dtype=float)).all()) or (df["compute"] <= 0).any():
        raise SystemExit("compute budgets must be finite and positive")
    if (df["role"] == "fit").sum() < 3:
        raise SystemExit("at least three fit budgets are required for power-law fits")
    if (df["role"] == "target").sum() != 1:
        raise SystemExit("exactly one target budget is required")
    if (df["role"] == "intermediate").sum() > 1:
        raise SystemExit("at most one intermediate budget is allowed")
    duplicates = df["compute"].duplicated(keep=False)
    if bool(duplicates.any()):
        budgets = sorted(df.loc[duplicates, "compute"].astype(float).unique().tolist())
        raise SystemExit(f"compute budgets must be unique; duplicates: {budgets}")
    target = float(df.loc[df["role"] == "target", "compute"].iloc[0])
    fit = df.loc[df["role"] == "fit", "compute"].astype(float)
    intermediate_rows = df.loc[df["role"] == "intermediate", "compute"].astype(float)
    fit_ceiling = float(intermediate_rows.iloc[0]) if len(intermediate_rows) else target
    if bool((fit >= fit_ceiling).any()):
        raise SystemExit("all fit budgets must be below the intermediate budget when present, otherwise the target")
    if len(intermediate_rows) and float(intermediate_rows.iloc[0]) >= target:
        raise SystemExit("the intermediate budget must be below the target budget")
    return df.sort_values("compute", kind="mergesort")


def make_manifest(interventions: pd.DataFrame, budgets: pd.DataFrame, seeds: int) -> pd.DataFrame:
    """Return one planned row per intervention, budget, and seed."""

    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    names = interventions["intervention"].astype(str)
    unsafe = sorted(name for name in names.unique() if SAFE_IDENTIFIER.fullmatch(name) is None)
    if unsafe:
        raise ValueError(f"unsafe intervention identifiers: {unsafe}")
    if names.duplicated().any():
        raise ValueError("intervention names must be unique")

    rows: list[dict[str, object]] = []
    for intervention in interventions.itertuples(index=False):
        for budget in budgets.itertuples(index=False):
            for seed in range(seeds):
                rows.append(
                    {
                        "run_id": f"{intervention.intervention}__c{float(budget.compute):.17g}__s{seed}",
                        "intervention": str(intervention.intervention),
                        "intervention_class": str(intervention.intervention_class),
                        "compute": float(budget.compute),
                        "role": str(budget.role),
                        "seed": int(seed),
                        "status": "pending",
                        "bpb": "",
                        "downstream": "",
                        "params_n": "",
                        "tokens_d": "",
                        "notes": "",
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest["run_id"].duplicated().any():
        raise ValueError("generated run_id values are not unique")
    return manifest


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a generated run manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            df.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    """Run the manifest generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interventions", default="data/interventions_template.csv")
    parser.add_argument("--budgets", default="data/budgets_template.csv")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="data/run_manifest.csv")
    args = parser.parse_args()

    interventions = _read_interventions(args.interventions)
    budgets = _read_budgets(args.budgets)
    manifest = make_manifest(interventions, budgets, args.seeds)
    out = Path(args.out)
    _write_manifest(manifest, out)
    target = budgets.loc[budgets["role"] == "target", "compute"].iloc[0]
    print(f"wrote {len(manifest)} planned runs to {out}")
    print(f"target compute: {target:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
