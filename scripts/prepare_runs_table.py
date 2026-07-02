"""Convert a real runs CSV into the canonical ASLA parquet table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

CANONICAL_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)


def _coerce_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce obvious CSV dtypes without fabricating missing values."""

    out = df.copy()
    if out.empty:
        raise SystemExit("input CSV has no run rows; fill data/runs_template.csv with real runs first")
    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    out["intervention"] = out["intervention"].astype(str)
    out["intervention_class"] = out["intervention_class"].astype(str)
    out["compute"] = pd.to_numeric(out["compute"], errors="raise").astype(float)
    seed = pd.to_numeric(out["seed"], errors="raise")
    if not (seed % 1 == 0).all():
        raise SystemExit("seed column must contain integer-like values")
    out["seed"] = seed.astype(int)
    out["bpb"] = pd.to_numeric(out["bpb"], errors="raise").astype(float)

    for col in OPTIONAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.Series([pd.NA] * len(out), dtype="Float64")
    return out.loc[:, CANONICAL_COLUMNS].copy()


def main() -> int:
    """Run the converter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to a real runs CSV.")
    parser.add_argument("--out", default="runs.parquet", help="Output parquet path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    runs = _coerce_runs(df)
    save_runs(runs, args.out)
    print(f"wrote {len(runs)} rows to {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
