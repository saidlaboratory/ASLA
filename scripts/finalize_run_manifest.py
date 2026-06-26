"""Convert a completed no-W&B run manifest into ASLA audit inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, validate


DONE_STATUSES = {"complete", "completed", "done"}
CANONICAL_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)


def _missing_message(df: pd.DataFrame, mask: pd.Series, label: str) -> str:
    """Return a compact message identifying bad manifest rows."""

    bad = df.loc[mask]
    if "run_id" in bad.columns:
        examples = bad["run_id"].astype(str).head(5).tolist()
    else:
        examples = bad.index.astype(str).head(5).tolist()
    suffix = "" if len(bad) <= 5 else f" and {len(bad) - 5} more"
    return f"{label}: {examples}{suffix}"


def manifest_to_runs(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return canonical ASLA rows from a completed run manifest.

    The conversion is intentionally strict: it never fills missing BPB values,
    never treats pending rows as data, and keeps only canonical audit columns.
    """

    missing = [col for col in REQUIRED_COLUMNS if col not in manifest.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    if "status" not in manifest.columns:
        raise ValueError("manifest missing required column: status")
    else:
        statuses = manifest["status"].astype(str).str.lower().str.strip()
        pending = ~statuses.isin(DONE_STATUSES)
        if bool(pending.any()):
            raise ValueError(_missing_message(manifest, pending, "manifest has rows that are not complete"))

    out = manifest.copy()
    out["bpb"] = pd.to_numeric(out["bpb"], errors="coerce")
    missing_bpb = out["bpb"].isna()
    if bool(missing_bpb.any()):
        raise ValueError(_missing_message(out, missing_bpb, "manifest has rows without measured BPB"))

    for col in ("compute", "downstream", "params_n", "tokens_d"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    seed = pd.to_numeric(out["seed"], errors="raise")
    if not (seed % 1 == 0).all():
        raise ValueError("seed column must contain integer-like values")
    out["seed"] = seed.astype(int)
    out["intervention"] = out["intervention"].astype(str)
    out["intervention_class"] = out["intervention_class"].astype(str)

    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="Float64")

    canonical = out.loc[:, CANONICAL_COLUMNS].copy()
    validate(canonical)
    return canonical


def main() -> int:
    """Run the manifest finalizer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/run_manifest.csv")
    parser.add_argument("--out-csv", default="data/runs_template.csv")
    parser.add_argument("--out-parquet", default="runs.parquet")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    try:
        runs = manifest_to_runs(manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(csv_path, index=False)
    save_runs(runs, args.out_parquet)
    print(f"wrote {len(runs)} completed runs to {csv_path}")
    print(f"wrote parquet table to {Path(args.out_parquet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
