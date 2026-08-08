"""Merge per-run HPC result JSON files back into an ASLA manifest."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from hpc.run_training_job import validate_result_json

RESULT_COLUMNS = ("bpb", "downstream", "params_n", "tokens_d", "notes")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def _read_result(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Read and validate one result JSON file."""

    return validate_result_json(path, expected_identity, require_identity=True)


def _write_csv_atomically(df: pd.DataFrame, path: Path) -> None:
    """Write CSV through a same-directory temporary file and atomic replace."""

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


def collect_results(
    manifest: pd.DataFrame,
    results_dir: str | Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return an updated manifest plus collected and missing run IDs."""

    required = ("run_id", "intervention", "compute", "seed")
    missing_columns = [column for column in required if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"manifest is missing result identity columns: {missing_columns}")
    duplicate_ids = manifest["run_id"].astype(str).duplicated(keep=False)
    if bool(duplicate_ids.any()):
        values = sorted(manifest.loc[duplicate_ids, "run_id"].astype(str).unique().tolist())
        raise ValueError(f"manifest run_id values must be unique; duplicates: {values[:5]}")
    out = manifest.copy()
    collected: list[str] = []
    missing: list[str] = []
    root = Path(results_dir)

    for idx, row in out.iterrows():
        run_id = str(row["run_id"])
        if SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(f"unsafe run_id in manifest: {run_id!r}")
        path = root / run_id / "result.json"
        if not path.exists():
            missing.append(run_id)
            continue
        existing_bpb = row.get("bpb", pd.NA)
        if not overwrite and not pd.isna(existing_bpb) and str(existing_bpb).strip() != "":
            raise ValueError(f"refusing to overwrite existing BPB for {run_id}; use --overwrite")

        expected_identity = {
            "run_id": run_id,
            "intervention": str(row["intervention"]),
            "compute": float(row["compute"]),
            "seed": int(row["seed"]),
        }
        payload = _read_result(path, expected_identity)
        out.at[idx, "status"] = "completed"
        for col in RESULT_COLUMNS:
            if col in payload:
                out.at[idx, col] = payload[col]
        collected.append(run_id)

    return out, collected, missing


def main() -> int:
    """Run result collection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/run_manifest.csv")
    parser.add_argument("--results-dir", default="results/hpc")
    parser.add_argument("--out", default="data/run_manifest.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    try:
        updated, collected, missing = collect_results(manifest, args.results_dir, overwrite=args.overwrite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if missing and not args.allow_missing:
        examples = missing[:5]
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise SystemExit(f"missing result files for {examples}{suffix}")

    out = Path(args.out)
    _write_csv_atomically(updated, out)
    print(f"collected {len(collected)} result files")
    print(f"missing {len(missing)} result files")
    print(f"wrote updated manifest to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
