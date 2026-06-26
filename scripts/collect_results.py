"""Merge per-run HPC result JSON files back into an ASLA manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULT_COLUMNS = ("bpb", "downstream", "params_n", "tokens_d", "notes")


def _read_result(path: Path) -> dict[str, Any]:
    """Read and validate one result JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "bpb" not in payload:
        raise ValueError(f"{path} is missing required key 'bpb'")
    bpb = float(payload["bpb"])
    if not np.isfinite(bpb):
        raise ValueError(f"{path} has non-finite bpb: {payload['bpb']!r}")
    payload["bpb"] = bpb
    for key in ("downstream", "params_n", "tokens_d"):
        if key in payload and payload[key] not in (None, ""):
            value = float(payload[key])
            if not np.isfinite(value):
                raise ValueError(f"{path} has non-finite {key}: {payload[key]!r}")
            payload[key] = value
    return payload


def collect_results(
    manifest: pd.DataFrame,
    results_dir: str | Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return an updated manifest plus collected and missing run IDs."""

    if "run_id" not in manifest.columns:
        raise ValueError("manifest must contain a run_id column")
    out = manifest.copy()
    collected: list[str] = []
    missing: list[str] = []
    root = Path(results_dir)

    for idx, row in out.iterrows():
        run_id = str(row["run_id"])
        path = root / run_id / "result.json"
        if not path.exists():
            missing.append(run_id)
            continue
        existing_bpb = row.get("bpb", pd.NA)
        if not overwrite and not pd.isna(existing_bpb) and str(existing_bpb).strip() != "":
            raise ValueError(f"refusing to overwrite existing BPB for {run_id}; use --overwrite")

        payload = _read_result(path)
        out.at[idx, "status"] = str(payload.get("status", "completed"))
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
    out.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(out, index=False)
    print(f"collected {len(collected)} result files")
    print(f"missing {len(missing)} result files")
    print(f"wrote updated manifest to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
