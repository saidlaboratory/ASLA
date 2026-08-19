"""ASLA training/eval entrypoint adapter.

This script has the exact CLI contract ASLA expects on HPC. It does not train a
model by itself. Instead, it runs a site-specific command template, then writes
the canonical ASLA result JSON from a measured metrics JSON. It fails loudly if
no finite BPB is produced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from hpc.run_training_job import render_site_command, validate_result_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_training_job import render_site_command, validate_result_json


def _lookup_metric(payload: dict[str, Any], key: str) -> Any:
    """Return a metric value by exact key or dotted nested path."""

    if key in payload:
        return payload[key]
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def result_from_metrics(
    metrics_path: str | Path,
    result_path: str | Path,
    bpb_key: str = "bpb",
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write canonical ASLA result JSON from a measured metrics JSON file."""

    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        raise ValueError(f"metrics JSON was not written: {metrics_file}")
    try:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read metrics JSON {metrics_file}: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"{metrics_file} must contain a JSON object")
    try:
        bpb = float(_lookup_metric(metrics, bpb_key))
    except KeyError as exc:
        raise ValueError(f"{metrics_file} is missing BPB key {bpb_key!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metrics_file} has non-numeric BPB at {bpb_key!r}") from exc
    if not np.isfinite(bpb) or bpb <= 0:
        raise ValueError(f"{metrics_file} has invalid or non-finite BPB at {bpb_key!r}: {bpb!r}")

    result: dict[str, Any] = {"bpb": bpb, "status": "completed"}
    if identity is not None:
        result.update(
            {
                "run_id": str(identity["run_id"]),
                "intervention": str(identity["intervention"]),
                "compute": float(identity["compute"]),
                "seed": int(identity["seed"]),
            }
        )
    for key in ("downstream", "params_n", "tokens_d"):
        if key in metrics and metrics[key] not in (None, ""):
            value = float(metrics[key])
            if not np.isfinite(value) or (key in {"params_n", "tokens_d"} and value <= 0):
                raise ValueError(f"{metrics_file} has invalid {key}: {metrics[key]!r}")
            result[key] = value
    if "notes" in metrics:
        result["notes"] = str(metrics["notes"])

    out = Path(result_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(out)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    validate_result_json(out, identity, require_identity=identity is not None)
    return result


def main() -> int:
    """Run the site command and write the canonical ASLA result JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--command-template-file", required=True)
    parser.add_argument("--metrics-json")
    parser.add_argument("--bpb-key", default="bpb")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-result", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_dir / "metrics.json"
    if metrics_path.resolve() == result_path.resolve():
        raise SystemExit("metrics JSON and result JSON must be different files")
    existing_outputs = [path for path in (result_path, metrics_path) if path.exists()]
    if existing_outputs and not args.overwrite_result:
        raise SystemExit(f"refusing to reuse existing output files: {[str(path) for path in existing_outputs]}")
    if args.overwrite_result:
        for path in existing_outputs:
            path.unlink()

    values = {
        "run_id": args.run_id,
        "intervention": args.intervention,
        "compute": args.compute,
        "compute_g": f"{float(args.compute):g}",
        "seed": args.seed,
        "seed_int": int(args.seed),
        "output_dir": str(output_dir),
        "metrics_json": str(metrics_path),
        "result_json": str(result_path),
        "result_path": str(result_path),
    }
    identity = {key: values[key] for key in ("run_id", "intervention", "compute", "seed")}
    template = Path(args.command_template_file).read_text(encoding="utf-8").strip()
    if not template:
        raise SystemExit(f"command template is empty: {args.command_template_file}")
    try:
        command = render_site_command(template, values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    (output_dir / "train_command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "asla_args.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(command)
        print(f"dry run only; command saved to {output_dir / 'train_command.txt'}")
        return 0

    completed = subprocess.run(command, shell=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    if result_path.exists():
        validate_result_json(result_path, identity, require_identity=True)
    else:
        result_from_metrics(metrics_path, result_path, bpb_key=args.bpb_key, identity=identity)
    print(f"wrote result JSON: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
