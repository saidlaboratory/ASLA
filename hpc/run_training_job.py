"""Site-adaptable training/eval adapter for one ASLA manifest row.

This script is intentionally not a trainer. It renders and launches the
site-specific command in ``hpc/site_train_command.template``, then validates
that the command wrote a result JSON with a finite measured BPB.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np

REQUIRED_RESULT_KEYS = ("bpb", "status")
IDENTITY_RESULT_KEYS = ("run_id", "intervention", "compute", "seed")


def _template_fields(template: str) -> set[str]:
    """Return named format fields used by a command template."""

    return {name for _, name, _, _ in Formatter().parse(template) if name}


def _load_template(path: str | Path) -> str:
    """Load a non-empty site command template."""

    template = Path(path).read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError(f"site command template is empty: {path}")
    return template


def render_site_command(template: str, values: dict[str, Any]) -> str:
    """Render a command with shell-quoted ``{name_q}`` placeholders."""

    rendered_values = {"overwrite_flag": "", **values}
    rendered_values.update({f"{name}_q": shlex.quote(str(value)) for name, value in values.items()})
    fields = _template_fields(template)
    missing = sorted(field for field in fields if field not in rendered_values)
    if missing:
        raise ValueError(f"site command template uses unknown placeholders: {missing}")
    return template.format(**rendered_values).strip()


def validate_result_json(
    path: str | Path,
    expected_identity: dict[str, Any] | None = None,
    *,
    require_identity: bool = False,
) -> dict[str, Any]:
    """Validate a completed result and optionally match its manifest identity."""

    result_path = Path(path)
    if not result_path.exists():
        raise ValueError(f"training command did not write required result file: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{result_path} is missing required keys: {missing}")
    bpb = float(payload["bpb"])
    if not np.isfinite(bpb) or bpb <= 0:
        raise ValueError(f"{result_path} has invalid or non-finite bpb: {payload['bpb']!r}")
    status = str(payload["status"]).lower().strip()
    if status not in {"complete", "completed", "done"}:
        raise ValueError(f"{result_path} status is not completed: {payload['status']!r}")
    payload["bpb"] = bpb
    payload["status"] = "completed"
    if require_identity or expected_identity is not None:
        identity_missing = [key for key in IDENTITY_RESULT_KEYS if key not in payload]
        if identity_missing:
            raise ValueError(f"{result_path} is missing run identity keys: {identity_missing}")
        payload["run_id"] = str(payload["run_id"])
        payload["intervention"] = str(payload["intervention"])
        if not payload["run_id"].strip() or not payload["intervention"].strip():
            raise ValueError(f"{result_path} has blank run identity")
        payload["compute"] = float(payload["compute"])
        seed_value = float(payload["seed"])
        if not np.isfinite(seed_value) or not seed_value.is_integer():
            raise ValueError(f"{result_path} has non-integer identity seed: {payload['seed']!r}")
        payload["seed"] = int(seed_value)
        if not np.isfinite(payload["compute"]) or payload["compute"] <= 0:
            raise ValueError(f"{result_path} has invalid identity compute: {payload['compute']!r}")
        if payload["seed"] < 0:
            raise ValueError(f"{result_path} has invalid identity seed: {payload['seed']!r}")
    for key in ("downstream", "params_n", "tokens_d"):
        if key in payload and payload[key] not in (None, ""):
            value = float(payload[key])
            if not np.isfinite(value) or (key in {"params_n", "tokens_d"} and value <= 0):
                raise ValueError(f"{result_path} has invalid {key}: {payload[key]!r}")
            payload[key] = value
    if expected_identity is not None:
        expected = {
            "run_id": str(expected_identity["run_id"]),
            "intervention": str(expected_identity["intervention"]),
            "compute": float(expected_identity["compute"]),
            "seed": int(expected_identity["seed"]),
        }
        mismatches = {
            key: {"expected": expected[key], "observed": payload[key]}
            for key in IDENTITY_RESULT_KEYS
            if (not np.isclose(payload[key], expected[key]) if key == "compute" else payload[key] != expected[key])
        }
        if mismatches:
            raise ValueError(f"{result_path} run identity does not match manifest row: {mismatches}")
    return payload


def main() -> int:
    """Run one site training command and validate its result JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site-command-template", default="hpc/site_train_command.template")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.overwrite_output:
        raise SystemExit(f"refusing to reuse existing result file: {result_path}")
    values = {
        "run_id": args.run_id,
        "intervention": args.intervention,
        "compute": args.compute,
        "compute_g": f"{float(args.compute):g}",
        "seed": args.seed,
        "seed_int": int(args.seed),
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "overwrite_flag": "--overwrite-result" if args.overwrite_output else "",
    }
    expected_identity = {key: values[key] for key in IDENTITY_RESULT_KEYS}

    try:
        site_template = _load_template(args.site_command_template)
        command = render_site_command(site_template, values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    (output_dir / "site_command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(command)
        print(f"dry run only; command saved to {output_dir / 'site_command.txt'}")
        return 0

    if result_path.exists() and args.overwrite_output:
        result_path.unlink()

    completed = subprocess.run(command, shell=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    try:
        validate_result_json(result_path, expected_identity, require_identity=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"validated result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
