"""Site-adaptable training/eval adapter for one ASLA manifest row.

This script is intentionally not a trainer. It renders and launches the
site-specific command in ``hpc/site_train_command.template``, then validates
that the command wrote a result JSON with a finite measured BPB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np


REQUIRED_RESULT_KEYS = ("bpb", "status")


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
    """Render the site-specific training command."""

    fields = _template_fields(template)
    missing = sorted(field for field in fields if field not in values)
    if missing:
        raise ValueError(f"site command template uses unknown placeholders: {missing}")
    return template.format(**values)


def validate_result_json(path: str | Path) -> dict[str, Any]:
    """Validate a completed run result file and return its payload."""

    result_path = Path(path)
    if not result_path.exists():
        raise ValueError(f"training command did not write required result file: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{result_path} is missing required keys: {missing}")
    bpb = float(payload["bpb"])
    if not np.isfinite(bpb):
        raise ValueError(f"{result_path} has non-finite bpb: {payload['bpb']!r}")
    status = str(payload["status"]).lower().strip()
    if status not in {"complete", "completed", "done"}:
        raise ValueError(f"{result_path} status is not completed: {payload['status']!r}")
    payload["bpb"] = bpb
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    values = {
        "run_id": args.run_id,
        "intervention": args.intervention,
        "compute": args.compute,
        "compute_g": f"{float(args.compute):g}",
        "seed": args.seed,
        "seed_int": int(args.seed),
        "output_dir": str(output_dir),
        "result_path": str(result_path),
    }

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

    completed = subprocess.run(command, shell=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    try:
        validate_result_json(result_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"validated result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
