"""Run local integrity checks for the ASLA no-W&B HPC scaffold."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ARRAY_DIRECTIVE = re.compile(r"^#SBATCH\s+--array=(\d+)-(\d+)(?:%(\d+))?\s*$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a preflight command and fail with its captured output."""

    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def _write_mock_trainer(path: Path) -> None:
    """Write a temporary plumbing-only trainer used by the local smoke check."""

    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--recipe', required=True)",
                "parser.add_argument('--compute', required=True)",
                "parser.add_argument('--seed', required=True)",
                "parser.add_argument('--output-dir', required=True)",
                "parser.add_argument('--metrics-json', required=True)",
                "args = parser.parse_args()",
                "Path(args.output_dir).mkdir(parents=True, exist_ok=True)",
                "payload = {'bpb': 1.234, 'notes': 'plumbing-only mock; not audit data'}",
                "Path(args.metrics_json).write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def validate_manifest_array(manifest_path: str | Path, slurm_path: str | Path) -> int:
    """Ensure unique safe manifest identities and exact one-based array coverage."""

    manifest = pd.read_csv(manifest_path)
    required = {"run_id", "intervention", "compute", "seed", "status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("manifest has no run rows")
    run_ids = manifest["run_id"].astype(str)
    if run_ids.duplicated().any():
        raise ValueError("manifest run_id values must be unique")
    unsafe = sorted(value for value in run_ids if SAFE_RUN_ID.fullmatch(value) is None)
    if unsafe:
        raise ValueError(f"manifest contains unsafe run_id values: {unsafe[:5]}")

    directive: tuple[int, int, int | None] | None = None
    for line in Path(slurm_path).read_text(encoding="utf-8").splitlines():
        match = ARRAY_DIRECTIVE.match(line.strip())
        if match:
            directive = (
                int(match.group(1)),
                int(match.group(2)),
                None if match.group(3) is None else int(match.group(3)),
            )
            break
    if directive is None:
        raise ValueError(f"SLURM template has no supported #SBATCH --array directive: {slurm_path}")
    start, end, concurrency = directive
    if start != 1 or end != len(manifest):
        raise ValueError(f"SLURM array must be 1-{len(manifest)} for the current manifest; found {start}-{end}")
    if concurrency is not None and concurrency <= 0:
        raise ValueError("SLURM array concurrency must be positive")
    return len(manifest)


def main() -> int:
    """Run manifest/array, shell-syntax, rendering, and mock-plumbing checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/run_manifest.csv")
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--site-command-template", default="hpc/site_command.template")
    parser.add_argument("--train-command-template", default="hpc/train_command.template")
    parser.add_argument("--slurm-template", default="hpc/slurm_array_template.sh")
    args = parser.parse_args()

    try:
        row_count = validate_manifest_array(args.manifest, args.slurm_template)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"array covers all {row_count} manifest rows exactly once")
    _run(["bash", "-n", args.slurm_template])

    with tempfile.TemporaryDirectory(prefix="asla_hpc_preflight_") as temporary:
        root = Path(temporary)
        mock_trainer = root / "mock_train_eval.py"
        _write_mock_trainer(mock_trainer)
        env = os.environ.copy()
        env["ASLA_SITE_COMMAND_TEMPLATE"] = args.site_command_template
        env["ASLA_REAL_TRAIN_EVAL"] = str(mock_trainer)
        common = [
            sys.executable,
            "scripts/run_one_manifest_row.py",
            "--manifest",
            args.manifest,
            "--row",
            str(args.row),
            "--index-base",
            "1",
            "--command-template-file",
            args.train_command_template,
        ]
        _run([*common, "--output-dir", str(root / "dry")], env=env)
        _run([*common, "--output-dir", str(root / "smoke"), "--execute"], env=env)
        results = sorted((root / "smoke").glob("*/result.json"))
        if len(results) != 1:
            raise SystemExit(f"expected exactly one smoke result JSON, found {len(results)}")
        payload = json.loads(results[0].read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or "bpb" not in payload:
            raise SystemExit(f"invalid smoke result payload: {payload}")
        print("validated plumbing-only smoke result (not audit data)")

    print("hpc preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
