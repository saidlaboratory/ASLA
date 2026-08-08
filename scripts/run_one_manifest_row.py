"""Run or print the command for one ASLA manifest row."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from string import Formatter

import pandas as pd

REQUIRED_PLACEHOLDERS = {"run_id", "intervention"}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def _template_fields(template: str) -> set[str]:
    """Return named format fields used by a command template."""

    return {name for _, name, _, _ in Formatter().parse(template) if name}


def load_template(args: argparse.Namespace) -> str:
    """Load a command template from a CLI argument or file."""

    if bool(args.command_template) == bool(args.command_template_file):
        raise ValueError("provide exactly one of --command-template or --command-template-file")
    if args.command_template_file:
        return Path(args.command_template_file).read_text(encoding="utf-8").strip()
    return str(args.command_template).strip()


def select_row(manifest: pd.DataFrame, row: int, index_base: int) -> pd.Series:
    """Return the requested manifest row using the configured index base."""

    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    idx = row - index_base
    if idx < 0 or idx >= len(manifest):
        raise IndexError(f"row {row} with index base {index_base} is outside manifest length {len(manifest)}")
    return manifest.iloc[idx]


def build_command(
    row: pd.Series,
    template: str,
    output_dir: str | Path,
    row_index: int,
    overwrite_output: bool = False,
) -> str:
    """Render a training command for a manifest row.

    Available placeholders are manifest columns plus ``output_dir`` and
    ``row_index``. The command is not executed unless the CLI receives
    ``--execute``.
    """

    fields = _template_fields(template)
    def _uses(name: str) -> bool:
        return name in fields or f"{name}_q" in fields

    missing = sorted(name for name in REQUIRED_PLACEHOLDERS if not _uses(name))
    if not _uses("compute") and not _uses("compute_g"):
        missing.append("compute or compute_g")
    if not _uses("seed") and not _uses("seed_int"):
        missing.append("seed or seed_int")
    if missing:
        raise ValueError(f"command template must include placeholders: {missing}")

    values = {key: "" if pd.isna(value) else value for key, value in row.to_dict().items()}
    run_id = str(values["run_id"])
    if SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"unsafe run_id in manifest: {run_id!r}")
    values["output_dir"] = str(output_dir)
    values["run_dir"] = str(Path(output_dir) / run_id)
    values["row_index"] = int(row_index)
    values["compute_g"] = f"{float(values['compute']):g}"
    values["seed_int"] = int(values["seed"])
    values["overwrite_flag"] = "--overwrite-output" if overwrite_output else ""
    values.update({f"{name}_q": shlex.quote(str(value)) for name, value in tuple(values.items())})
    return template.format(**values).strip()


def main() -> int:
    """Run the selected manifest row or print the command in dry-run mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/run_manifest.csv")
    parser.add_argument("--row", type=int, default=None, help="Manifest row index, usually $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--index-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--command-template")
    parser.add_argument("--command-template-file")
    parser.add_argument("--output-dir", default="results/hpc")
    parser.add_argument("--execute", action="store_true", help="Actually launch the rendered command.")
    parser.add_argument("--cwd", help="Working directory for the launched command.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow adapters to replace stale per-run metrics/results for a deliberate retry.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    row_arg = args.row
    if row_arg is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_id is None:
            raise SystemExit("--row is required when SLURM_ARRAY_TASK_ID is not set")
        row_arg = int(slurm_id)

    try:
        template = load_template(args)
        row = select_row(manifest, row_arg, args.index_base)
        if "run_id" not in manifest.columns or manifest["run_id"].astype(str).duplicated().any():
            raise ValueError("manifest run_id values must exist and be unique")
        command = build_command(row, template, args.output_dir, row_arg, overwrite_output=args.overwrite_output)
    except (KeyError, ValueError, IndexError) as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = Path(args.output_dir) / str(row["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    if not args.execute:
        print(command)
        print(f"dry run only; command saved to {run_dir / 'command.txt'}")
        return 0

    completed = subprocess.run(command, shell=True, cwd=args.cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
