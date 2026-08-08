#!/usr/bin/env bash
#SBATCH --job-name=asla-grid
#SBATCH --array=1-45%12
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/asla-%A-%a.out
#SBATCH --error=logs/asla-%A-%a.err
#
# Uncomment and adapt on GPU clusters:
##SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/hpc

echo "host: $(hostname)"
echo "task: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "workdir: $PWD"

# Site environment setup.
#
# Preferred usage is to pass these at submit time, for example:
#   sbatch --export=ALL,ASLA_MODULES="miniconda/3 cuda/12.1",ASLA_CONDA_ENV=asla hpc/slurm_array_template.sh
#
# If your cluster does not use environment modules or conda, leave these unset
# and make sure `python` on PATH is the environment where `pip install -e .`
# was run.
if [[ -n "${ASLA_MODULES:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ASLA_MODULES was set, but the module command is unavailable" >&2
    exit 2
  fi
  module purge
  # shellcheck disable=SC2086
  module load ${ASLA_MODULES}
fi

if [[ -n "${ASLA_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ASLA_CONDA_ENV was set, but conda is unavailable" >&2
    exit 2
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${ASLA_CONDA_ENV}"
fi

python - <<'PY'
import pandas
import numpy
import asla
print("python environment ok")
PY

# Dry-run first:
#   sbatch hpc/slurm_array_template.sh
#
RUN_ARGS=(
  --manifest data/run_manifest.csv
  --row "${SLURM_ARRAY_TASK_ID:-1}"
  --index-base 1
  --command-template-file hpc/train_command.template
  --output-dir results/hpc
)

# Default is dry-run. Execute only after rendered commands look correct:
#   sbatch --export=ALL,ASLA_EXECUTE=1 hpc/slurm_array_template.sh
if [[ "${ASLA_EXECUTE:-0}" == "1" ]]; then
  RUN_ARGS+=(--execute)
fi
if [[ "${ASLA_OVERWRITE_OUTPUT:-0}" == "1" ]]; then
  RUN_ARGS+=(--overwrite-output)
fi

python scripts/run_one_manifest_row.py "${RUN_ARGS[@]}"
