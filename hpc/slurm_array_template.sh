#!/usr/bin/env bash
#SBATCH --job-name=asla-grid
#SBATCH --array=1-36
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/asla-%A-%a.out
#SBATCH --error=logs/asla-%A-%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/hpc

# Dry-run first:
#   sbatch hpc/slurm_array_template.sh
#
# After hpc/train_command.template contains your real command, add --execute
# below or pass it through a copied site-specific script.
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row "${SLURM_ARRAY_TASK_ID}" \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
