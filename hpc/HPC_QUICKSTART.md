# HPC Quickstart

Run these checks before submitting real training jobs.

## 1. Local Preflight

From the repository root:

```bash
python scripts/hpc_preflight.py
```

This checks the SLURM shell syntax, renders one manifest row, and executes one
row against a temporary mock trainer in `/tmp`. It does not modify
`data/run_manifest.csv` and does not create audit data.

## 2. Connect Real Training Code

Your real training/eval command is referenced through:

```bash
ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py
ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template
```

The script named by `ASLA_REAL_TRAIN_EVAL` must accept:

```text
--recipe
--compute
--seed
--output-dir
--metrics-json
```

and write `{metrics_json}` with at least:

```json
{"bpb": 1.234}
```

That `1.234` is only a contract example. Do not copy it into the run
manifest.

## 3. Dry-Run One Row

```bash
ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template \
ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py \
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row 1 \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
```

## 4. Dry-Run SLURM

```bash
sbatch --export=ALL,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py hpc/slurm_array_template.sh
```

Inspect `logs/` and `results/hpc/*/command.txt`.

## 5. Execute

Only after the dry-run commands look correct:

```bash
sbatch --export=ALL,ASLA_EXECUTE=1,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py hpc/slurm_array_template.sh
```

For a deliberate retry after inspecting a failed run, add
`ASLA_OVERWRITE_OUTPUT=1`. This clears only that array row's stale
`metrics.json` and `result.json`; it is never enabled by default.

Retries refuse existing metrics and results. After confirming a run directory
contains only stale output from the same manifest row, opt in to replacement
with `ASLA_OVERWRITE_OUTPUT=1` on the retry submission.

## 6. Collect And Audit

```bash
python scripts/collect_results.py --manifest data/run_manifest.csv --results-dir results/hpc --out data/run_manifest.csv
python scripts/finalize_run_manifest.py --manifest data/run_manifest.csv --out-csv data/runs_template.csv --out-parquet runs.parquet
python scripts/check_runs_coverage.py --csv data/runs_template.csv --target 16 --fit-budgets 1 2 4 --intermediate-budget 8
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target 16 --budgets 1 2 4 --intermediate-budget 8 --estimand pairwise_decisions --fast --out runs_audit.json
```
