# Adapting The ASLA HPC Scaffold

The scaffold is intentionally conservative. It does not train anything until
you provide a real command and pass `--execute`.

## 1. Define The Site Training Command

Do not edit `hpc/train_command.template` unless you are changing the wrapper.
For normal use, set `ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template` and
set `ASLA_REAL_TRAIN_EVAL` to your real training/eval Python script. The default
`hpc/site_train_command.template` calls `hpc/train_and_eval.py`, which runs that
site command and converts its measured metrics JSON into ASLA's canonical
`result.json`.

The site command template may use these placeholders. Shell commands should use
the quoted `_q` variants:

- `{run_id_q}`
- `{intervention_q}`
- `{compute_g_q}`
- `{seed_int_q}`
- `{output_dir_q}`
- `{metrics_json_q}` or `{result_path_q}`

Your site command template should have this shape:

```bash
python /path/to/your_train_and_eval.py \
  --recipe {intervention_q} \
  --compute {compute_g_q} \
  --seed {seed_int_q} \
  --output-dir {output_dir_q} \
  --metrics-json {metrics_json_q}
```

See `hpc/site_command.template` and `hpc/site_command.template.example`.

Your training command should write measured metrics to:

```text
{metrics_json}
```

with at least:

```json
{
  "bpb": 1.234
}
```

Optional keys are `downstream`, `params_n`, `tokens_d`, and `notes`.
The ASLA adapter writes and validates `{result_path}` before the job is
considered successful.

Before submission, run `python scripts/hpc_preflight.py`. It checks safe unique
manifest identities, exact SLURM-array coverage, shell syntax, rendering, and a
temporary plumbing-only smoke execution.

Example dry-run with a specific entrypoint:

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

## 2. Dry Run

On the login node:

```bash
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row 1 \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
```

This prints the rendered command and saves it to
`results/hpc/<run_id>/command.txt`.

## 3. SLURM Dry Run

Submit the template as-is first. It does not execute the rendered command.

```bash
sbatch --export=ALL,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py,ASLA_MODULES="YOUR_MODULES",ASLA_CONDA_ENV=YOUR_ENV hpc/slurm_array_template.sh
```

Inspect `logs/` and `results/hpc/*/command.txt`.

If your cluster does not use modules or conda, omit `ASLA_MODULES` and
`ASLA_CONDA_ENV`:

```bash
sbatch hpc/slurm_array_template.sh
```

## 4. Execute

After the dry-run commands look correct and `hpc/site_train_command.template`
calls your real trainer, submit with `ASLA_EXECUTE=1`:

```bash
sbatch --export=ALL,ASLA_EXECUTE=1,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py,ASLA_MODULES="YOUR_MODULES",ASLA_CONDA_ENV=YOUR_ENV hpc/slurm_array_template.sh
```

Existing metrics and results are refused. A deliberate retry after inspection
must opt in with `ASLA_OVERWRITE_OUTPUT=1`; it is never the default.

## 5. Collect Results

After jobs finish:

```bash
python scripts/collect_results.py \
  --manifest data/run_manifest.csv \
  --results-dir results/hpc \
  --out data/run_manifest.csv
```

Then finalize:

```bash
python scripts/finalize_run_manifest.py \
  --manifest data/run_manifest.csv \
  --out-csv data/runs_template.csv \
  --out-parquet runs.parquet
```
