# Adapting The ASLA HPC Scaffold

The scaffold is intentionally conservative. It does not train anything until
you provide a real command and pass `--execute`.

## 1. Define The Command

Edit `hpc/train_command.template`. It must include these placeholders:

- `{run_id}`
- `{intervention}`
- either `{compute}` or `{compute_g}`
- either `{seed}` or `{seed_int}`

Useful optional placeholders:

- `{role}`
- `{output_dir}`
- `{row_index}`

Example shape:

```bash
python train.py \
  --recipe {intervention} \
  --compute {compute_g} \
  --seed {seed_int} \
  --output-dir {output_dir}/{run_id}
```

Your training command should write:

```text
results/hpc/<run_id>/result.json
```

with at least:

```json
{
  "bpb": 1.234,
  "status": "completed"
}
```

Optional keys are `downstream`, `params_n`, `tokens_d`, and `notes`.

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
sbatch hpc/slurm_array_template.sh
```

Inspect `logs/` and `results/hpc/*/command.txt`.

## 4. Execute

After the dry-run commands look correct, add `--execute` to the
`python scripts/run_one_manifest_row.py` call in your SLURM script.

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
