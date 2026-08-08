# No-W&B HPC Checklist

This project does not need W&B. Use `data/run_manifest.csv` as the source of
truth for the runs you need to execute.

## 1. On Your Mac

Confirm the planned grid:

```bash
python scripts/make_run_manifest.py --seeds 3 --out data/run_manifest.csv
```

Current default plan:

- `baseline_adamw`
- `adamw_lower_lr`
- `adamw_longer_warmup`
- fit budgets: `1`, `2`, `4`
- reserved intermediate budget: `8`
- target budget: `16`
- seeds: `0`, `1`, `2`

## 2. On HPC

For each row in `data/run_manifest.csv`, run one training/eval job using:

- `intervention`
- `compute`
- `seed`

The safest route is to adapt the scaffold in `hpc/`:

```bash
# Render one command without launching training.
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row 1 \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
```

Then set `ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template`, set
`ASLA_REAL_TRAIN_EVAL` to your real training/eval script, and adapt
`hpc/slurm_array_template.sh` for your cluster. See `hpc/ADAPTATION_CHECKLIST.md`.

The SLURM template defaults to dry-run. Submit it first without execution:

```bash
sbatch --export=ALL,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py hpc/slurm_array_template.sh
```

Only after inspecting `logs/` and `results/hpc/*/command.txt`, submit with:

```bash
sbatch --export=ALL,ASLA_EXECUTE=1,ASLA_SITE_COMMAND_TEMPLATE=hpc/site_command.template,ASLA_REAL_TRAIN_EVAL=/path/to/your_train_and_eval.py hpc/slurm_array_template.sh
```

When the job finishes, record:

- `status=completed`
- `bpb=<measured C4-EN bits-per-byte>`

Do not fill `bpb` with a placeholder. A row without measured BPB is not audit
data.

## 3. Back On Your Mac

After all rows are complete:

```bash
python scripts/collect_results.py \
  --manifest data/run_manifest.csv \
  --results-dir results/hpc \
  --out data/run_manifest.csv

python scripts/finalize_run_manifest.py \
  --manifest data/run_manifest.csv \
  --out-csv data/runs_template.csv \
  --out-parquet runs.parquet

python scripts/check_runs_coverage.py --csv data/runs_template.csv --target 16 --fit-budgets 1 2 4 --intermediate-budget 8
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target 16 --budgets 1 2 4 --intermediate-budget 8 --estimand pairwise_decisions --fast --out runs_audit.json
```

If the fast audit passes, run the full audit:

```bash
asla audit --runs runs.parquet --target 16 --budgets 1 2 4 --intermediate-budget 8 --estimand pairwise_decisions --out results/audit.json
```

## Safety Rules

- Keep `status=pending` until the run is actually done.
- Fill `bpb` only with measured evaluation results.
- Use the same compute scale for every row.
- Pass fitting budgets explicitly. Keep compute `8` reserved for the gate and
  compute `16` reserved for target truth.
