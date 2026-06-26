# ASLA Runs Data

`runs_template.csv` is intentionally empty except for the header. Fill it with
real completed runs before converting it to parquet. Do not use the toy rows in
`example_runs.csv` for research claims.

Each row is one completed run at one compute budget:

```csv
intervention,intervention_class,compute,seed,bpb,downstream,params_n,tokens_d
adamw_lr1e-3,optimizer,1e18,0,1.230,,,
adamw_lr1e-3,optimizer,2e18,0,1.110,,,
adamw_lr1e-3,optimizer,4e18,0,1.020,,,
adamw_lr1e-3,optimizer,6.4e19,0,0.940,,,
```

Required columns:

- `intervention`: recipe name being compared.
- `intervention_class`: coarse category, such as optimizer, architecture, data,
  or schedule.
- `compute`: FLOPs or a consistent relative compute unit.
- `seed`: integer seed.
- `bpb`: C4-EN bits per byte; lower is better.

Optional columns:

- `downstream`: downstream evaluation score.
- `params_n`: parameter count, required only for `--fit-form chinchilla`.
- `tokens_d`: training token count, required only for `--fit-form chinchilla`.

Minimum useful audit design:

- At least 3 interventions.
- At least 3 fitting budgets below the target for every intervention.
- Target-budget rows for every intervention.
- Ideally 3 or more seeds in every intervention/budget cell.

Build and check the table:

```bash
python scripts/prepare_runs_table.py --csv data/runs_template.csv --out runs.parquet
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target YOUR_TARGET_COMPUTE --fast --out runs_audit.json
```

Use the exact `compute` value from the held-out target rows as
`YOUR_TARGET_COMPUTE`.

## No-W&B Workflow

Use this path when you are collecting runs manually or on an HPC system without
Weights & Biases.

1. Edit `interventions_template.csv`.

Replace `baseline`, `candidate_a`, and `candidate_b` with the recipes you want
to compare.

2. Edit `budgets_template.csv`.

Replace `1`, `2`, `4`, and `64` with your real compute budgets. Keep exactly
one row marked `target`; the others should be marked `fit`.

3. Generate a run manifest.

```bash
python scripts/make_run_manifest.py \
  --interventions data/interventions_template.csv \
  --budgets data/budgets_template.csv \
  --seeds 3 \
  --out data/run_manifest.csv
```

`data/run_manifest.csv` is your HPC checklist. Each row should become one
completed run. When a run finishes, copy the measured BPB into the `bpb` column.

4. Build the ASLA input table.

After the manifest has real BPB values for every completed run, copy the
completed rows into `runs_template.csv` or save them as another CSV with the same
schema, then run:

```bash
python scripts/finalize_run_manifest.py \
  --manifest data/run_manifest.csv \
  --out-csv data/runs_template.csv \
  --out-parquet runs.parquet
python scripts/check_runs_coverage.py --csv data/runs_template.csv --target YOUR_TARGET_COMPUTE
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target YOUR_TARGET_COMPUTE --fast --out runs_audit.json
```

The finalizer refuses pending rows and blank BPB values.
