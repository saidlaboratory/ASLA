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
- `metric_name`: what `bpb` holds when a source has no bits-per-byte metric
  (must be a single value per table).
- `tuning_quality`: nullable note on how hyperparameters were tuned at scale.

## Harvested public tables

- `datadecide_runs.parquet` (C4-EN bits per token), `datadecide_runs_olmes_macro_error.parquet`,
  `datadecide_runs_olmes_correct_prob_per_char.parquet`: 25 recipes x 14 scales x 3 seeds
  from `allenai/DataDecide-*` (`asla harvest-datadecide`). Extra columns: `scale_label`,
  `step`, `tokens_per_param`, `seed_label`. 750M is off the 5xC trajectory; exclude it explicitly.
- `fantastic_optimizers_*.parquet`: single-seed ladders from the released Fantastic
  Optimizers `result.json` files (`asla harvest-fantastic-optimizers`); `fantastic_optimizers_cells.csv`
  is the raw grid (not a runs table). See the module docstrings under `asla/data/sources/`.
- `signal_and_noise_datadecide_c4_bpb.parquet`: Paloma C4-EN bits per byte for the final
  checkpoint of 25 recipes x 9 scales from `allenai/signal-and-noise` (`asla harvest-signal-and-noise`);
  one evaluated run per cell.
- `raw/` holds the downloaded artifacts and is git-ignored.

Minimum useful audit design:

- At least 3 interventions.
- At least 3 fitting budgets below the target for every intervention.
- At most one intermediate budget reserved for gate escalation.
- Target-budget rows for every intervention.
- Ideally 3 or more seeds in every intervention/budget cell.

Build and check the table:

```bash
python scripts/prepare_runs_table.py --csv data/runs_template.csv --out runs.parquet
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target YOUR_TARGET_COMPUTE --budgets YOUR_FIT_BUDGETS --intermediate-budget YOUR_INTERMEDIATE_COMPUTE --estimand YOUR_ESTIMAND --fast --out runs_audit.json
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

Replace `1`, `2`, `4`, `8`, and `16` with your real compute budgets. Keep
exactly one row marked `target`, at least three rows marked `fit`, and at most
one row marked `intermediate`. The intermediate budget stays out of projection
fitting.

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
python scripts/check_runs_coverage.py --csv data/runs_template.csv --target YOUR_TARGET_COMPUTE --fit-budgets YOUR_FIT_BUDGETS --intermediate-budget YOUR_INTERMEDIATE_COMPUTE
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target YOUR_TARGET_COMPUTE --budgets YOUR_FIT_BUDGETS --intermediate-budget YOUR_INTERMEDIATE_COMPUTE --estimand YOUR_ESTIMAND --fast --out runs_audit.json
```

The finalizer refuses pending rows and blank BPB values.
