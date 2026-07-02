# ASLA: Algorithm-Selection Leaderboard Audit

ASLA audits whether small-compute scaling-law projections select the same
training intervention that measured target-budget runs would select. The initial
package runs offline on deterministic synthetic scenarios or on existing run
tables converted into a canonical parquet table. W&B harvesting is optional.

## Install

```bash
pip install -e ".[test]"
```

Optional extras:

```bash
pip install -e ".[figures,wandb]"
```

## Demo

```bash
asla demo --scenario saturation_crossover
asla demo --scenario noise_close_call
```

Default demo and audit runs use paper-grade counts (`n_boot=1000`,
`n_trials=500`). For a quick local smoke test:

```bash
asla demo --scenario saturation_crossover --fast
asla demo --scenario noise_close_call --fast
```

The saturation demo should report a wrong projected winner, positive regret, a
detected crossover, and an explicit `fit is blind` message for the saturating
pair. The noise close-call demo should show the gate reducing mean regret and
wrong-pick rate versus plain projection. Demo and audit reports compare the
scaling-law projection ranker against the largest-single-scale baseline with
bootstrap confidence intervals.

## Validate and Audit Real Runs

Canonical run columns:

- `intervention` string
- `intervention_class` string
- `compute` float
- `seed` integer
- `bpb` float
- optional `downstream` float
- optional `params_n` float for Chinchilla two-axis fits
- optional `tokens_d` float for Chinchilla two-axis fits

```bash
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target 64 --out results/audit.json
asla figures --runs runs.parquet --out results/figures
```

Use `--fit-form compute_power_law` for leaderboard audits. Use
`--fit-form chinchilla` for controlled grids with `params_n` and `tokens_d`;
the command fails loudly if those columns are missing.

Fits are unit-invariant: `compute`, `params_n`, and `tokens_d` may be raw
counts (FLOPs, parameters, tokens) or consistent relative units. Inputs are
rescaled internally before optimization, so bounds never depend on the unit
convention.

## No-W&B Data Collection

On a laptop, create an HPC checklist:

```bash
python scripts/make_run_manifest.py \
  --interventions data/interventions_template.csv \
  --budgets data/budgets_template.csv \
  --seeds 3 \
  --out data/run_manifest.csv
```

Edit `data/interventions_template.csv` and `data/budgets_template.csv` before
using the manifest for real training. The generated manifest is not audit data;
it is a checklist for runs that still need measured BPB values.

After the runs finish and the BPB values are filled into a CSV:

```bash
python scripts/finalize_run_manifest.py \
  --manifest data/run_manifest.csv \
  --out-csv data/runs_template.csv \
  --out-parquet runs.parquet
python scripts/check_runs_coverage.py --csv data/runs_template.csv --target TARGET_COMPUTE
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target TARGET_COMPUTE --fast --out runs_audit.json
```

The finalizer refuses pending rows and blank BPB values, so incomplete manifests
cannot silently become audit data.

The optional HPC scaffold is in `hpc/`:

- `hpc/train_command.template`: replace with your real training command.
- `hpc/slurm_array_template.sh`: SLURM array template, dry-run by default.
- `hpc/ADAPTATION_CHECKLIST.md`: step-by-step cluster adaptation guide.

Dry-run one manifest row:

```bash
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row 1 \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
```

Collect per-run `result.json` files after HPC jobs finish:

```bash
python scripts/collect_results.py \
  --manifest data/run_manifest.csv \
  --results-dir results/hpc \
  --out data/run_manifest.csv
```

## Harvest W&B Runs

ASLA never guesses W&B field names. First inspect a finished run:

```bash
asla harvest --discover --entity-project ENTITY/PROJECT
```

Then provide an explicit JSON field map from schema column to logged W&B key:

```json
{
  "intervention": "intervention",
  "intervention_class": "intervention_class",
  "compute": "compute",
  "seed": "seed",
  "bpb": "eval/c4_en_bpb"
}
```

Harvest:

```bash
asla harvest --entity-project ENTITY/PROJECT --field-map field_map.json --out runs.parquet
```

## Tests

```bash
pytest
```
