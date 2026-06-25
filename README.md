# ASLA: Algorithm-Selection Leaderboard Audit

ASLA audits whether small-compute scaling-law projections select the same
training intervention that measured target-budget runs would select. The initial
package runs offline on deterministic synthetic scenarios or on existing
Weights & Biases runs harvested into a canonical parquet table.

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
