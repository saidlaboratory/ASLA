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

The saturation demo should report a wrong projected winner, positive regret, a
detected crossover, and an explicit `fit is blind` message for the saturating
pair. The noise close-call demo should show the gate reducing mean regret and
wrong-pick rate versus plain projection.

## Validate and Audit Real Runs

Canonical run columns:

- `intervention` string
- `intervention_class` string
- `compute` float
- `seed` integer
- `bpb` float
- optional `downstream` float

```bash
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target 64 --out results/audit.json
asla figures --runs runs.parquet --out results/figures
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

