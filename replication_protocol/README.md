# Replication protocol (v2, frozen 2026-10-31)

Pre-registration: `../PREDICTIONS_REPLICATION_PROTOCOL.md` (machine-readable: `preregistration.json`).

| File | Contents |
|---|---|
| `recipes.json` | Selection rule (E1 to E6, salted-crc32 rule), the search procedure, the 55 candidates checked against the HF API with licence, gate and revision SHA, the 10 selected corpora and 3 ordered reserves |
| `ladder.json` | DataDecide's setup at the 8 sizes, with sources |
| `configs/` | One config per run (240), written by `scripts/make_replication_configs.py` |
| `budget.json` | Runs, total tokens and FLOPs, GPU-hours |
| `MANIFEST.sha256` | SHA-256 of every protocol file at the freeze |

## After training

Collect one row per run into a CSV with the columns `recipe, size, seed, params_n, tokens_d, c4_en_bits_per_token`.
Use the recipe ids from `recipes.json`, the size labels and seeds from `ladder.json`, and bits per token on the
evaluation set in `ladder.json` at the final checkpoint.

Then run:

    .venv/bin/python scripts/analyze_replication.py runs.csv --out verdict.json

The script refuses to run if any frozen analysis module or protocol file has changed. Check the whole protocol
with `shasum -a 256 -c replication_protocol/MANIFEST.sha256`.

`scripts/dry_run_replication.py` exercises the whole path on simulator output.
