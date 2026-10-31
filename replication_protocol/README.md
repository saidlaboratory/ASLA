# Replication protocol (v2.4, frozen 2026-10-31)

Pre-registration: `../PREDICTIONS_REPLICATION_PROTOCOL.md` (machine-readable: `preregistration.json`).

| File | Contents |
|---|---|
| `recipes.json` | Selection rule (E1 to E6, salted-crc32 rule), the search procedure, the 55 candidates checked against the HF API with licence, gate and revision SHA, the 10 selected corpora and 3 ordered reserves |
| `ladder.json` | DataDecide's setup at the 8 sizes, with sources |
| `configs/` | One config per run (240), written by `scripts/make_replication_configs.py` |
| `eval_set.jsonl` | The evaluation set (DataDecide's in-loop c4_en-validation file): 2,099 document ids with text hashes, used for decontamination |
| `eval_13grams.npz` | 64-bit hashes of the evaluation set's normalized 13-grams, each with its evaluation document, used by the 13-gram decontamination pass |
| `olmo_patch/` | The pinned OLMo evaluator patch (`olmo_per_document.patch`, applies to allenai/OLMo@7094aab) and the module it adds (`per_document.py`), which log per-document summed loss and token counts |
| `budget.json` | Runs, total tokens and FLOPs, GPU-hours |
| `MANIFEST.sha256` | SHA-256 of every protocol file at the freeze |

## After training

Apply `olmo_patch/olmo_per_document.patch` to OLMo@7094aab and set each run's `ASLA_PER_DOCUMENT_LOG_DIR` as its config says. Build the runs table from the logs with `scripts/rescore_replication.py build --logs DIR --out runs.csv`. Otherwise, collect one row per run into a CSV with the columns `recipe, size, seed, params_n, tokens_d, c4_en_bits_per_token`.
Use the recipe ids from `recipes.json`, the size labels and seeds from `ladder.json`, and bits per token on the
evaluation set in `ladder.json` at the final checkpoint.

Then run:

    .venv/bin/python scripts/analyze_replication.py runs.csv --out verdict.json

For the v2.3 sensitivity check, derive the exclusion list with `scripts/rescore_replication.py exclusions --reports DECONTAM_REPORTS --out excl.json`, then add `--per-document-logs DIR --sensitivity-exclude excl.json`. The script refuses the sensitivity check if any run lacks per-document logs.

Before tokenizing each corpus, run `scripts/decontaminate.py --corpus <id> --report <file>` on it. It applies the exact-hash pass, then the 13-gram pass, and reports removals per pass with the evaluation ids hit.

The analysis script reports a numerical-stability diagnostic alongside the verdict. It refuses to run if any frozen analysis module or protocol file has changed. Check the whole protocol
with `shasum -a 256 -c replication_protocol/MANIFEST.sha256`.

`scripts/dry_run_replication.py` exercises the whole path on simulator output.
