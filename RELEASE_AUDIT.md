# Public-release audit (2026-10-27)

## The repository is already public

`https://github.com/saidlaboratory/ASLA` returns HTTP 200 to an unauthenticated
request. Everything below that is tracked or in history is therefore already
visible. Nothing here has been changed by this audit.

## Credentials

- **Result: none found.**
- **Scope:** every tracked file, and every added line in the full history
  (`git log --all -p`).
- **Patterns checked:**
  - provider keys: AWS keys, GitHub tokens, Hugging Face tokens, `sk-` keys,
    Slack tokens;
  - private-key blocks;
  - quoted assignments to api key, secret, password or token names;
  - a literal `WANDB_API_KEY=` value.
- **Hits, all false positives:**
  - the metric name `c4_en_bits_per_token`;
  - the usage line `WANDB_API_KEY=... python3 scripts/wandb_replicate_query.py`,
    which is a placeholder.
- **No credential files:** no `.env`, `.pem`, `.netrc` or similar file has ever
  been committed.

## Identities in commits

| Author | Email | Commits |
|---|---|---|
| Evan | evan.msaran@gmail.com | 187 |
| Duck Quang | quang.gateway@gmail.com | 7 |
| Quang Bui | 157789469+duckyquang@users.noreply.github.com | 1 |

- **Commit trailers:**
  - "Co-Authored-By" trailers name Claude models at noreply@anthropic.com.
  - One "Co-authored-by: Cursor <cursoragent@cursor.com>".
- **Rewriting history is out of scope.** Removing or changing these identities
  would mean rewriting published history, which is the owner's decision.

## Local paths in tracked results

- **What:** 18 occurrences of `/Users/Evan/...`, across five results files:
  - `results/adversarial/a2_dose_response.json`
  - `results/adversarial/floor_identifiability.json`
  - `results/common_mode/common_mode_gate.json`
  - `results/external/external_replication.json`
  - `results/target_scoring/target_scoring.json`
- **Exposure:** they reveal a local username and directory layout, nothing more.

## Data

Third-party data enters the repository in two ways.

**Tracked derived tables, committed under `data/`:**

| Files | Source | Licence |
|---|---|---|
| `datadecide_runs*.parquet` | `allenai/DataDecide-eval-results` (Hugging Face) | ODC-BY (per the dataset card) |
| `signal_and_noise_datadecide_c4_bpb.parquet` | `allenai/signal-and-noise` | Apache-2.0 (per the dataset card) |
| `fantastic_optimizers_*.parquet`, `fantastic_optimizers_cells.csv` | `kaiyue/optimizers` | Not determined: the Hugging Face API returned no record (possibly private, renamed or a different repo type); verify before relying on redistribution |

Both stated licences permit redistribution with attribution. The paper cites
both sources.

**Ignored, never committed, and should stay that way:**

- `data/raw/` (517 MB) holds raw downloads. They can be re-fetched by the
  `asla harvest-*` commands.
  - DataDecide and DataDecide-ppl: the ppl dataset card states no licence.
  - Signal and Noise.
  - PolyPythias.
  - Delphi.
  - ColPret: from `github.com/IBM/ColPret`.
  - compute-multipliers: from `huggingface.co/j23h67/compute-multipliers-checkpoints`.
  - The Marin speedrun board: its GitHub licence could not be read (the API
    returned Not Found for `marin-community/speedrun`).
  - The Fantastic Optimizers raw files.
- `data/cache/` (929 MB) holds resumable simulation caches. It also contains
  `speedrun_upstream`, a local clone of the Marin speedrun repository with an
  unpushed branch, `fix-power-law-convergence`. That branch is the owner's to
  propose upstream; it should not be published from here.

## Paper builds

- `scripts/build_paper.py` writes:
  - `build/icml/main.pdf`, the anonymized build;
  - `build/arxiv.tar.gz` and `build/arxiv/`, the preprint source.
- **Status:** `build/` is git-ignored.
- **Author block for the preprint:** it comes from `paper/authors_preprint.tex`.
  - The name and email are the git identity.
  - The affiliation "Independent" is a placeholder, not taken from the
    repository; confirm the author line before posting.
- **Paths the preprint cites:** the repository paths in its reproduction
  appendix resolve against this public repository.
