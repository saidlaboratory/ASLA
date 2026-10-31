# Pre-registration: the replication protocol

Fixed on 2026-10-31, before any training. Nothing has been trained. The machine-readable copy is
`replication_protocol/preregistration.json`. If it and this file disagree, the JSON governs.

## Design

The cheapest design from `results/replication/untuned_spec.json`:

- 10 recipes;
- 8 DataDecide sizes: 4M, 6M, 14M, 20M, 60M, 90M, 150M and 300M;
- 3 seeds: 6198, 14 and 15, DataDecide's "default", "small aux 2" and "small aux 3";
- 240 runs in total.

### Training

Each run follows DataDecide's own setup (`replication_protocol/ladder.json`). The sources are:

- DataDecide's released N, D and steps;
- the paper's Table 2;
- the OLMo ladder code on the DataDecide branch.

The settings are:

- 5xC tokens;
- tokenizer `allenai_gpt-neox-olmo-dolma-v1_5`;
- sequence length 2048;
- AdamW (0.9, 0.95), weight decay 0.05;
- a cosine schedule to 1% of peak, with N tokens of warmup;
- the seed controls both initialisation and data order.

`scripts/make_replication_configs.py` writes one config per run to `replication_protocol/configs/`. Its budget
(`replication_protocol/budget.json`) is:

- 1.92T tokens;
- 2.34e21 FLOPs;
- about 1,645 H100-hours at 40% utilisation.

### Evaluation

The evaluation metric is C4-en bits per token at the final checkpoint, on Paloma's `c4_en` validation split.
It is the same set for every run. Exact duplicates of Paloma `c4_en` documents are removed from every training
corpus.

One point is unverified: whether DataDecide's in-loop C4 file is exactly this split. Every replication run uses
Paloma, so the comparison within the replication does not depend on it.

## Recipe selection rule

The full rule and the per-candidate record are in `replication_protocol/recipes.json`.

### Eligibility

A candidate is a public Hugging Face corpus that meets all six conditions:

- **E1, licence:** it has an open licence, or its card states open or public-domain terms. Text crawled under
  Common Crawl's terms of use also qualifies.
- **E2, access:** the canonical repository is downloadable without manual approval.
- **E3, size:** it has an English portion of at least 30B tokens, selectable from metadata. This is enough for
  the 300M run without repeating data.
- **E4, distinctness:** it is not one of DataDecide's corpora, and at most 20% of it comes from those corpora or
  the pools they were filtered from. This is judged from the card's stated composition.
- **E5, scope:** it is general-domain.
- **E6, family:** it is the only corpus taken from its release family.

### Selection

If more than ten corpora are eligible, the ten with the smallest crc32 of `ASLA-replication-2026-10-31:<hf_id>`
are taken.

### Candidates

Twenty-seven candidates were checked against the HF API on 2026-10-31. Ten are eligible, so all ten are selected
and there are no reserves:

| Recipe | Corpus |
|---|---|
| madlad_400_en | MADLAD-400 en |
| comma_v0_1 | Comma v0.1 training set |
| open_license_corpus | Open License Corpus |
| finepdfs | FinePDFs eng_Latn |
| redpajama_v2 | RedPajama-V2 en |
| hplt_v2_en | HPLT 2.0 eng_Latn |
| txt360 | TxT360 |
| matrix_cc_en | Matrix cc_en |
| redpajama_1t | RedPajama-1T |
| common_corpus | Common Corpus English |

Each corpus is pinned to its HF revision SHA.

If a selected corpus becomes unavailable before training, the replication runs with the rest. Power is then
recomputed with the frozen simulator before training, and both facts are recorded.

## Analysis (frozen)

Run `scripts/analyze_replication.py runs.csv`. It needs one row per run, with these columns:

- recipe;
- size;
- seed;
- params_n;
- tokens_d;
- c4_en_bits_per_token.

The script checks the SHA-256 of the five frozen modules (confound, fixed_target, moderation, target_scoring,
models) and refuses to run on any change. The analysis then:

- computes the fixed-target pooled slope with its leave-one-recipe-out jackknife SE;
- uses the targets 150M and 300M, each admitted only when at least half its recipe pairs are determined.

### Missing runs

A failed run is rerun once with the same seed. If cells are still missing, the analysis runs with
`--allow-incomplete`, which keeps only recipes with at least two seeds at every size. That run is reported as a
deviation.

## Primary criterion

The slope is positive and its interval, estimate ± 1.96 SE, excludes zero. The verdict takes one of four values:

- **replicates:** the interval lies above zero;
- **inconclusive:** the interval contains zero;
- **contradicts:** the interval lies below zero;
- **no test:** no target is admitted.

Power under the validated simulator is 0.91 for random candidates and 0.97 for fixed candidates.

## Secondary: magnitude

| Quantity | Value |
|---|---|
| Predicted slope | +21.8 points per decade |
| Monte Carlo interval | 21.0 to 22.7 |
| Range across recipe draws | +10.8 to +31.7 |
| Curve-implied value | +11.0 (not the prediction) |

This prediction assumes the new recipes resemble DataDecide's 25. That means they are drawn from the same
distribution of power-law parameters and recipe-specific misfit as the 25. Three selected corpora are
openly licensed multi-domain mixtures and differ most from DataDecide's web-heavy recipes: Comma, Common Corpus
and the Open License Corpus. A magnitude outside the range is reported as evidence against that assumption, not
against the sign. Magnitude is not part of the pass criterion.

## What the outcomes would mean

### Replicates

Projection's excess over single-scale ranking grows with lever arm at fixed targets on recipes outside
DataDecide. The claim then extends beyond DataDecide's recipe family to these ten corpora, under DataDecide's
training setup.

### Inconclusive

The simulator expects this in 9% (the spec's 150 worlds) to 14% (the dry run's 150 worlds) of cases even when the effect is real at the predicted size. On its own,
an inconclusive result does not refute the claim. Two follow-ups apply:

- If the estimate is positive but small, below +10.8, it suggests the effect is weaker on these recipes than on
  DataDecide's.
- Before any follow-up design is chosen, power is recomputed at the observed magnitude, and that recomputation is
  reported.

An inconclusive result is not evidence that the effect is absent.

### Contradicts

The fixed-target slope is negative on new recipes, a sign the validated simulator treats as very unlikely under
its model. It would mean the DataDecide finding does not transfer. In that case:

- The claim would have to be restricted to DataDecide's recipes.
- The simulator's recipe model, a Gaussian over DataDecide's own recipes, would be refuted as a description of
  other corpora.

### No test

If no target has half its pairs determined, the ten recipes are too close in loss to rank at 150M or 300M. The
replication does not test the claim. It would point to a recipe set chosen for spread, a choice that would have to
be pre-registered separately.

## Dry run

`scripts/dry_run_replication.py` runs the full pipeline (configs, runs table, analysis entry) on simulator
outputs. It writes `results/replication/protocol_dry_run.json`. Dry runs train nothing.
