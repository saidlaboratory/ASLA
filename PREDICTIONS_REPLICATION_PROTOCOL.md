# Pre-registration: the replication protocol

Protocol v2.4, frozen on 2026-10-31, before any training. Nothing has been trained. The v2 to v2.4 changes are listed at the
end of this file. The machine-readable copy is
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

The evaluation metric is C4-en bits per token at the final checkpoint, on DataDecide's in-loop `c4_en` validation
file (SHA-256 c37d6b11...). It is the same set for every run. Its 2,099 documents are listed by id and text hash
in `replication_protocol/eval_set.jsonl`.

**Decontamination (v2.2)** runs before tokenization, in `scripts/decontaminate.py`, as two passes:

1. **Exact.** A training document whose NFC-normalized text hash equals an evaluation document's is removed.
2. **13-gram.** A remaining training document that shares at least one 13-gram with any evaluation document is
   removed.
   - Text is normalized first: NFC, casefold, and every character that is not a letter or digit becomes a
     space. It is then split on whitespace.
   - 13-grams are matched by 64-bit hash against `replication_protocol/eval_13grams.npz`, which holds 748,892
     distinct 13-grams.
   - The two evaluation documents shorter than 13 tokens are covered by the exact pass only.

Per corpus, the report gives the number of documents each pass removes and the evaluation ids each pass hit.

**C4 file check (v2):**

- All 2,099 documents of DataDecide's in-loop `c4_en-validation` file (public at olmo-data.org, SHA-256
  c37d6b11...) are C4-en validation documents: 2,097 byte-identical and 2 identical after Unicode NFC.
- Its metadata names Paloma `v3_small` c4_en/val source files.
- Document-for-document identity with HF Paloma is unverified, because Paloma is gated. This is provenance only:
  the evaluation set is the in-loop file, so every run shares DataDecide's own evaluation set.

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

Fifty-five candidates were checked against the HF API and dataset cards on 2026-10-31. The search covered:

- corpora known as public LM-pretraining corpora;
- the HF API listing of English text-generation datasets in each size category above 100M rows, top 100 by
  downloads.

Thirteen are eligible. The ten with the smallest keys are selected:

| Recipe | Corpus |
|---|---|
| madlad_400_en | MADLAD-400 en |
| comma_v0_1 | Comma v0.1 training set |
| open_license_corpus | Open License Corpus |
| txt360 | TxT360 (IFM/TxT360) |
| finepdfs | FinePDFs eng_Latn |
| redpajama_v2 | RedPajama-V2 en |
| mint_1t_html | MINT-1T HTML, text only |
| hplt_v2_en | HPLT 2.0 eng_Latn |
| omnicorpus_cc | OmniCorpus-CC, text only |
| cc100_en | CC-100 en |

The reserves, in replacement order, are matrix_cc_en, obelics and common_corpus. Each corpus is pinned to its HF
revision SHA.

If a selected corpus is unavailable or fails processing before training, it is replaced by the first unused
reserve. The replacement is recorded as a deviation, and power is recomputed with the frozen simulator before
training.

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

Power under the validated simulator, on the exact 6ND grid (v2 dry run, 150 worlds), is 0.85 ± 0.03 for random
candidates. v1 reported 0.91 random and 0.97 fixed on a rounded grid.

## Secondary: magnitude

| Quantity | Value |
|---|---|
| Predicted slope | +21.2 points per decade |
| Monte Carlo interval | 20.3 to 22.1 |
| Range across recipe draws | +8.2 to +31.9 |
| Curve-implied value | +11.0 (not the prediction) |

These are v2 values: the simulator on the exact 6ND grid, 150 worlds. v1 had +21.8, 21.0 to 22.7, and +10.8 to
+31.7.

This prediction assumes the new recipes resemble DataDecide's 25. That means they are drawn from the same
distribution of power-law parameters and recipe-specific misfit as the 25. Four selected corpora fit
that assumption least:

- Comma and the Open License Corpus are openly licensed multi-domain mixtures, which differ most from
  DataDecide's web-heavy recipes;
- MINT-1T HTML and OmniCorpus-CC are interleaved multimodal web corpora, of which only the text is used. A magnitude outside the range is reported as evidence against that assumption, not
against the sign. Magnitude is not part of the pass criterion.

## What the outcomes would mean

### Replicates

Projection's excess over single-scale ranking grows with lever arm at fixed targets on recipes outside
DataDecide. The claim then extends beyond DataDecide's recipe family to these ten corpora, under DataDecide's
training setup.

### Inconclusive

The v2 dry run expects this in 15% of cases (150 worlds) even when the effect is real at the predicted size. On its own,
an inconclusive result does not refute the claim. Two follow-ups apply:

- If the estimate is positive but small, below +8.2, it suggests the effect is weaker on these recipes than on
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

## v2 changes (2026-10-31)

- **Simulator compute.** The simulator uses exact log10(6ND) compute, matching the entry point. The dry run was
  rerun, and the power and magnitude figures above are its values.
- **Release families.** RedPajama-V2 and RedPajama-1T are one release family under E6. The salted crc32 rule keeps
  RedPajama-V2. The same within-family rule is applied to every family found.
- **Candidate search.** The search was extended under the same six conditions, from 27 to 55 candidates. There
  are 13 eligible and 3 reserves, and the replacement order is fixed above. The selection changes accordingly:
  - in: MINT-1T HTML, OmniCorpus-CC, CC-100;
  - moved to reserves: Matrix, Common Corpus;
  - dropped: RedPajama-1T.
- **C4 evaluation file.** It was checked by content against C4-en validation, as recorded above.
- **Within-regime secondary.** The conditional within-regime secondary is not added. Config steps do not
  contribute (PREDICTIONS_TASK_CONFIG_STEPS.md, outcome O1), and the slope would not be estimable on this ladder
  anyway.
- **Pins.** The analysis entry point pins the five frozen modules (unchanged) and, new in v2, `recipes.json`,
  `ladder.json` and `preregistration.json`. `replication_protocol/MANIFEST.sha256` lists every protocol file.

## v2.1 changes (2026-10-31)

- **Evaluation and decontamination target.** Both now name only DataDecide's in-loop `c4_en-validation` file.
  - `replication_protocol/eval_set.jsonl` lists its 2,099 documents by id, token span and SHA-256 of the exact and
    NFC text (`scripts/build_eval_set_manifest.py`, which checks the file's SHA-256).
  - `scripts/decontaminate.py` removes every training document whose NFC text hash matches the list, and reports
    the matched ids per corpus.
  - Every run config carries the eval-set path and hash in its `eval` and `decontamination` blocks.
  - Paloma appears only as provenance.
- **Numerical stability diagnostic (output only).** After the verdict is fixed, the entry point refits under 20
  independent relative perturbations of 1e-12 to every bpb value and each size's compute (seed 20261031). It
  reports the slope's min, max and SD, the largest deviation from the reported slope, and the number of draws
  whose verdict differs. It never changes the verdict, which uses the unperturbed inputs.
- **Pins.** The pins for `recipes.json`, `ladder.json` and `preregistration.json` are updated, and
  `MANIFEST.sha256` is regenerated.

## v2.2 changes (2026-10-31)

- **Decontamination.** The whole-document hash rule is replaced by two passes: an exact NFC-hash pass first, then
  removal of any training document sharing a 13-gram (over normalized whitespace tokens) with an evaluation
  document. This catches evaluation text embedded in longer documents, and near-duplicates that differ in
  whitespace, punctuation or case.
  - `replication_protocol/eval_13grams.npz` is the 13-gram index, built from the in-loop file by
    `scripts/build_eval_set_manifest.py`.
  - The per-corpus report counts removals by pass and lists the evaluation ids hit.
- **Updated files.** `ladder.json`, `recipes.json`, `preregistration.json` and every run config now name the new
  rule and the index's hash.
- **Pins.** The pins and `MANIFEST.sha256` are updated.

## v2.3 amendment (2026-10-31): decontamination sensitivity

This amendment adds a reporting rule and a sensitivity check. Nothing else changes.

### Trigger

A selected corpus triggers the check if either holds:

- it loses more than 3% of its documents to the 13-gram pass;
- it loses more than twice the median 13-gram loss share across the ten selected corpora.

The loss share is `removed_13gram / documents_in`, taken from `scripts/decontaminate.py`'s report.

### Report

Each triggering corpus is reported with its 13-gram loss share, the median, and the evaluation ids hit.

### Excluded documents

For each triggering corpus, the excluded documents are the smallest set of evaluation ids that accounts for more
than half of that corpus's 13-gram removals. The ids are taken in descending order of that corpus's removal
counts, with ties broken by id.

### Sensitivity analysis

- The primary analysis (`scripts/analyze_replication.py`, unchanged) is rerun with those documents excluded from
  scoring.
- Every run's `c4_en_bits_per_token` is recomputed on the evaluation set minus those documents, applying the same
  exclusion to all runs.
- There is one rerun per triggering corpus. If more than one corpus triggers, there is also one rerun with the
  union of the excluded documents.

### Logging requirement

So that this rescore is possible, every run records its summed token loss and token count for each evaluation
document at the final checkpoint.

### Primary verdict

The primary verdict still comes from the full evaluation set. The sensitivity results are reported beside it and
do not replace it.

## v2.4 changes (2026-10-31): per-document logging enforced where runs are produced

### Finding

The OLMo DataDecide branch (commit 7094aab) cannot log per-document loss. There is no config option for it, for
three reasons:

- `Trainer.eval_batch` reduces the per-token loss to a per-instance mean.
- The `Evaluator` keeps only a running mean per dataset label.
- Evaluation instances are fixed 2048-token chunks of the concatenated file, with no document boundaries.

### Evaluator patch

`replication_protocol/olmo_patch/olmo_per_document.patch` is pinned in the protocol, and every run config
requires it by SHA-256. It applies cleanly to 7094aab (checked with `git apply`), and the patched files compile.
It changes four things:

- **Chunk index.** The dataset adds each instance's chunk index to its metadata.
- **Per-token loss.** `eval_batch` keeps the per-token loss and label mask on the batch. The reported metric is
  unchanged.
- **Per-document rows.** The evaluator attributes each predicted token to its evaluation document, using the
  c4_en metadata spans. After every evaluation it writes one row per chunk and document: summed loss and token
  count, one file per rank.
- **Ladder hook.** `scripts/ladder.py` turns this on when `ASLA_PER_DOCUMENT_LOG_DIR` is set. Each config sets
  that variable to the run's own log directory and names the final-step log file.

### Rescore

`scripts/rescore_replication.py` works as follows:

1. It reads the final-step logs.
2. It keeps each chunk once across ranks, taking the lowest rank that scored it. This matters because which
   chunks get scored depends on world size, through `drop_last`.
3. It drops the excluded documents.
4. It computes bits per token as summed loss over summed tokens, divided by ln 2.

With no exclusion, this is the primary `c4_en_bits_per_token`: the runs table is built from the logs. Its
`exclusions` command applies the v2.3 rule to decontamination reports.

### Refusal

`scripts/analyze_replication.py --per-document-logs DIR --sensitivity-exclude FILE` refuses to run the
sensitivity analysis if any run in the runs table lacks final-checkpoint logs. The analysis itself, `analyse()`,
and the primary verdict are unchanged.

### Dry run

The dry run (`scripts/dry_run_rescore.py`, recorded in `results/replication/rescore_dry_run.json`) uses
synthetic logs for all 240 runs.

- **Empty exclusion.** It reproduces the primary bits per token bitwise.
- **Planted exclusion.** It changes exactly the runs that scored the planted documents, and leaves the others
  bitwise identical.
- **v2.3 rule.** It recovers the planted documents.
- **Missing logs.** The entry point refuses when one run's logs are missing.

### Pins

The entry point now also pins the rescore script and both patch files.
