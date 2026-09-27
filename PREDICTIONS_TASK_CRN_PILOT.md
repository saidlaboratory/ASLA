# Pre-registration: copied-initialisation CRN pilot (4 learning rates x 4 seed indices)

Written 2026-09-27, before any pilot run. The pilot runs on the user's RTX 4080,
not on this laptop; here it is only built and checked for correctness at toy
size on CPU.

## Framing (also in `results/crn/novelty.json`)

CRN for hyperparameter selection is established (Pearce, Poloczek & Branke,
Operations Research 2022). Seed pairing resolves otherwise inconclusive
differences in another domain (Sharma, arXiv:2512.24145). A positive pilot would
measure CRN's benefit for language-model recipe selection and give a practical
design. It would not introduce a method.

## Design

- **Model.** GPT-style decoder: 6 layers, 6 heads, width 384, context 256,
  byte-level vocabulary of 256. About 10.7M parameters. No dropout, tied
  embeddings, explicit (non-fused) causal attention.
- **Data.** One fixed public corpus sample: the first training shard of
  TinyStories (`roneneldan/TinyStories`), as bytes.
  - The evaluation sample is fixed and separate: the first 512 sequences of 257
    bytes from the TinyStories validation split.
  - Checksums of both are recorded in the manifest.
- **Arms.** 4 learning rates: 0.5x, 0.75x, 1x and 1.5x a base of 1e-3 (AdamW,
  betas 0.9/0.95, weight decay 0.1 on matrices, 1% linear warmup, cosine decay
  to 10%, gradient clip 1.0, batch 64 x 256). Crossed with 4 seed indices,
  16 runs in all.
- **Common random numbers, per seed index.**
  - The initial weights are generated once, saved, and loaded into all four
    learning-rate runs. The parameters are copied explicitly, not reproduced by
    re-seeding.
  - The data order (a permutation of training chunks) is generated once, saved,
    and reused by all four runs.
- **Determinism.**
  - Deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic
    cuDNN with no benchmarking.
  - No data-loader workers; a fixed batch order from the saved permutation.
  - Each run is a fresh process.
- **Steps.** A 50-step timing probe runs on the 4080 before any real run. The
  step count is the probe-projected number for about 5 minutes per run, rounded
  down to a multiple of 500 and clamped to [2000, 10000]. It is recorded in the
  manifest and is the same for all 16 runs.
- **Checkpoints.** Evaluation loss at step 0 and at 10 evenly spaced checkpoints.
- **Correctness, checked on the laptop (CPU, toy size) before hand-off.**
  - An identical configuration reproduces the loss curve exactly.
  - Step-0 parameter hashes are identical across the four learning rates for a
    seed index.
  - The batch sequence is identical across the four learning rates for a seed
    index.

## Primary statistic and decision rule

- **Pairs.** For each of the 6 learning-rate pairs (a, b), at the final
  checkpoint, from the same fully crossed runs:
  - matched gaps: y_{a,s} - y_{b,s}, 4 values;
  - mismatched gaps: y_{a,s} - y_{b,t} for s != t, 12 values.
- **Ratio.** R_ab = Var(matched gaps) / Var(mismatched gaps), both with ddof 1.
- **Decision.** If the median of R_ab over the 6 pairs is at most 0.5 (a
  reduction of at least 50%), recommend a full experiment. Otherwise close the
  CRN program.
- **Four seeds only detect large effects, and this is intentional.**
  - The matched variance of one pair has 3 degrees of freedom. A single pair's
    ratio of 0.5 cannot be told from 1.
  - Taking the median over 6 correlated pairs helps only a little.
  - The pilot is sized to find out whether the benefit is large, the only case
    that justifies a full experiment.
- **Reported alongside** (not part of the decision):
  - a bootstrap interval, resampling seed indices jointly across pairs;
  - the resolvable fraction of pairs under each arm. Matched: paired t, 3 df.
    Mismatched: the calibrated procedure's raw Welch determination. Both at
    alpha = 0.05 with Bonferroni over the 6 pairs;
  - the ratio at every checkpoint.

## Predictions

- **PP1.** At the first checkpoint after step 0, the median ratio is at most 0.2.
  The runs share initialisation and data, so early seed deviations are nearly
  common to all four learning rates.
  - At step 0 every gap is exactly zero by construction, so the ratio is
    undefined there.
- **PP2.** The median ratio rises over training as trajectories under different
  learning rates diverge: Spearman correlation with checkpoint index of at least
  0.6.
- **PP3.** At the final checkpoint the median ratio is in [0.2, 0.8]. Whether it
  clears 0.5 is the open question; my point guess is 0.5.
- **PP4.** More pairs are resolvable under matched than mismatched analysis at
  the final checkpoint.
