# Known-answer validation against DataDecide

Purpose: before trusting any new number from this pipeline, check that it
reproduces a decision-accuracy result someone else published on public data.
If it does, downstream numbers inherit that credibility; if it does not, the
pipeline has a bug that must be reported, not tuned away.

Run: `asla validate-known-answer --runs data/datadecide_runs_olmes_macro_error.parquet`
(also `pytest tests/test_known_answer_datadecide.py`). Report JSON:
`results/known_answer/datadecide_known_answer.json` (regenerated; not committed).

## The published number

DataDecide (Magnusson et al., ICML 2025, arXiv:2504.11393), abstract and
Figure 1: "the ranking of models at a single, small size (e.g., 150M
parameters) is a strong baseline for predicting best models at our larger
target scale (1B) (~80% of comparisons correct)". Their definition (Section
2.3, eq. 3): decision accuracy = fraction of the 300 unordered pairs of the 25
data recipes whose predicted order matches the observed order at 1B, where the
1B value is the OLMES 10-task macro-average ACCURACY averaged over 3 seeds.
Figure 1 plots, for each small scale, the average over three prediction
attempts (one per small-model seed), with the standard deviation shaded.

The value is approximate: it is stated as "~80%" and read from a figure, not
tabulated. **Acceptance band: [0.75, 0.85]** (±5 percentage points around
0.80), chosen before computing our number and encoded in
`asla.analysis.known_answer.PUBLISHED_SINGLE_SCALE_150M`.

## What we computed

Input: `data/datadecide_runs_olmes_macro_error.parquet`, harvested by
`asla harvest-datadecide --metric olmes_macro_error` from the released
`allenai/DataDecide-eval-results` macro-average table (`bpb` column = 1 -
macro-average ACCURACY; ranking on it is identical to ranking on accuracy).
Target = 1B at its last released checkpoint (step 69,369; 100B tokens); small
scales at the last checkpoint present for every recipe and seed.

Single-scale pairwise decision accuracy, 150M -> 1B, 300 pairs:

| protocol | computed | published | gap | in band |
|---|---|---|---|---|
| `seed_mean` (ASLA `single_scale_ranker`: rank by 3-seed mean at 150M) | **0.830** | ~0.80 | +0.030 | yes |
| `per_seed` (DataDecide Figure 1: one attempt per seed, averaged; sd over seeds 0.030) | **0.803** | ~0.80 | +0.003 | yes |

The full profile across scales (same JSON), `seed_mean` / `per_seed`:
4M 0.313/0.379, 6M 0.443/0.457, 8M 0.570/0.564, 10M 0.577/0.593,
14M 0.590/0.570, 16M 0.717/0.637, 20M 0.740/0.646, 60M 0.767/0.714,
90M 0.803/0.786, 150M 0.830/0.803, 300M 0.860/0.823, 530M 0.883/0.847,
750M 0.860/0.816 (750M is off-trajectory at ~45 tokens/param; listed for
completeness). The shape - accuracy rising from near-chance at 4M through
~80% at 150M and only slowly beyond - matches Figure 1 qualitatively.

**Outcome: reproduced.** Both protocols land inside the band, and the protocol
that matches the paper's exactly is within 0.3 points of the published value.

## Explaining the (small) gap

- `seed_mean` is 2.7 points above `per_seed`. Averaging three seeds before
  ranking removes some small-scale noise, so it is expected to be slightly
  higher; the paper's headline uses single-seed attempts. Both are reported
  because ASLA's audits use the seed-mean ranker.
- Checkpoint choice: we use the last checkpoint shared by all 75 (recipe,
  seed) cells at 150M (step 37,500, 97 tokens/param); DataDecide's code takes
  the maximum step per model. Any residual difference is at the level of one
  late checkpoint.
- Task subset: identical (the released `olmes_10_macro_avg` rows).

## Secondary comparison: our projection ranker versus their released scaling-law fits

DataDecide's released `scaling_law_fit` table carries `decision_acc` for each
of their scaling-law setups (read from the artifact by
`asla.data.sources.datadecide.load_published_scaling_law_decision_acc`; not
used as a test because their fits are two-stage compute->loss->accuracy fits,
not our single power law in compute). On the OLMES macro-average ACCURACY:

| method | pairwise decision accuracy |
|---|---|
| DataDecide `2_param-default` | 72.7% |
| DataDecide `3_param-default` (all scales < 1B) | 78.3% |
| DataDecide `3_param-no_750M` | 79.7% |
| DataDecide `5_param-ai2` | 78.7% |
| ASLA `projection_ranker` (3-parameter compute power law, all scales < 1B) | 85.0% |
| ASLA `projection_ranker` (750M excluded) | 84.0% |

On the continuous proxy `correct_prob_per_char`: DataDecide `3_param-default`
88.7%, `3_param-no_750M` 90.3%; ASLA projection 88.7% (all) / 86.0% (no 750M).
Our fits are in the same range as theirs on both metrics, which is the
expected outcome for a different but reasonable functional form. Nothing in
this table was tuned to match.

## What this does and does not license

It licenses trusting the harvest (recipes, seeds, checkpoints, compute
derivation) and the pairwise decision machinery on this table. It does not
validate the crossover significance tests or the bootstrap intervals, which
have no published counterpart; those are checked by the synthetic scenarios
and unit tests, and their DataDecide values are reported in FIRST_AUDIT.md as
new measurements.
