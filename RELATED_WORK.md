# Related work and positioning

Each paragraph states what the work did (verified against the abstract, paper
text, or released artifacts on 2026-08-23) and the precise axis on which ASLA
differs. ASLA's claimed axis: prior work measures how well small experiments
predict large-scale winners for *data* recipes (DataDecide), characterises
*benchmarks* (Signal and Noise), or compares individual *optimizers*
(Fantastic Optimizers). ASLA measures decision accuracy of the same projection
procedure across *algorithmic intervention classes* and tests whether
crossover risk differs between classes in a way that can be predicted from
mechanism, with seed-aware significance testing of every crossover.

## DataDecide (Magnusson et al., ICML 2025; arXiv:2504.11393)

DataDecide releases 1,050 models: 25 pretraining data recipes x 14 scales
(4M to 1B non-embedding parameters, 5x Chinchilla tokens) x 3 seeds, with
OLMES evaluations of every checkpoint. It defines *decision accuracy* as the
fraction of recipe pairs whose predicted order matches the observed order at
the 1B target (3-seed mean), and finds that ranking a single small scale
(e.g. 150M) is a strong baseline (~80% of pairs correct) that none of eight
scaling-law baselines beat on the compute-vs-accuracy frontier. It explicitly
notes that single-scale ranking cannot predict crossovers, that crossovers
bound its decision accuracy, that "it is difficult to distinguish evaluation
variance from true crossovers, but the scaling trends we empirically observe
cross over frequently", and that improved methods can be measured on the
suite. **Difference:** DataDecide varies data only. ASLA uses it as the
data-axis measurement (and as a known-answer test: KNOWN_ANSWER.md), adds the
seed-noise-controlled crossover test the paper says is missing, and contrasts
the data axis against algorithmic axes under the same estimand.

## Signal and Noise (Heineman et al., NeurIPS 2025; arXiv:2508.13144)

Defines *signal* (a benchmark's spread across models) and *noise* (its
sensitivity to seed / late-training variation), shows that signal-to-noise
ratio predicts both small-scale decision accuracy and scaling-law prediction
error, and proposes interventions (continuous metrics, averaging checkpoints,
filtering noisy subtasks). Releases ~900K evaluation results over 375
open-weight models (60M-32B) on 30 benchmarks (`allenai/signal-and-noise`).
**Difference:** their unit of analysis is the *benchmark*; ours is the
*intervention class*. We hold the metric fixed (C4-EN bits per token, or the
OLMES macro average) and ask whether the decision procedure fails differently
for data versus optimizer interventions. Their noise estimates are the natural
input to our seed-noise band and are the third source planned in Task 6.

## Fantastic Pretraining Optimizers (Wen, Hall, Ma, Liang, 2025; arXiv:2509.02046)

Re-benchmarks eleven optimizers (AdamW, NAdamW, Mars, Cautious, Lion,
Adam-mini, Muon, Scion, Kron, SOAP, Sophia) at 130M-1.2B parameters and 1-8x
(some 16x) Chinchilla tokens with a three-phase per-optimizer, per-scale
coordinate-descent tuning protocol. Finds that speedups over a well-tuned
AdamW are smaller than claimed and shrink with scale (1.4x at 130M to ~1.1x at
1.2B), that rankings shift with the data-to-model ratio (Muon best at low
ratios, Kron/SOAP at 8x+), and that loss curves of different optimizers cross
during learning-rate decay. Released artifacts: per-cell `result.json` files
with the tuned final C4-EN validation loss and one-hyperparameter ablations
(single run per cell, no seeds). **Difference:** they compare optimizers to
each other; we use their grid as the optimizer-axis *decision* problem - would
a projection from 130M-520M have picked the 1.2B winner? - and their
careful tuning is what lets us separate crossover from under-tuning
(NOTES_TUNING_CONFOUND.md). The absence of seeds in their release is a stated
limitation of our optimizer-axis numbers.

## Small-Scale Experiments: Are We There Yet? (Lourie, Cho, Ullrich, Lotfi, 2026; arXiv:2608.11859)

Argues that scaling laws have looked unreliable at small scale because small
models are far more hyperparameter-sensitive than large ones, so studies that
did not tune to the frontier saw inconsistent trends; shows the hyperparameter
loss surface becomes lower-dimensional with scale; and demonstrates a
tuning-first methodology that recovers a known large-scale result
(pre-normalisation beats post-normalisation as models grow) from small
experiments, while noting that extrapolation still hits statistical limits.
**Difference:** this is the main threat to any crossover claim, not a
complementary result. If crossovers are under-tuning artifacts, our optimizer
axis is measuring tuning rather than mechanism. ASLA therefore records
`tuning_quality` per run, prefers the coordinate-descent-tuned optimizer grid
for headline numbers, stratifies rank stability by tuning condition, and
states what the available metadata cannot rule out (NOTES_TUNING_CONFOUND.md).
Their positive result is on an architecture intervention; whether the same
holds for optimizers at fixed tuning effort is exactly what our contrast asks.

## Scaling-law crossover and extrapolation reliability

Hoffmann et al. (Chinchilla, 2022) established compute-optimal fits and their
sensitivity to fitting choices; later re-analyses (e.g. Besiroglu et al. 2024
on the Chinchilla fit, Porian et al. 2024 on reconciling Kaplan vs Chinchilla)
show that the fitted exponents and therefore the projected orderings depend on
data range, warmup, and hyperparameter handling. Work on downstream-task
scaling (e.g. Gadre et al. 2024; Bhagia et al. 2024's two-step loss-to-task
fits used by DataDecide) documents that task metrics are less predictable than
loss. **Difference:** these study curve-fit accuracy for one recipe at a time.
ASLA measures the *decision* error of comparing two fitted curves at a target
none of them reached, reports the fitted crossover budget with a bootstrap
interval, and treats a crossover as a statistical claim to be tested against
seed noise rather than a visual reading of two curves.

## Maximal-update parameterisation (muP; Yang et al. 2022, Tensor Programs V)

muP chooses width-dependent scalings so that optimal hyperparameters (notably
the learning rate) transfer across width, making small-scale tuning results
valid at large scale. Follow-ups (e.g. depth-muP, muP for other optimizers)
extend the transfer. **Difference:** muP addresses the *hyperparameter*
component of Lourie et al.'s confound and is the mechanism behind the planned
`mechanism_crossover_budget` predictor, which stays an explicit
`NotImplementedError` stub until it can be derived and validated rather than
invented. None of the public grids used here were trained under muP; the
DataDecide ladder uses per-scale heuristic hyperparameters and Fantastic
Optimizers tunes per scale directly, so hyperparameter transfer is not assumed
anywhere in the current measurements.
