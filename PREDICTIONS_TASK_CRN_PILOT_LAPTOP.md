# Deviation: the CRN pilot moves from an RTX 4080 to the MacBook (M-series, 16 GB)

Written 2026-10-11, before any pilot or gate run. It amends
`PREDICTIONS_TASK_CRN_PILOT.md`, which assumed hardware the user does not have.

## Unchanged

- **Design.** 4 learning rates (0.5x, 0.75x, 1x and 1.5x of 1e-3) crossed with 4
  seed indices.
- **Common random numbers.** Copied initial weights and a saved data order, per
  seed index.
- **Data.** TinyStories bytes and the fixed evaluation sample.
- **Schedule.** Evaluation at step 0 and 10 checkpoints.
- **Analysis.** `scripts/analyze_crn_pilot.py`.
- **Decision rule.** Recommend a full experiment only if the median
  matched/mismatched gap-variance ratio at the final checkpoint is at most 0.5;
  otherwise close the CRN program.

## Laptop safety

- One training process at a time, `nice -n 19`.
- At most 2 CPU threads for any CPU work. MPS runs use the GPU from one process.
- Every run is resumable; a finished run is never repeated.
- **Total budget** for the whole pilot is at most about 6 hours. Step counts are
  chosen for 5.5 hours, leaving margin.

## Path 1: MPS, gated on measured nondeterminism

1. **Step count.** At full size (the 10.8M-parameter model, batch 64 x 256), time
   50 training steps on MPS. S is the largest multiple of 100, at most 10,000,
   with 16 x S x step time x 1.1 <= 5.5 h.
2. **Gate runs.** Three runs, each with the full S-step schedule but stopped after
   the 3rd checkpoint (3S/10 steps):
   - A: seed index 0 at 1x;
   - B: seed index 0 at 1x again;
   - C: seed index 1 at 1x.
3. **Gate statistic.** Over checkpoints 1 to 3 (step 0 excluded):
   - v_nd = mean over c of (A_c - B_c)^2 / 2, the per-run nondeterminism variance;
   - v_seed = mean over c of (A_c - C_c)^2 / 2, minus v_nd: the seed-effect
     variance, with the nondeterminism inside A - C removed.
4. **Pass if v_nd < 0.10 x v_seed.** A non-positive v_seed is a fail.
5. **Rejected operations.** If deterministic mode
   (`torch.use_deterministic_algorithms(True)`) rejects an operation on MPS, the
   operations and whether each can be swapped for a deterministic equivalent
   are reported. The gate fails unless every rejected operation is swapped and
   the gate is rerun.
6. **If it passes,** the pilot runs on MPS at full size. v_nd is reported next to
   the ratio. Each matched gap carries two runs' nondeterminism (2 v_nd) that
   CRN cannot remove, so nondeterminism biases the ratio upward, toward
   concluding CRN does not help.

## Path 2: CPU fallback, if the MPS gate fails

1. **Candidate sizes,** largest first, at 2 threads under `nice -n 19`:

   | Label | Layers | Width | Heads | Context | Batch |
   |---|---|---|---|---|---|
   | C0 | 6 | 384 | 6 | 256 | 64 |
   | C1 | 4 | 256 | 4 | 256 | 32 |
   | C2 | 4 | 192 | 4 | 128 | 32 |
   | C3 | 3 | 128 | 4 | 128 | 32 |
   | C4 | 2 | 96 | 4 | 128 | 16 |

2. **Steps.** Time 50 steps for each candidate. Its step count is the largest
   multiple of 100, at most 10,000, with 16 x steps x step time x 1.1 <= 5.5 h.
3. **Choice.** The largest candidate whose step count is at least 1,000. If none
   reaches 1,000 steps, do not run.
4. **Floor** (pre-registered; trains meaningfully). It is checked after seed
   indices 0 and 1 are complete (8 runs), and again on all 16.
   - **F1.** Every run's final evaluation loss is at most 0.75 x its step-0
     evaluation loss.
   - **F2.** The range of the four learning-rate means at the final checkpoint
     exceeds the largest within-learning-rate range across seed indices (the
     spread across matched seeds).
   - If either fails after 8 runs, stop: do not complete the pilot, and do not
     try a smaller size, since the budget would be exceeded.
   - CPU is deterministic here; the toy-size tests show exact reproduction.
     v_nd is taken as 0 and verified by one repeated run at the chosen size.
5. **Scale change.** Any scale change is reported with the result.

## Reporting

- The device, the measured nondeterminism (v_nd and v_nd / v_seed), any scale
  change, and floors F1 and F2 (descriptive on MPS) are reported next to the
  decision.
- **If neither path passes,** stop and report. The user will run the unchanged
  launch on Colab.
