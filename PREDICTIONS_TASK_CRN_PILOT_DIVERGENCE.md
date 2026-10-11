# Pre-registration: does MPS nondeterminism explain the pilot's ratio near 1?

Written 2026-10-11, after the MPS pilot (median ratio 1.04, [0.29, 2.81]) and
before any rerun. It closes a gap in the determinism gate, which measured only
the first three checkpoints. Over those, the identical-run gap grew from 0.0004
to 0.010 nats.

## Check

- **Reruns.** Rerun (seed index 0, 1x) and (seed index 0, 1.5x) for the full 500
  steps on MPS, with the pilot's exact configuration, initial state, data order
  and code.
  - One process at a time, `nice -n 19`, at most 2 CPU threads.
- **Comparison.** Each rerun is compared with the original pilot run of the same
  configuration at every checkpoint.
- **Identical-run gap curve.** |original - rerun| at each checkpoint, for both
  learning rates.
- **Identical-run variance at the final checkpoint.** v_lr = (original - rerun)^2 / 2.
  It has one degree of freedom, so it is a noisy estimate; that is stated with
  the result.
- **Fractions.** For each pair (a, b) involving a measured learning rate, at the
  final checkpoint:
  - F_ab = 2 v_lr / Var(matched gaps of the pair), using the pilot's 4 matched
    gaps;
  - for the pair 1x|1.5x, where both are measured, F = (v_1 + v_1.5) / Var(matched
    gaps). The 2 v versions are also reported.
- **Decision statistic.** The maximum F over the five pairs that involve 1x or
  1.5x. Taking the maximum is the conservative choice.

## Decision, fixed now

- **Maximum F < 10%: the negative stands.**
  - Record it as no large benefit: point estimate 1.04, with the interval
    [0.29, 2.81], which does not exclude a moderate reduction.
  - Close the CRN program.
- **Maximum F >= 10%: MPS cannot test CRN at this training length.**
  - Record the pilot as inconclusive.
  - Leave the unchanged launch (`python -m pilot.crn.launch`) ready for Colab.

## Recorded now

The 0.5x/0.75x pair's ratio of 0.075 is exploratory: one of six pairs, with no
follow-up.
