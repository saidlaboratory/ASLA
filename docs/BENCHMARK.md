# ASLA-Bench: controlled selection problems with known ground truth

ASLA-Bench generates algorithm-selection problems whose noiseless truth is
known, so any selection rule can be scored on wrong-pick rate, regret, and
compute spent — with difficulty controlled by interpretable knobs instead of
opaque random draws.

## Problem families

| family | construction | what it stresses |
|--------|--------------|------------------|
| `close_call` | runner-up trails the winner by a constant `gap_to_noise x noise` at every budget | pure noise-vs-signal decisions |
| `late_crossover` | runner-up beats the winner before `C_x` and loses by the planted gap at target | extrapolation across an order flip |
| `saturating` | runner-up is a saturating curve crossing the winner at `C_x`; `saturation_strength <= 1` reliably fools one-shot power-law projection | functional-form misspecification |

## Knobs

- `gap_to_noise` — target-budget gap between the top two, in seed-noise SDs.
- `crossover_position` — `log10(C_x / C_max_fit)`; negative puts the crossing
  inside the fitting ladder (visible), positive puts it in the extrapolation
  gap (deceptive). Must keep `C_x` below the target.
- `saturation_strength` — half-saturation budget as a multiple of the
  ladder's geometric mean. `<= 1`: the bend happens at or before the ladder,
  masquerading as a shallow power law (hard); `> 1`: the bend is visible in
  the upper ladder (easier).
- `n_interventions`, `noise`, `n_seeds`, `budgets`, `seed` (curve-sampling
  seed — one config is one reproducible problem; trials resample noise only).

Knob combinations that are geometrically infeasible (e.g. a planted gap
larger than the winner's decay between crossing and target) raise
`ValueError` at construction; `benchmark_grid` skips them.

## Running

```bash
# quick smoke (10 trials, small bootstrap)
asla benchmark --fast --out results/benchmark

# paper-grade sweep over all three families
asla benchmark --trials 200 --n-boot 500 --problem-seeds 3 --out results/benchmark
```

Outputs: `benchmark_results.parquet`/`.csv` (one tidy row per problem x rule),
`benchmark_summary.json` (per-family x rule means), and wrong-pick-rate heat
maps over `(gap_to_noise x crossover_position)` per rule when matplotlib is
installed.

## Companion studies

```bash
# compute-regret Pareto frontier: gate tau sweep vs race beta sweep
python scripts/run_pareto_study.py --scenario noise_close_call --out results/pareto

# empirical coverage of bootstrap vs conformal projection intervals
python scripts/run_calibration_study.py --out results/calibration

# everything the paper needs, in one command
python scripts/run_paper_experiments.py --out results/paper   # add --fast to smoke
```

## Using it in code

```python
from asla.data.benchmark import BenchmarkConfig, benchmark_grid, evaluate_configs, make_scenario

problem = BenchmarkConfig(family="late_crossover", gap_to_noise=2.0, crossover_position=0.3, seed=0)
scenario = make_scenario(problem)      # rng -> runs table with ScenarioTruth attrs
table = evaluate_configs(benchmark_grid(), n_trials=100, n_boot=300)
```

Every generated table carries `ScenarioTruth` in `df.attrs`, so noiseless
regret is always measurable; the tables also pass `asla validate` and work
with the whole audit pipeline (`audit_with_ci`, `detect_crossovers_fdr`,
`ensemble_report`, `race_pick`).
