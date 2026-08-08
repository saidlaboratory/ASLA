# ASLA: Algorithm-Selection Leaderboard Audit

Small-compute scaling-law extrapolation is routinely used as a *decision
procedure*: fit each training intervention on cheap runs, project to the
target budget, ship the projected winner. ASLA audits that procedure. It
treats intervention selection as a fixed-budget decision problem and
measures, with known or measured ground truth, when extrapolation-based
selection fails — noise close-calls, functional-form misspecification, and
crossovers that happen after the largest fitting budget — and what better
decision rules buy.

Everything runs offline on deterministic synthetic scenarios, on the
ASLA-Bench problem generator, or on your own run tables converted to a
canonical parquet schema. W&B harvesting is optional.

## What ASLA provides

- **Decision-level audit** (`asla audit`): top-1 accuracy, regret, pairwise
  accuracy, rank correlations for each decision rule, with cell-wise seed
  bootstrap confidence intervals; seed-noise bands; under-seeded-cell
  warnings; statistical ties at the target (`truth_ties`); per-intervention
  fit diagnostics (R², RMSE, dof).
- **FDR-controlled crossover detection**: per-pair Welch tests at the target
  with Benjamini–Hochberg correction (`crossovers_fdr`); single-seed pairs
  are reported as untestable rather than significant.
- **Misspecification-aware ensemble projection**: power-law, saturating, and
  damped-power-law fits weighted by leave-largest-budget-out extrapolation
  loss, with an **extrapolation reliability score** ρ = cross-family
  disagreement ÷ seed-noise band. ρ ≫ 1 is the quantitative version of
  "the fit is blind here".
- **Cost-aware sequential selection (racing)**: advance interventions up the
  budget ladder, eliminating by projected-interval dominance; every decision
  rule reports `compute_spent`, so rules are compared on the compute–regret
  Pareto frontier, not a single operating point.
- **Conformal projection intervals** and a deterministic conformal gate, plus
  an empirical calibration study of bootstrap vs. conformal coverage.
- **ASLA-Bench** (`asla benchmark`): parameterized problem families
  (close-call, late-crossover, saturating) with interpretable difficulty
  knobs and known ground truth. See [docs/BENCHMARK.md](docs/BENCHMARK.md).
- **Markdown reports** (`asla report` or `asla audit --report`).
- Theory notes — a detectability lower bound and racing correctness — in
  [docs/THEORY.md](docs/THEORY.md).

## Install

```bash
pip install -e ".[test]"
```

Optional extras:

```bash
pip install -e ".[figures,wandb]"
```

## Demo

```bash
asla demo --scenario saturation_crossover --estimand single_design_seed_sensitivity
asla demo --scenario noise_close_call --estimand pairwise_decisions
```

Default demo and audit runs use paper-grade counts (`n_boot=1000`,
`n_trials=500`). For a quick local smoke test add `--fast`.

The saturation demo reports a wrong projected winner, positive regret, a
detected crossover, an explicit `fit is blind` message, and large ensemble
reliability scores ρ for the saturating pair. The noise close-call demo shows
the gate and the race reducing mean regret and wrong-pick rate versus plain
projection, with compute costs alongside. Demo and audit reports compare the
scaling-law projection ranker, the ensemble ranker, and the
largest-single-scale baseline with bootstrap confidence intervals.

## Validate and audit real runs

Canonical run columns:

- `intervention` string
- `intervention_class` string
- `compute` float
- `seed` integer
- `bpb` float
- optional `downstream` float
- optional `params_n` float for Chinchilla two-axis fits
- optional `tokens_d` float for Chinchilla two-axis fits

```bash
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target 64 --budgets 1 2 4 8 --intermediate-budget 16 --estimand pairwise_decisions --out results/audit.json --report
asla figures --runs runs.parquet --target 64 --budgets 1 2 4 8 --out results/figures
```

`--estimand` is required. Choose `single_design_seed_sensitivity` for one
complete-candidate-set decision or `pairwise_decisions` for the finite-table
average over every unordered intervention pair. Their assumptions and licensed
claims are documented in [NOTES_ESTIMAND.md](NOTES_ESTIMAND.md); the software
does not select a paper headline. Pass fitting budgets explicitly for paper
results. A reserved `--intermediate-budget` is excluded from fitting and makes
the top-1-only gate available.

Useful audit flags:

- `--weighted` — weight fits by per-cell seed standard errors.
- `--crossover-q` — BH-FDR level for crossover significance (default 0.05).
- `--report` — write a markdown report next to `--out`.
- `--budgets` — explicit projection-fitting budgets.
- `--intermediate-budget` — held-out gate exploration budget, never fit by the
  baseline projection.
- `--fit-form compute_power_law` (leaderboard audits) or `--fit-form
  chinchilla` (controlled grids with `params_n`/`tokens_d`; fails loudly if
  those columns are missing).

Fits are unit-invariant: `compute`, `params_n`, and `tokens_d` may be raw
counts (FLOPs, parameters, tokens) or consistent relative units.

## Benchmark and paper experiments

```bash
asla benchmark --fast --out results/benchmark        # difficulty sweep, 4 rules
python scripts/run_pareto_study.py --fast            # compute-regret frontier
python scripts/run_calibration_study.py --fast       # interval coverage
python scripts/run_paper_experiments.py --estimand pairwise_decisions --fast  # explicit human-selected estimand
```

Drop `--fast` for paper-grade counts. See [docs/BENCHMARK.md](docs/BENCHMARK.md).

## No-W&B data collection

On a laptop, create an HPC checklist:

```bash
python scripts/make_run_manifest.py \
  --interventions data/interventions_template.csv \
  --budgets data/budgets_template.csv \
  --seeds 3 \
  --out data/run_manifest.csv
```

Edit `data/interventions_template.csv` and `data/budgets_template.csv` before
using the manifest for real training. The generated manifest is not audit data;
it is a checklist for runs that still need measured BPB values.

After the runs finish and the BPB values are filled into a CSV:

```bash
python scripts/finalize_run_manifest.py \
  --manifest data/run_manifest.csv \
  --out-csv data/runs_template.csv \
  --out-parquet runs.parquet
python scripts/check_runs_coverage.py --csv data/runs_template.csv --target TARGET_COMPUTE
asla validate --runs runs.parquet
asla audit --runs runs.parquet --target TARGET_COMPUTE --budgets FIT_BUDGETS --intermediate-budget INTERMEDIATE_COMPUTE --estimand ESTIMAND --fast --out runs_audit.json
```

The finalizer refuses pending rows and blank BPB values, so incomplete manifests
cannot silently become audit data.

The optional HPC scaffold is in `hpc/`:

- `hpc/train_command.template`: wrapper command used by the row runner.
- `hpc/site_train_command.template`: calls `hpc/train_and_eval.py`, which uses `ASLA_SITE_COMMAND_TEMPLATE`.
- `hpc/site_command.template`: local template that calls the script named by `ASLA_REAL_TRAIN_EVAL`.
- `hpc/slurm_array_template.sh`: SLURM array template, dry-run by default.
- `hpc/ADAPTATION_CHECKLIST.md`: step-by-step cluster adaptation guide.

Dry-run one manifest row:

```bash
python scripts/run_one_manifest_row.py \
  --manifest data/run_manifest.csv \
  --row 1 \
  --index-base 1 \
  --command-template-file hpc/train_command.template \
  --output-dir results/hpc
```

Collect per-run `result.json` files after HPC jobs finish:

```bash
python scripts/collect_results.py \
  --manifest data/run_manifest.csv \
  --results-dir results/hpc \
  --out data/run_manifest.csv
```

Run `python scripts/hpc_preflight.py` before submission. It verifies that the
SLURM array covers every manifest row exactly once and smoke-tests only the
plumbing in a temporary directory. Result JSON must contain `run_id`,
`intervention`, `compute`, `seed`, finite positive measured `bpb`, and completed
status. Collection rejects identity mismatches. Existing metrics/results are
never reused; a deliberate reviewed retry requires `ASLA_OVERWRITE_OUTPUT=1`.

## Harvest W&B runs

ASLA never guesses W&B field names. First inspect a finished run:

```bash
asla harvest --discover --entity-project ENTITY/PROJECT
```

Then provide an explicit JSON field map from schema column to logged W&B key:

```json
{
  "intervention": "config.recipe_name",
  "intervention_class": "config.recipe_class",
  "compute": "summary.train_flops",
  "seed": "config.seed",
  "bpb": "summary.eval/c4_en_bpb"
}
```

`config.` and `summary.` prefixes select the W&B store explicitly and support
nested dotted paths. An unprefixed key is accepted only when it resolves in
exactly one store; ambiguity is an error.

Harvest:

```bash
asla harvest --entity-project ENTITY/PROJECT --field-map field_map.json --out runs.parquet
```

## Development

```bash
pytest        # test suite
ruff check .  # lint (also run in CI)
```

The enhancement roadmap and its status live in [PLAN.md](PLAN.md).
