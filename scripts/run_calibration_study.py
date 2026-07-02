"""Empirical coverage of bootstrap vs. conformal projection intervals.

For each benchmark problem, simulate noise resamples, build both interval
types per intervention from fit-range data only, and check whether the
noiseless true target BPB lands inside. Nominal levels: bootstrap uses the
5th-95th percentile band (90%); conformal uses ``1 - alpha``.

Writes a tidy CSV and prints per-family coverage. Empirically (fast run,
4-rung ladder, target 8x beyond the ladder) BOTH interval types undercover:
bootstrap ~0.82-0.86 and conformal ~0.55 against a 0.90 nominal. The
conformal gap is structural on short ladders — leave-one-budget-out scores
measure in-ladder prediction error, which is smaller than 8x-extrapolation
error. This undercoverage, and how it widens with extrapolation ratio and
under misspecification, is itself a headline audit number for the paper: no
off-the-shelf interval on fit-range data is calibrated for the decision
actually being made.

Example:
    python scripts/run_calibration_study.py --fast --out results/calibration
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from asla.analysis.conformal import conformal_projection_interval
from asla.analysis.fits import projection_with_uncertainty
from asla.data.benchmark import BenchmarkConfig, make_scenario
from asla.models import FitError

DEFAULT_PROBLEMS: tuple[BenchmarkConfig, ...] = (
    BenchmarkConfig(family="close_call", gap_to_noise=2.0, seed=0),
    BenchmarkConfig(family="late_crossover", gap_to_noise=2.0, crossover_position=0.30, seed=0),
    BenchmarkConfig(family="saturating", gap_to_noise=2.0, crossover_position=0.30, saturation_strength=1.0, seed=0),
)


def run_study(n_trials: int, n_boot: int, alpha: float, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root_rng = np.random.default_rng(seed)
    for config in DEFAULT_PROBLEMS:
        scenario = make_scenario(config)
        fit_budgets = config.budgets.fit
        target = config.budgets.target
        for trial in range(n_trials):
            data_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
            boot_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
            df = scenario(data_rng, None)
            truth_fns = df.attrs["scenario_truth"].functions
            budget_values = np.asarray(fit_budgets, dtype=float)
            for intervention, fn in truth_fns.items():
                true_value = float(fn(target))
                group = df[df["intervention"] == intervention]
                mask = group["compute"].astype(float).apply(lambda c: bool(np.any(np.isclose(c, budget_values))))
                fit_rows = group[mask]
                x = fit_rows["compute"].to_numpy(dtype=float)
                y = fit_rows["bpb"].to_numpy(dtype=float)
                try:
                    _, _, b_lo, b_hi = projection_with_uncertainty(group, fit_budgets, target, n_boot, boot_rng)
                    _, c_lo, c_hi = conformal_projection_interval(x, y, target, alpha=alpha)
                except FitError:
                    continue
                for method, lo, hi in (("bootstrap", b_lo, b_hi), ("conformal", c_lo, c_hi)):
                    rows.append(
                        {
                            "family": config.family,
                            "intervention": intervention,
                            "trial": trial,
                            "method": method,
                            "lo": float(lo),
                            "hi": float(hi),
                            "true_target": true_value,
                            "covered": int(lo <= true_value <= hi),
                            "width": float(hi - lo),
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default="results/calibration")
    args = parser.parse_args()
    n_trials = 25 if args.fast else args.trials
    n_boot = 60 if args.fast else args.n_boot
    rows = run_study(n_trials, n_boot, args.alpha, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "interval_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    print(f"nominal: bootstrap 90%, conformal {100 * (1 - args.alpha):.0f}%")
    by: dict[tuple[str, str], list[int]] = {}
    widths: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["method"]))
        by.setdefault(key, []).append(int(row["covered"]))
        widths.setdefault(key, []).append(float(row["width"]))
    for (family, method), covered in sorted(by.items()):
        print(
            f"{family:16s} {method:9s} coverage={np.mean(covered):.3f} "
            f"mean_width={np.mean(widths[(family, method)]):.4f} n={len(covered)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
