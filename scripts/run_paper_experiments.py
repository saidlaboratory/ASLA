"""Regenerate every number and figure the paper needs, in one command.

Runs, in order:
1. Demo audits on the three planted synthetic scenarios (JSON + markdown).
2. The ASLA-Bench difficulty sweep (results table, summary, heat maps).
3. The compute-regret Pareto study (gate tau vs race beta).
4. The interval calibration study (bootstrap vs conformal coverage).

Use ``--fast`` for a minutes-scale smoke run; omit it for paper-grade counts.

Example:
    python scripts/run_paper_experiments.py --fast --out results/paper
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from asla.analysis.audit import ESTIMANDS
from asla.cli import _write_json_atomically


def _run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/paper")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--estimand", choices=ESTIMANDS, required=True)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fast = ["--fast"] if args.fast else []
    py = [sys.executable]

    # 1. Scenario audits: generate each planted scenario, audit it, render reports.
    scenario_dir = out / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    make_tables = f"""
import numpy as np
from asla.config import AuditConfig
from asla.data.io import save_runs
from asla.data.synthetic import SCENARIOS
cfg = AuditConfig()
for name, fn in SCENARIOS.items():
    save_runs(fn(np.random.default_rng({args.seed}), cfg), r"{scenario_dir}/" + name + ".parquet")
print("wrote scenario tables to {scenario_dir}")
"""
    _run(py + ["-c", make_tables])
    for scenario in ("clean_crossover", "saturation_crossover", "noise_close_call", "negative_controls_only"):
        _run(
            py
            + [
                "-m",
                "asla.cli",
                "audit",
                "--runs",
                str(scenario_dir / f"{scenario}.parquet"),
                "--target",
                "64",
                "--budgets",
                "1",
                "2",
                "4",
                "8",
                "--intermediate-budget",
                "16",
                "--estimand",
                args.estimand,
                "--seed",
                str(args.seed),
                "--report",
                "--out",
                str(scenario_dir / f"{scenario}_audit.json"),
            ]
            + fast
        )
        _run(
            py
            + [
                "-m",
                "asla.cli",
                "figures",
                "--runs",
                str(scenario_dir / f"{scenario}.parquet"),
                "--target",
                "64",
                "--budgets",
                "1",
                "2",
                "4",
                "8",
                "--out",
                str(scenario_dir / f"{scenario}_figures"),
            ]
        )

    # 2. Benchmark sweep.
    _run(py + ["-m", "asla.cli", "benchmark", "--seed", str(args.seed), "--out", str(out / "benchmark")] + fast)

    # 3. Pareto study.
    _run(py + ["scripts/run_pareto_study.py", "--seed", str(args.seed), "--out", str(out / "pareto")] + fast)

    # 4. Calibration study.
    _run(py + ["scripts/run_calibration_study.py", "--seed", str(args.seed), "--out", str(out / "calibration")] + fast)

    manifest = {
        "fast": bool(args.fast),
        "seed": int(args.seed),
        "estimand": args.estimand,
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    _write_json_atomically(manifest, out / "MANIFEST.json")
    print(f"\nall paper artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
