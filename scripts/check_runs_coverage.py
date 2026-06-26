"""Check whether a completed ASLA CSV is ready for audit."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from scripts.prepare_runs_table import _coerce_runs
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from prepare_runs_table import _coerce_runs


def coverage_report(df: pd.DataFrame, target: float, min_seeds: int) -> tuple[list[str], pd.DataFrame]:
    """Return coverage problems and per-cell seed counts."""

    problems: list[str] = []
    computes = sorted(df["compute"].astype(float).unique())
    fit_budgets = [c for c in computes if not np.isclose(c, target) and c < target]
    interventions = sorted(df["intervention"].astype(str).unique())

    if len(interventions) < 3:
        problems.append(f"need at least 3 interventions; found {len(interventions)}")
    if len(fit_budgets) < 3:
        problems.append(f"need at least 3 fit budgets below target {target:g}; found {len(fit_budgets)}")
    if not any(np.isclose(computes, target)):
        problems.append(f"target compute {target:g} is not present in the CSV")

    counts = (
        df.groupby(["intervention", "compute"], sort=True)["seed"]
        .nunique()
        .reset_index(name="seed_count")
        .sort_values(["intervention", "compute"], kind="mergesort")
    )
    expected = [(name, budget) for name in interventions for budget in [*fit_budgets, target]]
    for name, budget in expected:
        match = counts[(counts["intervention"] == name) & np.isclose(counts["compute"].astype(float), budget)]
        if match.empty:
            problems.append(f"missing cell: intervention={name}, compute={budget:g}")
        elif int(match["seed_count"].iloc[0]) < min_seeds:
            problems.append(
                f"under-seeded cell: intervention={name}, compute={budget:g}, "
                f"seeds={int(match['seed_count'].iloc[0])}, required={min_seeds}"
            )
    return problems, counts


def main() -> int:
    """Run the coverage checker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/runs_template.csv")
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()

    df = _coerce_runs(pd.read_csv(args.csv))
    problems, counts = coverage_report(df, args.target, args.min_seeds)
    print(counts.to_string(index=False))
    if problems:
        print("\nnot ready for paper-grade audit:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("\nready for audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
