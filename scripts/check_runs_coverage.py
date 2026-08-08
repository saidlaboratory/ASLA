"""Check whether a completed ASLA CSV is ready for audit."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from asla.data.schema import validate

try:
    from scripts.prepare_runs_table import _coerce_runs
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from prepare_runs_table import _coerce_runs


def coverage_report(
    df: pd.DataFrame,
    target: float,
    min_seeds: int,
    fit_budgets: tuple[float, ...] | None = None,
    intermediate_budget: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Return coverage problems and per-cell seed counts."""

    validate(df)
    if min_seeds < 1:
        raise ValueError("min_seeds must be at least 1")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a finite positive number")
    problems: list[str] = []
    computes = sorted(df["compute"].astype(float).unique())
    resolved_fit_budgets = (
        [
            c
            for c in computes
            if not np.isclose(c, target)
            and c < target
            and (intermediate_budget is None or not np.isclose(c, intermediate_budget))
        ]
        if fit_budgets is None
        else list(fit_budgets)
    )
    interventions = sorted(df["intervention"].astype(str).unique())

    if len(interventions) < 3:
        problems.append(f"need at least 3 interventions; found {len(interventions)}")
    if len(set(resolved_fit_budgets)) != len(resolved_fit_budgets):
        problems.append("fitting budgets must be unique")
    invalid_fit = [
        budget
        for budget in resolved_fit_budgets
        if not np.isfinite(budget)
        or budget <= 0
        or budget >= target
        or (intermediate_budget is not None and budget >= intermediate_budget)
    ]
    if invalid_fit:
        problems.append(f"invalid fitting budgets: {invalid_fit}")
    missing_fit = [
        budget for budget in resolved_fit_budgets if not any(np.isclose(budget, observed) for observed in computes)
    ]
    if missing_fit:
        problems.append(f"fitting budgets absent from table: {missing_fit}")
    if intermediate_budget is not None and (
        not np.isfinite(intermediate_budget)
        or intermediate_budget <= 0
        or intermediate_budget >= target
        or not any(np.isclose(intermediate_budget, observed) for observed in computes)
    ):
        problems.append(f"invalid or absent intermediate budget: {intermediate_budget}")
    if len(resolved_fit_budgets) < 3:
        problems.append(f"need at least 3 fitting budgets below target {target:g}; found {len(resolved_fit_budgets)}")
    if not any(np.isclose(computes, target)):
        problems.append(f"target compute {target:g} is not present in the CSV")

    counts = (
        df.groupby(["intervention", "compute"], sort=True)["seed"]
        .nunique()
        .reset_index(name="seed_count")
        .sort_values(["intervention", "compute"], kind="mergesort")
    )
    expected_budgets = [*resolved_fit_budgets]
    if intermediate_budget is not None:
        expected_budgets.append(float(intermediate_budget))
    expected_budgets.append(float(target))
    expected = [(name, budget) for name in interventions for budget in expected_budgets]
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
    parser.add_argument("--fit-budgets", type=float, nargs="+", help="Explicit fitting budgets.")
    parser.add_argument("--intermediate-budget", type=float, help="Reserved gate exploration budget.")
    args = parser.parse_args()

    df = _coerce_runs(pd.read_csv(args.csv))
    problems, counts = coverage_report(
        df,
        args.target,
        args.min_seeds,
        fit_budgets=None if args.fit_budgets is None else tuple(args.fit_budgets),
        intermediate_budget=args.intermediate_budget,
    )
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
