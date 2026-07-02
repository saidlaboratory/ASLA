"""Create a no-W&B run manifest for ASLA data collection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_INTERVENTION_COLUMNS = ("intervention", "intervention_class")
REQUIRED_BUDGET_COLUMNS = ("compute", "role")


def _read_interventions(path: str) -> pd.DataFrame:
    """Read and validate the intervention plan."""

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_INTERVENTION_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"intervention file missing required columns: {missing}")
    if df.empty:
        raise SystemExit("intervention file has no rows")
    if df["intervention"].astype(str).str.len().eq(0).any():
        raise SystemExit("intervention names must be non-empty")
    duplicates = df["intervention"].astype(str).duplicated(keep=False)
    if bool(duplicates.any()):
        names = sorted(df.loc[duplicates, "intervention"].astype(str).unique().tolist())
        raise SystemExit(f"intervention names must be unique; duplicates: {names}")
    return df


def _read_budgets(path: str) -> pd.DataFrame:
    """Read and validate the budget plan."""

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_BUDGET_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"budget file missing required columns: {missing}")
    if df.empty:
        raise SystemExit("budget file has no rows")
    df = df.copy()
    df["compute"] = pd.to_numeric(df["compute"], errors="raise").astype(float)
    df["role"] = df["role"].astype(str)
    if not set(df["role"]).issubset({"fit", "target"}):
        raise SystemExit("budget role must be either 'fit' or 'target'")
    if (df["compute"] <= 0).any():
        raise SystemExit("compute budgets must be positive")
    if (df["role"] == "fit").sum() < 3:
        raise SystemExit("at least three fit budgets are required for power-law fits")
    if (df["role"] == "target").sum() != 1:
        raise SystemExit("exactly one target budget is required")
    duplicates = df["compute"].duplicated(keep=False)
    if bool(duplicates.any()):
        budgets = sorted(df.loc[duplicates, "compute"].astype(float).unique().tolist())
        raise SystemExit(f"compute budgets must be unique; duplicates: {budgets}")
    target = float(df.loc[df["role"] == "target", "compute"].iloc[0])
    fit = df.loc[df["role"] == "fit", "compute"].astype(float)
    if bool((fit >= target).any()):
        raise SystemExit("all fit budgets must be below the target budget")
    return df.sort_values("compute", kind="mergesort")


def make_manifest(interventions: pd.DataFrame, budgets: pd.DataFrame, seeds: int) -> pd.DataFrame:
    """Return one planned row per intervention, budget, and seed."""

    if seeds < 1:
        raise ValueError("seeds must be at least 1")

    rows: list[dict[str, object]] = []
    for intervention in interventions.itertuples(index=False):
        for budget in budgets.itertuples(index=False):
            for seed in range(seeds):
                rows.append(
                    {
                        "run_id": f"{intervention.intervention}__c{budget.compute:g}__s{seed}",
                        "intervention": str(intervention.intervention),
                        "intervention_class": str(intervention.intervention_class),
                        "compute": float(budget.compute),
                        "role": str(budget.role),
                        "seed": int(seed),
                        "status": "pending",
                        "bpb": "",
                        "downstream": "",
                        "params_n": "",
                        "tokens_d": "",
                        "notes": "",
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    """Run the manifest generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interventions", default="data/interventions_template.csv")
    parser.add_argument("--budgets", default="data/budgets_template.csv")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="data/run_manifest.csv")
    args = parser.parse_args()

    interventions = _read_interventions(args.interventions)
    budgets = _read_budgets(args.budgets)
    manifest = make_manifest(interventions, budgets, args.seeds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    target = budgets.loc[budgets["role"] == "target", "compute"].iloc[0]
    print(f"wrote {len(manifest)} planned runs to {out}")
    print(f"target compute: {target:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
