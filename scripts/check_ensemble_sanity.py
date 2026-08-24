"""Sanity-check the ensemble ranker on synthetic data with a known answer.

Question: is the ensemble ranker's poor DataDecide result (18.7% pairwise
mis-selection vs 5.3% for the plain power law) an implementation defect, or
the expected cost of extra model flexibility on a well-specified problem?

Design. A DataDecide-shaped grid: 25 interventions, the 11 harvested fit
budgets (4M..300M compute) plus the 1B target, 3 seeds per cell, additive
Gaussian seed noise. Three truths:

* ``noiseless``: exact power laws, zero seed noise. Any implementation of a
  family-weighted ensemble must reproduce the plain fit here (the power-law
  family has zero leave-largest-budget-out loss); a gap is a code defect.
* ``well_specified``: the same exact power laws with DataDecide-level seed
  noise. The plain fit's family is correct, so any ensemble excess is the
  variance cost of weighting flexible families from noisy extrapolation losses.
* ``saturating_half``: for half the interventions the truth is the saturating
  curve fitted to that intervention's power law over the fit range (so it
  agrees on the fit range and flattens toward the target) - the
  misspecification the ensemble was built for.

Both rankers are scored against the noiseless truth at the target (no
measurement noise in the truth), averaged over independent noise draws.
Writes ``results/ensemble_check/ensemble_check.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.analysis.ensemble import ensemble_rank  # noqa: E402
from asla.analysis.fits import fit_all, project_ranking  # noqa: E402
from asla.analysis.metrics import decision_metrics  # noqa: E402
from asla.cli import _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402
from asla.models import bpb_power_law, bpb_saturating, fit_saturating  # noqa: E402

DD_TABLE = Path("data/datadecide_runs.parquet")


def fitted_truth_curves(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, tuple[float, float, float]]:
    """Fit a power law per DataDecide recipe on the fit budgets to use as a noiseless truth."""

    params = fit_all(df, budgets)
    return {name: (float(p[0]), float(p[1]), float(p[2])) for name, p in params.items()}


def synthetic_grid(
    curves: dict[str, tuple[float, float, float]],
    budgets: tuple[float, ...],
    target: float,
    sigma: float,
    rng: np.random.Generator,
    n_seeds: int = 3,
    saturating_names: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a runs table with additive noise and the noiseless target truth."""

    rows = []
    truth = {}
    all_budgets = (*budgets, target)
    x_fit = np.asarray(budgets, dtype=float)
    for name, (e, a, alpha) in curves.items():
        sat_params = None
        if name in saturating_names:
            sat_params = fit_saturating(x_fit, np.asarray(bpb_power_law(x_fit, e, a, alpha), dtype=float))
        for compute in all_budgets:
            if sat_params is not None:
                value = float(bpb_saturating(compute, *sat_params))
            else:
                value = float(bpb_power_law(compute, e, a, alpha))
            if np.isclose(compute, target):
                truth[name] = value
            for seed in range(n_seeds):
                rows.append(
                    {
                        "intervention": name,
                        "intervention_class": "data",
                        "compute": float(compute),
                        "seed": seed,
                        "bpb": value + rng.normal(0.0, sigma),
                    }
                )
    return pd.DataFrame(rows), pd.Series(truth).sort_values()


def run_check(n_draws: int = 5, seed: int = 0) -> dict[str, Any]:
    df = load_runs(DD_TABLE)
    df = df[df["scale_label"] != "750M"]
    target = float(df[df["scale_label"] == "1B"]["compute"].iloc[0])
    fit = df[(df["compute"] < target) & (df["scale_label"] != "530M")]
    budgets = tuple(sorted(float(b) for b in fit["compute"].unique()))
    curves = fitted_truth_curves(df, budgets)
    sds = df[df["compute"].isin(budgets)].groupby(["intervention", "compute"])["bpb"].std(ddof=1)
    sigma = float(np.sqrt((sds**2).mean()))
    names = sorted(curves)
    scenarios = {"noiseless": (), "well_specified": (), "saturating_half": tuple(names[::2])}
    out: dict[str, Any] = {
        "n_interventions": len(curves),
        "n_fit_budgets": len(budgets),
        "n_seeds": 3,
        "sigma_bits_per_token": sigma,
        "n_draws": n_draws,
        "scenarios": {},
    }
    rng = np.random.default_rng(seed)
    for scenario, sat_names in scenarios.items():
        acc = {"projection_ranker": [], "ensemble_ranker": []}
        for _ in range(n_draws):
            noise = 0.0 if scenario == "noiseless" else sigma
            table, truth = synthetic_grid(curves, budgets, target, noise, rng, saturating_names=sat_names)
            proj = project_ranking(table, budgets, target)
            ens = ensemble_rank(table, budgets, target)
            acc["projection_ranker"].append(decision_metrics(proj, truth, k=1)["pairwise_acc"])
            acc["ensemble_ranker"].append(decision_metrics(ens, truth, k=1)["pairwise_acc"])
        entry: dict[str, Any] = {"n_saturating": len(sat_names)}
        for ranker, values in acc.items():
            entry[ranker] = {
                "pairwise_mis_selection_mean": float(1.0 - np.mean(values)),
                "pairwise_mis_selection_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "draws": [float(1.0 - v) for v in values],
            }
        out["scenarios"][scenario] = entry
    ws = out["scenarios"]["well_specified"]
    nl = out["scenarios"]["noiseless"]
    out["verdict"] = {
        "implementation_ok_noiseless_within_1pp": bool(
            abs(
                nl["ensemble_ranker"]["pairwise_mis_selection_mean"] - nl["projection_ranker"]["pairwise_mis_selection_mean"]
            )
            <= 0.01
        ),
        "well_specified_ensemble_within_2pp_of_projection": bool(
            ws["ensemble_ranker"]["pairwise_mis_selection_mean"] - ws["projection_ranker"]["pairwise_mis_selection_mean"]
            <= 0.02
        ),
        "ensemble_helps_under_misspecification": bool(
            out["scenarios"]["saturating_half"]["ensemble_ranker"]["pairwise_mis_selection_mean"]
            < out["scenarios"]["saturating_half"]["projection_ranker"]["pairwise_mis_selection_mean"]
        ),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ensemble_check")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_check(n_draws=args.draws, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(result, out / "ensemble_check.json")
    for scenario, entry in result["scenarios"].items():
        print(
            f"{scenario}: projection {100 * entry['projection_ranker']['pairwise_mis_selection_mean']:.1f}% "
            f"vs ensemble {100 * entry['ensemble_ranker']['pairwise_mis_selection_mean']:.1f}% pairwise mis-selection"
        )
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
