"""Impact of the adaptive-bounds fix on the audit: what changed and what did not.

The corrected re-run reproduced every reported decision number, on all four
data-axis designs, including the OLMES metrics whose fits were previously
pinned. That needs explaining rather than merely asserting, because the fits
demonstrably changed.

The explanation is that the mis-specified fit was *monotonically* wrong. Pinning
``alpha`` at a floor above its true value biases every intervention's projected
target value in the same direction by a similar amount, because the
interventions share a fit range and differ mainly in level. Decision metrics
depend only on the ordering of projections, so a monotone distortion leaves them
untouched. This module measures that directly: it reports how much the fitted
parameters and the projected values moved, alongside the rank correlation
between the old and new projections.

This is a fact about this dataset, not a general licence. A monotone distortion
is a coincidence of the geometry here; the same defect on a suite where
interventions differ in curvature rather than level would reorder them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import independent_rederivation as ir
from asla.models import fit_power_law

REPO = Path(__file__).resolve().parents[1]


def _cell_means(df: pd.DataFrame, budgets: list[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fit = df[df["compute"].isin(budgets)]
    out = {}
    for name, group in fit.groupby("intervention", sort=True):
        means = group.groupby("compute", sort=True)["bpb"].mean()
        out[str(name)] = (means.index.to_numpy(dtype=float), means.to_numpy(dtype=float))
    return out


def compare(metric: str) -> dict[str, Any]:
    df, budgets, target = ir.design_frame(metric)
    cells = _cell_means(df, budgets)
    old_params, new_params = {}, {}
    for name, (x, y) in cells.items():
        old_params[name] = fit_power_law(x, y, adaptive_bounds=False)
        new_params[name] = fit_power_law(x, y, adaptive_bounds=True)
    old_proj = {n: float(e + a * target ** (-al)) for n, (e, a, al) in old_params.items()}
    new_proj = {n: float(e + a * target ** (-al)) for n, (e, a, al) in new_params.items()}
    names = sorted(old_proj)
    old_alpha = np.array([old_params[n][2] for n in names])
    new_alpha = np.array([new_params[n][2] for n in names])
    old_values = np.array([old_proj[n] for n in names])
    new_values = np.array([new_proj[n] for n in names])
    return {
        "metric": metric,
        "n_interventions": len(names),
        "old_alpha_all_pinned_at_0.05": bool(np.allclose(old_alpha, 0.05, atol=1e-9)),
        "old_alpha_range": [float(old_alpha.min()), float(old_alpha.max())],
        "new_alpha_range": [float(new_alpha.min()), float(new_alpha.max())],
        "median_alpha_change": float(np.median(np.abs(new_alpha - old_alpha))),
        "max_abs_projection_change": float(np.max(np.abs(new_values - old_values))),
        "median_abs_projection_change": float(np.median(np.abs(new_values - old_values))),
        "projection_change_relative_to_between_spread": float(
            np.median(np.abs(new_values - old_values)) / np.std(new_values, ddof=1)
        ),
        "spearman_old_vs_new_projection": float(spearmanr(old_values, new_values).statistic),
        "kendall_old_vs_new_projection": float(kendalltau(old_values, new_values).statistic),
        "ranking_identical": bool(
            [n for n in sorted(names, key=lambda k: old_proj[k])] == [n for n in sorted(names, key=lambda k: new_proj[k])]
        ),
    }


def main() -> int:
    out = {metric: compare(metric) for metric in ir.TABLES}
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "bounds_fix_impact.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    for metric, row in out.items():
        print(f"=== {metric}")
        print(f"    old alpha {row['old_alpha_range']} (all pinned: {row['old_alpha_all_pinned_at_0.05']})")
        print(f"    new alpha {row['new_alpha_range']}")
        print(f"    median |projection change| = {row['median_abs_projection_change']:.3e} "
              f"({row['projection_change_relative_to_between_spread']:.1%} of the between-intervention spread)")
        print(f"    spearman(old, new) = {row['spearman_old_vs_new_projection']:.6f}; "
              f"ranking identical: {row['ranking_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
