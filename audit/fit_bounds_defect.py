"""Quantify the fit-bound defect that A1 surfaced, for the adversarial report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np

from asla.analysis.fits import fit_all

REPO = Path(__file__).resolve().parents[1]
ALPHA_BOUNDS = (0.05, 2.0)


def per_metric() -> list[dict[str, Any]]:
    rows = []
    for metric in ir.TABLES:
        df, budgets, target = ir.design_frame(metric)
        params = fit_all(df, tuple(budgets))
        ymin = df[df["compute"].isin(budgets)].groupby("intervention")["bpb"].min()
        alphas = [al for _, _, al in params.values()]
        rows.append(
            {
                "metric": metric,
                "n_interventions": len(params),
                "at_alpha_lo": sum(abs(al - ALPHA_BOUNDS[0]) < 1e-6 for al in alphas),
                "at_alpha_hi": sum(abs(al - ALPHA_BOUNDS[1]) < 1e-6 for al in alphas),
                "at_E_hi": sum(abs(e - float(ymin[n])) < 1e-9 for n, (e, _, _) in params.items()),
                "alpha_min": float(min(alphas)),
                "alpha_max": float(max(alphas)),
            }
        )
    return rows


def olmes_detail() -> dict[str, Any]:
    df, budgets, target = ir.design_frame("olmes_macro_error")
    means = df[df["compute"].isin(budgets)].groupby(["intervention", "compute"])["bpb"].mean().reset_index()
    one = means[means["intervention"] == "C4"].sort_values("compute")
    lo, hi = one.iloc[0], one.iloc[-1]
    free_alphas = []
    for _, group in df[df["compute"].isin(budgets)].groupby("intervention"):
        free_alphas.append(ir.fit_power_law_independent(group["compute"].to_numpy(), group["bpb"].to_numpy())[2])
    median_free = float(np.median(free_alphas))
    return {
        "olmes_total_decay": float(lo["bpb"] - hi["bpb"]),
        "olmes_compute_span": float(hi["compute"] / lo["compute"]),
        "olmes_free_alpha": median_free,
        "orders_of_magnitude": float(np.log10(ALPHA_BOUNDS[0] / median_free)),
        "n_free_alphas_below_bound": int(sum(a < ALPHA_BOUNDS[0] for a in free_alphas)),
    }


def main() -> int:
    rows = per_metric()
    detail = olmes_detail()
    olmes = next(r for r in rows if r["metric"] == "olmes_macro_error")
    out = {
        "alpha_bounds": list(ALPHA_BOUNDS),
        "per_metric": rows,
        "n_pinned_olmes": olmes["at_alpha_lo"],
        "n_interventions": olmes["n_interventions"],
        **detail,
    }
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "defect_fit_bounds.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
