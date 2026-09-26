"""The lever-arm test: does projection's excess mis-selection grow with extrapolation distance?

One function is used for the real data and for every simulated world, so the
test that is calibrated is the test that is reported.

A *design* is a (largest fitted scale, target scale) pair on the ladder. For
each design, the kernel is the per-pair difference in expected-error charge,
projection minus single-scale ranking at the top fitted scale, scored against
calibrated evidence at the design's target. The statistic is the Spearman
correlation between log lever arm (target / largest fitted compute) and the
per-design mean kernel.

Designs share recipes and rungs, so they are not independent observations.
Leave-one-recipe-out drops the recipe's pairs from every design at once, and
the standard error is the jackknife of the correlation's Fisher z.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from asla.analysis.moderation import evidence_from_cells
from asla.analysis.target_scoring import expected_charges
from asla.models import FitError, bpb_power_law, fit_power_law

MIN_FIT_BUDGETS = 3
Z95 = 1.959963984540054


@dataclass
class Design:
    target_label: str
    max_fit_label: str
    lever_arm: float
    kernel: dict[tuple[str, str], float]  # projection minus single-scale charge, in points
    replicate_kernel: dict[tuple[str, str], float] | None = None  # null comparator
    true_excess: float | None = None  # mis-selection difference against the noiseless target, in points


def _gaps(values: Mapping[str, float], names: list[str]) -> dict[tuple[str, str], float]:
    return {(a, b): float(values[a] - values[b]) for a, b in combinations(names, 2)}


def _mis(pred: Mapping[tuple[str, str], float], truth: Mapping[tuple[str, str], float]) -> float:
    scored = [(p, truth[k]) for k, p in pred.items() if np.isfinite(p) and p != 0 and truth[k] != 0]
    return float(np.mean([np.sign(p) != np.sign(t) for p, t in scored]))


def designs(
    frame: pd.DataFrame,
    truth_targets: Mapping[str, pd.Series] | None = None,
    replicate: pd.DataFrame | None = None,
    with_projection: bool = True,
) -> list[Design]:
    """Every design on the ladder, with its kernel, and optionally its true excess and a null kernel.

    ``truth_targets`` maps a target label to the noiseless target values per
    recipe. ``replicate`` is an independent noise draw of the same truth; its
    top-rung ranking is the null comparator, whose true excess is zero.
    ``with_projection=False`` skips the projection fits and leaves the main
    kernel empty; only the null comparator is computed.
    """

    order = frame.groupby("scale_label")["compute"].first().sort_values()
    labels = list(order.index)
    means = frame.groupby(["intervention", "compute"])["bpb"].mean()
    rep_means = replicate.groupby(["intervention", "compute"])["bpb"].mean() if replicate is not None else None
    names = sorted(frame["intervention"].astype(str).unique())
    projections: dict[tuple[int, int], dict[str, float]] = {}
    out = []
    for t_index in range(3, len(labels)):
        target = float(order.iloc[t_index])
        rows = frame[np.isclose(frame["compute"], target)]
        evidence = evidence_from_cells({str(r): g["bpb"].to_list() for r, g in rows.groupby("intervention")})
        for f_index in range(2, t_index):
            max_compute = float(order.iloc[f_index])
            budgets = [float(c) for c in order.iloc[: f_index + 1]]
            if len(budgets) < MIN_FIT_BUDGETS:
                continue
            key = (f_index, t_index)
            if with_projection and key not in projections:
                projected = {}
                for name in names:
                    y = means.loc[name].reindex(budgets).to_numpy(dtype=float)
                    try:
                        params = fit_power_law(np.asarray(budgets), y)
                        projected[name] = float(bpb_power_law(target, *params))
                    except (FitError, RuntimeError, ValueError):
                        projected[name] = float("nan")
                projections[key] = projected
            single = {name: float(means.loc[(name, max_compute)]) for name in names}
            single_gaps = _gaps(single, names)
            right = expected_charges(single_gaps, evidence)
            kernel: dict[tuple[str, str], float] = {}
            if with_projection:
                proj_gaps = _gaps(projections[key], names)
                left = expected_charges(proj_gaps, evidence)
                kernel = {p: 100 * (left[p] - right[p]) for p in left if p in right}
            design = Design(labels[t_index], labels[f_index], target / max_compute, kernel)
            if rep_means is not None:
                rep = {name: float(rep_means.loc[(name, max_compute)]) for name in names}
                rep_charges = expected_charges(_gaps(rep, names), evidence)
                design.replicate_kernel = {p: 100 * (rep_charges[p] - right[p]) for p in rep_charges if p in right}
            if truth_targets is not None and with_projection:
                truth_gaps = _gaps(truth_targets[labels[t_index]].to_dict(), names)
                design.true_excess = 100 * (_mis(proj_gaps, truth_gaps) - _mis(single_gaps, truth_gaps))
            out.append(design)
    return out


def _rho(log_lever: np.ndarray, values: np.ndarray) -> float:
    return float(stats.spearmanr(log_lever, values).statistic)


def loro_test(all_designs: list[Design], replicate: bool = False) -> dict[str, float | list[float]]:
    """Spearman rho of log lever arm against per-design mean kernel, with a leave-one-recipe-out jackknife."""

    kernels = [d.replicate_kernel if replicate else d.kernel for d in all_designs]
    if any(k is None for k in kernels):
        raise ValueError("replicate kernels were not computed")
    log_lever = np.log([d.lever_arm for d in all_designs])
    rho = _rho(log_lever, np.array([np.mean(list(k.values())) for k in kernels]))  # type: ignore[union-attr]
    names = sorted({n for k in kernels for pair in k for n in pair})  # type: ignore[union-attr]
    loo = []
    for drop in names:
        values = np.array([np.mean([v for p, v in k.items() if drop not in p]) for k in kernels])  # type: ignore[union-attr]
        loo.append(_rho(log_lever, values))
    z = np.arctanh(np.clip(np.asarray(loo), -0.999999, 0.999999))
    n = len(names)
    se = float(np.sqrt((n - 1) / n * np.sum((z - z.mean()) ** 2)))
    fisher = float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
    return {
        "rho": rho,
        "fisher_z": fisher,
        "se_fisher_z": se,
        "p_two_sided": float(2 * stats.norm.sf(abs(fisher) / se)) if se > 0 else 0.0,
        "ci": [float(np.tanh(fisher - Z95 * se)), float(np.tanh(fisher + Z95 * se))],
        "n_designs": len(all_designs),
        "leave_one_out_rho": [float(r) for r in loo],
    }


__all__ = ["Design", "designs", "loro_test"]
