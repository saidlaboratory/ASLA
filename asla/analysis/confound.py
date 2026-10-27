"""Separating lever arm from top-rung size and rung count (PREDICTIONS_TASK_CONFOUND.md).

The lever-arm test of :mod:`asla.analysis.lever_arm` uses fit ladders that all
start at the smallest model, so rung count is fixed by the top rung and long lever
arms come only from small top rungs. Here a design is a fit *window* of rungs
s..f (at least three) and a target t > f, so the three factors vary separately.

The kernel is the one :func:`asla.analysis.lever_arm.designs` uses: per recipe pair,
projection's expected-error charge minus single-scale ranking's at the top fitted
rung, scored against the target's own three-seed evidence, in points.

Every statistic is computed by one function, :func:`statistics`, on a list of
designs and a per-design excess vector, so the real data, the leave-one-recipe-out
jackknife and every calibration world run the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd

from asla.analysis.moderation import evidence_from_cells
from asla.analysis.target_scoring import expected_charges
from asla.models import FitError, bpb_power_law, fit_power_law

MIN_RUNGS = 3
MIN_SLICE = 3
LEVER_BIN = 0.2  # decades: width of the lever-arm bins that define slice (ii)
MIN_TOP_SPAN = 1.0  # decades of top-rung compute a slice (ii) must span
STATISTICS = ("joint_lever", "joint_top", "joint_rungs", "slice_lever", "slice_top", "slice_rungs")


@dataclass
class WindowDesign:
    start: int
    top: int
    target: int
    lever: float  # log10(C_target / C_top)
    log_top: float  # log10 C_top
    rungs: int
    kernel: dict[tuple[str, str], float] = field(default_factory=dict)
    true_excess: float | None = None
    determined_share: float | None = None
    determined_pairs: frozenset[tuple[str, str]] = frozenset()
    true_excess_determined: float | None = None  # true excess over the determined pairs only


def enumerate_designs(n_scales: int) -> list[tuple[int, int, int]]:
    return [
        (s, f, t)
        for t in range(MIN_RUNGS, n_scales)
        for f in range(MIN_RUNGS - 1, t)
        for s in range(0, f - MIN_RUNGS + 2)
    ]


def _gaps(values: Mapping[str, float], names: list[str]) -> dict[tuple[str, str], float]:
    return {(a, b): float(values[a] - values[b]) for a, b in combinations(names, 2)}


def _mis(pred: Mapping[tuple[str, str], float], truth: Mapping[tuple[str, str], float]) -> float:
    scored = [(p, truth[k]) for k, p in pred.items() if np.isfinite(p) and p != 0 and truth[k] != 0]
    return float(np.mean([np.sign(p) != np.sign(t) for p, t in scored]))


def window_designs(
    frame: pd.DataFrame,
    truth_targets: Mapping[str, pd.Series] | None = None,
) -> list[WindowDesign]:
    """Every window design on the frame's ladder, with its kernel and optionally its true excess."""

    order = frame.groupby("scale_label")["compute"].first().sort_values()
    labels = list(order.index)
    log_c = np.log10(order.to_numpy(dtype=float))
    means = frame.groupby(["intervention", "compute"])["bpb"].mean()
    names = sorted(frame["intervention"].astype(str).unique())
    evidence = {}
    for t in range(MIN_RUNGS, len(labels)):
        rows = frame[np.isclose(frame["compute"], float(order.iloc[t]))]
        evidence[t] = evidence_from_cells({str(r): g["bpb"].to_list() for r, g in rows.groupby("intervention")})
    params: dict[tuple[int, int], dict[str, tuple[float, ...] | None]] = {}
    out = []
    for s, f, t in enumerate_designs(len(labels)):
        if (s, f) not in params:
            budgets = order.iloc[s : f + 1].to_numpy(dtype=float)
            fitted: dict[str, tuple[float, ...] | None] = {}
            for name in names:
                y = means.loc[name].reindex(budgets).to_numpy(dtype=float)
                try:
                    fitted[name] = tuple(fit_power_law(budgets, y))
                except (FitError, RuntimeError, ValueError):
                    fitted[name] = None
            params[(s, f)] = fitted
        target = float(order.iloc[t])
        projected = {
            n: (float(bpb_power_law(target, *p)) if p is not None else float("nan")) for n, p in params[(s, f)].items()
        }
        single = {n: float(means.loc[(n, float(order.iloc[f]))]) for n in names}
        proj_gaps, single_gaps = _gaps(projected, names), _gaps(single, names)
        left, right = expected_charges(proj_gaps, evidence[t]), expected_charges(single_gaps, evidence[t])
        design = WindowDesign(
            start=s,
            top=f,
            target=t,
            lever=float(log_c[t] - log_c[f]),
            log_top=float(log_c[f]),
            rungs=f - s + 1,
            kernel={p: 100 * (left[p] - right[p]) for p in left if p in right},
            determined_share=float(np.mean([e.determined for e in evidence[t]])),
            determined_pairs=frozenset((e.left, e.right) for e in evidence[t] if e.determined),
        )
        if truth_targets is not None:
            truth_gaps = _gaps(truth_targets[labels[t]].to_dict(), names)
            design.true_excess = 100 * (_mis(proj_gaps, truth_gaps) - _mis(single_gaps, truth_gaps))
            keep = design.determined_pairs
            if keep:
                sub_truth = {p: v for p, v in truth_gaps.items() if p in keep}
                design.true_excess_determined = 100 * (
                    _mis({p: v for p, v in proj_gaps.items() if p in keep}, sub_truth)
                    - _mis({p: v for p, v in single_gaps.items() if p in keep}, sub_truth)
                )
        out.append(design)
    return out


def slices(all_designs: list[WindowDesign]) -> dict[str, list[list[int]]]:
    """Index groups for the three factorial slices, as pre-registered."""

    lever: dict[tuple[int, int], list[int]] = {}
    rungs: dict[tuple[int, int], list[int]] = {}
    top: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(all_designs):
        lever.setdefault((d.start, d.top), []).append(i)
        rungs.setdefault((d.top, d.target), []).append(i)
        top.setdefault((d.rungs, int(round(d.lever / LEVER_BIN))), []).append(i)

    def keep_top(group: list[int]) -> bool:
        tops = {all_designs[i].top for i in group}
        span = max(all_designs[i].log_top for i in group) - min(all_designs[i].log_top for i in group)
        return len(tops) >= MIN_SLICE and span >= MIN_TOP_SPAN - 1e-9

    return {
        "lever": [g for g in lever.values() if len(g) >= MIN_SLICE],
        "top": [g for g in top.values() if keep_top(g)],
        "rungs": [g for g in rungs.values() if len(g) >= MIN_SLICE],
    }


def _within(y: np.ndarray, x: np.ndarray, groups: list[list[int]], covariate: np.ndarray | None = None) -> float:
    """Pooled within-group OLS slope of y on x (group fixed effects), optionally with one covariate."""

    rows_y, rows_x, rows_c = [], [], []
    for g in groups:
        idx = np.asarray(g)
        rows_y.append(y[idx] - y[idx].mean())
        rows_x.append(x[idx] - x[idx].mean())
        if covariate is not None:
            rows_c.append(covariate[idx] - covariate[idx].mean())
    yy, xx = np.concatenate(rows_y), np.concatenate(rows_x)
    if covariate is None:
        return float(xx @ yy / (xx @ xx))
    design = np.column_stack([xx, np.concatenate(rows_c)])
    return float(np.linalg.lstsq(design, yy, rcond=None)[0][0])


def statistics(all_designs: list[WindowDesign], excess: np.ndarray, groups: dict[str, list[list[int]]]) -> dict[str, float]:
    lever = np.array([d.lever for d in all_designs])
    log_top = np.array([d.log_top for d in all_designs])
    rungs = np.array([d.rungs for d in all_designs], dtype=float)
    design = np.column_stack([np.ones_like(lever), lever, log_top, rungs])
    beta = np.linalg.lstsq(design, excess, rcond=None)[0]
    return {
        "joint_lever": float(beta[1]),
        "joint_top": float(beta[2]),
        "joint_rungs": float(beta[3]),
        "slice_lever": _within(excess, lever, groups["lever"]),
        "slice_top": _within(excess, log_top, groups["top"], covariate=lever),
        "slice_rungs": _within(excess, rungs, groups["rungs"]),
    }


def excess_vector(all_designs: list[WindowDesign], drop: str | None = None) -> np.ndarray:
    return np.array(
        [np.mean([v for p, v in d.kernel.items() if drop is None or drop not in p]) for d in all_designs]
    )


def jackknife(all_designs: list[WindowDesign], groups: dict[str, list[list[int]]]) -> dict[str, dict[str, float]]:
    """Each statistic with its leave-one-recipe-out jackknife standard error."""

    full = statistics(all_designs, excess_vector(all_designs), groups)
    names = sorted({n for d in all_designs for pair in d.kernel for n in pair})
    loo = [statistics(all_designs, excess_vector(all_designs, drop), groups) for drop in names]
    n = len(names)
    out = {}
    for key in STATISTICS:
        values = np.array([row[key] for row in loo])
        out[key] = {
            "estimate": full[key],
            "se": float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2))),
        }
    return out


__all__ = [
    "STATISTICS",
    "WindowDesign",
    "enumerate_designs",
    "excess_vector",
    "jackknife",
    "slices",
    "statistics",
    "window_designs",
]
