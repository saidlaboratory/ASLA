"""Fixed-target designs on a budget-indexed ladder (Porian et al. 2024), with the frozen kernel and statistic.

A rung is a FLOP budget; a recipe's rung loss is its isoFLOP vertex. Projection fits
asla.models.fit_power_law to a window of consecutive admitted budgets and evaluates at the target
budget; single-scale ranking orders recipes by their loss at the window's top budget; the kernel is
the frozen expected-error charge of projection minus single-scale, in points. Target evidence uses the
known-variance limit of the frozen evidence rule (the suite has no seeds at the target), as in
PREDICTIONS_TASK_REPLICATION.md. The designs are asla.analysis.confound.WindowDesign objects, so the
statistics and jackknife of asla.analysis.fixed_target apply unchanged.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from asla.analysis.confound import MIN_RUNGS, WindowDesign
from asla.analysis.target_scoring import PairEvidence, expected_charges
from asla.models import FitError, bpb_power_law, fit_power_law

ALPHA = 0.05


def known_variance_evidence(target: Mapping[str, float], sds: Mapping[str, float]) -> list[PairEvidence]:
    names = sorted(target)
    m = len(names) * (len(names) - 1) // 2
    crit = float(stats.norm.ppf(1 - ALPHA / (2 * m)))
    out = []
    for a, b in combinations(names, 2):
        gap = float(target[a] - target[b])
        se = float(np.sqrt(sds[a] ** 2 + sds[b] ** 2))
        z = abs(gap) / se if se > 0 else np.inf
        out.append(
            PairEvidence(
                left=a,
                right=b,
                gap=gap,
                standard_error=se,
                welch_df=float("inf"),
                p_value=float(2 * stats.norm.sf(z)),
                determined=bool(z > crit),
                probability_observed_order_correct=0.5 if gap == 0 else float(stats.norm.cdf(z)),
            )
        )
    return out


def _mis(pred: Mapping[tuple[str, str], float], truth: Mapping[tuple[str, str], float]) -> float:
    scored = [(p, truth[k]) for k, p in pred.items() if np.isfinite(p) and p != 0 and truth[k] != 0]
    return float(np.mean([np.sign(p) != np.sign(t) for p, t in scored])) if scored else 0.0


def designs(
    loss: Mapping[str, Sequence[float]],
    log10_flops: Sequence[float],
    target_sds: Mapping[int, Mapping[str, float]],
    targets: Sequence[int],
    truth: Mapping[str, Sequence[float]] | None = None,
) -> list[WindowDesign]:
    """Every window design (>= MIN_RUNGS consecutive rungs, target above) for the given target indices.

    ``loss[recipe]`` and ``truth[recipe]`` are aligned with ``log10_flops`` (the admitted budgets, in
    order). ``target_sds[t][recipe]`` is the noise SD of the recipe's loss at target index t.
    """

    names = sorted(loss)
    compute = 10.0 ** np.asarray(log10_flops, dtype=float)
    out = []
    for t in targets:
        target = {r: float(loss[r][t]) for r in names}
        ev = known_variance_evidence(target, target_sds[t])
        determined = frozenset((e.left, e.right) for e in ev if e.determined)
        truth_gaps = None
        if truth is not None:
            tt = {r: float(truth[r][t]) for r in names}
            truth_gaps = {(a, b): tt[a] - tt[b] for a, b in combinations(names, 2)}
        for f in range(MIN_RUNGS - 1, t):
            for s in range(0, f - MIN_RUNGS + 2):
                projected, single = {}, {}
                for r in names:
                    y = np.asarray(loss[r][s : f + 1], dtype=float)
                    try:
                        projected[r] = float(bpb_power_law(compute[t], *fit_power_law(compute[s : f + 1], y)))
                    except (FitError, RuntimeError, ValueError):
                        projected[r] = float("nan")
                    single[r] = float(loss[r][f])
                pg = {(a, b): projected[a] - projected[b] for a, b in combinations(names, 2)}
                sg = {(a, b): single[a] - single[b] for a, b in combinations(names, 2)}
                left, right = expected_charges(pg, ev), expected_charges(sg, ev)
                d = WindowDesign(
                    start=s,
                    top=f,
                    target=t,
                    lever=float(np.log10(compute[t] / compute[f])),
                    log_top=float(np.log10(compute[f])),
                    rungs=f - s + 1,
                    kernel={p: 100 * (left[p] - right[p]) for p in left if p in right},
                    determined_share=float(np.mean([e.determined for e in ev])),
                    determined_pairs=determined,
                )
                if truth_gaps is not None:
                    d.true_excess = 100 * (_mis(pg, truth_gaps) - _mis(sg, truth_gaps))
                    keep = {k: v for k, v in truth_gaps.items() if k in determined}
                    if keep:
                        d.true_excess_determined = 100 * (
                            _mis({k: v for k, v in pg.items() if k in keep}, keep)
                            - _mis({k: v for k, v in sg.items() if k in keep}, keep)
                        )
                out.append(d)
    return out


__all__ = ["designs", "known_variance_evidence"]
