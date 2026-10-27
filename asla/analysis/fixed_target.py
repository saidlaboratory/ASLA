"""The lever-arm effect at a fixed target (PREDICTIONS_TASK_FIXED_TARGET.md).

Built on the window designs of :mod:`asla.analysis.confound`. At a fixed target the
reference, and so its determinacy, is the same for every design, so a slope of
projection's excess on lever arm within a target cannot come from target size.

Two statistics families:

* fixed-target slices: target t and rung count k fixed, the window slides; the
  pooled within-slice slope of excess on log10 lever arm, overall and per target;
* the target-varied slice (i) rescored on the pairs determined at each design's
  target only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from asla.analysis.confound import MIN_SLICE, WindowDesign, _within

ADMIT_SHARE = 0.5  # pre-registered: targets whose determined share is at least this


def admitted_targets(all_designs: Sequence[WindowDesign]) -> list[int]:
    shares = {d.target: d.determined_share for d in all_designs}
    return sorted(t for t, share in shares.items() if share is not None and share >= ADMIT_SHARE)


def fixed_target_slices(all_designs: Sequence[WindowDesign], targets: Sequence[int]) -> dict[int, list[list[int]]]:
    """Per admitted target, index groups of designs sharing that target and a rung count."""

    groups: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(all_designs):
        if d.target in targets:
            groups.setdefault((d.target, d.rungs), []).append(i)
    out: dict[int, list[list[int]]] = {t: [] for t in targets}
    for (t, _), idx in sorted(groups.items()):
        if len(idx) >= MIN_SLICE:
            out[t].append(idx)
    return out


def target_varied_slices(all_designs: Sequence[WindowDesign]) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(all_designs):
        groups.setdefault((d.start, d.top), []).append(i)
    return [g for g in groups.values() if len(g) >= MIN_SLICE]


def excess_all(all_designs: Sequence[WindowDesign], drop: str | None = None) -> np.ndarray:
    return np.array(
        [np.mean([v for p, v in d.kernel.items() if drop is None or drop not in p]) for d in all_designs]
    )


def excess_determined(all_designs: Sequence[WindowDesign], drop: str | None = None) -> np.ndarray:
    """Mean kernel over the pairs determined at each design's target; nan if none remain."""

    out = []
    for d in all_designs:
        values = [v for p, v in d.kernel.items() if p in d.determined_pairs and (drop is None or drop not in p)]
        out.append(np.mean(values) if values else np.nan)
    return np.array(out)


def _slope(excess: np.ndarray, lever: np.ndarray, groups: list[list[int]]) -> float:
    kept = [[i for i in g if np.isfinite(excess[i])] for g in groups]
    kept = [g for g in kept if len(g) >= 2]
    return _within(np.nan_to_num(excess), lever, kept)


def statistics(
    all_designs: Sequence[WindowDesign],
    excess: np.ndarray,
    excess_det: np.ndarray,
    ft_groups: dict[int, list[list[int]]],
    tv_groups: list[list[int]],
    labels: Sequence[str],
) -> dict[str, float]:
    lever = np.array([d.lever for d in all_designs])
    out = {"fixed_target_pooled": _slope(excess, lever, [g for gs in ft_groups.values() for g in gs])}
    for t, gs in ft_groups.items():
        out[f"fixed_target_{labels[t]}"] = _slope(excess, lever, gs)
    out["target_varied_all_pairs"] = _slope(excess, lever, tv_groups)
    out["target_varied_determined"] = _slope(excess_det, lever, tv_groups)
    return out


def jackknife(
    all_designs: Sequence[WindowDesign],
    ft_groups: dict[int, list[list[int]]],
    tv_groups: list[list[int]],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    def stat(drop: str | None) -> dict[str, float]:
        return statistics(
            all_designs, excess_all(all_designs, drop), excess_determined(all_designs, drop), ft_groups, tv_groups, labels
        )

    full = stat(None)
    names = sorted({n for d in all_designs for pair in d.kernel for n in pair})
    loo = [stat(drop) for drop in names]
    n = len(names)
    out = {}
    for key in full:
        values = np.array([row[key] for row in loo])
        out[key] = {"estimate": full[key], "se": float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))}
    return out


def true_statistics(
    all_designs: Sequence[WindowDesign],
    true_excess: np.ndarray,
    true_excess_det: np.ndarray,
    ft_groups: dict[int, list[list[int]]],
    tv_groups: list[list[int]],
    labels: Sequence[str],
) -> dict[str, float]:
    return statistics(all_designs, true_excess, true_excess_det, ft_groups, tv_groups, labels)


__all__: list[str] = [
    "ADMIT_SHARE",
    "admitted_targets",
    "excess_all",
    "excess_determined",
    "fixed_target_slices",
    "jackknife",
    "statistics",
    "target_varied_slices",
    "true_statistics",
]

