"""Fixed-target slopes for any subset of DataDecide's sizes, from kernels computed once per window.

A window is a set of three or more sizes (consecutive within the geometry); its kernel at a target depends only
on that set and the target, so it is computed once with the frozen functions (asla.models.fit_power_law on
seed means, single-scale ranking at the window's top, the target's evidence_from_cells, expected_charges) and
reused across every geometry that contains the window. The slope is the frozen pooled within-slice slope
(slices: target and rung count with at least three windows), and its SE is the leave-one-recipe-out
jackknife. tests/test_geometry_scan.py checks this against asla.analysis.confound.window_designs with
asla.analysis.fixed_target on real geometries.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from asla.analysis.moderation import evidence_from_cells
from asla.analysis.target_scoring import expected_charges
from asla.models import FitError, bpb_power_law, fit_power_law

MIN_RUNGS = 3
MIN_SLICE = 3


class Scanner:
    def __init__(self, frame: pd.DataFrame):
        order = frame.groupby("scale_label")["compute"].first().sort_values()
        self.labels = list(order.index)
        self.compute = order.to_numpy(dtype=float)
        self.log_c = np.log10(self.compute)
        means = frame.groupby(["intervention", "compute"])["bpb"].mean()
        self.recipes = sorted(frame["intervention"].astype(str).unique())
        self.means = {r: np.array([float(means.loc[(r, c)]) for c in self.compute]) for r in self.recipes}
        self.pairs = list(combinations(self.recipes, 2))
        self.pair_index = {p: i for i, p in enumerate(self.pairs)}
        self.member = np.zeros((len(self.recipes), len(self.pairs)), dtype=bool)
        for i, (a, b) in enumerate(self.pairs):
            self.member[self.recipes.index(a), i] = True
            self.member[self.recipes.index(b), i] = True
        self._evidence: dict[int, list] = {}
        self._fits: dict[tuple[int, ...], dict[str, tuple[float, ...] | None]] = {}
        self._kernel: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        self._frame = frame

    def evidence(self, t: int) -> list:
        if t not in self._evidence:
            rows = self._frame[np.isclose(self._frame["compute"], self.compute[t])]
            self._evidence[t] = evidence_from_cells({str(r): g["bpb"].to_list() for r, g in rows.groupby("intervention")})
        return self._evidence[t]

    def kernel(self, window: tuple[int, ...], t: int) -> np.ndarray:
        """Per-pair kernel (points) in self.pairs order; NaN where the frozen charge skips the pair."""

        key = (window, t)
        if key in self._kernel:
            return self._kernel[key]
        if window not in self._fits:
            budgets = self.compute[list(window)]
            fits = {}
            for r in self.recipes:
                try:
                    fits[r] = tuple(fit_power_law(budgets, self.means[r][list(window)]))
                except (FitError, RuntimeError, ValueError):
                    fits[r] = None
            self._fits[window] = fits
        proj = {r: (float(bpb_power_law(self.compute[t], *p)) if p is not None else float("nan")) for r, p in self._fits[window].items()}
        single = {r: float(self.means[r][window[-1]]) for r in self.recipes}
        pg = {(a, b): proj[a] - proj[b] for a, b in self.pairs}
        sg = {(a, b): single[a] - single[b] for a, b in self.pairs}
        ev = self.evidence(t)
        left, right = expected_charges(pg, ev), expected_charges(sg, ev)
        k = np.full(len(self.pairs), np.nan)
        for p in left:
            if p in right and p in self.pair_index:
                k[self.pair_index[p]] = 100 * (left[p] - right[p])
        self._kernel[key] = k
        return k

    def designs(self, sizes: Sequence[int], targets: Iterable[int]) -> list[tuple[tuple[int, ...], int]]:
        sizes = sorted(sizes)
        out = []
        for t in targets:
            below = [s for s in sizes if s < t]
            for f in range(MIN_RUNGS - 1, len(below)):
                for s0 in range(0, f - MIN_RUNGS + 2):
                    out.append((tuple(below[s0 : f + 1]), t))
        return out

    def slices(self, ds: list[tuple[tuple[int, ...], int]]) -> list[list[int]]:
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (w, t) in enumerate(ds):
            groups.setdefault((t, len(w)), []).append(i)
        return [g for g in groups.values() if len(g) >= MIN_SLICE]

    def lever(self, ds) -> np.ndarray:
        return np.array([self.log_c[t] - self.log_c[w[-1]] for w, t in ds])

    @staticmethod
    def pooled_slope(y: np.ndarray, x: np.ndarray, groups: list[list[int]]) -> float:
        num = den = 0.0
        for g in groups:
            xg, yg = x[g] - x[g].mean(), y[g] - y[g].mean()
            num += float(xg @ yg)
            den += float(xg @ xg)
        return num / den if den > 0 else float("nan")

    def slope(self, sizes: Sequence[int], targets: Iterable[int]) -> dict:
        ds = self.designs(sizes, targets)
        groups = self.slices(ds)
        if not groups:
            return {"estimate": None, "se": None, "n_designs": 0}
        used = sorted({i for g in groups for i in g})
        x = self.lever(ds)
        kern = np.array([self.kernel(w, t) for w, t in ds])  # designs x pairs
        valid = np.isfinite(kern)
        k0 = np.where(valid, kern, 0.0)
        excess = k0.sum(axis=1) / valid.sum(axis=1)
        est = self.pooled_slope(excess, x, groups)
        loo = []
        for r in range(len(self.recipes)):
            keep = valid & ~self.member[r][None, :]
            ex_r = np.where(keep, kern, 0.0).sum(axis=1) / keep.sum(axis=1)
            loo.append(self.pooled_slope(ex_r, x, groups))
        loo = np.array(loo)
        n = len(loo)
        se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
        return {"estimate": est, "se": se, "n_designs": len(used), "excess": excess, "lever": x, "designs": ds, "groups": groups}


__all__ = ["Scanner"]
