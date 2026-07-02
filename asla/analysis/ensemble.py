"""Misspecification-aware ensemble projection across curve families.

A single functional form extrapolated beyond its fitting range can be
confidently wrong: on saturating truths the pure power law projects
improvements that never arrive. This module fits a small dictionary of curve
families per intervention, weights each family by its leave-largest-budget-out
(LLBO) extrapolation error, and reports both a weighted ensemble projection
and a *disagreement* — the weighted spread of family projections at the
target. Disagreement divided by the target seed-noise band gives the
extrapolation **reliability score** ``rho``: when ``rho >> 1`` the fitting
data cannot distinguish families whose target predictions differ by more than
run-to-run noise, so any single-family projection is untrustworthy there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets
from asla.data.schema import validate
from asla.models import (
    FitError,
    bpb_damped_power_law,
    bpb_power_law,
    bpb_saturating,
    fit_damped_power_law,
    fit_power_law,
    fit_saturating,
)


@dataclass(frozen=True)
class CurveFamily:
    """A fit/predict pair with the distinct-budget count it needs to be identifiable."""

    name: str
    fit: Callable[[np.ndarray, np.ndarray], tuple[float, ...]]
    predict: Callable[[float, tuple[float, ...]], float]
    min_distinct_budgets: int


FAMILIES: tuple[CurveFamily, ...] = (
    CurveFamily(
        name="power_law",
        fit=lambda x, y: fit_power_law(x, y),
        predict=lambda target, params: float(bpb_power_law(target, *params)),
        min_distinct_budgets=3,
    ),
    CurveFamily(
        name="saturating",
        fit=lambda x, y: fit_saturating(x, y),
        predict=lambda target, params: float(bpb_saturating(target, *params)),
        min_distinct_budgets=3,
    ),
    CurveFamily(
        name="damped_power_law",
        fit=lambda x, y: fit_damped_power_law(x, y),
        predict=lambda target, params: float(bpb_damped_power_law(target, *params)),
        min_distinct_budgets=4,
    ),
)


@dataclass(frozen=True)
class FamilyFit:
    """One family's full fit, LLBO loss, pseudo-BMA weight, and target projection."""

    name: str
    params: tuple[float, ...]
    llbo_loss: float
    weight: float
    projection: float


@dataclass(frozen=True)
class EnsembleProjection:
    """Weighted cross-family projection with its disagreement at the target."""

    point: float
    disagreement: float
    families: tuple[FamilyFit, ...]
    failed_families: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "point": float(self.point),
            "disagreement": float(self.disagreement),
            "families": {
                fam.name: {
                    "llbo_loss": None if not np.isfinite(fam.llbo_loss) else float(fam.llbo_loss),
                    "weight": float(fam.weight),
                    "projection": float(fam.projection),
                }
                for fam in self.families
            },
            "failed_families": list(self.failed_families),
        }


def pseudo_bma_weights(losses: np.ndarray) -> np.ndarray:
    """Softmax of negative losses with a scale-invariant temperature.

    ``w_m \\propto exp(-loss_m / T)`` with ``T = mean(loss)``, so multiplying
    all losses by a constant leaves the weights unchanged and equal losses
    (including all-zero) give uniform weights.
    """

    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        raise ValueError("at least one loss is required")
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("losses must be finite and non-negative")
    temperature = float(np.mean(losses))
    if temperature == 0.0:
        return np.full(len(losses), 1.0 / len(losses))
    logits = -losses / temperature
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def ensemble_projection(
    compute: np.ndarray,
    bpb: np.ndarray,
    target: float,
    families: Iterable[CurveFamily] = FAMILIES,
) -> EnsembleProjection:
    """Project one intervention's target BPB with pseudo-BMA family weighting.

    Every family is fit on all points for its projection. Weights come from
    the leave-largest-budget-out loss: refit without the largest distinct
    budget and score the absolute error against the held-out budget's mean.
    Families that need more distinct budgets than the LLBO training set offers,
    or whose fits fail, are excluded and recorded in ``failed_families``.
    """

    x = np.asarray(compute, dtype=float)
    y = np.asarray(bpb, dtype=float)
    distinct = np.unique(x)
    if len(distinct) < 4:
        raise FitError(
            f"ensemble projection requires at least 4 distinct compute budgets for "
            f"leave-largest-budget-out weighting, got {len(distinct)}"
        )
    largest = float(distinct[-1])
    held = np.isclose(x, largest)
    x_train, y_train = x[~held], y[~held]
    held_mean = float(np.mean(y[held]))
    train_distinct = len(distinct) - 1

    fits: list[FamilyFit] = []
    failed: list[str] = []
    for family in families:
        if train_distinct < family.min_distinct_budgets:
            failed.append(family.name)
            continue
        try:
            full_params = family.fit(x, y)
            llbo_params = family.fit(x_train, y_train)
        except FitError:
            failed.append(family.name)
            continue
        loss = abs(family.predict(largest, llbo_params) - held_mean)
        fits.append(
            FamilyFit(
                name=family.name,
                params=tuple(float(p) for p in full_params),
                llbo_loss=float(loss),
                weight=float("nan"),
                projection=family.predict(float(target), full_params),
            )
        )
    if not fits:
        raise FitError(f"no curve family could be fit; failures: {failed}")

    weights = pseudo_bma_weights(np.asarray([fam.llbo_loss for fam in fits]))
    fits = [
        FamilyFit(fam.name, fam.params, fam.llbo_loss, float(w), fam.projection)
        for fam, w in zip(fits, weights)
    ]
    projections = np.asarray([fam.projection for fam in fits], dtype=float)
    point = float(np.sum(weights * projections))
    disagreement = float(np.sqrt(np.sum(weights * (projections - point) ** 2)))
    return EnsembleProjection(
        point=point,
        disagreement=disagreement,
        families=tuple(fits),
        failed_families=tuple(failed),
    )


def ensemble_rank(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    families: Iterable[CurveFamily] = FAMILIES,
) -> pd.Series:
    """Rank interventions by ensemble target projection, ascending."""

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    budget_values = np.asarray(fit_budgets, dtype=float)
    values: dict[str, float] = {}
    for intervention, group in df.groupby("intervention", sort=True):
        mask = group["compute"].astype(float).apply(lambda c: bool(np.any(np.isclose(c, budget_values))))
        rows = group[mask]
        if rows.empty:
            raise ValueError(f"no fitting rows found for intervention {str(intervention)!r}")
        proj = ensemble_projection(
            rows["compute"].to_numpy(dtype=float),
            rows["bpb"].to_numpy(dtype=float),
            target,
            families=families,
        )
        values[str(intervention)] = proj.point
    return pd.Series(values).sort_values(kind="mergesort")


def ensemble_report(
    df: pd.DataFrame,
    budgets: Iterable[float],
    target: float,
    noise_band: float | None,
    families: Iterable[CurveFamily] = FAMILIES,
) -> dict[str, dict[str, object]]:
    """Per-intervention ensemble projections with reliability scores.

    ``reliability`` is ``disagreement / noise_band``; it is ``None`` when the
    noise band is unestimated rather than fabricated from a guess.
    """

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    budget_values = np.asarray(fit_budgets, dtype=float)
    report: dict[str, dict[str, object]] = {}
    for intervention, group in df.groupby("intervention", sort=True):
        mask = group["compute"].astype(float).apply(lambda c: bool(np.any(np.isclose(c, budget_values))))
        rows = group[mask]
        if rows.empty:
            raise ValueError(f"no fitting rows found for intervention {str(intervention)!r}")
        proj = ensemble_projection(
            rows["compute"].to_numpy(dtype=float),
            rows["bpb"].to_numpy(dtype=float),
            target,
            families=families,
        )
        entry = proj.as_dict()
        if noise_band is not None and noise_band > 0:
            entry["reliability"] = float(proj.disagreement / noise_band)
        else:
            entry["reliability"] = None
        report[str(intervention)] = entry
    return report
