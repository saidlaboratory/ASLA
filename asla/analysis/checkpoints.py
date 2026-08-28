"""Checkpoint-augmented projection: fitting on intermediate checkpoints, correlation-aware.

Choshen, Zhang & Andreas (arXiv:2410.11840) report that fitting scaling laws to
intermediate checkpoints rather than only final losses substantially improves
*fit* accuracy, and that discarding roughly the first 10% of a run's checkpoints
matters. This module evaluates that technique as a *decision* rule, which the
paper does not do.

The statistical trap, handled explicitly here: checkpoints within a training run
are serially correlated. Treating ``n`` checkpoints as ``n`` independent
observations overstates the information they carry, understates every standard
error, and over-weights checkpoint-dense scales relative to their true
information content. Two weightings are provided:

* ``naive`` - every checkpoint is one observation. This is what a
  straightforward reading of the technique gives, and is the comparison arm.
* ``ar1`` (default) - residual autocorrelation is estimated per run, and each
  run's checkpoints are down-weighted by the AR(1) effective sample size
  ``n_eff = n (1 - rho) / (1 + rho)``, so a run contributes information rather
  than row count. This is the headline weighting.

Where the correction has to bite. A *uniform* per-point weight cancels in a
least-squares fit, so scaling every checkpoint of a run by the same factor is a
no-op for that run's parameters (verified in the tests). The correlation
correction must therefore act *within* a fit, by down-weighting scales that
contribute many densely-spaced checkpoints relative to scales that contribute
few. This is exactly the bias the trap creates: on the DataDecide ladder the
large scales carry 30-42 checkpoints each while the small ones carry 4-11, so
naive fitting silently reweights the ladder toward its top. The AR(1) weighting
allocates each *scale* an information budget of ``n_eff`` points, spread over
its ``n`` checkpoints, so a scale contributes what it knows rather than how
often it was evaluated.

Fitting data never includes the target budget; that is asserted, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from asla.analysis import splits
from asla.analysis.fits import normalize_budgets
from asla.data.schema import validate
from asla.models import FitError, bpb_power_law, fit_power_law

Weighting = Literal["naive", "ar1"]


@dataclass(frozen=True)
class RunAutocorrelation:
    """Estimated serial correlation of one run's residuals about its fitted curve."""

    intervention: str
    n_points: int
    rho: float
    effective_sample_size: float
    weight: float


def lag1_autocorrelation(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of residuals ordered by compute, clipped to [0, 0.99).

    Negative estimates are clipped to zero: they imply the effective sample size
    exceeds the row count, which we decline to claim. The upper clip keeps the
    effective sample size bounded away from zero.
    """

    values = np.asarray(residuals, dtype=float)
    if len(values) < 3:
        return 0.0
    centred = values - values.mean()
    denominator = float(np.sum(centred**2))
    if denominator <= 0:
        return 0.0
    rho = float(np.sum(centred[:-1] * centred[1:]) / denominator)
    return float(min(0.99, max(0.0, rho)))


def effective_sample_size(n_points: int, rho: float) -> float:
    """AR(1) effective sample size ``n (1 - rho) / (1 + rho)``, floored at 1."""

    if n_points <= 0:
        return 0.0
    return float(max(1.0, n_points * (1.0 - rho) / (1.0 + rho)))


def scale_of(compute: np.ndarray, scale_edges: np.ndarray) -> np.ndarray:
    """Assign each compute value to a scale bucket, used to group checkpoints by run."""

    return np.searchsorted(scale_edges, np.asarray(compute, dtype=float), side="right")


def estimate_run_autocorrelation(compute: np.ndarray, values: np.ndarray, intervention: str) -> RunAutocorrelation:
    """Fit a run's curve, then measure the serial correlation of its residuals."""

    order = np.argsort(compute)
    x, y = np.asarray(compute, dtype=float)[order], np.asarray(values, dtype=float)[order]
    try:
        params = fit_power_law(x, y)
        residuals = y - np.asarray(bpb_power_law(x, *params), dtype=float)
    except (FitError, ValueError):
        # Fall back to a log-log line, which is enough to expose serial structure.
        design = np.vstack([np.ones_like(x), np.log(x)]).T
        coefficients, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
        residuals = np.log(y) - design @ coefficients
    rho = lag1_autocorrelation(residuals)
    n_eff = effective_sample_size(len(x), rho)
    return RunAutocorrelation(
        intervention=str(intervention),
        n_points=int(len(x)),
        rho=rho,
        effective_sample_size=n_eff,
        weight=float(n_eff / len(x)),
    )


def _scale_labels(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Per-intervention scale label for each fitting compute value, aligned with ``checkpoint_cells``.

    Scales are the natural run boundary: all checkpoints of one model size come
    from one training run and share its serial correlation.
    """

    if "scale_label" not in df.columns:
        return {}
    values = np.asarray(budgets, dtype=float)
    mask = df["compute"].astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))
    fit_df = df[mask]
    out: dict[str, np.ndarray] = {}
    for name, group in fit_df.groupby("intervention", sort=True):
        by_compute = group.groupby("compute", sort=True)["scale_label"].first()
        out[str(name)] = by_compute.to_numpy()
    return out


def checkpoint_cells(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-intervention (compute, seed-mean value) over every fitting checkpoint."""

    values = np.asarray(budgets, dtype=float)
    mask = df["compute"].astype(float).apply(lambda x: bool(np.any(np.isclose(x, values))))
    fit_df = df[mask]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, group in fit_df.groupby("intervention", sort=True):
        cells = group.groupby("compute", sort=True)["bpb"].mean()
        out[str(name)] = (cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))
    return out


def _per_scale_sigma(compute: np.ndarray, values: np.ndarray, labels: np.ndarray | None) -> np.ndarray | None:
    """Per-point sigma giving each scale an AR(1) effective-sample-size information budget.

    Within a scale, ``n`` serially correlated checkpoints carry ``n_eff`` points
    of information, so each is weighted ``w = n_eff / n`` and enters the fit as
    ``sigma = 1/sqrt(w)``. Because the weight differs *between* scales (dense
    scales are penalised more), this does not cancel the way a uniform weight
    does.
    """

    if labels is None or len(labels) != len(compute):
        return None
    sigma = np.ones(len(compute), dtype=float)
    for label in np.unique(labels):
        mask = labels == label
        n_points = int(mask.sum())
        if n_points < 3:
            continue
        order = np.argsort(compute[mask])
        x, y = compute[mask][order], values[mask][order]
        design = np.vstack([np.ones_like(x), np.log(x)]).T
        coefficients, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
        rho = lag1_autocorrelation(np.log(y) - design @ coefficients)
        weight = effective_sample_size(n_points, rho) / n_points
        sigma[mask] = 1.0 / np.sqrt(max(weight, 1e-6))
    return sigma


def fit_checkpoint_augmented(
    df: pd.DataFrame,
    budgets: Iterable[float],
    weighting: Weighting = "ar1",
) -> tuple[dict[str, tuple[float, float, float]], dict[str, RunAutocorrelation]]:
    """Fit each intervention on all supplied checkpoints, optionally correlation-aware.

    With ``weighting="ar1"`` each *scale* within an intervention is given an
    AR(1) effective-sample-size information budget, so checkpoint-dense scales
    do not silently dominate the fit.
    """

    validate(df)
    splits.assert_not_test(df)
    fit_budgets = normalize_budgets(budgets)
    cells = checkpoint_cells(df, fit_budgets)
    if not cells:
        raise FitError("checkpoint-augmented fit found no fitting rows")
    scale_labels = _scale_labels(df, fit_budgets)
    params: dict[str, tuple[float, float, float]] = {}
    diagnostics: dict[str, RunAutocorrelation] = {}
    for name, (x, y) in cells.items():
        autocorrelation = estimate_run_autocorrelation(x, y, name)
        diagnostics[name] = autocorrelation
        if weighting == "naive":
            sigma = None
        elif weighting == "ar1":
            sigma = _per_scale_sigma(x, y, scale_labels.get(name))
        else:
            raise ValueError(f"unknown weighting {weighting!r}; choose 'naive' or 'ar1'")
        params[name] = fit_power_law(x, y, sigma=sigma)
    return params, diagnostics


def make_checkpoint_ranker(weighting: Weighting = "ar1"):
    """Return a ranker that fits on every checkpoint below the target."""

    def _ranker(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
        # Use every checkpoint strictly below the target, not only the ladder
        # budgets the caller passed: that is the whole point of the technique.
        # The target itself can never enter the fit.
        available = np.asarray(sorted(float(c) for c in df["compute"].unique()), dtype=float)
        fitting = tuple(c for c in available if c < float(target) and not np.isclose(c, float(target)))
        if not fitting:
            raise FitError("no checkpoints below the target budget")
        if any(c >= float(target) for c in fitting):
            raise AssertionError("target budget leaked into the checkpoint fitting set")
        params, _ = fit_checkpoint_augmented(df, fitting, weighting=weighting)
        projected = {name: float(bpb_power_law(target, *par)) for name, par in params.items()}
        return pd.Series(projected).sort_values(kind="mergesort")

    _ranker.__name__ = f"checkpoint_ranker_{weighting}"
    return _ranker


def autocorrelation_summary(diagnostics: dict[str, RunAutocorrelation]) -> dict[str, float | int]:
    """Aggregate the per-run serial-correlation diagnostics."""

    rhos = np.asarray([d.rho for d in diagnostics.values()], dtype=float)
    weights = np.asarray([d.weight for d in diagnostics.values()], dtype=float)
    counts = np.asarray([d.n_points for d in diagnostics.values()], dtype=float)
    return {
        "n_runs": int(len(diagnostics)),
        "mean_rho": float(rhos.mean()),
        "median_rho": float(np.median(rhos)),
        "max_rho": float(rhos.max()),
        "mean_weight": float(weights.mean()),
        "mean_n_points": float(counts.mean()),
        "mean_effective_points": float((counts * weights).mean()),
    }


__all__ = [
    "RunAutocorrelation",
    "autocorrelation_summary",
    "checkpoint_cells",
    "effective_sample_size",
    "estimate_run_autocorrelation",
    "fit_checkpoint_augmented",
    "lag1_autocorrelation",
    "make_checkpoint_ranker",
]
