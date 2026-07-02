"""Scaling-law models and low-level fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import curve_fit


class FitError(RuntimeError):
    """Raised when fitting a scaling law fails."""


FitForm = Literal["compute_power_law", "chinchilla"]


@dataclass(frozen=True)
class Projection:
    """A fitted projection with bootstrap uncertainty."""

    point: float
    std: float
    lo: float
    hi: float


@dataclass(frozen=True)
class FitDiagnostics:
    """Goodness-of-fit summary for a fitted scaling law."""

    r_squared: float
    rmse: float
    max_abs_residual: float
    n_points: int
    dof: int


def fit_diagnostics(y: np.ndarray, y_pred: np.ndarray, n_params: int) -> FitDiagnostics:
    """Summarize residuals of a fit; ``r_squared`` is 1 for a perfect constant fit."""

    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y) != len(y_pred):
        raise ValueError(f"y and y_pred must have the same length, got {len(y)} and {len(y_pred)}")
    residuals = y - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot)
    return FitDiagnostics(
        r_squared=float(r_squared),
        rmse=float(np.sqrt(ss_res / len(y))) if len(y) else float("nan"),
        max_abs_residual=float(np.max(np.abs(residuals))) if len(y) else float("nan"),
        n_points=int(len(y)),
        dof=int(len(y) - n_params),
    )


def bpb_power_law(C: np.ndarray | float, E: float, A: float, alpha: float) -> np.ndarray | float:
    """Compute ``E + A * C**(-alpha)`` for BPB scaling-law projections."""

    return E + A * np.asarray(C, dtype=float) ** (-alpha)


def bpb_saturating(C: np.ndarray | float, floor: float, drop: float, c_half: float) -> np.ndarray | float:
    """Compute a saturating BPB curve used only by synthetic misspecification tests."""

    return floor + drop / (1.0 + np.asarray(C, dtype=float) / c_half)


def bpb_chinchilla(
    N: np.ndarray | float,
    D: np.ndarray | float,
    E: float,
    A: float,
    a: float,
    B: float,
    b: float,
) -> np.ndarray | float:
    """Compute ``E + A * N**(-a) + B * D**(-b)`` for two-axis controlled grids."""

    return E + A * np.asarray(N, dtype=float) ** (-a) + B * np.asarray(D, dtype=float) ** (-b)


def _check_sigma(sigma: np.ndarray | None, n: int) -> np.ndarray | None:
    """Validate optional per-point standard errors for weighted fitting."""

    if sigma is None:
        return None
    s = np.asarray(sigma, dtype=float)
    if len(s) != n:
        raise FitError(f"sigma must have the same length as the data, got {len(s)} and {n}")
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise FitError("sigma values must be finite and positive")
    return s


def fit_power_law(compute: np.ndarray, bpb: np.ndarray, sigma: np.ndarray | None = None) -> Tuple[float, float, float]:
    """Fit the BPB power law and return ``(E, A, alpha)``.

    At least three distinct compute budgets are required. ``FitError`` is
    raised with context for invalid inputs or optimizer failures.

    Compute is rescaled internally by its smallest value so the optimizer
    bounds are unit-invariant: raw FLOP counts and O(1) relative units fit
    equally well. Returned parameters are in the caller's raw compute units.

    ``sigma`` gives per-point standard errors for weighted least squares,
    e.g. seed-count-aware cell standard errors.
    """

    x = np.asarray(compute, dtype=float)
    y = np.asarray(bpb, dtype=float)
    if len(x) != len(y):
        raise FitError(f"compute and bpb must have the same length, got {len(x)} and {len(y)}")
    if len(x) == 0:
        raise FitError("fit_power_law requires at least one row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FitError("compute and BPB values must be finite")
    if len(np.unique(x)) < 3:
        raise FitError("fit_power_law requires at least 3 distinct compute values")
    if np.any(x <= 0):
        raise FitError("compute values must be positive")
    y_min = float(np.min(y))
    if not np.isfinite(y_min) or y_min <= 0:
        raise FitError("BPB values must be finite and positive")
    s = _check_sigma(sigma, len(x))

    x_ref = float(np.min(x))
    x_scaled = x / x_ref
    e0 = max(0.0, min(y_min * 0.9, y_min - 1e-6))
    a0 = min(5.0, max(0.01, float(np.max(y) - e0)))
    p0 = (e0, a0, 0.3)
    bounds = ([0.0, 0.0, 0.05], [y_min, 5.0, 2.0])
    try:
        params, _ = curve_fit(
            bpb_power_law,
            x_scaled,
            y,
            p0=p0,
            bounds=bounds,
            sigma=s,
            maxfev=50000,
        )
    except Exception as exc:  # pragma: no cover - exact scipy exception varies
        raise FitError(f"power-law fit failed: {exc}") from exc
    if not np.all(np.isfinite(params)):
        raise FitError("power-law fit produced non-finite parameters")
    E, A_scaled, alpha = (float(p) for p in params)
    A = A_scaled * x_ref**alpha
    if not np.isfinite(A):
        raise FitError("power-law fit produced non-finite parameters after unit conversion")
    return (E, A, alpha)


def fit_power_law_diagnostics(
    compute: np.ndarray,
    bpb: np.ndarray,
    sigma: np.ndarray | None = None,
) -> tuple[Tuple[float, float, float], FitDiagnostics]:
    """Fit the BPB power law and return parameters with residual diagnostics."""

    params = fit_power_law(compute, bpb, sigma=sigma)
    y_pred = np.asarray(bpb_power_law(np.asarray(compute, dtype=float), *params), dtype=float)
    return params, fit_diagnostics(np.asarray(bpb, dtype=float), y_pred, n_params=3)


def fit_chinchilla(
    params_n: np.ndarray,
    tokens_d: np.ndarray,
    bpb: np.ndarray,
    sigma: np.ndarray | None = None,
) -> Tuple[float, float, float, float, float]:
    """Fit ``E + A * N**(-a) + B * D**(-b)`` and return ``(E, A, a, B, b)``.

    ``params_n`` and ``tokens_d`` are rescaled internally by their smallest
    values so the optimizer bounds are unit-invariant; returned parameters are
    in the caller's raw units.
    """

    n = np.asarray(params_n, dtype=float)
    d = np.asarray(tokens_d, dtype=float)
    y = np.asarray(bpb, dtype=float)
    if not (len(n) == len(d) == len(y)):
        raise FitError("params_n, tokens_d, and bpb must have the same length")
    if len(n) == 0:
        raise FitError("fit_chinchilla requires at least one row")
    if not np.isfinite(n).all() or not np.isfinite(d).all() or not np.isfinite(y).all():
        raise FitError("params_n, tokens_d, and BPB values must be finite")
    if np.any(n <= 0) or np.any(d <= 0):
        raise FitError("params_n and tokens_d values must be positive")
    if len(np.unique(n)) < 3 or len(np.unique(d)) < 3:
        raise FitError("fit_chinchilla requires at least 3 distinct params_n and tokens_d values")
    y_min = float(np.min(y))
    if not np.isfinite(y_min) or y_min <= 0:
        raise FitError("BPB values must be finite and positive")
    s = _check_sigma(sigma, len(n))

    def _model(xdata: tuple[np.ndarray, np.ndarray], E: float, A: float, a: float, B: float, b: float) -> np.ndarray:
        n_values, d_values = xdata
        return np.asarray(bpb_chinchilla(n_values, d_values, E, A, a, B, b), dtype=float)

    n_ref = float(np.min(n))
    d_ref = float(np.min(d))
    e0 = max(0.0, min(y_min * 0.8, y_min - 1e-6))
    residual = max(0.02, float(np.max(y) - e0))
    p0 = (e0, min(5.0, residual / 2.0), 0.3, min(5.0, residual / 2.0), 0.3)
    bounds = ([0.0, 0.0, 0.05, 0.0, 0.05], [y_min, 5.0, 2.0, 5.0, 2.0])
    try:
        params, _ = curve_fit(
            _model,
            (n / n_ref, d / d_ref),
            y,
            p0=p0,
            bounds=bounds,
            sigma=s,
            maxfev=100000,
        )
    except Exception as exc:  # pragma: no cover - exact scipy exception varies
        raise FitError(f"chinchilla fit failed: {exc}") from exc
    if not np.all(np.isfinite(params)):
        raise FitError("chinchilla fit produced non-finite parameters")
    E, A_scaled, a, B_scaled, b = (float(p) for p in params)
    A = A_scaled * n_ref**a
    B = B_scaled * d_ref**b
    if not np.isfinite(A) or not np.isfinite(B):
        raise FitError("chinchilla fit produced non-finite parameters after unit conversion")
    return (E, A, a, B, b)


def bootstrap_projection(
    compute: np.ndarray,
    bpb: np.ndarray,
    target: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Bootstrap a target-budget projection.

    Rows are resampled nonparametrically. Failed resamples are skipped. Returns
    ``(point, std, lo, hi)`` where ``lo`` and ``hi`` are the 5th and 95th
    percentiles of successful bootstrap projections.
    """

    x = np.asarray(compute, dtype=float)
    y = np.asarray(bpb, dtype=float)
    if n_boot <= 0:
        raise FitError("n_boot must be positive")
    params = fit_power_law(x, y)
    point = float(bpb_power_law(target, *params))
    projections: list[float] = []
    n = len(x)
    min_successes = max(10, int(np.ceil(0.25 * n_boot)))
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_params = fit_power_law(x[idx], y[idx])
        except FitError:
            continue
        projections.append(float(bpb_power_law(target, *boot_params)))

    if len(projections) < min_successes:
        raise FitError(
            f"too few bootstrap resamples fit successfully: {len(projections)}/{n_boot} "
            f"(minimum {min_successes})"
        )
    arr = np.asarray(projections, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    lo, hi = np.percentile(arr, [5.0, 95.0])
    return point, std, float(lo), float(hi)
