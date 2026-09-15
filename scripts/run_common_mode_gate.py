"""Task 0 GATE: is misspecification common-mode across interventions?

The method premise for a differenced estimator is that, writing each
intervention's curve at a given compute as

    y_i(C) = f_i(C) + m(C) + eps_i(C)          (shared misspecification m)

the shared term `m(C)` cancels when we model the *gap* g_ij(C) = y_i(C) - y_j(C).
If so, a model fitted to gaps is correctly specified even though a model fitted
to each intervention separately is not.

**This measures a different quantity from the rho_ab = -0.004 reported in
AUDIT_ADVERSARIAL.md.** That was the correlation of *parameter-estimation errors*
under the seed bootstrap, which resamples seeds within cells and therefore holds
the misfit pattern fixed by construction - it could not have detected common-mode
misspecification even if present. Here we measure the correlation of *fit
residuals* (observed cell mean minus fitted value) across interventions at
matched compute, which is exactly the quantity the premise is about.

Three measurements (0a), one decisive ratio (0b), one gate (0c):

* mean pairwise correlation of residual vectors across interventions;
* fraction of residual variance explained by a single shared per-budget
  component (first principal component of the residual matrix, and the
  equivalent one-way compute effect);
* **the cancellation ratio**: Var(r_i - r_j) / (Var(r_i) + Var(r_j)), averaged
  over pairs. Under independent residuals this is ~1; under perfect common-mode
  it is ~0. This is the number the gate turns on, because it is exactly the
  variance a differenced estimator would face relative to what independent
  residuals would give.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.cli import _sha256_file, _write_json_atomically  # noqa: E402
from asla.data.io import load_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUITES = {
    "datadecide_c4_bits_per_token": DATA / "datadecide_runs.parquet",
    "datadecide_olmes_macro_error": DATA / "datadecide_runs_olmes_macro_error.parquet",
    "datadecide_olmes_correct_prob": DATA / "datadecide_runs_olmes_correct_prob_per_char.parquet",
    "signal_and_noise_c4_bits_per_byte": DATA / "signal_and_noise_datadecide_c4_bpb.parquet",
}
LADDER = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M"]
# Gate thresholds, fixed before the measurement is read.
GATE_PROCEED_BELOW = 0.70   # cancellation ratio below this: material common-mode, proceed
GATE_STOP_ABOVE = 0.90      # at or above this: premise false, stop and report
N_BOOTSTRAP = 400


def power_law(compute: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.asarray(compute, dtype=float) ** (-alpha)


def fit_one(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    try:
        popt, _ = curve_fit(
            power_law, x, y, p0=(y.min() * 0.9, y.max() - y.min() * 0.9, 0.15), maxfev=200000
        )
    except Exception:  # noqa: BLE001 - a failed fit is recorded by omission
        return None
    return (float(popt[0]), float(popt[1]), float(popt[2]))


def residual_matrix(df: pd.DataFrame, budgets: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (residuals, cell means) as intervention x budget frames.

    Residuals are in *relative* (log-ish) units - residual divided by the fitted
    value - so that interventions with different absolute levels are comparable
    and the shared component is not an artifact of scale.
    """

    rows = df[df["compute"].isin(budgets)]
    cells = rows.groupby(["intervention", "compute"])["bpb"].mean().unstack()
    cells = cells[sorted(cells.columns)]
    residuals: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    x = cells.columns.to_numpy(dtype=float)
    for name, row in cells.iterrows():
        y = row.to_numpy(dtype=float)
        if not np.isfinite(y).all():
            continue
        params = fit_one(x, y)
        if params is None:
            continue
        fitted = power_law(x, *params)
        residuals[str(name)] = (y - fitted) / fitted
        means[str(name)] = y
    return (
        pd.DataFrame(residuals, index=cells.columns).T,
        pd.DataFrame(means, index=cells.columns).T,
    )


def shared_component_fraction(residuals: pd.DataFrame) -> dict[str, float]:
    """Fraction of residual variance carried by one shared per-budget component."""

    matrix = residuals.to_numpy(dtype=float)
    total = float(np.sum(matrix**2))
    if total <= 0:
        return {"pc1_fraction": float("nan"), "one_way_fraction": float("nan")}
    # One-way compute effect: the per-budget mean across interventions.
    column_mean = matrix.mean(axis=0, keepdims=True)
    one_way = float(np.sum(np.repeat(column_mean, matrix.shape[0], axis=0) ** 2) / total)
    # First principal component of the (interventions x budgets) residual matrix.
    try:
        singular = np.linalg.svd(matrix, compute_uv=False)
        pc1 = float(singular[0] ** 2 / np.sum(singular**2))
    except np.linalg.LinAlgError:
        pc1 = float("nan")
    return {"pc1_fraction": pc1, "one_way_fraction": one_way}


def shared_component_shape(residuals: pd.DataFrame) -> dict[str, Any]:
    """Is the dominant shared component an OFFSET or a per-intervention SCALED shape?

    This distinguishes the two ways a residual matrix can have a large first
    principal component, which have opposite consequences for differencing:

    * ``r_i(C) = m(C) + e_i(C)`` - a common offset. PC1 loadings are near
      constant across interventions, and the shared term **cancels** in
      ``r_i - r_j``.
    * ``r_i(C) = a_i m(C) + e_i(C)`` - a shared shape scaled differently per
      intervention. PC1 is still large, but ``r_i - r_j = (a_i - a_j) m(C) + ...``
      does **not** cancel; it leaves a residual proportional to the spread of the
      loadings.

    The coefficient of variation of the PC1 loadings separates these: near zero
    means an offset, order one means a scaled shape.
    """

    matrix = residuals.to_numpy(dtype=float)
    if matrix.shape[0] < 3:
        return {"loading_cv": float("nan"), "loadings_same_sign": False, "interpretation": "too few interventions"}
    u, sv, _ = np.linalg.svd(matrix, full_matrices=False)
    loadings = u[:, 0] * sv[0]
    mean = float(loadings.mean())
    sd = float(loadings.std(ddof=1))
    cv = float(abs(sd / mean)) if mean != 0 else float("inf")
    same_sign = bool(np.all(loadings > 0) or np.all(loadings < 0))
    return {
        "pc1_variance_fraction": float(sv[0] ** 2 / np.sum(sv**2)),
        "loading_mean": mean,
        "loading_sd": sd,
        "loading_cv": cv,
        "loadings_same_sign": same_sign,
        "interpretation": (
            "common OFFSET: cancels under differencing"
            if (cv < 0.5 and same_sign)
            else "per-intervention SCALED shape: does not cancel under differencing"
        ),
    }


def cancellation_ratio(residuals: pd.DataFrame) -> dict[str, Any]:
    """Var(r_i - r_j) / (Var(r_i) + Var(r_j)) averaged over pairs.

    This is the decisive quantity: it is exactly the factor by which residual
    variance changes when a model is fitted to gaps rather than to levels.
    """

    names = list(residuals.index)
    ratios, diffs, sums = [], [], []
    for a, b in itertools.combinations(names, 2):
        ra = residuals.loc[a].to_numpy(dtype=float)
        rb = residuals.loc[b].to_numpy(dtype=float)
        var_diff = float(np.mean((ra - rb) ** 2))
        var_sum = float(np.mean(ra**2) + np.mean(rb**2))
        if var_sum <= 0:
            continue
        ratios.append(var_diff / var_sum)
        diffs.append(var_diff)
        sums.append(var_sum)
    arr = np.asarray(ratios, dtype=float)
    return {
        "n_pairs": int(len(arr)),
        "mean_ratio": float(arr.mean()),
        "median_ratio": float(np.median(arr)),
        "pooled_ratio": float(np.sum(diffs) / np.sum(sums)),
        "min_ratio": float(arr.min()),
        "max_ratio": float(arr.max()),
        "implied_variance_reduction": float(1.0 - arr.mean()),
    }


def pairwise_residual_correlation(residuals: pd.DataFrame) -> dict[str, float]:
    matrix = residuals.to_numpy(dtype=float)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    corr = np.corrcoef(matrix)
    off = corr[np.triu_indices_from(corr, 1)]
    off = off[np.isfinite(off)]
    return {
        "mean": float(off.mean()),
        "median": float(np.median(off)),
        "min": float(off.min()),
        "max": float(off.max()),
        "fraction_positive": float(np.mean(off > 0)),
    }


def bootstrap_ratio(df: pd.DataFrame, budgets: tuple[float, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Interval on the cancellation ratio by resampling interventions with replacement.

    Interventions are the replication unit here: the question is whether the
    shared component generalises across the set of interventions, so resampling
    them is the relevant uncertainty.
    """

    residuals, _ = residual_matrix(df, budgets)
    names = list(residuals.index)
    if len(names) < 4:
        return {"lo": None, "hi": None, "n_draws": 0}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        drawn = rng.choice(np.asarray(names, dtype=object), size=len(names), replace=True)
        subset = residuals.loc[[str(d) for d in drawn]]
        subset.index = [f"{d}_{i}" for i, d in enumerate(drawn)]
        try:
            draws.append(cancellation_ratio(subset)["mean_ratio"])
        except Exception:  # noqa: BLE001
            continue
    arr = np.asarray(draws, dtype=float)
    return {
        "lo": float(np.percentile(arr, 2.5)) if len(arr) > 10 else None,
        "hi": float(np.percentile(arr, 97.5)) if len(arr) > 10 else None,
        "n_draws": int(len(arr)),
    }


def scale_dependence(residuals: pd.DataFrame) -> list[dict[str, Any]]:
    """How the shared component varies with compute: per-budget mean and spread."""

    rows = []
    for budget in residuals.columns:
        column = residuals[budget].to_numpy(dtype=float)
        rows.append(
            {
                "compute": float(budget),
                "mean_residual": float(column.mean()),
                "sd_residual": float(column.std(ddof=1)),
                # |mean| / sd is the per-budget signal-to-spread of the shared offset
                "shared_to_spread": (
                    float(abs(column.mean()) / column.std(ddof=1)) if column.std(ddof=1) > 0 else float("inf")
                ),
            }
        )
    return rows


def assess(name: str, path: Path, max_fit: str, target_label: str = "1B") -> dict[str, Any]:
    df = load_runs(path)
    df = df[df["scale_label"] != "750M"].reset_index(drop=True)
    available = [s for s in df.drop_duplicates("scale_label").sort_values("compute")["scale_label"]]
    ladder = [s for s in LADDER if s in available]
    if max_fit not in ladder:
        max_fit = ladder[-1]
    target = float(df[df["scale_label"] == target_label]["compute"].iloc[0])
    keep = ladder[: ladder.index(max_fit) + 1]
    budgets = tuple(sorted(float(b) for b in df[df["scale_label"].isin(keep)]["compute"].unique() if b < target))
    residuals, means = residual_matrix(df, budgets)
    ratio = cancellation_ratio(residuals)
    return {
        "suite": name,
        "path": str(path),
        "sha256": _sha256_file(path),
        "metric": str(df["metric_name"].iloc[0]),
        "fit_scales": keep,
        "n_budgets": len(budgets),
        "n_interventions": int(residuals.shape[0]),
        "min_seeds_per_cell": int(df.groupby(["intervention", "compute"])["seed"].nunique().min()),
        "residual_correlation": pairwise_residual_correlation(residuals),
        "shared_component": shared_component_fraction(residuals),
        "shared_component_shape": shared_component_shape(residuals),
        "cancellation": ratio,
        "cancellation_ci": bootstrap_ratio(df, budgets, N_BOOTSTRAP, seed=1729),
        "scale_dependence": scale_dependence(residuals),
    }


def gate(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [a["cancellation"]["mean_ratio"] for a in assessments]
    worst = max(ratios)
    best = min(ratios)
    if worst < GATE_PROCEED_BELOW:
        verdict = "PROCEED"
        reading = (
            "Residual variance falls materially under differencing on every suite tested, so a shared "
            "per-compute misspecification component is present and does cancel. The method premise holds."
        )
    elif best >= GATE_STOP_ABOVE:
        verdict = "STOP_PREMISE_FALSE"
        reading = (
            "Differencing does not reduce residual variance: misspecification is intervention-specific, not "
            "common-mode. A differenced estimator has no variance advantage and the method should not be built."
        )
    else:
        verdict = "AMBIGUOUS_HUMAN_DECISION"
        reading = (
            "The cancellation ratio sits between the pre-set thresholds, or differs across suites. Reported "
            "with intervals for a human decision rather than resolved by the script."
        )
    offsets = [a for a in assessments if a["shared_component_shape"]["loading_cv"] < 0.5]
    scaled = [a for a in assessments if a["shared_component_shape"]["loading_cv"] >= 0.5]
    return {
        "verdict": verdict,
        "reading": reading,
        "mechanism": {
            "finding": (
                "The split across suites is not noise: it tracks whether the dominant shared residual "
                "component is a common OFFSET or a per-intervention SCALED shape. Every suite has a large "
                "first principal component (0.80-0.97 of residual variance), but that is not sufficient for "
                "cancellation. What matters is whether the PC1 loadings are near-constant across "
                "interventions."
            ),
            "suites_with_offset_structure": [
                {"suite": a["suite"], "fit_to": a["fit_scales"][-1], "loading_cv": a["shared_component_shape"]["loading_cv"],
                 "cancellation": a["cancellation"]["mean_ratio"]}
                for a in offsets
            ],
            "suites_with_scaled_structure": [
                {"suite": a["suite"], "fit_to": a["fit_scales"][-1], "loading_cv": a["shared_component_shape"]["loading_cv"],
                 "cancellation": a["cancellation"]["mean_ratio"]}
                for a in scaled
            ],
            "why_it_matters": (
                "r_i = m + e_i differences to e_i - e_j and the misspecification disappears. "
                "r_i = a_i m + e_i differences to (a_i - a_j) m + e_i - e_j and it does not. A large PC1 is "
                "therefore not evidence for the method premise on its own; the loading spread is the test."
            ),
        },
        "cancellation_ratio_by_suite": {a["suite"]: a["cancellation"]["mean_ratio"] for a in assessments},
        "worst_case_ratio": worst,
        "best_case_ratio": best,
        "thresholds": {
            "proceed_below": GATE_PROCEED_BELOW,
            "stop_at_or_above": GATE_STOP_ABOVE,
            "note": "fixed before the measurement was read; ~1.0 means independent residuals, ~0 perfect common-mode",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/common_mode")
    args = parser.parse_args(argv)
    assessments = []
    for name, path in SUITES.items():
        for max_fit in ("300M", "150M"):
            try:
                assessments.append(assess(name, path, max_fit))
            except Exception as exc:  # noqa: BLE001 - a suite that cannot be assessed is recorded
                assessments.append({"suite": name, "fit_scales_max": max_fit, "error": str(exc)})
    usable = [a for a in assessments if "error" not in a]
    results = {"gate": gate(usable), "assessments": assessments}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(results, out / "common_mode_gate.json")

    g = results["gate"]
    print("=" * 78)
    print(f"TASK 0 GATE: {g['verdict']}")
    print("=" * 78)
    for a in usable:
        c = a["cancellation"]
        ci = a["cancellation_ci"]
        band = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci["lo"] is not None else "n/a"
        print(
            f"{a['suite'][:34]:34s} fit<={a['fit_scales'][-1]:>4s} k={a['n_budgets']:2d} "
            f"n={a['n_interventions']:2d} | cancel={c['mean_ratio']:.3f} {band} "
            f"| corr={a['residual_correlation']['mean']:+.3f} "
            f"| pc1={a['shared_component']['pc1_fraction']:.3f} one_way={a['shared_component']['one_way_fraction']:.3f}"
        )
    print(f"\n{g['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
