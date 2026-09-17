"""Task 3: abstention rate versus delta and budget, on real leaderboards.

Scores the pre-registered predictions A1-A5 in ``PREDICTIONS_TASK_ABSTENTION.md``.

Three parts:

* **Coverage (A1, A5).** Do parametric projection intervals built on fit-range
  data actually cover the observed target value? Unlike
  ``scripts/run_calibration_study.py``, which measures coverage on synthetic
  scenarios where truth is known by construction, this measures it on real
  DataDecide cells where the target value is *observed*. That is the quantity
  the guarantee depends on.
* **Abstention (A2, A3).** Sweep delta and seed budget, report the fraction of
  adjacent leaderboard orderings the delta-PAC rule declines to certify.
* **Cost of correctness (A4).** Among pairs single-scale ranking gets right, how
  many does the rule abstain on --- how much apparent accuracy is uncertified.

Usage::

    python scripts/run_abstention_study.py --out results/abstention/abstention.json
    python scripts/run_abstention_study.py --fast
    python scripts/run_abstention_study.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.abstention import (
    Decision,
    abstention_rate,
    certify_leaderboard,
    effective_error_rate,
    seeds_to_resolve,
)
from asla.analysis.conformal import conformal_projection_interval
from asla.analysis.fits import cell_means_and_sigma, projection_with_uncertainty
from asla.models import FitError

OFF_TRAJECTORY_SCALES = ("750M",)
NOMINAL_COVERAGE = 0.90
DELTAS = (0.20, 0.10, 0.05, 0.01)
# A single seed provides no within-cell variance estimate, so nothing can be
# certified from it at any delta. It is kept in the grid and reported as
# abstention 1.0 rather than dropped, because "one seed certifies nothing" is
# a result a practitioner needs, not a missing row.
SEED_BUDGETS = (1, 2, 3, 5, 10, 20)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=float)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[~df["scale_label"].isin(OFF_TRAJECTORY_SCALES)].copy()


def measure_coverage(
    df: pd.DataFrame,
    fit_budgets: tuple[float, ...],
    target: float,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Empirical coverage of projection intervals against the OBSERVED target.

    The true target value here is the seed mean of real runs at ``C*``, not a
    synthetic ground truth. Coverage is the fraction of interventions whose
    interval contains it.
    """

    means = cell_means_and_sigma(df)
    rows: list[dict[str, Any]] = []
    for name in sorted(df["intervention"].unique()):
        target_cells = df[(df["intervention"] == name) & (np.isclose(df["compute"], target))]
        if target_cells.empty:
            continue
        true_value = float(target_cells["bpb"].mean())
        group = df[df["intervention"] == name]
        fit_rows = means[(means["intervention"] == name) & (means["compute"].isin(fit_budgets))]
        if len(fit_rows) < 4:
            continue
        compute = fit_rows["compute"].to_numpy(dtype=float)
        bpb = fit_rows["bpb_mean"].to_numpy(dtype=float)
        try:
            _, _, b_lo, b_hi = projection_with_uncertainty(group, fit_budgets, target, n_boot, rng)
        except (FitError, ValueError):
            b_lo = b_hi = float("nan")
        try:
            _, c_lo, c_hi = conformal_projection_interval(compute, bpb, target, alpha=alpha)
        except (FitError, ValueError):
            c_lo = c_hi = float("nan")
        for method, lo, hi in (("bootstrap", b_lo, b_hi), ("conformal", c_lo, c_hi)):
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            rows.append(
                {
                    "intervention": name,
                    "method": method,
                    "lo": float(lo),
                    "hi": float(hi),
                    "true_target": true_value,
                    "covered": bool(lo <= true_value <= hi),
                    "width": float(hi - lo),
                }
            )

    # Record the centring error alongside the widths. A zero coverage reading is
    # only interpretable next to how far the point projection sits from truth:
    # if the error dwarfs the width, the interval is precisely wrong rather than
    # merely narrow, and that distinction is the whole finding.
    summary: dict[str, Any] = {"per_interval": rows, "nominal_coverage": NOMINAL_COVERAGE}
    for method in ("bootstrap", "conformal"):
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        centres = np.asarray([(r["lo"] + r["hi"]) / 2 for r in subset], dtype=float)
        truths = np.asarray([r["true_target"] for r in subset], dtype=float)
        widths = np.asarray([r["width"] for r in subset], dtype=float)
        target_sd = df[np.isclose(df["compute"], target)].groupby("intervention")["bpb"].std(ddof=1)
        seed_counts = df[np.isclose(df["compute"], target)].groupby("intervention")["bpb"].count()
        median_seed_se = float((target_sd / np.sqrt(seed_counts)).median())
        summary[f"{method}_centring"] = {
            "median_seed_standard_error_at_target": median_seed_se,
            "centring_error_in_seed_standard_errors": (
                float(np.mean(np.abs(centres - truths)) / median_seed_se) if median_seed_se > 0 else float("nan")
            ),
            "mean_signed_error": float(np.mean(centres - truths)),
            "mean_absolute_error": float(np.mean(np.abs(centres - truths))),
            "mean_width": float(np.mean(widths)),
            "error_to_width_ratio": float(np.mean(np.abs(centres - truths)) / np.mean(widths)),
            "n_overshooting": int(np.sum(centres > truths)),
            "n": len(subset),
        }
    for method in ("bootstrap", "conformal"):
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            summary[method] = {"n": 0, "coverage": None}
            continue
        covered = sum(1 for r in subset if r["covered"])
        n = len(subset)
        coverage = covered / n
        # Wilson interval, so a small n cannot masquerade as a precise estimate.
        z = 1.959963985
        denominator = 1 + z**2 / n
        centre = (coverage + z**2 / (2 * n)) / denominator
        half = z * np.sqrt(coverage * (1 - coverage) / n + z**2 / (4 * n**2)) / denominator
        summary[method] = {
            "n": n,
            "coverage": coverage,
            "ci_lo": float(max(0.0, centre - half)),
            "ci_hi": float(min(1.0, centre + half)),
            "median_width": float(np.median([r["width"] for r in subset])),
            "effective_error_at_nominal_delta_0_05": effective_error_rate(0.05, coverage, NOMINAL_COVERAGE),
        }
    return summary


def leaderboard_at_scale(df: pd.DataFrame, compute: float) -> tuple[dict[str, float], dict[str, float]]:
    """Entry means and per-entry seed standard errors at one scale."""

    rows = df[np.isclose(df["compute"], compute)]
    entries: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for name, group in rows.groupby("intervention"):
        values = group["bpb"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        entries[str(name)] = float(values.mean())
        sigmas[str(name)] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return entries, sigmas


def sweep_abstention(
    entries: dict[str, float],
    sigmas: dict[str, float],
    deltas: tuple[float, ...],
    seed_budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Abstention rate over the (delta, budget) grid.

    The seed budget enters through the standard error: with ``n`` seeds the
    per-entry standard error is ``sigma / sqrt(n)``. Sigma is the measured
    run-to-run spread, so this asks what *would* happen at a budget the suite
    did not buy.
    """

    pooled_sigma = float(np.median([s for s in sigmas.values() if s > 0])) if sigmas else 0.0
    rows = []
    n_pairs_adjacent = max(len(entries) - 1, 0)
    for delta in deltas:
        for budget in seed_budgets:
            if budget < 2:
                rows.append(
                    {
                        "delta": delta,
                        "seed_budget": budget,
                        "abstention_rate": 1.0,
                        "reference_distribution": "none",
                        "n_pairs": n_pairs_adjacent,
                        "n_certified": 0,
                        "note": "one seed gives no variance estimate; nothing is certifiable",
                    }
                )
                continue
            errors = {name: (sigmas[name] if sigmas[name] > 0 else pooled_sigma) / np.sqrt(budget) for name in entries}
            verdicts = certify_leaderboard(entries, errors, delta, n_seeds=budget)
            rows.append(
                {
                    "delta": delta,
                    "seed_budget": budget,
                    "abstention_rate": abstention_rate(verdicts),
                    "reference_distribution": "student_t",
                    "n_pairs": len(verdicts),
                    "n_certified": sum(1 for v in verdicts if not v.abstained),
                }
            )
    return rows


def cost_of_correctness(
    df: pd.DataFrame,
    rung: float,
    target: float,
    delta: float,
    seed_budget: int,
) -> dict[str, Any]:
    """A4: how many of the baseline's correct calls does the rule not certify?"""

    entries, sigmas = leaderboard_at_scale(df, rung)
    truth_entries, _ = leaderboard_at_scale(df, target)
    shared = [name for name in entries if name in truth_entries]
    if len(shared) < 2:
        return {"error": "too few shared interventions"}

    pooled = float(np.median([s for s in sigmas.values() if s > 0])) if sigmas else 0.0
    errors = {name: (sigmas[name] if sigmas[name] > 0 else pooled) / np.sqrt(seed_budget) for name in shared}
    verdicts = certify_leaderboard({name: entries[name] for name in shared}, errors, delta, adjacent_only=False)

    baseline_correct = 0
    abstained_on_correct = 0
    certified_correct = 0
    certified_wrong = 0
    for verdict in verdicts:
        left, right = verdict.left, verdict.right
        baseline_sign = np.sign(entries[left] - entries[right])
        truth_sign = np.sign(truth_entries[left] - truth_entries[right])
        is_correct = baseline_sign == truth_sign
        if is_correct:
            baseline_correct += 1
            if verdict.abstained:
                abstained_on_correct += 1
        if not verdict.abstained:
            certified_sign = -1.0 if verdict.decision is Decision.LEFT else 1.0
            if certified_sign == truth_sign:
                certified_correct += 1
            else:
                certified_wrong += 1

    n_certified = certified_correct + certified_wrong
    return {
        "rung": rung,
        "target": target,
        "delta": delta,
        "seed_budget": seed_budget,
        "n_pairs": len(verdicts),
        "baseline_correct": baseline_correct,
        "abstained_on_correct": abstained_on_correct,
        "fraction_of_correct_calls_abstained": (
            abstained_on_correct / baseline_correct if baseline_correct else float("nan")
        ),
        "n_certified": n_certified,
        "certified_correct": certified_correct,
        "certified_wrong": certified_wrong,
        "certified_error_rate": (certified_wrong / n_certified if n_certified else float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/datadecide_runs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("results/abstention/abstention.json"))
    parser.add_argument("--target-scale", default="1B")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.fast:
        args.n_boot = 20

    df = load_runs(args.data)
    target_rows = df[df["scale_label"] == args.target_scale]
    if target_rows.empty:
        raise SystemExit(f"target scale {args.target_scale} not present")
    target = float(target_rows["compute"].iloc[0])
    ladder = tuple(sorted(float(c) for c in df["compute"].unique() if c < target))

    existing: dict[str, Any] = {}
    if args.resume and args.out.exists():
        existing = json.loads(args.out.read_text())

    payload: dict[str, Any] = dict(existing)
    payload.update(
        {
            "inputs": {str(args.data): _file_sha(args.data)},
            "target_scale": args.target_scale,
            "target_compute": target,
            "ladder": list(ladder),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "fast": args.fast,
            "deltas": list(DELTAS),
            "seed_budgets": list(SEED_BUDGETS),
            "excluded_scales": list(OFF_TRAJECTORY_SCALES),
        }
    )

    # A1/A5: coverage on real data.
    if not (args.resume and "coverage" in payload):
        payload["coverage"] = measure_coverage(df, ladder, target, args.n_boot, np.random.default_rng(args.seed))
        _write_json_atomically(args.out, payload)

    # A2/A3: abstention over the (delta, budget) grid at every scale.
    if not (args.resume and "abstention_by_scale" in payload):
        by_scale: dict[str, Any] = {}
        for label in sorted(df["scale_label"].unique()):
            compute = float(df[df["scale_label"] == label]["compute"].iloc[0])
            entries, sigmas = leaderboard_at_scale(df, compute)
            if len(entries) < 2:
                continue
            grid = sweep_abstention(entries, sigmas, DELTAS, SEED_BUDGETS)
            ordered = sorted(entries.items(), key=lambda kv: kv[1])
            pooled = float(np.median([s for s in sigmas.values() if s > 0])) if sigmas else 0.0
            n_pairs_all = len(entries) * (len(entries) - 1) // 2
            required = []
            for (better, better_value), (worse, worse_value) in zip(ordered[:-1], ordered[1:]):
                sigma = sigmas[better] if sigmas[better] > 0 else pooled
                required.append(seeds_to_resolve(abs(worse_value - better_value), sigma, 0.05, n_pairs_all))
            resolvable = [r for r in required if r is not None]
            by_scale[label] = {
                "compute": compute,
                "n_entries": len(entries),
                "grid": grid,
                "seeds_required_adjacent": required,
                "median_seeds_required": (float(np.median(resolvable)) if resolvable else None),
                "n_never_resolvable": sum(1 for r in required if r is None),
            }
        payload["abstention_by_scale"] = by_scale
        _write_json_atomically(args.out, payload)

    # A4: the cost of correctness at the top rung.
    if not (args.resume and "cost_of_correctness" in payload):
        payload["cost_of_correctness"] = [
            cost_of_correctness(df, max(ladder), target, delta, budget) for delta in (0.05,) for budget in (3, 10)
        ]
        _write_json_atomically(args.out, payload)

    _write_json_atomically(args.out, payload)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
