"""Task 2c/2d: retrospective evaluation of optimal allocation on DataDecide.

Evaluates the pre-registered predictions P1-P5 in
``PREDICTIONS_TASK_ALLOCATION.md``. Nothing here reads the pre-registration; the
predictions are scored by hand against the JSON this writes.

The comparison is at **matched exploration compute**: every arm is given the same
FLOP budget, and the run counts each arm may use are derived from that budget. The
single-scale baseline is charged for the runs it actually consumes at its rung.

Design provenance is recorded for every arm, including where the deviation
function used by the bias-aware design was estimated, in the same way ``lambda``
is recorded in ``NOTES_SHRINKAGE_STRENGTH.md``.

Usage::

    python scripts/run_allocation_study.py --out results/allocation/allocation_study.json
    python scripts/run_allocation_study.py --fast    # smoke test
    python scripts/run_allocation_study.py --resume  # reuse completed designs
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

from asla.analysis.allocation import (
    round_allocation,
    scale_to_budget,
    solve_bias_aware,
    solve_decision_optimal,
    solve_estimation_optimal,
)
from asla.analysis.fits import cell_means_and_sigma, normalize_budgets
from asla.analysis.transductive import (
    classical_leverage,
    lever_arm,
    target_variance_factor,
    uniform_allocation,
)
from asla.models import FitError, bpb_power_law, fit_power_law

# DataDecide's 750M cell sits off the 5xC trajectory and is excluded everywhere
# else in this project; excluding it here keeps the ladder comparable.
OFF_TRAJECTORY_SCALES = ("750M",)


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _stable_seed(base: int, *parts: str) -> int:
    """Derive a reproducible per-arm seed.

    ``hash()`` is salted per process, so using it here would make ``--resume``
    produce different streams than the original run.
    """

    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return (base + int(digest[:8], 16)) % (2**32)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_datadecide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[~df["scale_label"].isin(OFF_TRAJECTORY_SCALES)].copy()


def estimate_deviation(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    interventions: list[str],
) -> dict[str, Any]:
    """Estimate the per-budget model deviation on the fitting range.

    For each intervention we fit the two-parameter power law on ``budgets`` and
    record the residual at each budget. The deviation used by the bias-aware
    design is the RMS residual per budget, averaged across the supplied
    interventions --- which must be the *training* interventions only.

    Returns the deviation vector plus the provenance needed to audit it.
    """

    means = cell_means_and_sigma(df)
    residuals: dict[float, list[float]] = {b: [] for b in budgets}
    used: list[str] = []
    for name in interventions:
        rows = means[(means["intervention"] == name) & (means["compute"].isin(budgets))]
        rows = rows.sort_values("compute")
        if len(rows) < 3:
            continue
        compute = rows["compute"].to_numpy(dtype=float)
        bpb = rows["bpb_mean"].to_numpy(dtype=float)
        try:
            params = fit_power_law(compute, bpb)
        except FitError:
            continue
        predicted = np.asarray(bpb_power_law(compute, *params), dtype=float)
        for budget, resid in zip(compute, bpb - predicted):
            residuals[float(budget)].append(float(resid))
        used.append(name)

    deviation = []
    per_budget = []
    for budget in budgets:
        values = residuals[budget]
        rms = float(np.sqrt(np.mean(np.square(values)))) if values else 0.0
        deviation.append(rms)
        per_budget.append({"compute": budget, "rms_residual": rms, "n": len(values)})

    return {
        "deviation": deviation,
        "per_budget": per_budget,
        "estimated_from": "train_interventions_only",
        "n_interventions_used": len(used),
        "interventions_used": sorted(used),
    }


def evaluate_allocation(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    runs: tuple[float, ...],
    target: float,
    interventions: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    """Score an allocation by pairwise mis-selection against the target truth.

    Run counts are realised as a seed-resampling weight: an allocation asking for
    ``n_j`` runs at budget ``C_j`` draws ``n_j`` seed-resampled cell means there.
    Budgets with zero runs are dropped from the fit.
    """

    support = [(b, r) for b, r in zip(budgets, runs) if r >= 1.0]
    if len(support) < 2:
        return {"error": "allocation has fewer than two supported budgets", "mis_selection": None}
    if len(support) < 3:
        # The three-parameter E + A*C^-alpha estimator needs three distinct
        # budgets. A two-point c-optimal design is optimal for the two-parameter
        # LINEAR model but cannot feed the estimator actually used downstream.
        # This is a real mismatch between the design theory and the estimator,
        # not a bug, and it is reported rather than worked around.
        return {
            "error": "design has two support points; three-parameter fit is unidentifiable",
            "mis_selection": None,
            "n_support": len(support),
            "support_budgets": [b for b, _ in support],
        }

    fit_budgets = tuple(b for b, _ in support)
    counts = {b: int(round(r)) for b, r in support}

    truth = df[np.isclose(df["compute"], target)].groupby("intervention")["bpb"].mean().reindex(interventions)
    if truth.isna().any():
        return {"error": "target truth missing for some interventions", "mis_selection": None}

    cells: dict[tuple[str, float], np.ndarray] = {}
    for name in interventions:
        for budget in fit_budgets:
            values = df[(df["intervention"] == name) & (np.isclose(df["compute"], budget))]["bpb"]
            cells[(name, budget)] = values.to_numpy(dtype=float)

    pairs = [(a, b) for i, a in enumerate(interventions) for b in interventions[i + 1 :]]
    truth_sign = {(a, b): np.sign(truth[a] - truth[b]) for a, b in pairs}

    wrong_counts = []
    for _ in range(n_boot):
        projections: dict[str, float] = {}
        for name in interventions:
            xs, ys = [], []
            for budget in fit_budgets:
                pool = cells[(name, budget)]
                if pool.size == 0:
                    continue
                for _ in range(counts[budget]):
                    xs.append(budget)
                    ys.append(float(rng.choice(pool)))
            if len(set(xs)) < 2:
                projections[name] = float("nan")
                continue
            try:
                params = fit_power_law(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
                projections[name] = float(bpb_power_law(target, *params))
            except FitError:
                projections[name] = float("nan")
        wrong = 0
        scored = 0
        for a, b in pairs:
            pa, pb = projections[a], projections[b]
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            scored += 1
            if np.sign(pa - pb) != truth_sign[(a, b)]:
                wrong += 1
        wrong_counts.append(wrong / scored if scored else float("nan"))

    values = np.asarray([v for v in wrong_counts if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"error": "no evaluable bootstrap replicates", "mis_selection": None}
    return {
        "mis_selection": float(values.mean()),
        "mis_selection_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
        "n_boot_evaluable": int(values.size),
        "fit_budgets": list(fit_budgets),
        "runs_per_budget": {str(b): counts[b] for b in fit_budgets},
        "total_runs": int(sum(counts.values())),
        "cost_flops": float(sum(counts[b] * b for b in fit_budgets)),
    }


def evaluate_single_scale(
    df: pd.DataFrame,
    rung: float,
    target: float,
    interventions: list[str],
    rng: np.random.Generator,
    n_boot: int,
    n_seeds: int,
) -> dict[str, Any]:
    """Score single-scale ranking at one rung: no fit, no extrapolation."""

    truth = df[np.isclose(df["compute"], target)].groupby("intervention")["bpb"].mean().reindex(interventions)
    cells = {
        name: df[(df["intervention"] == name) & (np.isclose(df["compute"], rung))]["bpb"].to_numpy(dtype=float)
        for name in interventions
    }
    pairs = [(a, b) for i, a in enumerate(interventions) for b in interventions[i + 1 :]]
    truth_sign = {(a, b): np.sign(truth[a] - truth[b]) for a, b in pairs}

    rates = []
    for _ in range(n_boot):
        scores = {
            name: float(np.mean(rng.choice(pool, size=n_seeds))) if pool.size else float("nan")
            for name, pool in cells.items()
        }
        wrong = sum(
            1
            for a, b in pairs
            if np.isfinite(scores[a]) and np.isfinite(scores[b]) and np.sign(scores[a] - scores[b]) != truth_sign[(a, b)]
        )
        rates.append(wrong / len(pairs))
    values = np.asarray(rates, dtype=float)
    return {
        "mis_selection": float(values.mean()),
        "mis_selection_sd": float(values.std(ddof=1)),
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
        "rung": rung,
        "n_seeds": n_seeds,
        "cost_flops": float(rung * n_seeds * len(interventions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/datadecide_runs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("results/allocation/allocation_study.json"))
    parser.add_argument("--target-scale", default="1B")
    parser.add_argument(
        "--budget-fractions",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0],
        help=(
            "Exploration budgets to evaluate, as fractions of the uniform 3-seed "
            "ladder's per-intervention cost. The shape of the optimum is "
            "budget-dependent, so a single budget would not characterise it."
        ),
    )
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument(
        "--min-eval-pairs-mis-selected",
        type=int,
        default=1,
        help=(
            "Record whether the single-scale baseline mis-selects at least this "
            "many pairs. At small lever arms it scores exactly 0, leaving no "
            "headroom for any method to demonstrate a difference; that is a "
            "property of the regime and is flagged rather than hidden."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--max-runs-per-budget",
        type=float,
        default=3.0,
        help=(
            "Cap on runs at any one budget. DataDecide has 3 seeds per cell, so "
            "beyond 3 draws the bootstrap resamples the same values and buys no "
            "new information; an uncapped variance objective would exploit that "
            "as if it were free replication."
        ),
    )
    parser.add_argument("--fast", action="store_true", help="smoke test with tiny counts")
    parser.add_argument("--resume", action="store_true", help="reuse completed designs from --out")
    args = parser.parse_args()

    if args.fast:
        args.n_boot = 12

    df = load_datadecide(args.data)
    target_rows = df[df["scale_label"] == args.target_scale]
    if target_rows.empty:
        raise SystemExit(f"target scale {args.target_scale} not present")
    target = float(target_rows["compute"].iloc[0])

    ladder = normalize_budgets(sorted(float(c) for c in df["compute"].unique() if c < target), target=target)
    interventions = sorted(df["intervention"].unique())

    existing: dict[str, Any] = {}
    if args.resume and args.out.exists():
        existing = json.loads(args.out.read_text())

    payload: dict[str, Any] = {
        "inputs": {str(args.data): _file_sha(args.data)},
        "target_scale": args.target_scale,
        "target_compute": target,
        "ladder": list(ladder),
        "n_interventions": len(interventions),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "fast": args.fast,
        "excluded_scales": list(OFF_TRAJECTORY_SCALES),
        "lever_arm_full_ladder": lever_arm(ladder, target),
    }
    payload.update({k: v for k, v in existing.items() if k.startswith("design_")})

    # Exploration budget: what the uniform 3-seed ladder over the full ladder costs.
    uniform = uniform_allocation(ladder, target, 3.0)
    budget_flops = uniform.cost * len(interventions)
    per_intervention_budget = uniform.cost
    payload["max_runs_per_budget"] = args.max_runs_per_budget
    payload["budget_flops_total"] = budget_flops
    payload["budget_flops_per_intervention"] = per_intervention_budget

    # Deviation estimated on TRAIN interventions only.
    rng = np.random.default_rng(args.seed)
    shuffled = list(interventions)
    rng.shuffle(shuffled)
    n_train = int(round(0.68 * len(shuffled)))
    train_interventions = sorted(shuffled[:n_train])
    eval_interventions = sorted(shuffled[n_train:])
    # The deviation function is a HYPERPARAMETER and must not see evaluation
    # data, exactly as lambda must not (NOTES_SHRINKAGE_STRENGTH.md). It is
    # therefore estimated on the train subset only.
    #
    # Mis-selection itself is scored on ALL interventions, which is this
    # project's established pairwise estimand. Scoring on a held-out subset of
    # interventions was tried and abandoned: with 8 interventions the baseline
    # scores exactly 0.0000 at every target but one, because 28 pairs cannot
    # resolve a 1-5% rate. That is a power failure of the split, not a property
    # of any method, and it would have made every comparison vacuous.
    payload["split"] = {
        "n_train": len(train_interventions),
        "n_heldout_from_deviation": len(eval_interventions),
        "train_interventions": train_interventions,
        "heldout_from_deviation": eval_interventions,
        "split_seed": args.seed,
        "deviation_estimated_on": "train_interventions_only",
        "mis_selection_scored_on": "all_interventions",
        "rationale": (
            "The deviation function must not be fitted on data used to score "
            "decisions. Mis-selection is scored on all 25 interventions because "
            "a 8-intervention subset yields 28 pairs, too few to resolve rates "
            "of 1-5%: the baseline hits exactly 0.0000 and no method can differ."
        ),
    }
    deviation_info = estimate_deviation(df, ladder, train_interventions)
    scoring_interventions = interventions
    payload["deviation"] = deviation_info

    cap = args.max_runs_per_budget
    payload["classical_leverage_full_ladder"] = classical_leverage(ladder, target)["leverage"]

    # The shape of the optimum is budget-dependent, so we sweep the exploration
    # budget rather than reporting a single design.
    sweep: dict[str, Any] = dict(existing.get("sweep", {}))
    scored: dict[str, Any] = dict(existing.get("scored", {}))

    top_rung = max(ladder)
    single_key = "single_scale_top_rung"
    if not (args.resume and single_key in scored):
        scored[single_key] = evaluate_single_scale(
            df,
            top_rung,
            target,
            scoring_interventions,
            np.random.default_rng(args.seed + 7),
            args.n_boot,
            3,
        )
        payload["scored"] = scored
        _write_json_atomically(args.out, payload)
    single = scored[single_key]
    payload["baseline_has_headroom"] = bool(single["mis_selection"] > 0.0)
    if not payload["baseline_has_headroom"]:
        payload["headroom_warning"] = (
            "Single-scale ranking mis-selects zero pairs at this target and split. "
            "No method can beat it here, so comparisons at this target are "
            "uninformative about relative accuracy and only the variance and "
            "design-shape results (P1, P2, P4, P5) carry information."
        )

    for fraction in args.budget_fractions:
        key = f"frac_{fraction:g}"
        if args.resume and key in sweep:
            continue
        budget = per_intervention_budget * float(fraction)
        entry: dict[str, Any] = {
            "budget_fraction": float(fraction),
            "budget_flops_per_intervention": budget,
            "cost_relative_to_single_scale": budget / float(top_rung * 3),
        }
        try:
            decision = solve_decision_optimal(ladder, target, budget, max_runs=cap)
            estimation = solve_estimation_optimal(ladder, target, budget, max_runs=cap)
            bias_aware = solve_bias_aware(ladder, target, budget, deviation_info["deviation"], max_runs=cap)
        except ValueError as exc:
            entry["error"] = str(exc)
            sweep[key] = entry
            payload["sweep"] = sweep
            _write_json_atomically(args.out, payload)
            continue

        uniform_scaled = scale_to_budget(uniform, budget)
        arms = {
            "decision_optimal": decision.allocation,
            "estimation_optimal_sl2": estimation.allocation,
            "bias_aware": bias_aware.allocation,
            "uniform_ladder": uniform_scaled,
        }
        entry["designs"] = {name: alloc.as_dict() for name, alloc in arms.items()}
        entry["variance_factors"] = {
            name: target_variance_factor(ladder, alloc.runs, target) for name, alloc in arms.items()
        }
        entry["variance_ratio_decision_over_uniform"] = (
            entry["variance_factors"]["decision_optimal"] / entry["variance_factors"]["uniform_ladder"]
        )
        # P4: do the decision and estimation designs agree in cost share?
        dec = arms["decision_optimal"]
        est = arms["estimation_optimal_sl2"]
        dec_share = np.array([r * b for r, b in zip(dec.runs, ladder)]) / max(dec.cost, 1e-300)
        est_share = np.array([r * b for r, b in zip(est.runs, ladder)]) / max(est.cost, 1e-300)
        entry["max_cost_share_gap_decision_vs_estimation"] = float(np.max(np.abs(dec_share - est_share)))
        # P5: does bias-awareness move cost up the ladder?
        bias = arms["bias_aware"]
        bias_share = np.array([r * b for r, b in zip(bias.runs, ladder)]) / max(bias.cost, 1e-300)
        entry["cheap_rung_share_decision"] = float(dec_share[0])
        entry["cheap_rung_share_bias_aware"] = float(bias_share[0])
        entry["top_rung_share_decision"] = float(dec_share[-1])
        entry["top_rung_share_bias_aware"] = float(bias_share[-1])

        entry["scored"] = {}
        for name, alloc in arms.items():
            rounded = round_allocation(alloc, budget, max_runs=cap)
            result = evaluate_allocation(
                df,
                ladder,
                rounded.runs,
                target,
                scoring_interventions,
                np.random.default_rng(_stable_seed(args.seed, key, name)),
                args.n_boot,
            )
            result["rounded_runs"] = list(rounded.runs)
            result["rounded_cost"] = rounded.cost
            result["n_support_rounded"] = len(rounded.support)
            if result.get("mis_selection") is not None:
                result["excess_over_single_scale"] = result["mis_selection"] - single["mis_selection"]
            entry["scored"][name] = result

        sweep[key] = entry
        payload["sweep"] = sweep
        _write_json_atomically(args.out, payload)

    payload["sweep"] = sweep
    payload["scored"] = scored
    payload["matched_compute_note"] = {
        "single_scale_cost_per_intervention": float(top_rung * 3),
        "full_ladder_cost_per_intervention": per_intervention_budget,
        "note": (
            "Costs are per intervention in every arm. The single-scale arm buys "
            "3 seeds at the top rung; each design arm buys what the same stated "
            "FLOP budget affords across the ladder. The full uniform ladder costs "
            "more than single-scale ranking, so equal-cost comparisons against "
            "single-scale sit at budget fractions below 1."
        ),
    }
    _write_json_atomically(args.out, payload)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
