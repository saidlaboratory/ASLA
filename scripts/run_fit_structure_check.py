"""Do the fit-structure findings stand without the decision-accuracy gap?

Pre-registered in PREDICTIONS_TASK_FIT_STRUCTURE.md. Each finding is re-read from
its committed JSON and checked against the criterion its own source states. The
output records whether the finding's evidence uses target-scale reference
labels at all; one that does not is untouched by the expected-error re-scoring.
Findings that also carry a decision-level claim get that claim's re-scored
verdict attached, from ``results/target_scoring/rescoring.json``.

The design theorem's numerical companion figures, previously quoted in
FORMULATION.md with no committed source, are recomputed here.

Writes ``results/target_scoring/fit_structure.json``.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from asla.analysis.allocation import solve_decision_optimal
from asla.analysis.transductive import design_features, feature, information_matrix, pair_variance_factor
from asla.provenance import Source

REPO = Path(__file__).resolve().parents[1]
N_RANDOM_DESIGNS = 2000
SEED = 0


def _region_objective(runs: np.ndarray, budgets: tuple[float, ...], target: float, decades: float) -> float:
    """Mean prediction-variance factor over a log-uniform region of ``decades`` centred on the target."""

    half = decades / 2.0
    region = np.exp(np.linspace(np.log(target) - half * np.log(10.0), np.log(target) + half * np.log(10.0), 9))
    features = design_features(region)
    matrix = information_matrix(budgets, runs)
    try:
        solved = np.linalg.solve(matrix, features.T)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.mean(np.sum(features.T * solved, axis=0)))


def _point_objective(runs: np.ndarray, budgets: tuple[float, ...], target: float) -> float:
    matrix = information_matrix(budgets, runs)
    x_star = feature(target)
    try:
        return float(x_star @ np.linalg.solve(matrix, x_star))
    except np.linalg.LinAlgError:
        return float("inf")


def _optimal(objective: Any, budgets: tuple[float, ...], target: float, budget_flops: float) -> np.ndarray:
    """Global optimum under the FLOP constraint, by exhaustive small-support search.

    Run counts span orders of magnitude across a ladder, which stalls a
    gradient solver working in run-count space. Both objectives here are linear
    criteria of a two-parameter information matrix, so an optimal design needs
    at most three support points (Caratheodory). Every two- and three-point
    support is searched in cost-share coordinates, where the problem is well
    scaled, and the best design is returned in run counts.
    """

    costs = np.asarray(budgets, dtype=float)
    k = len(costs)

    def runs_from(support: tuple[int, ...], shares: np.ndarray) -> np.ndarray:
        runs = np.zeros(k)
        runs[list(support)] = shares * budget_flops / costs[list(support)]
        return runs

    def value(support: tuple[int, ...], free: np.ndarray) -> float:
        shares = np.exp(np.append(free, 0.0))
        return float(objective(runs_from(support, shares / shares.sum()), budgets, target))

    best: tuple[float, np.ndarray] | None = None
    for size in (2, 3):
        for support in combinations(range(k), size):
            res = minimize(
                lambda z, sp=support: value(sp, z),
                np.zeros(size - 1),
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000},
            )
            if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
                shares = np.exp(np.append(res.x, 0.0))
                best = (float(res.fun), runs_from(support, shares / shares.sum()))
    if best is None:
        raise RuntimeError("no identifiable design found")
    return best[1]


def design_theorem() -> dict[str, Any]:
    """The identity, and how far the SL2 region objective departs from it."""

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"]
    target = float(frame[frame["scale_label"] == "1B"]["compute"].iloc[0])
    budgets = tuple(sorted(float(c) for c in frame["compute"].unique() if c < target and c > 0))
    budgets = tuple(b for b in budgets if not np.isclose(b, float(frame[frame["scale_label"] == "530M"]["compute"].iloc[0])))
    budget_flops = float(sum(budgets) * 3)

    rng = np.random.default_rng(SEED)
    point, region, pair_over_level = [], [], []
    for _ in range(N_RANDOM_DESIGNS):
        runs = rng.dirichlet(np.ones(len(budgets))) * len(budgets) * 3
        p = _point_objective(runs, budgets, target)
        point.append(p)
        region.append(_region_objective(runs, budgets, target, 1.0))
        pair_over_level.append(pair_variance_factor(budgets, runs, target) / p)
    point_arr, region_arr = np.asarray(point), np.asarray(region)
    ratio = region_arr / point_arr

    optimum_runs = _optimal(_point_objective, budgets, target, budget_flops)
    optimum = _point_objective(optimum_runs, budgets, target)
    project = np.asarray(solve_decision_optimal(budgets, target, budget_flops).allocation.runs)
    by_region: dict[str, Any] = {}
    for decades in (1.0, 3.0):

        def region(runs: np.ndarray, b: tuple[float, ...], t: float, d: float = decades) -> float:
            return _region_objective(runs, b, t, d)

        runs = _optimal(region, budgets, target, budget_flops)
        by_region[f"{decades:g}_decades"] = {
            "excess_point_variance_of_region_optimal_design": _point_objective(runs, budgets, target) / optimum - 1.0,
            # A region optimum that loses on its own objective to the point
            # optimum would mean the search failed; this must be <= 0.
            "region_gain_over_point_optimal_design": region(runs, budgets, target) / region(optimum_runs, budgets, target)
            - 1.0,
        }
    costs = np.asarray(budgets, dtype=float)

    return {
        "statement": (
            "Recipes fitted independently by least squares on one shared design, with homoscedastic "
            "noise and a well-specified linear log-space model: Var(gap at C*) = 2 Var(level at C*) "
            "for every design, so the two point objectives share an argmin and, under Gaussian noise, "
            "that design minimises every pair's mis-selection probability simultaneously."
        ),
        "identity_check_pair_over_level": {
            "min": float(np.min(pair_over_level)),
            "max": float(np.max(pair_over_level)),
        },
        "n_random_designs": N_RANDOM_DESIGNS,
        "n_budgets": len(budgets),
        "target_compute": target,
        "region_vs_point_correlation": float(np.corrcoef(point_arr, region_arr)[0, 1]),
        "region_over_point_ratio_relative_range": float((ratio.max() - ratio.min()) / np.median(ratio)),
        "region_optimal_designs": by_region,
        "point_optimal_cost_shares": [float(v) for v in optimum_runs * costs / budget_flops],
        # The project's SLSQP solver works in run-count space, which is badly
        # scaled across a ladder; its "optimum" is this far above the true one.
        "project_solver_excess_point_variance": _point_objective(project, budgets, target) / optimum - 1.0,
        "proof": "paper/main.tex, appendix: design theorem",
        "uses_target_labels": False,
    }


def _finding(
    evidence: dict[str, Any], criterion: str, stands: bool, uses_target_labels: bool, **extra: Any
) -> dict[str, Any]:
    return {
        "evidence": evidence,
        "criterion": criterion,
        "stands": bool(stands),
        "uses_target_labels": uses_target_labels,
        **extra,
    }


def findings() -> dict[str, Any]:
    theory = Source.load("theory_v2/theory_v2.json")
    gate = Source.load("common_mode/common_mode_gate.json")
    hcurv = Source.load("common_mode/hnorm_hcurv_test.json")
    bounds = Source.load("adversarial/bounds_fix_impact.json")
    demo = Source.load("adversarial/curvature_boundary_demo.json")
    floor = Source.load("adversarial/floor_profile.json")
    checkpoints = Source.load("checkpoints/checkpoint_study.json")
    resolution = Source.load("resolution/resolution_study.json")
    multipliers = Source.load("external/compute_multipliers_gate.json")
    rescoring = Source.load("target_scoring/rescoring.json")

    misspec_key = "FINDING_misspecification_is_structured.evidence.1_misspecification_is_real"
    scatter = theory.number(f"{misspec_key}.residual_scatter_over_seed_se")
    floor_ratio = floor.number("c4_en_bits_per_token.median_misspecification_ratio")

    shared_range = hcurv.get("synthetic_control.summary.shared.cancellation_range")
    separate_range = hcurv.get("synthetic_control.summary.per_intervention.cancellation_range")
    c4_cancel = gate.number("gate.cancellation_ratio_by_suite.datadecide_c4_bits_per_token")

    per_metric = {row["metric"]: row for row in checkpoints.get("technique_critique.per_metric")}
    deflation = {metric: float(row["information_deflation_factor"]) for metric, row in per_metric.items()}
    unresolved = resolution.number("summary.total_unresolved_at_3_seeds")
    adjacent = resolution.number("summary.total_adjacent_pairs")

    def rescored(ranker: str) -> dict[str, str]:
        return {
            regime_metric: rescoring.get(f"cells.{regime_metric}.rankers.{ranker}.verdict")
            for regime_metric in ("c4_en_bits_per_token", "olmes_macro_error")
        }

    return {
        "misspecification": _finding(
            {"residual_scatter_over_seed_se": scatter, "floor_profile_c4_dispersion_over_seed_variance": floor_ratio},
            "residual scatter exceeds seed standard error, on two independent computations",
            scatter > 1 and floor_ratio > 1,
            False,
        ),
        "amplitude_sharing": _finding(
            {
                "synthetic_shared_amplitude_cancellation_range": shared_range,
                "synthetic_per_intervention_amplitude_cancellation_range": separate_range,
                "c4_cancellation_ratio": c4_cancel,
            },
            "shared amplitude cancels (range below 0.1) and per-intervention amplitude does not (above 1)",
            max(shared_range) < 0.1 and min(separate_range) > 1,
            False,
        ),
        "monotone_distortion": _finding(
            {
                "olmes_error_spearman_old_vs_new_projection": bounds.number(
                    "olmes_macro_error.spearman_old_vs_new_projection"
                ),
                "olmes_error_projection_change_over_between_spread": bounds.number(
                    "olmes_macro_error.projection_change_relative_to_between_spread"
                ),
                "reordering_ratio_curvature_over_level": demo.number("contrast.reordering_ratio_curvature_over_level"),
            },
            "a large shared distortion leaves the projection ranking identical, and does not under curvature geometry",
            bool(bounds.get("olmes_macro_error.ranking_identical"))
            and demo.number("contrast.reordering_ratio_curvature_over_level") > 1,
            False,
            note="ranking identity is between two fits' projections, not against the target reference",
        ),
        "floor_identifiability": _finding(
            {
                "c4_interval_excludes_zero": floor.number("c4_en_bits_per_token.n_interval_excludes_zero"),
                "olmes_error_interval_excludes_zero": floor.number("olmes_macro_error.n_interval_excludes_zero"),
                "olmes_error_width_relative_to_ymin": floor.number(
                    "olmes_macro_error.median_interval_width_relative_to_ymin"
                ),
                "condition_ratio": floor.number("predictions.F3_conditioning.measured_ratio"),
            },
            "pre-registered F1-F3 (PREDICTIONS_TASK_FIT_STRUCTURE.md)",
            all(
                bool(floor.get(f"predictions.{key}.confirmed"))
                for key in ("F1_c4_floor_identified", "F2_accuracy_floor_unidentified", "F3_conditioning")
            ),
            False,
            refuted=["F4_fix_tightens_c4"] if not floor.get("predictions.F4_fix_tightens_c4.confirmed") else [],
        ),
        "checkpoint_critique": _finding(
            {
                "information_deflation_factor": deflation,
                "largest_uniform_weight_effect": checkpoints.number(
                    "technique_critique.uniform_weight_noop.largest_uniform_effect"
                ),
                "largest_varying_weight_effect": checkpoints.number(
                    "technique_critique.uniform_weight_noop.largest_varying_effect"
                ),
            },
            "correlated checkpoints carry fewer independent observations than they count, and a uniform weight is a no-op",
            all(v > 1 for v in deflation.values())
            and bool(checkpoints.get("technique_critique.uniform_weight_noop.varying_dominates_uniform")),
            False,
            decision_claim_rescored=rescored("checkpoint_augmented"),
        ),
        "resolvability": _finding(
            {"unresolved_at_3_seeds": unresolved, "adjacent_pairs": adjacent},
            "most adjacent orderings are not resolvable at the seed budget used",
            unresolved / adjacent > 0.5,
            False,
        ),
        "metric_instability": _finding(
            {
                "corpus_min_pairwise_spearman": multipliers.number(
                    "metric_agreement_at_top_budget.corpus.min_pairwise_spearman"
                ),
                "corpus_distinct_winners": multipliers.number("metric_agreement_at_top_budget.corpus.n_distinct_winners"),
            },
            "evaluation metrics disagree on the winner at the same budget",
            multipliers.number("metric_agreement_at_top_budget.corpus.n_distinct_winners") > 1,
            False,
        ),
    }


def main() -> None:
    out = {"findings": findings(), "design_theorem": design_theorem()}
    out["all_stand"] = all(f["stands"] for f in out["findings"].values())
    path = REPO / "results" / "target_scoring" / "fit_structure.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for name, row in out["findings"].items():
        print(f"{name:24s} stands={row['stands']} target_labels={row['uses_target_labels']}")
    theorem = out["design_theorem"]
    print(
        "theorem corr",
        theorem["region_vs_point_correlation"],
        "ratio range",
        theorem["region_over_point_ratio_relative_range"],
    )
    print("region-optimal excess", theorem["region_optimal_designs"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
