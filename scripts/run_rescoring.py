"""Priority 4: re-score every decision-level result under expected-error scoring.

Every ranker comparison in this project was scored against observed target
means. Here each is re-scored under observed-target and expected-error rules,
split by noise regime, and each ranker's difference from single-scale ranking is
tested by resampling candidates (the independent unit) rather than pairs.

Uses the project's own ranker implementations, not reimplementations, so the
numbers are comparable with the audit. Verdict per result is one of SURVIVES,
WEAKENS, or VANISHES, decided mechanically below.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from asla.analysis.abstention import certify_leaderboard
from asla.analysis.fits import normalize_budgets
from asla.analysis.moderation import calibrated_target_evidence
from asla.analysis.pooled import make_shrinkage_ranker, shared_exponent_ranker
from asla.analysis.rankers import ensemble_ranker, projection_ranker, single_scale_ranker
from asla.analysis.target_scoring import (
    PairEvidence,
    bootstrap_two_sided_p,
    candidate_bootstrap,
    expected_charges,
    jackknife_standard_error,
    score_ranker,
    u_statistic_test,
)
from asla.models import FitError
from asla.provenance import Source

REPO = Path(__file__).resolve().parents[1]
N_RESAMPLES = 4000
N_SEEDS = 3
CERT_DELTA = 0.05
# Regime by metric, from the Bonferroni determined fraction measured in
# results/target_scoring/expected_error.json: C4 is above one half on every
# design, the accuracy metrics below one fifth.
METRIC_REGIME = {"c4_en_bits_per_token": "low_noise", "olmes_macro_error": "high_noise"}
Ranker = Callable[[pd.DataFrame, tuple[float, ...], float], pd.Series]


def _load_checkpoint_builder() -> Callable[..., dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("run_checkpoint_study", REPO / "scripts" / "run_checkpoint_study.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_checkpoint_study"] = module
    spec.loader.exec_module(module)
    return module.build


def _gaps(series: pd.Series) -> dict[tuple[str, str], float]:
    values = {str(k): float(v) for k, v in series.items() if np.isfinite(float(v))}
    return {(a, b): values[a] - values[b] for a, b in combinations(sorted(values), 2)}


def _evidence(table: pd.DataFrame, target: float) -> list[PairEvidence]:
    cells = table[np.isclose(table["compute"], target)].groupby("intervention")["bpb"]
    return calibrated_target_evidence(
        {str(k): float(v) for k, v in cells.mean().items()},
        {str(k): float(v) for k, v in cells.std(ddof=1).items()},
        {str(k): int(v) for k, v in cells.count().items()},
        alpha=0.05,
        multiplicity="bonferroni",
    )


def paired_candidate_test(
    comparator: dict[tuple[str, str], float],
    baseline: dict[tuple[str, str], float],
    evidence: list[PairEvidence],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Expected-error difference versus the baseline, with candidates as the unit.

    The headline interval is the unbiased U-statistic test, which simulation at
    this project's variance components puts within a few percent of the true
    standard error (results/target_scoring/expected_error.json,
    ``estimator_calibration``). The multiplicity-weighted candidate bootstrap is
    reported alongside; it is conservative here. The first version of this test
    scored only the unique candidates in each bootstrap draw.
    """

    left = expected_charges(comparator, evidence)
    right = expected_charges(baseline, evidence)
    kernel = {pair: 100 * (left[pair] - right[pair]) for pair in left if pair in right}
    u_test = u_statistic_test(kernel)
    draws = candidate_bootstrap(kernel, rng, N_RESAMPLES)
    return {
        "point_pp": float(u_test["mean"]),
        "standard_error_pp": float(u_test["standard_error"]),
        "ci_low_pp": float(u_test["ci_low"]),
        "ci_high_pp": float(u_test["ci_high"]),
        "excludes_zero": bool(u_test["excludes_zero"]),
        "p_two_sided": float(u_test["p_two_sided"]),
        "n_pairs": len(kernel),
        "jackknife_se_pp": jackknife_standard_error(kernel),
        "weighted_bootstrap_conservative": {
            "ci_low_pp": float(np.percentile(draws, 2.5)),
            "ci_high_pp": float(np.percentile(draws, 97.5)),
            "p_two_sided": bootstrap_two_sided_p(draws),
        },
    }


def verdict(observed_pp: float, expected_pp: float, test: dict[str, Any]) -> str:
    """SURVIVES / WEAKENS / VANISHES, decided mechanically.

    SURVIVES: expected-error keeps the sign and at least half the observed
    magnitude, and the candidate-level interval excludes zero. The interval is
    the U-statistic one (see :func:`paired_candidate_test`).
    VANISHES: expected-error flips the sign or keeps under a quarter of it.
    WEAKENS: everything between, including a retained point estimate whose
    interval includes zero.
    """

    if observed_pp == 0:
        return "NO OBSERVED DIFFERENCE"
    same_sign = np.sign(observed_pp) == np.sign(expected_pp)
    retained = abs(expected_pp) / abs(observed_pp) if same_sign else 0.0
    if not same_sign or retained < 0.25:
        return "VANISHES"
    if retained >= 0.5 and test["excludes_zero"]:
        return "SURVIVES"
    return "WEAKENS"


def score_cell(
    table: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    rankers: dict[str, Ranker],
    rng: np.random.Generator,
    checkpoint_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    evidence = _evidence(table, target)
    predictions: dict[str, dict[tuple[str, str], float]] = {}
    failures: dict[str, str] = {}
    for name, ranker in rankers.items():
        source = checkpoint_table if name == "checkpoint_augmented" and checkpoint_table is not None else table
        try:
            predictions[name] = _gaps(ranker(source, budgets, target))
        except (FitError, ValueError, RuntimeError) as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    baseline = predictions["single_scale"]
    base_scores = score_ranker(baseline, evidence)
    out: dict[str, Any] = {
        "determined_fraction_bonferroni": sum(e.determined for e in evidence) / len(evidence),
        "single_scale": {k: base_scores[k] for k in ("observed_rate", "expected_rate", "n_scored")},
        "rankers": {},
        "failures": failures,
    }
    for name, gaps in predictions.items():
        if name == "single_scale":
            continue
        scores = score_ranker(gaps, evidence)
        observed = 100 * (scores["observed_rate"] - base_scores["observed_rate"])
        expected = 100 * (scores["expected_rate"] - base_scores["expected_rate"])
        test = paired_candidate_test(gaps, baseline, evidence, rng)
        out["rankers"][name] = {
            "observed_rate": scores["observed_rate"],
            "expected_rate": scores["expected_rate"],
            "excess_observed_pp": observed,
            "excess_expected_pp": expected,
            "candidate_test": test,
            "verdict": verdict(observed, expected, test),
        }
    return out


def rescore_certification() -> dict[str, Any]:
    """Abstention: how many certified orderings are wrong in expectation?

    The committed result is 292 of 300 pairs certified at 1B with zero certified
    errors against the observed target. Zero observed errors is not the same as
    zero expected errors when the reference is itself uncertain.
    """

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    target = float(frame[frame["scale_label"] == "1B"]["compute"].iloc[0])
    rung = max(float(c) for c in frame["compute"].unique() if c < target)
    evidence = {(e.left, e.right): e for e in _evidence(frame, target)}
    target_names = set(frame[np.isclose(frame["compute"], target)]["intervention"].astype(str))

    # Same inputs as run_abstention_study.cost_of_correctness: the shared set and
    # the pooled-median fallback for a zero spread.
    cells = frame[np.isclose(frame["compute"], rung)].groupby("intervention")["bpb"]
    entries = {str(k): float(v) for k, v in cells.mean().items() if str(k) in target_names}
    sigmas = {str(k): float(v) for k, v in cells.std(ddof=1).items() if str(k) in target_names}
    pooled = float(np.median([s for s in sigmas.values() if s > 0]))
    errors = {k: (s if s > 0 else pooled) / np.sqrt(N_SEEDS) for k, s in sigmas.items()}

    rules: dict[str, Any] = {}
    # "normal" is the committed rule (292 certified); "student_t" applies the
    # small-sample critical value the Student-t defect says it should have used.
    for rule, n_seeds in (("normal", None), ("student_t", N_SEEDS)):
        verdicts = certify_leaderboard(entries, errors, CERT_DELTA, adjacent_only=False, n_seeds=n_seeds)
        certified = observed_wrong = 0
        expected_wrong = 0.0
        min_probability = 1.0
        for v in verdicts:
            if v.abstained:
                continue
            key = (v.left, v.right) if (v.left, v.right) in evidence else (v.right, v.left)
            item = evidence[key]
            certified += 1
            rung_sign = np.sign(entries[item.left] - entries[item.right])
            agrees = rung_sign == np.sign(item.gap)
            if not agrees:
                observed_wrong += 1
            p = item.probability_observed_order_correct
            min_probability = min(min_probability, p)
            expected_wrong += (1 - p) if agrees else p
        rules[rule] = {
            # Phi saturates to 1.0 in float64 beyond |gap|/se of about 8, so an
            # expected count of exactly zero means every certified pair is there.
            "min_target_order_probability_on_certified": min_probability,
            "n_certified": certified,
            "certified_errors_observed": observed_wrong,
            "certified_errors_expected": expected_wrong,
            "verdict": certification_verdict(expected_wrong),
        }
    return {
        "rung_compute": rung,
        "target_compute": target,
        "delta": CERT_DELTA,
        "seed_budget": N_SEEDS,
        "rules": rules,
        "note": (
            "Zero observed certified errors was never evidence of a guarantee. Under expected-error "
            "scoring the certified set carries this many errors in expectation; it is the reference's "
            "residual uncertainty on certified pairs, not a property of the rule."
        ),
    }


def certification_verdict(expected_errors: float) -> str:
    """Fewer than one certified error in expectation keeps the zero-error claim."""

    return "SURVIVES" if expected_errors < 1.0 else "WEAKENS"


def rescore_allocation(rng: np.random.Generator) -> dict[str, Any]:
    """Optimal allocation: point fit on each design's supported budgets.

    The allocation study scored by seed-resampling run counts. Re-scoring needs
    point predictions, so each sweep design is fitted on the cell means of the
    budgets it supports. That drops the run-count weighting, which is stated.
    """

    alloc = Source.load("allocation/allocation_1B.json")
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    target = alloc.number("target_compute")
    evidence = _evidence(frame, target)
    ladder = [float(b) for b in alloc.get("ladder")]
    baseline = _gaps(single_scale_ranker(frame, tuple(ladder), target))

    rows = []
    for key, entry in sorted(alloc.get("sweep").items(), key=lambda kv: kv[1]["budget_fraction"]):
        runs = entry.get("scored", {}).get("decision_optimal", {}).get("rounded_runs")
        if not runs:
            continue
        support = tuple(b for b, r in zip(ladder, runs) if r >= 1)
        if len(support) < 3:
            continue
        try:
            gaps = _gaps(projection_ranker(frame, support, target))
        except (FitError, ValueError):
            continue
        scores = score_ranker(gaps, evidence)
        base = score_ranker(baseline, evidence)
        observed = 100 * (scores["observed_rate"] - base["observed_rate"])
        expected = 100 * (scores["expected_rate"] - base["expected_rate"])
        test = paired_candidate_test(gaps, baseline, evidence, rng)
        rows.append(
            {
                "budget_fraction": entry["budget_fraction"],
                "n_support_budgets": len(support),
                "excess_observed_pp": observed,
                "excess_expected_pp": expected,
                "candidate_test": test,
                "verdict": verdict(observed, expected, test),
            }
        )
    return {
        "target_compute": target,
        "note": "point fit on supported budgets; the study's run-count weighting is not reproduced here",
        "by_budget_fraction": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "target_scoring" / "rescoring.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    training = Source.load("task_b/task_b.json").get("shrinkage_strength_protocol.training_interventions")
    build = _load_checkpoint_builder()
    from asla.analysis.checkpoints import make_checkpoint_ranker

    payload: dict[str, Any] = {
        "primary_rule": "expected_error",
        "design": "4M-300M fit, 530M target (the audit's primary design)",
        "eb_strength_provenance": "estimated on Task B training interventions only (results/task_b/task_b.json)",
        "verdict_rule": verdict.__doc__,
        "cells": {},
    }
    for metric, regime in METRIC_REGIME.items():
        built = build(metric, "300M", "530M")
        rankers: dict[str, Ranker] = {
            "single_scale": single_scale_ranker,
            "projection": projection_ranker,
            "shared_exponent": shared_exponent_ranker,
            "eb_shrinkage": make_shrinkage_ranker(None, strength_interventions=training),
            "ensemble": ensemble_ranker,
            "checkpoint_augmented": make_checkpoint_ranker("ar1"),
        }
        budgets = normalize_budgets(built["budgets"], target=built["target"])
        cell = score_cell(built["final_table"], budgets, built["target"], rankers, rng, built["ckpt_table"])
        cell["metric"] = metric
        cell["regime"] = regime
        payload["cells"][metric] = cell
        print(f"scored {metric}", flush=True)

    payload["certification"] = rescore_certification()
    payload["optimal_allocation"] = rescore_allocation(rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
