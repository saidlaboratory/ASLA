"""Tests for the observed-vs-expected-error re-scoring of decision results."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from asla.analysis.target_scoring import target_evidence

REPO = Path(__file__).resolve().parents[1]
rescoring = importlib.import_module("scripts.run_rescoring")

EXCLUDES = {"excludes_zero": True}
INCLUDES = {"excludes_zero": False}


def test_verdict_survives_needs_retained_sign_magnitude_and_interval() -> None:
    assert rescoring.verdict(2.0, 1.5, EXCLUDES) == "SURVIVES"
    # Same magnitude, but the candidate-resampled interval includes zero.
    assert rescoring.verdict(2.0, 1.5, INCLUDES) == "WEAKENS"
    # Interval excludes zero, but under half the magnitude is kept.
    assert rescoring.verdict(2.0, 0.8, EXCLUDES) == "WEAKENS"


def test_verdict_vanishes_on_sign_flip_or_small_remainder() -> None:
    assert rescoring.verdict(2.0, -0.1, EXCLUDES) == "VANISHES"
    assert rescoring.verdict(-2.0, 0.1, EXCLUDES) == "VANISHES"
    assert rescoring.verdict(2.0, 0.4, EXCLUDES) == "VANISHES"


def test_verdict_reports_no_difference_rather_than_a_ratio_of_zero() -> None:
    assert rescoring.verdict(0.0, -0.2, INCLUDES) == "NO OBSERVED DIFFERENCE"


def test_certification_verdict_threshold_is_one_expected_error() -> None:
    assert rescoring.certification_verdict(0.18) == "SURVIVES"
    assert rescoring.certification_verdict(1.0) == "WEAKENS"


def _evidence(n: int, spread: float) -> list:
    rng = np.random.default_rng(0)
    means = {f"c{i}": float(i) for i in range(n)}
    sds = {name: spread * (1 + 0.1 * rng.random()) for name in means}
    counts = {name: 3 for name in means}
    return target_evidence(means, sds, counts, alpha=0.05, multiplicity="bonferroni")


def test_paired_candidate_test_detects_a_ranker_that_is_always_wrong() -> None:
    evidence = _evidence(8, spread=0.05)
    truth = {(e.left, e.right): e.gap for e in evidence}
    wrong = {pair: -gap for pair, gap in truth.items()}
    result = rescoring.paired_candidate_test(wrong, truth, evidence, np.random.default_rng(1))
    assert result["point_pp"] == pytest.approx(100.0, abs=1e-6)
    assert result["excludes_zero"]
    assert 0.0 <= result["p_two_sided"] <= 1.0


def test_paired_candidate_test_identical_rankers_give_zero_and_valid_p() -> None:
    evidence = _evidence(6, spread=2.0)
    truth = {(e.left, e.right): e.gap for e in evidence}
    result = rescoring.paired_candidate_test(truth, truth, evidence, np.random.default_rng(2))
    assert result["point_pp"] == pytest.approx(0.0)
    assert not result["excludes_zero"]
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_shipped_certification_reproduces_the_committed_count() -> None:
    """The normal-critical rule must certify what the abstention study shipped."""

    import json

    from asla.provenance import Source

    rescored = Source.load("target_scoring/rescoring.json")
    committed = json.loads((REPO / "results" / "abstention" / "abstention_1B.json").read_text(encoding="utf-8"))
    three_seed = next(row for row in committed["cost_of_correctness"] if row["seed_budget"] == 3)
    assert rescored.number("certification.rules.normal.n_certified") == three_seed["n_certified"]
    # The Student-t rule is strictly more conservative.
    assert rescored.number("certification.rules.student_t.n_certified") <= rescored.number(
        "certification.rules.normal.n_certified"
    )
