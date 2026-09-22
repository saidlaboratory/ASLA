"""Tests for the threshold sweep and equivalence bounds.

The load-bearing one is :func:`test_zero_error_subset_is_flagged_vacuous`: a
difference of zero between two rankers is only informative if the subset
contained errors to differ on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("run_robustness_sweep", REPO / "scripts" / "run_robustness_sweep.py")
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["run_robustness_sweep"] = sweep
_SPEC.loader.exec_module(sweep)


def test_zero_error_subset_is_flagged_vacuous() -> None:
    """Zero versus zero bounds nothing; the width comes from n alone."""

    bound = sweep.equivalence_bound(0, 0, 246)
    assert bound["vacuous"] is True
    assert bound["difference_pp"] == 0.0
    assert bound["method"].startswith("rule of three")
    # Rule of three: 3/246 = 1.22%.
    assert bound["largest_difference_ruled_out_pp"] == pytest.approx(1.2195, abs=1e-3)


def test_real_difference_is_not_flagged_vacuous() -> None:
    bound = sweep.equivalence_bound(6, 2, 294)
    assert bound["vacuous"] is False
    assert bound["difference_pp"] > 0
    assert bound["ci_low_pp"] < bound["difference_pp"] < bound["ci_high_pp"]


def test_equivalence_bound_handles_empty_subset() -> None:
    assert sweep.equivalence_bound(0, 0, 0)["n_pairs"] == 0


def test_rule_of_three_narrows_with_more_pairs() -> None:
    """A larger zero-error subset rules out more, but still only by size."""

    small = sweep.equivalence_bound(0, 0, 50)["largest_difference_ruled_out_pp"]
    large = sweep.equivalence_bound(0, 0, 500)["largest_difference_ruled_out_pp"]
    assert large < small


def test_shipped_threshold_curve_shows_the_vanishing_is_vacuous() -> None:
    """The committed curve must show zero gap only where there are no errors."""

    path = REPO / "results" / "target_scoring" / "robustness.json"
    if not path.exists():  # pragma: no cover - results not always materialised
        pytest.skip("robustness results not present")
    cell = json.loads(path.read_text())["cells"]["c4_en_bits_per_token/primary_4M-300M_gate530M"]
    curve = cell["threshold_curve"]
    zero_gap = [r for r in curve if abs(r["gap_pp"]) < 1e-9]
    assert zero_gap, "expected at least one zero-gap threshold"
    # Every zero-gap row must be a subset in which neither ranker erred.
    assert all(r["comparison_is_vacuous"] for r in zero_gap)
    # And once errors appear the gap must be positive.
    with_errors = [r for r in curve if not r["comparison_is_vacuous"]]
    assert with_errors and all(r["gap_pp"] > 0 for r in with_errors)


def test_shipped_published_figure_shifts_upward() -> None:
    """The protocol's application to the published number must be recorded."""

    path = REPO / "results" / "target_scoring" / "robustness.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("robustness results not present")
    published = json.loads(path.read_text())["published_figure"]
    for rule in ("bonferroni", "bh"):
        block = published[rule]
        assert block["accuracy_on_determined"] > block["accuracy_overall"]
        assert block["shift_pp"] > 0
