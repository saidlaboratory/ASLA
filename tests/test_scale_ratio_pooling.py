"""Regression tests for the scale-ratio pooling defect (methods practice instance 5).

A ratio of two *scales* estimated from per-cell standard errors must pool those
errors as ``sqrt(mean(se**2))``, not ``mean(se)``. The two differ by Jensen's
inequality whenever the standard errors vary, always in the same direction, and
using the mean understates the denominator and inflates the ratio.

This is the same class of error as the ``phi = 0`` analytic-limit defect caught
earlier in the project. That one was fixed at its site; this one recurred in a
different file, which is why the check now lives in a shared test rather than
next to a single caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def pooled_scale(standard_errors: np.ndarray) -> float:
    """The correct denominator for a ratio of scales."""

    return float(np.sqrt(np.mean(np.square(standard_errors))))


def test_mean_of_errors_understates_pooled_scale() -> None:
    """Jensen: mean(se) <= sqrt(mean(se^2)), with equality only when se is constant."""

    rng = np.random.default_rng(0)
    for _ in range(200):
        errors = np.abs(rng.normal(0.01, 0.004, size=11))
        assert float(np.mean(errors)) <= pooled_scale(errors) + 1e-15

    constant = np.full(11, 0.01)
    assert float(np.mean(constant)) == pytest.approx(pooled_scale(constant))


def test_gap_grows_with_dispersion() -> None:
    """The error is negligible for uniform noise and large for dispersed noise."""

    tight = np.array([0.010, 0.0101, 0.0099, 0.0100])
    spread = np.array([0.001, 0.004, 0.020, 0.050])
    assert pooled_scale(tight) / float(np.mean(tight)) < 1.01
    assert pooled_scale(spread) / float(np.mean(spread)) > 1.4


def test_committed_misspecification_ratio_is_self_consistent() -> None:
    """The shipped figure must equal sqrt of its own shipped variance inflation.

    This is the cross-check that the superseded hardcoded pair (4.22 and 17.8)
    would also have passed --- 4.22^2 = 17.81 --- which is precisely why it is
    not sufficient on its own. The test below adds the part that pair failed.
    """

    path = RESULTS / "theory_v2" / "theory_v2.json"
    if not path.exists():  # pragma: no cover - results are not always materialised
        pytest.skip("theory_v2 results not present")
    evidence = json.loads(path.read_text())["FINDING_misspecification_is_structured"]["evidence"]
    real = evidence["1_misspecification_is_real"]
    assert real["residual_scatter_over_seed_se"] == pytest.approx(np.sqrt(real["variance_inflation"]), rel=1e-9)


def test_committed_misspecification_ratio_traces_to_its_inputs() -> None:
    """The figure must be reconstructible from the variances shipped beside it.

    A hardcoded literal cannot satisfy this, which is the property that would
    have caught instance 5 at the time it was introduced.
    """

    path = RESULTS / "theory_v2" / "theory_v2.json"
    if not path.exists():  # pragma: no cover - results are not always materialised
        pytest.skip("theory_v2 results not present")
    evidence = json.loads(path.read_text())["FINDING_misspecification_is_structured"]["evidence"]
    real = evidence["1_misspecification_is_real"]
    for key in ("pooled_residual_var", "pooled_seed_var", "n_interventions", "design"):
        assert key in real, f"the shipped figure must carry its input {key!r}"
    implied = real["pooled_residual_var"] / real["pooled_seed_var"]
    assert real["variance_inflation"] == pytest.approx(implied, rel=1e-9)
    assert real["residual_scatter_over_seed_se"] == pytest.approx(np.sqrt(implied), rel=1e-9)


def test_superseded_figure_is_not_reintroduced() -> None:
    """Guard against the old pair reappearing in the shipped results."""

    path = RESULTS / "theory_v2" / "theory_v2.json"
    if not path.exists():  # pragma: no cover - results are not always materialised
        pytest.skip("theory_v2 results not present")
    evidence = json.loads(path.read_text())["FINDING_misspecification_is_structured"]["evidence"]
    real = evidence["1_misspecification_is_real"]
    assert real["residual_scatter_over_seed_se"] != pytest.approx(4.22, abs=5e-3)
    assert real["variance_inflation"] != pytest.approx(17.8, abs=5e-2)
