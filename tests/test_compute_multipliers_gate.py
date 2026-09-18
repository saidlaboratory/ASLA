"""Tests for the compute-multipliers Task 0 gate.

The gate's job is to fail loudly when a suite advertises a ladder it does not
release. These tests pin that behaviour on synthetic frames, so the verdict logic
is exercised without a network call, and check the shipped assessment against the
real artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "assess_compute_multipliers", REPO / "scripts" / "assess_compute_multipliers.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["assess_compute_multipliers"] = gate
_SPEC.loader.exec_module(gate)


def _frame(budgets: list[float], seeds: list[int] = [6198, 6199, 6200]) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        for recipe in ("v2019_gpt2", "v2025_olmo2_control"):
            for seed in seeds:
                rows.append(
                    {
                        "run_id": f"{recipe}-{budget:.0e}-{seed}",
                        "roles": "model_axis",
                        "recipe": recipe,
                        "corpus": "fineweb_edu",
                        "budget_flops": budget,
                        "rung": "s8",
                        "seed": seed,
                        "native_heldout_nll": 3.0,
                        "olmes10": 0.4,
                        "heldout7_mean_nll": 3.2,
                        "alt8": 0.4,
                    }
                )
    return pd.DataFrame(rows)


def test_single_budget_fails_the_gate(tmp_path: Path) -> None:
    """One budget is an endpoint, not a ladder: there is nothing to extrapolate from."""

    path = tmp_path / "checkpoints.csv"
    _frame([1e19]).to_csv(path, index=False)
    coverage = gate.assess(path)
    assert coverage["n_budgets_public"] == 1
    assert coverage["has_usable_ladder"] is False
    assert coverage["lever_arms_available"] == []


def test_full_ladder_passes_the_gate(tmp_path: Path) -> None:
    """The gate must not be unconditionally negative; a real ladder passes it."""

    path = tmp_path / "checkpoints.csv"
    _frame([1e17, 3.16e17, 1e18, 3.16e18, 1e19]).to_csv(path, index=False)
    coverage = gate.assess(path)
    assert coverage["n_budgets_public"] == 5
    assert coverage["has_usable_ladder"] is True
    assert len(coverage["lever_arms_available"]) == 4
    assert max(coverage["lever_arms_available"]) == pytest.approx(100.0, rel=1e-6)


def test_exactly_the_minimum_ladder_passes(tmp_path: Path) -> None:
    """Three fitting budgets plus a held-out target is the stated requirement."""

    path = tmp_path / "checkpoints.csv"
    _frame([1e17, 1e18, 1e19, 1e20]).to_csv(path, index=False)
    assert gate.assess(path)["has_usable_ladder"] is True

    path_short = tmp_path / "short.csv"
    _frame([1e17, 1e18, 1e19]).to_csv(path_short, index=False)
    assert gate.assess(path_short)["has_usable_ladder"] is False


def test_replicate_counting_requires_two_seeds(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.csv"
    _frame([1e19], seeds=[6198]).to_csv(path, index=False)
    assert gate.assess(path)["cells_with_replicates"] == 0


def test_metric_agreement_detects_an_inversion() -> None:
    """A planted disagreement must show up as a low rank correlation."""

    rows = []
    for index, name in enumerate(("a", "b", "c", "d")):
        for seed in (0, 1):
            rows.append(
                {
                    "roles": "data_axis",
                    "recipe": "r",
                    "corpus": name,
                    "seed": seed,
                    # Perfectly opposed orderings across the two metric families.
                    "native_heldout_nll": float(index),
                    "olmes10": float(index),
                    "heldout7_mean_nll": float(index),
                    "alt8": float(index),
                }
            )
    report = gate.metric_agreement(pd.DataFrame(rows))
    # The recipe axis is absent from this frame; it must be reported as empty
    # rather than crashing on an argmin over nothing.
    assert report["recipe"]["n_candidates"] == 0
    # olmes10 and alt8 are higher-is-better, the NLLs lower-is-better, so an
    # identical column ranks in opposite directions.
    assert report["corpus"]["min_pairwise_spearman"] == pytest.approx(-1.0)
    assert report["corpus"]["n_distinct_winners"] == 2


def test_shipped_gate_records_a_stop_with_its_reasoning() -> None:
    path = REPO / "results" / "external" / "compute_multipliers_gate.json"
    if not path.exists():  # pragma: no cover - results not always materialised
        pytest.skip("gate results not present")
    payload = json.loads(path.read_text())
    assert payload["gate"]["passes"] is False
    assert "STOP" in payload["gate"]["verdict"]
    assert payload["coverage"]["n_budgets_public"] == 1
    # The suite is still worth citing for what it does answer.
    assert payload["metric_agreement_at_top_budget"]["corpus"]["n_distinct_winners"] >= 2


# --- Chart harvest: the compute-scaling curves recovered from the write-up ---

_CURVES_SPEC = importlib.util.spec_from_file_location(
    "harvest_compute_multipliers_charts",
    REPO / "scripts" / "harvest_compute_multipliers_charts.py",
)
assert _CURVES_SPEC is not None and _CURVES_SPEC.loader is not None
curves = importlib.util.module_from_spec(_CURVES_SPEC)
sys.modules["harvest_compute_multipliers_charts"] = curves
_CURVES_SPEC.loader.exec_module(curves)


def _curve_frame(values: dict[str, list[float]], sd: float = 0.001) -> pd.DataFrame:
    """A scaling-curve frame in the chart's own layout, with +/-1 sd bands."""

    data: dict[str, list[float]] = {"compute": [1e17, 3.16e17, 1e18, 3.16e18, 1e19]}
    for name, series in values.items():
        data[name] = series
        data[f"{name} −1 sd"] = [v - sd for v in series]
        data[f"{name} +1 sd"] = [v + sd for v in series]
    return pd.DataFrame(data)


def test_resolved_reversal_excludes_noise_sized_flips() -> None:
    """A flip inside the seed band must not count as a crossover.

    This is the distinction the whole estimand rests on: two candidates that
    swap order by less than their noise have not crossed over.
    """

    # b overtakes a by 0.0002, far inside a 0.001 sd.
    frame = _curve_frame({"a": [0.40, 0.40, 0.40, 0.40, 0.4000], "b": [0.3999] * 4 + [0.4002]}, sd=0.001)
    report = curves.analyse_axis(frame, "synthetic")
    last = report["by_fit_budget"][-1]
    assert last["raw_reversals"] == 1
    assert last["resolved_pairs"] == 0
    assert last["resolved_reversal_rate"] is None


def test_resolved_reversal_counts_a_real_crossover() -> None:
    """A flip far outside the band must count."""

    frame = _curve_frame({"a": [0.30, 0.32, 0.34, 0.36, 0.38], "b": [0.40, 0.41, 0.42, 0.43, 0.44]}, sd=0.0005)
    # b leads throughout: no reversal.
    assert curves.analyse_axis(frame, "synthetic")["by_fit_budget"][0]["resolved_reversals"] == 0

    crossing = _curve_frame({"a": [0.30, 0.34, 0.38, 0.42, 0.50], "b": [0.40, 0.41, 0.42, 0.43, 0.44]}, sd=0.0005)
    first = curves.analyse_axis(crossing, "synthetic")["by_fit_budget"][0]
    assert first["resolved_pairs"] == 1
    assert first["resolved_reversals"] == 1
    assert first["resolved_reversal_rate"] == pytest.approx(1.0)


def test_spread_in_seed_sds_is_scale_free() -> None:
    """The power diagnostic must compare spread to noise, not report spread alone."""

    tight = _curve_frame({"a": [0.40] * 5, "b": [0.402] * 5}, sd=0.001)
    wide = _curve_frame({"a": [0.40] * 5, "b": [0.460] * 5}, sd=0.001)
    assert (
        curves.analyse_axis(wide, "w")["endpoint"]["spread_in_seed_sds"]
        > curves.analyse_axis(tight, "t")["endpoint"]["spread_in_seed_sds"]
    )


def test_shipped_curves_record_the_underpowered_verdict() -> None:
    path = REPO / "results" / "external" / "compute_multipliers_curves.json"
    if not path.exists():  # pragma: no cover - results not always materialised
        pytest.skip("curve results not present")
    payload = json.loads(path.read_text())
    contrast = payload["class_contrast"]
    assert "UNDERPOWERED" in contrast["verdict"]
    # The recipe axis must retain too few resolved pairs to support a rate.
    assert contrast["resolved_reversals_total"]["recipe_resolved_pairs"] < 10
    # And the corpus axis must have many, which is what makes the asymmetry real.
    assert contrast["resolved_reversals_total"]["corpus_resolved_pairs"] > 20
    assert payload["recipe_axis"]["budgets"] == payload["corpus_axis"]["budgets"]
    assert len(payload["recipe_axis"]["budgets"]) == 5
