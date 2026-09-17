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
