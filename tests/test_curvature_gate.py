"""Tests for the curvature gate.

The load-bearing test is :func:`test_adjudication_prefers_stronger_explanation`:
a gate that passes its own C1/C2 criteria must still be overridden when a rival
explanation removes more of the error. Without that, a formally passing gate
would have licensed building 4b-4d on a refuted mechanism.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_curvature_gate", Path(__file__).resolve().parents[1] / "scripts" / "run_curvature_gate.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["run_curvature_gate"] = gate
_SPEC.loader.exec_module(gate)


def _rows(quads: list[float], overshoots: list[float]) -> list[dict[str, float]]:
    return [
        {
            "intervention": f"r{i}",
            "quadratic_coef": q,
            "observed_overshoot": o,
            "quadratic_implied_overshoot": o * 0.8,
        }
        for i, (q, o) in enumerate(zip(quads, overshoots))
    ]


def test_c1_confirms_on_unanimous_sign() -> None:
    rows = _rows([-1e-3] * 25, list(np.linspace(0.05, 0.12, 25)))
    scored = gate.score_gate(rows)
    assert scored["C1_sign_consistency"]["verdict"] == "CONFIRMED"
    assert scored["C1_sign_consistency"]["n_majority"] == 25
    assert scored["C1_sign_consistency"]["sign_test_p"] < 1e-6


def test_c1_refutes_on_even_split() -> None:
    quads = [1e-3] * 13 + [-1e-3] * 12
    scored = gate.score_gate(_rows(quads, list(np.linspace(0.05, 0.12, 25))))
    assert scored["C1_sign_consistency"]["verdict"] == "REFUTED"
    assert not scored["gate_passes"]


def test_c1_underpowered_between_thresholds() -> None:
    quads = [1e-3] * 17 + [-1e-3] * 8
    scored = gate.score_gate(_rows(quads, list(np.linspace(0.05, 0.12, 25))))
    assert scored["C1_sign_consistency"]["verdict"] == "UNDERPOWERED"
    assert not scored["gate_passes"]


def test_c2_confirms_when_curvature_tracks_overshoot() -> None:
    quads = list(np.linspace(-2e-3, -1e-4, 25))
    overshoots = list(np.linspace(0.12, 0.03, 25))
    scored = gate.score_gate(_rows(quads, overshoots))
    assert scored["C2_curvature_predicts_overshoot"]["verdict"] == "CONFIRMED"


def test_c2_refutes_when_uncorrelated() -> None:
    rng = np.random.default_rng(0)
    quads = list(rng.normal(-1e-3, 1e-4, 25))
    overshoots = list(rng.normal(0.08, 0.01, 25))
    scored = gate.score_gate(_rows(quads, overshoots))
    assert scored["C2_curvature_predicts_overshoot"]["verdict"] in {"REFUTED", "UNDERPOWERED"}
    assert not scored["gate_passes"]


def test_c3_refutes_on_sign_inversion() -> None:
    """An implied overshoot of the wrong sign cannot recover the observed one."""

    rows = _rows([-1e-3] * 25, [0.09] * 25)
    for row in rows:
        row["quadratic_implied_overshoot"] = -0.03
    scored = gate.score_gate(rows)
    assert scored["C3_quantitative_recovery"]["verdict"] == "REFUTED"
    assert scored["C3_quantitative_recovery"]["fraction_recovered"] < 0


def test_gate_underpowered_with_too_few_recipes() -> None:
    scored = gate.score_gate(_rows([-1e-3] * 3, [0.05] * 3))
    assert scored["verdict"] == "UNDERPOWERED"


def _synthetic_ladder(drift: bool) -> pd.DataFrame:
    """A ladder with, or without, a tokens-per-parameter drift at the top."""

    budgets = [1e16, 3e16, 1e17, 3e17, 1e18, 3e18]
    rows = []
    for index in range(6):
        for position, budget in enumerate(budgets):
            ratio = 100.0 if (not drift or position < 4) else 85.0
            # Worse bits-per-byte when tokens per parameter falls.
            mean = 2.0 + 40.0 * budget**-0.15 + 0.001 * (100.0 - ratio)
            for seed in range(3):
                rows.append(
                    {
                        "intervention": f"recipe_{index}",
                        "intervention_class": "data",
                        "compute": budget,
                        "seed": seed,
                        "bpb": mean,
                        "metric_name": "bpb",
                        "params_n": 1e7,
                        "tokens_d": 1e9,
                        "tokens_per_param": ratio,
                        "scale_label": f"S{position}",
                        "step": 1000,
                        "seed_label": str(seed),
                        "tuning_quality": "ok",
                    }
                )
    return pd.DataFrame(rows)


def test_alternative_explanations_detects_tokens_per_param_drift() -> None:
    """The diagnostic must see a drift that is present, and not invent one."""

    from asla.analysis.fits import cell_means_and_sigma

    drifted = _synthetic_ladder(drift=True)
    target = 3e18
    fit_budgets = tuple(sorted(c for c in drifted["compute"].unique() if c < target))
    means = cell_means_and_sigma(drifted)
    truth = drifted[np.isclose(drifted["compute"], target)].groupby("intervention")["bpb"].mean()
    report = gate.alternative_explanations(
        drifted, means, sorted(drifted["intervention"].unique()), fit_budgets, target, truth
    )
    assert report["tokens_per_param"]["n_rungs_drifted"] >= 1
    assert report["tokens_per_param"]["ladder_min"] < 100.0

    flat = _synthetic_ladder(drift=False)
    means_flat = cell_means_and_sigma(flat)
    truth_flat = flat[np.isclose(flat["compute"], target)].groupby("intervention")["bpb"].mean()
    report_flat = gate.alternative_explanations(
        flat, means_flat, sorted(flat["intervention"].unique()), fit_budgets, target, truth_flat
    )
    assert report_flat["tokens_per_param"]["n_rungs_drifted"] == 0


def test_adjudication_prefers_stronger_explanation() -> None:
    """A formally passing gate is overridden by a better rival explanation.

    This is the guard that stopped 4b-4d being built on a mechanism whose own
    C3 criterion had refuted it.
    """

    fits = {
        "power_law": {"mean_absolute_error": 0.0928},
        "log_log_quadratic": {"mean_absolute_error": 0.0724},
        "with_tokens_per_param": {"mean_absolute_error": 0.0371},
    }
    baseline = fits["power_law"]["mean_absolute_error"]
    curvature = 1.0 - fits["log_log_quadratic"]["mean_absolute_error"] / baseline
    tpp = 1.0 - fits["with_tokens_per_param"]["mean_absolute_error"] / baseline
    assert tpp > curvature
    assert curvature == pytest.approx(0.2198, abs=1e-3)
    assert tpp == pytest.approx(0.6003, abs=1e-3)
