import numpy as np
import pandas as pd
import pytest

from asla.analysis.crossover import crossover_budget_ci, detect_crossovers, fitted_crossover_for_pair, naive_crossover_budget
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, negative_controls_only, saturation_crossover
from asla.models import bpb_power_law


def _pairs(crossovers):
    return {frozenset((a, b)) for a, b, _ in crossovers}


def test_detector_flags_clean_crossover():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(1729), cfg)
    cross = detect_crossovers(df, cfg.budgets.fit, cfg.budgets.target)
    assert frozenset(("fast_start", "late_scaler")) in _pairs(cross)


def test_detector_flags_saturation_crossover():
    cfg = AuditConfig()
    df = saturation_crossover(np.random.default_rng(1729), cfg)
    cross = detect_crossovers(df, cfg.budgets.fit, cfg.budgets.target)
    assert frozenset(("saturating_early", "late_power")) in _pairs(cross)


def test_detection_and_crossover_budget_are_unit_invariant_at_flop_scale():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(1729), cfg)
    unit = 1e18
    df_raw = df.copy()
    df_raw["compute"] = df_raw["compute"] * unit
    budgets = tuple(b * unit for b in cfg.budgets.fit)
    target = cfg.budgets.target * unit
    cross = detect_crossovers(df_raw, budgets, target)
    assert frozenset(("fast_start", "late_scaler")) in _pairs(cross)
    root_raw = fitted_crossover_for_pair(df_raw, "fast_start", "late_scaler", budgets)
    root_rel = fitted_crossover_for_pair(df, "fast_start", "late_scaler", cfg.budgets.fit)
    assert root_raw is not None
    assert root_rel is not None
    assert np.isclose(root_raw / unit, root_rel, rtol=0.05)


def test_negative_control_false_positive_rate_low():
    cfg = AuditConfig()
    n = 40
    flagged = 0
    for seed in range(n):
        df = negative_controls_only(np.random.default_rng(seed), cfg)
        flagged += int(bool(detect_crossovers(df, cfg.budgets.fit, cfg.budgets.target)))
    assert flagged / n < 0.1


def test_non_crossing_power_laws_are_not_roots_by_absolute_tolerance():
    assert naive_crossover_budget((0.9, 0.2, 0.5), (0.9, 0.3, 0.5)) is None


def test_genuine_power_law_crossing_uses_bracketed_root():
    root = naive_crossover_budget((0.9, 0.2, 0.5), (0.8, 0.4, 0.5))
    assert root == pytest.approx(4.0, rel=1e-8)


def _crossover_bootstrap_frame(crossing: bool) -> pd.DataFrame:
    rows = []
    params = {
        "a": (0.9, 0.2, 0.5),
        "b": (0.8, 0.4, 0.5) if crossing else (0.9, 0.3, 0.5),
    }
    for intervention, curve in params.items():
        for compute in (1.0, 2.0, 4.0, 8.0, 16.0):
            for seed in range(3):
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": "x",
                        "compute": compute,
                        "seed": seed,
                        "bpb": float(bpb_power_law(compute, *curve)) + (seed - 1) * 1e-5,
                    }
                )
    return pd.DataFrame(rows)


def test_crossover_bootstrap_reports_selection_probability_without_conditioning_failures_away():
    report = crossover_budget_ci(
        _crossover_bootstrap_frame(True),
        "a",
        "b",
        (1.0, 2.0, 4.0, 8.0, 16.0),
        n_boot=8,
        rng=np.random.default_rng(5),
    )
    assert report["n_boot"] == 8
    assert report["fit_successes"] == 8
    assert report["crossings_found"] == 8
    assert report["crossing_found_fraction"] == 1.0
    assert report["median"] == pytest.approx(4.0, rel=0.05)

    absent = crossover_budget_ci(
        _crossover_bootstrap_frame(False),
        "a",
        "b",
        (1.0, 2.0, 4.0, 8.0, 16.0),
        n_boot=8,
        rng=np.random.default_rng(5),
    )
    assert 0.0 <= absent["crossing_found_fraction"] < 1.0
    assert absent["crossing_found_fraction"] == absent["crossings_found"] / absent["n_boot"]
    assert absent["fit_failures"] + absent["fit_successes"] == absent["n_boot"]
    assert absent["non_crossing_draws"] == absent["fit_successes"] - absent["crossings_found"]
