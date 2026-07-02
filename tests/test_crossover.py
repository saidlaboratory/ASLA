import numpy as np

from asla.analysis.crossover import detect_crossovers, fitted_crossover_for_pair
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, negative_controls_only, saturation_crossover


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

