"""WS3 conformal intervals: construction, gate contract, racing integration."""

import numpy as np
import pytest

from asla.analysis.conformal import conformal_gate_pick, conformal_projection_interval
from asla.analysis.racing import race_pick
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, noise_close_call
from asla.models import FitError, bpb_power_law

PRE_TARGET = (1.0, 2.0, 4.0, 8.0, 16.0)


def _noisy_power_law(rng, budgets=(1.0, 2.0, 4.0, 8.0, 16.0), seeds=4, noise=0.005):
    x = np.repeat(np.asarray(budgets, dtype=float), seeds)
    y = np.asarray(bpb_power_law(x, 0.9, 0.3, 0.4), dtype=float) + rng.normal(0, noise, len(x))
    return x, y


def test_interval_contains_truth_on_clean_power_law():
    x, y = _noisy_power_law(np.random.default_rng(0), noise=1e-6)
    point, lo, hi = conformal_projection_interval(x, y, 64.0)
    truth = float(bpb_power_law(64.0, 0.9, 0.3, 0.4))
    assert lo <= truth <= hi
    assert np.isclose(point, truth, atol=1e-3)
    assert hi - lo < 0.01


def test_interval_widens_with_smaller_alpha():
    x, y = _noisy_power_law(np.random.default_rng(1))
    _, lo_wide, hi_wide = conformal_projection_interval(x, y, 64.0, alpha=0.05)
    _, lo_narrow, hi_narrow = conformal_projection_interval(x, y, 64.0, alpha=0.5)
    assert (hi_wide - lo_wide) >= (hi_narrow - lo_narrow)


def test_interval_requires_four_budgets_and_valid_alpha():
    x, y = _noisy_power_law(np.random.default_rng(2), budgets=(1.0, 2.0, 4.0))
    with pytest.raises(FitError):
        conformal_projection_interval(x, y, 64.0)
    x4, y4 = _noisy_power_law(np.random.default_rng(3))
    with pytest.raises(ValueError):
        conformal_projection_interval(x4, y4, 64.0, alpha=1.5)


def test_conformal_gate_is_deterministic_and_returns_valid_pick():
    cfg = AuditConfig()
    df = noise_close_call(np.random.default_rng(4), cfg)
    pick1 = conformal_gate_pick(df, cfg.budgets.fit, cfg.budgets.target, cfg.budgets.intermediate)
    pick2 = conformal_gate_pick(df, cfg.budgets.fit, cfg.budgets.target, cfg.budgets.intermediate)
    assert pick1 == pick2
    assert pick1 in set(df["intervention"])


def test_conformal_gate_escalation_needs_intermediate_rows():
    cfg = AuditConfig()
    df = noise_close_call(np.random.default_rng(5), cfg)
    trimmed = df[~np.isclose(df["compute"].astype(float), cfg.budgets.intermediate)]
    # noise_close_call top-2 overlap, so the gate must escalate and fail loudly.
    with pytest.raises(ValueError):
        conformal_gate_pick(trimmed, cfg.budgets.fit, cfg.budgets.target, cfg.budgets.intermediate)


def test_race_with_conformal_intervals_runs_and_eliminates():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(6), cfg)
    result = race_pick(
        df,
        PRE_TARGET,
        cfg.budgets.target,
        n_boot=10,
        rng=np.random.default_rng(7),
        interval_source="conformal",
    )
    assert result.pick in set(df["intervention"])
    eliminated = {name for record in result.rungs for name in record.eliminated}
    assert "weak_control" in eliminated
    with pytest.raises(ValueError):
        race_pick(df, PRE_TARGET, cfg.budgets.target, n_boot=10, rng=np.random.default_rng(8), interval_source="magic")
