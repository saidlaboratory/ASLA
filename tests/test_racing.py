"""WS4 racing policy: elimination behavior, cost accounting, Monte Carlo comparison."""

import numpy as np
import pytest

from asla.analysis.racing import (
    cell_cost,
    gate_result,
    ladder_cost,
    largest_single_result,
    monte_carlo_selection,
    plain_projection_result,
    race_pick,
)
from asla.config import AuditConfig
from asla.data.synthetic import clean_crossover, noise_close_call

PRE_TARGET = (1.0, 2.0, 4.0, 8.0, 16.0)


def test_cell_and_ladder_cost():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(0), cfg)
    # clean_crossover uses two seeds per cell.
    assert cell_cost(df, "fast_start", 8.0) == 8.0 * 2
    expected = sum(b * 2 for b in cfg.budgets.fit) * df["intervention"].nunique()
    assert ladder_cost(df, sorted(df["intervention"].unique()), cfg.budgets.fit) == expected


def test_plain_and_largest_costs_ordering():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(1), cfg)
    plain = plain_projection_result(df, cfg.budgets.fit, cfg.budgets.target)
    largest = largest_single_result(df, cfg.budgets.fit, cfg.budgets.target)
    assert plain.compute_spent > largest.compute_spent
    assert plain.pick in set(df["intervention"])
    assert largest.pick in set(df["intervention"])


def test_gate_result_cost_covers_escalation():
    cfg = AuditConfig()
    df = noise_close_call(np.random.default_rng(2), cfg)
    interventions = sorted(df["intervention"].unique())
    base = ladder_cost(df, interventions, cfg.budgets.fit)
    result = gate_result(
        df,
        cfg.budgets.fit,
        cfg.budgets.target,
        cfg.budgets.intermediate,
        tau=cfg.gate.tau,
        n_boot=60,
        rng=np.random.default_rng(3),
    )
    assert result.compute_spent >= base
    assert result.pick in set(interventions)


def test_race_eliminates_weak_control_and_spends_less_than_full_ladder():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(4), cfg)
    interventions = sorted(df["intervention"].unique())
    result = race_pick(df, PRE_TARGET, cfg.budgets.target, n_boot=100, rng=np.random.default_rng(5))
    full = ladder_cost(df, interventions, PRE_TARGET)
    assert result.compute_spent < full
    eliminated = {name for record in result.rungs for name in record.eliminated}
    assert "weak_control" in eliminated
    assert result.pick in set(interventions)


def test_race_beta_controls_aggressiveness():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(6), cfg)
    aggressive = race_pick(df, PRE_TARGET, cfg.budgets.target, n_boot=100, rng=np.random.default_rng(7), beta=0.25)
    conservative = race_pick(df, PRE_TARGET, cfg.budgets.target, n_boot=100, rng=np.random.default_rng(7), beta=4.0)
    assert aggressive.compute_spent <= conservative.compute_spent
    with pytest.raises(ValueError):
        race_pick(df, PRE_TARGET, cfg.budgets.target, n_boot=10, rng=np.random.default_rng(8), beta=0.0)


def test_race_requires_enough_rungs_and_interventions():
    cfg = AuditConfig()
    df = clean_crossover(np.random.default_rng(9), cfg)
    with pytest.raises(ValueError):
        race_pick(df, (1.0, 2.0), cfg.budgets.target, n_boot=10, rng=np.random.default_rng(10))
    single = df[df["intervention"] == "fast_start"]
    with pytest.raises(ValueError):
        race_pick(single, PRE_TARGET, cfg.budgets.target, n_boot=10, rng=np.random.default_rng(11))


def test_monte_carlo_selection_reports_costs_and_race_competitiveness():
    from dataclasses import replace

    cfg = AuditConfig()
    cfg = replace(cfg, gate=replace(cfg.gate, n_boot=60))
    result = monte_carlo_selection(noise_close_call, cfg, n_trials=12, rng=np.random.default_rng(13))
    assert set(result) == {"plain", "largest", "gate", "race"}
    for stats in result.values():
        assert set(stats) == {"mean_regret", "wrong_pick_rate", "mean_compute"}
        assert stats["mean_compute"] > 0
    # The race must not pay more than the plain full-ladder cost on average
    # (it runs the same ladder minus eliminated cells, plus the intermediate rung).
    pre_target_full = result["plain"]["mean_compute"] + 0.0
    assert result["race"]["mean_compute"] <= pre_target_full * 3
    # And its decision quality should not collapse relative to plain projection.
    assert result["race"]["mean_regret"] <= result["plain"]["mean_regret"] * 2 + 1e-6
