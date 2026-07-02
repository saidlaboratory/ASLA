"""WS5 ASLA-Bench: generator determinism, knob semantics, grid feasibility."""

import numpy as np
import pandas as pd
import pytest

from asla.analysis.fits import project_ranking, truth_ranking
from asla.data.benchmark import BenchmarkConfig, benchmark_grid, evaluate_configs, make_scenario
from asla.data.schema import validate
from asla.data.synthetic import true_ranking


def test_generator_is_deterministic():
    config = BenchmarkConfig(family="late_crossover", crossover_position=0.3, seed=7)
    scenario = make_scenario(config)
    df1 = scenario(np.random.default_rng(5), None)
    df2 = scenario(np.random.default_rng(5), None)
    pd.testing.assert_frame_equal(df1, df2)
    validate(df1)


def test_winner_is_true_best_and_gap_matches_knob():
    config = BenchmarkConfig(family="close_call", gap_to_noise=4.0, noise=0.01, seed=3)
    scenario = make_scenario(config)
    df = scenario(np.random.default_rng(0), None)
    truth = true_ranking(df)
    assert truth.index[0] == "winner"
    assert truth.index[1] == "rival"
    assert np.isclose(truth.loc["rival"] - truth.loc["winner"], 4.0 * 0.01)


def test_late_crossover_rival_leads_inside_ladder():
    config = BenchmarkConfig(family="late_crossover", gap_to_noise=3.0, crossover_position=0.3, seed=11)
    scenario = make_scenario(config)
    df = scenario(np.random.default_rng(1), None)
    truth_fns = df.attrs["scenario_truth"].functions
    c_max_fit = max(config.budgets.fit)
    assert truth_fns["rival"](c_max_fit) < truth_fns["winner"](c_max_fit)
    assert truth_fns["rival"](config.budgets.target) > truth_fns["winner"](config.budgets.target)


def test_saturating_family_fools_projection_but_not_truth():
    config = BenchmarkConfig(
        family="saturating", gap_to_noise=4.0, crossover_position=0.3, saturation_strength=1.0, noise=0.002, seed=5
    )
    scenario = make_scenario(config)
    df = scenario(np.random.default_rng(2), None)
    truth = truth_ranking(df, config.budgets.target)
    assert truth.index[0] == "winner"
    projected = project_ranking(df, config.budgets.fit, config.budgets.target)
    assert projected.index[0] == "rival"


def test_infeasible_knobs_raise():
    with pytest.raises(ValueError):
        make_scenario(BenchmarkConfig(family="late_crossover", crossover_position=None))
    with pytest.raises(ValueError):
        make_scenario(BenchmarkConfig(family="late_crossover", crossover_position=2.0))
    with pytest.raises(ValueError):
        make_scenario(BenchmarkConfig(family="unknown"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_scenario(BenchmarkConfig(n_interventions=1))


def test_benchmark_grid_skips_infeasible_and_names_are_unique():
    configs = benchmark_grid(gaps=(2.0, 100.0), positions=(0.3, 0.85))
    names = [c.name() for c in configs]
    assert len(names) == len(set(names))
    assert all(isinstance(c, BenchmarkConfig) for c in configs)
    for config in configs:
        make_scenario(config)  # every surviving config must be feasible


def test_evaluate_configs_produces_tidy_rows():
    configs = [
        BenchmarkConfig(family="close_call", gap_to_noise=6.0, seed=1),
        BenchmarkConfig(family="late_crossover", gap_to_noise=2.0, crossover_position=0.3, seed=2),
    ]
    table = evaluate_configs(configs, n_trials=3, n_boot=25, seed=9)
    assert set(table["rule"]) == {"plain", "largest", "gate", "race"}
    assert len(table) == len(configs) * 4
    assert {"mean_regret", "wrong_pick_rate", "mean_compute"} <= set(table.columns)
    assert (table["mean_compute"] > 0).all()


def test_harder_positions_are_not_easier_for_plain_projection():
    # Crossovers far beyond the ladder must not be easier for one-shot
    # projection than crossovers visible inside it.
    easy = BenchmarkConfig(family="late_crossover", gap_to_noise=2.0, crossover_position=-0.30, seed=2)
    hard = BenchmarkConfig(family="late_crossover", gap_to_noise=2.0, crossover_position=0.30, seed=2)
    table = evaluate_configs([easy, hard], n_trials=8, n_boot=25, seed=4)
    plain = table[table["rule"] == "plain"].set_index("crossover_position")
    assert (
        plain.loc[0.30, "wrong_pick_rate"] >= plain.loc[-0.30, "wrong_pick_rate"]
    )
