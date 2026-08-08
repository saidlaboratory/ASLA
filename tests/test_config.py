import numpy as np

from asla.analysis.gate import monte_carlo
from asla.cli import _runtime_config, build_parser
from asla.config import AuditConfig, CountConfig, GateConfig
from asla.data.synthetic import negative_controls_only


def test_default_counts_are_paper_grade():
    cfg = AuditConfig()
    assert cfg.counts.n_boot == 1000
    assert cfg.counts.n_trials == 500


def test_fast_flag_uses_small_counts():
    parser = build_parser()
    args = parser.parse_args(["demo", "--estimand", "single_design_seed_sensitivity", "--fast"])
    _, n_boot, n_trials = _runtime_config(args)
    cfg = AuditConfig()
    assert n_boot == cfg.counts.fast_n_boot
    assert n_trials == cfg.counts.fast_n_trials


def test_monte_carlo_uses_config_trial_default():
    cfg = AuditConfig(
        counts=CountConfig(n_trials=2, n_boot=10, fast_n_trials=1, fast_n_boot=5),
        gate=GateConfig(n_boot=12),
    )
    result = monte_carlo(negative_controls_only, cfg, rng=np.random.default_rng(1))
    assert set(result) == {"plain", "largest", "gate"}
