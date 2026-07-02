import numpy as np
import pytest

from asla.analysis.gate import gate_pick, monte_carlo
from asla.config import AuditConfig, GateConfig
from asla.data.synthetic import noise_close_call, saturation_crossover


def test_gate_escalation_fails_loudly_without_intermediate_rows():
    cfg = AuditConfig()
    df = noise_close_call(np.random.default_rng(3), cfg)
    no_intermediate = df[~np.isclose(df["compute"].astype(float), cfg.budgets.intermediate)]
    with pytest.raises(ValueError, match="intermediate budget"):
        gate_pick(
            no_intermediate,
            cfg.budgets.fit,
            cfg.budgets.target,
            cfg.budgets.intermediate,
            tau=1e9,
            n_boot=40,
            rng=np.random.default_rng(4),
        )


def test_gate_beats_plain_in_noise_close_call():
    cfg = AuditConfig(gate=GateConfig(tau=0.8, n_boot=80))
    result = monte_carlo(noise_close_call, cfg, n_trials=45, rng=np.random.default_rng(11))
    assert result["gate"]["mean_regret"] < result["plain"]["mean_regret"] * 0.85
    assert result["gate"]["wrong_pick_rate"] < result["plain"]["wrong_pick_rate"]


def test_gate_does_not_materially_help_saturation_crossover():
    cfg = AuditConfig(gate=GateConfig(tau=2.0, n_boot=80))
    result = monte_carlo(saturation_crossover, cfg, n_trials=30, rng=np.random.default_rng(12))
    assert result["plain"]["wrong_pick_rate"] >= 0.8
    assert result["gate"]["wrong_pick_rate"] >= result["plain"]["wrong_pick_rate"] - 0.1
    assert result["gate"]["mean_regret"] >= result["plain"]["mean_regret"] * 0.8
