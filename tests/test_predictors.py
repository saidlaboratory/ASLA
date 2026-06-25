import numpy as np
import pytest

from asla.analysis.crossover import mechanism_crossover_budget, naive_crossover_budget
from asla.analysis.fits import fit_all
from asla.config import AuditConfig
from asla.data.synthetic import saturation_crossover


def test_naive_predictor_returns_none_on_saturating_pair():
    cfg = AuditConfig()
    df = saturation_crossover(np.random.default_rng(1729), cfg)
    params = fit_all(df[df["intervention"].isin(["saturating_early", "late_power"])], cfg.budgets.fit)
    assert naive_crossover_budget(params["saturating_early"], params["late_power"]) is None


def test_mechanism_predictor_is_explicit_stub():
    with pytest.raises(NotImplementedError):
        mechanism_crossover_budget()

