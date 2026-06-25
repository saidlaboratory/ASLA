import numpy as np
import pytest

from asla.analysis.fits import project_ranking
from asla.config import AuditConfig
from asla.data.synthetic import negative_controls_only


def test_project_ranking_rejects_target_budget_leakage():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    with pytest.raises(ValueError, match="cannot be included"):
        project_ranking(df, (*cfg.budgets.fit, cfg.budgets.target), cfg.budgets.target)
