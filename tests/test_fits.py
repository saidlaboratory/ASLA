import numpy as np
import pytest

from asla.analysis.fits import project_ranking, truth_ranking
from asla.analysis.gate import largest_single_run_pick
from asla.config import AuditConfig
from asla.data.synthetic import negative_controls_only


def test_project_ranking_rejects_target_budget_leakage():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    with pytest.raises(ValueError, match="cannot be included"):
        project_ranking(df, (*cfg.budgets.fit, cfg.budgets.target), cfg.budgets.target)


def test_largest_single_run_pick_rejects_target_budget_leakage():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    with pytest.raises(ValueError, match="cannot be included"):
        largest_single_run_pick(df, (*cfg.budgets.fit, cfg.budgets.target), cfg.budgets.target)


def test_project_ranking_rejects_intervention_without_fitting_rows():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    bad = df[~((df["intervention"] == "control_a") & (df["compute"].isin(cfg.budgets.fit)))]
    with pytest.raises(ValueError, match="no fitting rows"):
        project_ranking(bad, cfg.budgets.fit, cfg.budgets.target)


def test_truth_ranking_rejects_intervention_without_target_rows():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    bad = df[~((df["intervention"] == "control_a") & (df["compute"] == cfg.budgets.target))]
    with pytest.raises(ValueError, match="no target rows"):
        truth_ranking(bad, cfg.budgets.target)
