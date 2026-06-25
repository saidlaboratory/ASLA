import pandas as pd
import pytest

from asla.analysis.metrics import decision_metrics


def test_regret_zero_iff_top1_correct():
    projected = pd.Series({"a": 1.0, "b": 2.0})
    truth = pd.Series({"a": 1.5, "b": 2.5})
    m = decision_metrics(projected, truth, k=1)
    assert m["top1_acc"] == 1.0
    assert m["regret"] == 0.0

    projected_bad = pd.Series({"b": 1.0, "a": 2.0})
    m_bad = decision_metrics(projected_bad, truth, k=1)
    assert m_bad["top1_acc"] == 0.0
    assert m_bad["regret"] > 0.0


def test_hand_computed_metrics():
    projected = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    truth = pd.Series({"a": 1.0, "c": 2.0, "b": 3.0})
    m = decision_metrics(projected, truth, k=2)
    assert m["top1_acc"] == 1.0
    assert m["topk_recall"] == 0.5
    assert m["pairwise_acc"] == 2 / 3
    assert m["regret"] == 0.0


def test_metrics_reject_invalid_k():
    projected = pd.Series({"a": 1.0, "b": 2.0})
    truth = pd.Series({"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="k must be positive"):
        decision_metrics(projected, truth, k=0)


def test_metrics_reject_mismatched_interventions():
    projected = pd.Series({"a": 1.0, "b": 2.0})
    truth = pd.Series({"a": 1.0, "c": 2.0})
    with pytest.raises(ValueError, match="identical interventions"):
        decision_metrics(projected, truth, k=1)
