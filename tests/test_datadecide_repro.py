"""Guards on the DataDecide reproduction package sent to the authors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "repro" / "datadecide_decision_acc" / "out"


def test_decision_accuracy_scores_ties_and_direction() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("repro", REPO / "repro" / "datadecide_decision_acc" / "reproduce.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = pd.Series({"a": 3.0, "b": 2.0, "c": 2.0})
    pred = pd.Series({"a": 1.0, "b": 0.0, "c": 0.5})
    assert module.decision_accuracy(pred, target, higher=True, tie=0.5) == pytest.approx((1 + 1 + 0.5) / 3)
    assert module.decision_accuracy(pred, target, higher=False, tie=0.0) == pytest.approx(0.0)


def test_shipped_outputs_are_consistent() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    nonmatching = pd.read_csv(OUT / "nonmatching_rows.csv")
    assert len(nonmatching) == summary["rows_not_matching_reference_convention"]
    assert summary["rows_matching_reference_convention"] + len(nonmatching) == summary["rows"]
    check = summary["target_seed_check"]
    assert check["tasks_where_it_does_not"] == []
    assert check["max_abs_stacked_y_minus_three_seed_mean"] > check["max_abs_stacked_y_minus_default_seed"]
