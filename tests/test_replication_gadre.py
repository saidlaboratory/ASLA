"""The Gadre replication: the derived table matches the pre-registered ladder, and the evidence rule is sound."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_replication_gadre import RECIPES, evidence

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "data" / "gadre_scaling_runs.csv"


@pytest.mark.skipif(not TABLE.exists(), reason="derived Gadre table not built")
def test_ladder_matches_preregistration():
    t = pd.read_csv(TABLE)
    assert len(t) == 104 and set(t["recipe"]) == set(RECIPES)
    ladder = t[np.isclose(t["chinchilla_multiplier"], 1.0)]
    sizes = sorted(ladder["params"].unique())
    assert len(sizes) == 6 and sizes[-1] > 6.8e9
    assert all(len(ladder[ladder["params"] == s]) == 3 for s in sizes)
    assert set(t["seed"].astype(str)) == {"124"}


def test_evidence_is_the_known_variance_limit():
    target = pd.Series({"a": 1.0, "b": 1.5, "c": 1.01})
    ev = {(e.left, e.right): e for e in evidence(target, {"a": 0.01, "b": 0.01, "c": 0.01})}
    assert ev[("a", "b")].determined and not ev[("a", "c")].determined
    assert np.isclose(ev[("a", "c")].standard_error, np.sqrt(2) * 0.01)
    assert 0.5 < ev[("a", "c")].probability_observed_order_correct < 1.0
