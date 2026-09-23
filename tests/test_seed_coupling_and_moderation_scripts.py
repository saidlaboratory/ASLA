"""Tests for the seed-coupling and moderated-variance scripts."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

coupling = importlib.import_module("scripts.run_seed_coupling")
moderated = importlib.import_module("scripts.run_moderated_variance")


def _table(rho: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(30):
        shared = rng.standard_normal(3)
        for scale in ("a", "b"):
            own = rng.standard_normal(3)
            noise = rho * shared + np.sqrt(1 - rho**2) * own if scale == "b" else shared
            for s, label in enumerate(("default", "small aux 2", "small aux 3")):
                rows.append({"intervention": f"r{r}", "scale_label": scale, "seed_label": label, "bpb": 1.0 + noise[s]})
    return pd.DataFrame(rows)


def test_coupling_detects_planted_correlation_and_not_its_absence() -> None:
    rng = np.random.default_rng(0)
    planted = coupling.coupling(coupling.deviations(_table(0.8, 1)), "a", "b", rng)
    null = coupling.coupling(coupling.deviations(_table(0.0, 2)), "a", "b", rng)
    assert planted is not None and null is not None
    assert planted["rho"] > 0.5 and planted["p_two_sided"] < 0.01
    assert abs(null["rho"]) < 0.3 and null["p_two_sided"] > 0.01


def test_null_p_values_are_valid_probabilities_for_every_procedure() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, (25, 3)), rng.normal(0, 1, (25, 3))
    out = moderated._null_p_values(a, b)
    assert set(out) == {"raw_welch", "moderated_t", "moderated_se_raw_df"}
    for values in out.values():
        assert values.shape == (25,) and np.all((values >= 0) & (values <= 1))


def test_weights_are_probabilities_and_handle_zero_variance() -> None:
    values = np.array([[1.0, 1.0, 1.0], [2.0, 2.1, 1.9], [3.0, 3.2, 2.9], [0.5, 0.6, 0.4]])
    weights, gap = moderated._weights(values, 3)
    for w in weights.values():
        assert np.all((w >= 0.5) & (w <= 1.0)) and np.all(np.isfinite(w))
    assert gap.shape == (6,)
