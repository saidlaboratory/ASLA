import numpy as np
import pandas as pd
import pytest

from asla.analysis.fits import project_ranking
from asla.models import FitError, bpb_chinchilla, fit_chinchilla


def test_fit_chinchilla_recovers_known_parameters():
    params = (0.80, 0.30, 0.40, 0.25, 0.35)
    n_values, d_values = np.meshgrid(np.array([1.0, 2.0, 4.0, 8.0]), np.array([1.0, 3.0, 9.0, 27.0]))
    n = n_values.ravel()
    d = d_values.ravel()
    y = bpb_chinchilla(n, d, *params)
    fitted = fit_chinchilla(n, d, y)
    assert np.allclose(fitted, params, rtol=0.15, atol=0.06)


def test_fit_chinchilla_is_unit_invariant_at_realistic_scale():
    a, b = 0.40, 0.35
    params = (0.80, 0.30 * (1e8) ** a, a, 0.25 * (1e9) ** b, b)
    n_values, d_values = np.meshgrid(np.array([1e8, 2e8, 4e8, 8e8]), np.array([1e9, 3e9, 9e9, 2.7e10]))
    n = n_values.ravel()
    d = d_values.ravel()
    y = bpb_chinchilla(n, d, *params)
    fitted = fit_chinchilla(n, d, y)
    truth = float(bpb_chinchilla(1e9, 1e11, *params))
    projected = float(bpb_chinchilla(1e9, 1e11, *fitted))
    assert abs(projected - truth) < 1e-3


def test_chinchilla_projection_requires_columns():
    df = pd.DataFrame(
        {
            "intervention": ["a", "a", "a"],
            "intervention_class": ["x", "x", "x"],
            "compute": [1.0, 2.0, 4.0],
            "seed": [0, 0, 0],
            "bpb": [1.2, 1.1, 1.0],
        }
    )
    with pytest.raises(ValueError, match="requires columns"):
        project_ranking(df, (1.0, 2.0, 4.0), 8.0, fit_form="chinchilla")


def test_chinchilla_project_ranking_on_two_dimensional_table():
    rows = []
    budgets = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
    designs = {
        1.0: (1.0, 1.0),
        2.0: (2.0, 9.0),
        3.0: (3.0, 2.0),
        4.0: (4.0, 16.0),
        5.0: (5.0, 3.0),
        8.0: (8.0, 27.0),
    }
    params_by_intervention = {
        "a": (0.80, 0.30, 0.40, 0.25, 0.35),
        "b": (0.84, 0.30, 0.40, 0.25, 0.35),
    }
    for intervention, params in params_by_intervention.items():
        for compute in budgets:
            for seed in range(2):
                n, d = designs[compute]
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": "x",
                        "compute": compute,
                        "seed": seed,
                        "params_n": n,
                        "tokens_d": d,
                        "bpb": float(bpb_chinchilla(n, d, *params)),
                    }
                )
    df = pd.DataFrame(rows)
    ranking = project_ranking(df, (1.0, 2.0, 3.0, 4.0, 5.0), 8.0, fit_form="chinchilla")
    assert ranking.index[0] == "a"


def test_chinchilla_projection_rejects_missing_target_inputs():
    rows = []
    params = (0.80, 0.30, 0.40, 0.25, 0.35)
    fit_budgets = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    fit_designs = [(1.0, 1.0), (2.0, 9.0), (3.0, 2.0), (4.0, 16.0), (5.0, 3.0), (6.0, 12.0)]
    for compute, (n_value, d_value) in [*zip(fit_budgets, fit_designs), (8.0, (np.nan, 64.0))]:
        rows.append(
            {
                "intervention": "a",
                "intervention_class": "x",
                "compute": compute,
                "seed": 0,
                "params_n": n_value,
                "tokens_d": d_value,
                "bpb": float(bpb_chinchilla(compute, compute**2, *params)),
            }
        )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="target Chinchilla inputs"):
        project_ranking(df, fit_budgets, 8.0, fit_form="chinchilla")


def test_chinchilla_rejects_collinear_axes_even_with_distinct_values():
    n = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = n**2
    y = np.asarray(bpb_chinchilla(n, d, 0.8, 0.3, 0.4, 0.2, 0.3))
    with pytest.raises(FitError, match="independently varied"):
        fit_chinchilla(n, d, y)


def test_chinchilla_requires_five_distinct_design_points_not_duplicate_seeds():
    n = np.repeat(np.array([1.0, 2.0, 4.0]), 3)
    d = np.repeat(np.array([1.0, 9.0, 3.0]), 3)
    y = np.asarray(bpb_chinchilla(n, d, 0.8, 0.3, 0.4, 0.2, 0.3))
    with pytest.raises(FitError, match="5 distinct"):
        fit_chinchilla(n, d, y)


def test_chinchilla_projection_rejects_inconsistent_target_design_across_seeds():
    rows = []
    params = (0.80, 0.30, 0.40, 0.25, 0.35)
    fit_designs = [(1.0, 1.0), (1.0, 3.0), (2.0, 1.0), (2.0, 3.0), (4.0, 9.0), (8.0, 3.0)]
    for compute, (n_value, d_value) in enumerate(fit_designs, start=1):
        rows.append(
            {
                "intervention": "a",
                "intervention_class": "x",
                "compute": float(compute),
                "seed": 0,
                "params_n": n_value,
                "tokens_d": d_value,
                "bpb": float(bpb_chinchilla(n_value, d_value, *params)),
            }
        )
    for seed, n_value in enumerate((10.0, 11.0)):
        rows.append(
            {
                "intervention": "a",
                "intervention_class": "x",
                "compute": 8.0,
                "seed": seed,
                "params_n": n_value,
                "tokens_d": 12.0,
                "bpb": float(bpb_chinchilla(n_value, 12.0, *params)),
            }
        )
    with pytest.raises(ValueError, match="identical across seeds"):
        project_ranking(pd.DataFrame(rows), tuple(range(1, 7)), 8.0, fit_form="chinchilla")
