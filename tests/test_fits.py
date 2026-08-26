import numpy as np
import pytest

from asla.analysis.fits import project_ranking, resolve_fit_budgets, truth_ranking
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


def test_resolve_fit_budgets_holds_out_intermediate():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(4), cfg)
    budgets = resolve_fit_budgets(df, cfg.budgets.target, None, cfg.budgets.intermediate)
    assert budgets == cfg.budgets.fit
    with pytest.raises(ValueError, match="must not include the reserved intermediate"):
        resolve_fit_budgets(
            df,
            cfg.budgets.target,
            (*cfg.budgets.fit, cfg.budgets.intermediate),
            cfg.budgets.intermediate,
        )


def test_adaptive_alpha_floor_scales_with_dynamic_range():
    from asla.models import adaptive_alpha_floor

    x = np.array([1.0, 10.0, 100.0, 1000.0, 1e4])
    # a BPB-like curve with a large relative decay keeps the historical floor
    strong = 0.5 + 3.0 * x ** (-0.15)
    assert adaptive_alpha_floor(x, strong) <= 0.05
    # an accuracy-like curve decaying a few percent gets a much lower floor
    weak = 0.65 - 0.02 * np.log10(x) / 4.0
    assert adaptive_alpha_floor(x, weak) < adaptive_alpha_floor(x, strong)
    assert adaptive_alpha_floor(x, weak) > 0.0
    # degenerate inputs fall back to the fixed floor
    assert adaptive_alpha_floor(x, np.full_like(x, 0.5)) == 0.05


def test_adaptive_bounds_recover_a_slow_exponent_that_the_fixed_floor_pins():
    from asla.models import fit_power_law

    x = np.array([1e15, 1e16, 1e17, 1e18, 1e19], dtype=float)
    truth_alpha = 0.004
    y = 0.55 + 0.5 * x ** (-truth_alpha)
    _, _, alpha_fixed = fit_power_law(x, y, adaptive_bounds=False)
    _, _, alpha_adaptive = fit_power_law(x, y, adaptive_bounds=True)
    assert alpha_fixed == pytest.approx(0.05, abs=1e-9)  # pinned at the old floor
    assert alpha_adaptive < 0.05  # freed by the adaptive floor
    predicted = fit_power_law(x, y, adaptive_bounds=True)
    assert np.allclose(predicted[0] + predicted[1] * x ** (-predicted[2]), y, rtol=1e-4)


def test_require_interior_raises_loudly_on_a_pinned_fit():
    from asla.models import BoundPinError, fit_power_law

    x = np.array([1e15, 1e16, 1e17, 1e18], dtype=float)
    y = 0.55 + 0.5 * x ** (-0.004)
    with pytest.raises(BoundPinError, match="landed on a bound"):
        fit_power_law(x, y, adaptive_bounds=False, require_interior=True)
    # with adaptive bounds the same data fits in the interior and does not raise
    fit_power_law(x, y, adaptive_bounds=True, require_interior=True)


def test_detect_bound_pins_reports_parameter_and_side():
    from asla.models import detect_bound_pins

    pins = detect_bound_pins((0.0, 1.0, 2.0), [0.0, 0.0, 0.05], [1.0, 5.0, 2.0], ("E", "A", "alpha"))
    found = {(p.parameter, p.side) for p in pins}
    assert ("E", "lower") in found and ("alpha", "upper") in found
    assert not detect_bound_pins((0.5, 1.0, 0.3), [0.0, 0.0, 0.05], [1.0, 5.0, 2.0], ("E", "A", "alpha"))
