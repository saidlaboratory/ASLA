import numpy as np
import pandas as pd
import pytest

from asla.analysis.audit import audit_with_ci, seed_noise_report
from asla.analysis.rankers import projection_ranker, single_scale_ranker
from asla.config import AuditConfig
from asla.data.synthetic import negative_controls_only


def test_audit_with_ci_reports_intervals_and_under_seeded_cells():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(1), cfg)
    ragged = df[~((df["intervention"] == "control_a") & (df["compute"] == 1.0) & (df["seed"] > 0))]
    result = audit_with_ci(
        ragged,
        cfg.budgets.fit,
        cfg.budgets.target,
        {"projection_ranker": projection_ranker, "single_scale_ranker": single_scale_ranker},
        n_boot=10,
        rng=np.random.default_rng(2),
    )
    assert result["under_seeded_cells"]
    assert result["noise"]["adequately_seeded_cells"] > 0
    assert np.isfinite(result["noise"]["noise_band"])
    assert result["rankers"]["projection_ranker"]["mis_selection_rate"]["lo"] <= 0.0


def test_audit_with_ci_requires_at_least_one_ranker():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(1), cfg)
    with pytest.raises(ValueError, match="at least one ranker"):
        audit_with_ci(df, cfg.budgets.fit, cfg.budgets.target, {}, n_boot=5, rng=np.random.default_rng(2))


def test_noise_report_marks_band_unestimated_when_all_target_cells_have_one_seed():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(1), cfg)
    one_seed = df[df["seed"] == 0].copy()
    result = audit_with_ci(
        one_seed,
        cfg.budgets.fit,
        cfg.budgets.target,
        {"single_scale_ranker": single_scale_ranker},
        n_boot=5,
        rng=np.random.default_rng(3),
    )
    assert result["noise"]["noise_band"] is None
    assert result["noise"]["noise_band_estimated"] is False
    assert result["noise"]["target_under_seeded_cells"]


def test_noise_band_pools_only_target_cells_when_available():
    rng = np.random.default_rng(0)
    rows = []
    for intervention, offset in (("a", 0.0), ("b", 0.02)):
        for compute, sd in ((1.0, 0.3), (2.0, 0.3), (4.0, 0.3), (8.0, 0.001)):
            for seed in range(3):
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": "x",
                        "compute": compute,
                        "seed": seed,
                        "bpb": 1.5 - 0.1 * np.log2(compute) + offset + rng.normal(0.0, sd),
                    }
                )
    df = pd.DataFrame(rows)
    report = seed_noise_report(df, target=8.0)
    assert report["noise_band_source"] == "target_cells"
    assert report["adequately_seeded_cells"] == 2
    assert report["noise_band"] < 0.05


def test_noise_report_uses_non_target_cells_when_target_is_under_seeded():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(1), cfg)
    target_one_seed = df[~((df["compute"] == cfg.budgets.target) & (df["seed"] > 0))].copy()
    result = audit_with_ci(
        target_one_seed,
        cfg.budgets.fit,
        cfg.budgets.target,
        {"single_scale_ranker": single_scale_ranker},
        n_boot=5,
        rng=np.random.default_rng(4),
    )
    assert result["noise"]["noise_band_estimated"] is True
    assert result["noise"]["noise_band"] is not None
    assert result["noise"]["target_under_seeded_cells"]


def test_audit_ci_contains_known_zero_misselection_rate():
    cfg = AuditConfig()
    contains = 0
    trials = 8
    for seed in range(trials):
        df = negative_controls_only(np.random.default_rng(seed), cfg)
        result = audit_with_ci(
            df,
            cfg.budgets.fit,
            cfg.budgets.target,
            {"projection_ranker": projection_ranker},
            n_boot=20,
            rng=np.random.default_rng(100 + seed),
        )
        interval = result["rankers"]["projection_ranker"]["mis_selection_rate"]
        contains += int(interval["lo"] <= 0.0 <= interval["hi"])
    assert contains / trials >= 0.8


def _constructed_seed_count_case(n_seeds: int) -> pd.DataFrame:
    rows = []
    for intervention, offset in [("a", 0.0), ("b", 0.03)]:
        for compute in [1.0, 2.0, 4.0, 8.0]:
            for seed in range(n_seeds):
                wobble = 0.12 if seed % 2 == 0 else -0.12
                shrink = 1.0 / np.sqrt(n_seeds)
                if compute == 4.0 and intervention == "a":
                    bpb = 1.00 + wobble * shrink
                elif compute == 4.0 and intervention == "b":
                    bpb = 1.02 - wobble * shrink
                elif compute == 8.0:
                    bpb = 0.90 + offset + 0.02 * wobble * shrink
                else:
                    bpb = 1.2 + offset - 0.05 * compute
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": "x",
                        "compute": compute,
                        "seed": seed,
                        "bpb": bpb,
                    }
                )
    return pd.DataFrame(rows)


def test_audit_ci_regret_interval_tends_to_narrow_with_more_seeds():
    low = audit_with_ci(
        _constructed_seed_count_case(2),
        (1.0, 2.0, 4.0),
        8.0,
        {"single_scale_ranker": single_scale_ranker},
        n_boot=40,
        rng=np.random.default_rng(6),
    )
    high = audit_with_ci(
        _constructed_seed_count_case(16),
        (1.0, 2.0, 4.0),
        8.0,
        {"single_scale_ranker": single_scale_ranker},
        n_boot=40,
        rng=np.random.default_rng(6),
    )
    low_regret = low["rankers"]["single_scale_ranker"]["mean_regret"]
    high_regret = high["rankers"]["single_scale_ranker"]["mean_regret"]
    low_width = low_regret["hi"] - low_regret["lo"]
    high_width = high_regret["hi"] - high_regret["lo"]
    assert high_width <= low_width
