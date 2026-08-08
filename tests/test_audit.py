import numpy as np
import pandas as pd
import pytest

from asla.analysis.audit import audit_with_ci, seed_noise_report
from asla.analysis.rankers import make_gate_ranker, projection_ranker, single_scale_ranker
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
        estimand="single_design_seed_sensitivity",
    )
    assert result["under_seeded_cells"]
    assert result["noise"]["adequately_seeded_cells"] > 0
    assert np.isfinite(result["noise"]["noise_band"])
    assert result["rankers"]["projection_ranker"]["mis_selection_rate"]["lo"] <= 0.0


def test_audit_with_ci_requires_at_least_one_ranker():
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(1), cfg)
    with pytest.raises(ValueError, match="at least one ranker"):
        audit_with_ci(
            df,
            cfg.budgets.fit,
            cfg.budgets.target,
            {},
            n_boot=5,
            rng=np.random.default_rng(2),
            estimand="single_design_seed_sensitivity",
        )


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
        estimand="single_design_seed_sensitivity",
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
        estimand="single_design_seed_sensitivity",
    )
    assert result["noise"]["noise_band_estimated"] is True
    assert result["noise"]["noise_band"] is not None
    assert result["noise"]["target_under_seeded_cells"]


def test_noise_report_matches_ragged_seed_hand_calculation():
    df = pd.DataFrame(
        {
            "intervention": ["a", "a", "b", "b", "b", "b"],
            "intervention_class": ["x"] * 6,
            "compute": [8.0] * 6,
            "seed": [0, 1, 0, 1, 2, 3],
            "bpb": [1.0, 3.0, 1.0, 1.0, 1.0, 5.0],
        }
    )
    report = seed_noise_report(df, target=8.0, k=2.0)
    assert report["pooled_variance"] == pytest.approx(3.5)
    assert report["pooled_degrees_of_freedom"] == 4
    mean_inverse_n = (1 / 2 + 1 / 4) / 2
    expected_sem = np.sqrt(3.5 * mean_inverse_n)
    assert report["effective_seed_count"] == pytest.approx(1 / mean_inverse_n)
    assert report["representative_mean_sem"] == pytest.approx(expected_sem)
    assert report["representative_gap_sem"] == pytest.approx(np.sqrt(2) * expected_sem)
    assert report["noise_band"] == pytest.approx(2 * expected_sem)


def test_noise_report_borrows_variance_but_uses_one_seed_target_counts():
    df = pd.DataFrame(
        {
            "intervention": ["a", "a", "a", "b", "b", "b"],
            "intervention_class": ["x"] * 6,
            "compute": [4.0, 4.0, 8.0, 4.0, 4.0, 8.0],
            "seed": [0, 1, 0, 0, 1, 0],
            "bpb": [1.0, 3.0, 0.9, 2.0, 4.0, 1.0],
        }
    )
    report = seed_noise_report(df, target=8.0, k=2.0)
    assert report["noise_band_source"] == "all_cells"
    assert report["effective_seed_count"] == pytest.approx(1.0)
    assert report["target_under_seeded_cells"]
    assert report["representative_mean_sem"] == pytest.approx(np.sqrt(2.0))


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
            estimand="single_design_seed_sensitivity",
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
        estimand="single_design_seed_sensitivity",
    )
    high = audit_with_ci(
        _constructed_seed_count_case(16),
        (1.0, 2.0, 4.0),
        8.0,
        {"single_scale_ranker": single_scale_ranker},
        n_boot=40,
        rng=np.random.default_rng(6),
        estimand="single_design_seed_sensitivity",
    )
    low_regret = low["rankers"]["single_scale_ranker"]["mean_regret"]
    high_regret = high["rankers"]["single_scale_ranker"]["mean_regret"]
    low_width = low_regret["hi"] - low_regret["lo"]
    high_width = high_regret["hi"] - high_regret["lo"]
    assert high_width <= low_width


def _estimand_case() -> pd.DataFrame:
    rows = []
    target_values = {"a": 0.90, "b": 0.95, "c": 1.00}
    for intervention in target_values:
        for compute in (1.0, 2.0, 4.0, 8.0):
            for seed in range(2):
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": "x",
                        "compute": compute,
                        "seed": seed,
                        "bpb": target_values[intervention] + 0.1 / compute,
                    }
                )
    return pd.DataFrame(rows)


def _fixed_b_before_a_ranker(df, budgets, target):
    values = {"a": 1.0, "b": 0.0, "c": 2.0}
    names = sorted(df["intervention"].astype(str).unique())
    return pd.Series({name: values[name] for name in names}).sort_values(kind="mergesort")


def test_both_estimands_report_their_actual_replication_units():
    common = {
        "df": _estimand_case(),
        "budgets": (1.0, 2.0, 4.0),
        "target": 8.0,
        "rankers": {"fixed": _fixed_b_before_a_ranker},
        "n_boot": 5,
    }
    single = audit_with_ci(
        **common,
        rng=np.random.default_rng(10),
        estimand="single_design_seed_sensitivity",
    )
    pairwise = audit_with_ci(
        **common,
        rng=np.random.default_rng(10),
        estimand="pairwise_decisions",
    )
    assert single["estimand"]["replication_count"] == 1
    assert "complete-candidate-set" in single["estimand"]["replication_unit"]
    assert single["rankers"]["fixed"]["mis_selection_rate"]["point"] == 1.0
    assert pairwise["estimand"]["replication_count"] == 3
    assert pairwise["rankers"]["fixed"]["mis_selection_rate"]["point"] == pytest.approx(1 / 3)
    assert pairwise["estimand"]["assumptions"]


def test_audit_rejects_unknown_estimand():
    with pytest.raises(ValueError, match="unknown estimand"):
        audit_with_ci(
            _estimand_case(),
            (1.0, 2.0, 4.0),
            8.0,
            {"fixed": _fixed_b_before_a_ranker},
            n_boot=2,
            rng=np.random.default_rng(1),
            estimand="paper_headline",  # type: ignore[arg-type]
        )


def test_gate_is_top1_only_and_omits_full_ranking_metrics(monkeypatch):
    monkeypatch.setattr("asla.analysis.rankers.gate_pick", lambda *args, **kwargs: "b")
    gate = make_gate_ranker(6.0, tau=1.0, n_boot=2, seed=3)
    point = gate(_estimand_case(), (1.0, 2.0, 4.0), 8.0)
    assert list(point.index) == ["b"]
    result = audit_with_ci(
        _estimand_case(),
        (1.0, 2.0, 4.0),
        8.0,
        {"gate_ranker": gate},
        n_boot=2,
        rng=np.random.default_rng(4),
        estimand="single_design_seed_sensitivity",
    )
    assert set(result["rankers"]["gate_ranker"]) == {
        "top1_acc",
        "regret",
        "mis_selection_rate",
        "mean_regret",
    }
    omitted = result["ranker_metric_availability"]["gate_ranker"]["omitted"]
    assert set(omitted) == {"kendall_tau", "pairwise_acc", "spearman", "topk_recall"}
