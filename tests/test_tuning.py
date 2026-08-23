import pandas as pd
import pytest

from asla.analysis.tuning import single_scale_pairwise_agreement, stratify_by_tuning, tuning_quality_levels


def _table(values: dict[str, tuple[float, float]], tuning: str | None) -> pd.DataFrame:
    rows = []
    for name, (small, large) in values.items():
        rows.append({"intervention": name, "intervention_class": "optimizer", "compute": 1.0, "seed": 0, "bpb": small})
        rows.append({"intervention": name, "intervention_class": "optimizer", "compute": 8.0, "seed": 0, "bpb": large})
    df = pd.DataFrame(rows)
    if tuning is not None:
        df["tuning_quality"] = tuning
    return df


def test_single_scale_agreement_counts_order_flips():
    df = _table({"a": (1.0, 1.0), "b": (2.0, 3.0), "c": (3.0, 2.0)}, "tuned")
    out = single_scale_pairwise_agreement(df, 1.0, 8.0)
    assert out["n_pairs"] == 3
    assert out["n_disagreeing"] == 1
    assert out["disagreeing_pairs"][0]["a"] == "b" and out["disagreeing_pairs"][0]["b"] == "c"
    assert out["pairwise_agreement"] == pytest.approx(2 / 3)
    assert out["min_seeds_per_cell"] == 1


def test_stratify_by_tuning_reports_each_condition():
    tuned = _table({"a": (1.0, 1.0), "b": (2.0, 2.0), "c": (3.0, 3.0)}, "tuned")
    mistuned = _table({"a": (3.0, 1.0), "b": (2.0, 2.0), "c": (1.0, 3.0)}, "mistuned")
    out = stratify_by_tuning({"tuned": tuned, "mistuned": mistuned}, 1.0, 8.0)
    assert out["agreement_by_condition"] == {"tuned": 1.0, "mistuned": 0.0}


def test_stratify_by_tuning_rejects_mislabelled_tables():
    df = _table({"a": (1.0, 1.0), "b": (2.0, 2.0)}, "other")
    with pytest.raises(ValueError, match="tuning_quality levels"):
        stratify_by_tuning({"tuned": df}, 1.0, 8.0)


def test_tuning_quality_levels_handles_missing_column_and_nulls():
    df = _table({"a": (1.0, 1.0), "b": (2.0, 2.0)}, None)
    assert tuning_quality_levels(df) == [None]
    df["tuning_quality"] = ["x", None, "x", None]
    assert tuning_quality_levels(df) == ["x", None]
