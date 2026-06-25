import pytest

from asla.data.harvest import _coerce_harvested_rows
from asla.data.schema import validate


def test_coerce_harvested_rows_drops_missing_after_type_coercion():
    rows = [
        {
            "intervention": "a",
            "intervention_class": "class",
            "compute": "1.0",
            "seed": "0",
            "bpb": "1.2",
        },
        {
            "intervention": "b",
            "intervention_class": "class",
            "compute": "not-a-number",
            "seed": "0",
            "bpb": "1.1",
        },
    ]
    df, dropped = _coerce_harvested_rows(rows)
    validate(df)
    assert dropped == 1
    assert df["intervention"].tolist() == ["a"]


def test_coerce_harvested_rows_rejects_non_integer_seed():
    rows = [
        {
            "intervention": "a",
            "intervention_class": "class",
            "compute": "1.0",
            "seed": "0.5",
            "bpb": "1.2",
        }
    ]
    with pytest.raises(ValueError, match="integer-like"):
        _coerce_harvested_rows(rows)
