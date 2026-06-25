import pandas as pd
import pytest

from asla.data.schema import SchemaError, validate


def good_df():
    return pd.DataFrame(
        {
            "intervention": ["a"],
            "intervention_class": ["c"],
            "compute": [1.0],
            "seed": [0],
            "bpb": [1.0],
        }
    )


def test_validate_accepts_schema():
    validate(good_df())


def test_validate_rejects_missing_columns():
    df = good_df().drop(columns=["bpb", "seed"])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate(df)


def test_validate_rejects_wrong_dtypes():
    df = good_df()
    df["seed"] = [1.2]
    df["compute"] = ["big"]
    with pytest.raises(SchemaError) as exc:
        validate(df)
    msg = str(exc.value)
    assert "seed" in msg
    assert "compute" in msg


def test_validate_rejects_nulls_and_nonfinite_values():
    df = good_df()
    df.loc[0, "bpb"] = float("inf")
    with pytest.raises(SchemaError, match="finite"):
        validate(df)

    df = good_df()
    df.loc[0, "intervention"] = None
    with pytest.raises(SchemaError, match="must not contain null"):
        validate(df)

    df = good_df()
    df.loc[0, "compute"] = 0.0
    with pytest.raises(SchemaError, match="positive"):
        validate(df)
