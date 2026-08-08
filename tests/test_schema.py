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


def test_validate_rejects_empty_table():
    with pytest.raises(SchemaError, match="at least one row"):
        validate(good_df().iloc[0:0])


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


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("intervention", "  ", "blank"),
        ("intervention_class", "", "blank"),
        ("bpb", 0.0, "positive"),
        ("bpb", -1.0, "positive"),
        ("seed", -1, "non-negative"),
    ],
)
def test_validate_rejects_invalid_required_values(column, value, match):
    df = good_df()
    df.loc[0, column] = value
    with pytest.raises(SchemaError, match=match):
        validate(df)


def test_validate_rejects_duplicate_run_identity_and_inconsistent_class():
    duplicate = pd.concat([good_df(), good_df()], ignore_index=True)
    with pytest.raises(SchemaError, match="uniquely identify"):
        validate(duplicate)
    inconsistent = pd.concat(
        [good_df(), good_df().assign(intervention_class="other", seed=1)],
        ignore_index=True,
    )
    with pytest.raises(SchemaError, match="exactly one intervention_class"):
        validate(inconsistent)


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("downstream", [float("inf")], "finite"),
        ("params_n", [0.0], "positive"),
        ("tokens_d", [-1.0], "positive"),
    ],
)
def test_validate_rejects_invalid_optional_numeric_values(column, values, match):
    df = good_df()
    df[column] = values
    with pytest.raises(SchemaError, match=match):
        validate(df)
