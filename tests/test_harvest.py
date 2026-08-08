from types import SimpleNamespace

import pytest

from asla.data.harvest import _coerce_harvested_rows, _run_value
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


def test_run_value_supports_prefixed_nested_keys_and_exact_slash_keys():
    run = SimpleNamespace(
        config={"model": {"recipe": "adamw"}, "seed": 1},
        summary={"eval": {"bpb": 1.2}, "eval/c4_en_bpb": 1.1, "seed": 2},
    )
    assert _run_value(run, "config.model.recipe") == "adamw"
    assert _run_value(run, "summary.eval.bpb") == 1.2
    assert _run_value(run, "summary.eval/c4_en_bpb") == 1.1
    assert _run_value(run, "config.seed") == 1
    assert _run_value(run, "summary.seed") == 2


def test_run_value_rejects_unprefixed_ambiguity_including_nested_paths():
    run = SimpleNamespace(
        config={"model": {"name": "config-value"}},
        summary={"model": {"name": "summary-value"}},
    )
    with pytest.raises(ValueError, match="both config and summary"):
        _run_value(run, "model.name")
