import numpy as np
import pytest

from asla.analysis.splits import assert_not_test, make_split, test_rows, train_rows
from asla.data.synthetic import negative_controls_only


def test_split_is_deterministic_and_persisted(tmp_path):
    df = negative_controls_only(np.random.default_rng(1))
    path = tmp_path / "split.json"
    s1 = make_split(df, seed=7, path=path)
    s2 = make_split(df, seed=999, path=path)
    assert s1 == s2
    assert s1["seed"] == 7
    assert path.exists()


def test_stale_persisted_split_is_rejected_for_different_table(tmp_path):
    df = negative_controls_only(np.random.default_rng(1))
    path = tmp_path / "split.json"
    make_split(df, seed=7, path=path)
    other = negative_controls_only(np.random.default_rng(2)).iloc[:-5].copy()
    with pytest.raises(ValueError, match="does not match"):
        make_split(other, seed=7, path=path)


def test_assert_not_test_raises_for_test_rows(tmp_path):
    df = negative_controls_only(np.random.default_rng(1))
    split = make_split(df, seed=7, path=tmp_path / "split.json")
    assert_not_test(train_rows(df, split))
    with pytest.raises(AssertionError):
        assert_not_test(test_rows(df, split))
