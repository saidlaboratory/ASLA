import numpy as np

from asla.data.schema import validate
from asla.data.synthetic import SCENARIOS, true_ranking


def test_scenarios_validate_and_have_truth():
    for name, fn in SCENARIOS.items():
        df = fn(np.random.default_rng(123), None)
        validate(df)
        ranking = true_ranking(df)
        assert len(ranking) >= 2
        assert ranking.is_monotonic_increasing
        assert df.attrs["scenario_truth"].name == name

