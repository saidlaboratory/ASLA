import pandas as pd

from asla.analysis.rankers import single_scale_ranker


def test_single_scale_ranker_uses_best_mean_at_largest_budget():
    df = pd.DataFrame(
        {
            "intervention": ["a", "a", "b", "b", "a", "b"],
            "intervention_class": ["x"] * 6,
            "compute": [1.0, 2.0, 1.0, 2.0, 2.0, 2.0],
            "seed": [0, 0, 0, 0, 1, 1],
            "bpb": [1.0, 0.9, 0.8, 0.95, 0.91, 0.96],
        }
    )
    ranking = single_scale_ranker(df, (1.0, 2.0), 4.0)
    assert ranking.index[0] == "a"

