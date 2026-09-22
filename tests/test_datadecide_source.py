import json

import numpy as np
import pandas as pd
import pytest

from asla.data.schema import validate
from asla.data.sources import datadecide as dd


def _synthetic_artifacts():
    """Tiny tables mimicking the released DataDecide artifact layout."""

    scales = {
        "4M": (3.7e6, 65536.0, (1250, 2500, 5725)),
        "150M": (1.5e8, 393216.0, (1250, 37500)),
        "1B": (1.18e9, 1441792.0, (2500, 69369)),
    }
    recipes = ["C4", "Dolma1.7", "DCLM-Baseline"]
    eval_rows, ppl_rows = [], []
    rng = np.random.default_rng(0)
    for scale, (n, tps, steps) in scales.items():
        seeds = ["default", "large aux 2", "large aux 3"] if scale == "1B" else ["default", "small aux 2", "small aux 3"]
        for recipe in recipes:
            for seed in seeds:
                for step in steps:
                    tokens = step * tps
                    acc = 0.3 + 0.2 * np.log10(n) / 10 + rng.normal(0, 0.005)
                    eval_rows.append(
                        {
                            "params": scale,
                            "data": recipe,
                            "task": "olmes_10_macro_avg",
                            "step": step,
                            "seed": seed,
                            "chinchilla": "5xC",
                            "tokens": int(tokens),
                            "compute": 6.0 * n * tokens,
                            "metrics": json.dumps({"primary_metric": acc, "correct_prob_per_char": acc + 0.1}),
                        }
                    )
                    ppl_rows.append(
                        {
                            "step": float(step),
                            dd.C4_PPL_COLUMN: 30.0 / np.log10(n),
                            "data": recipe,
                            "params": scale,
                            "seed": seed,
                        }
                    )
            if scale == "1B":  # truncated auxiliary rows that must be dropped
                for seed in ("small aux 2", "small aux 3"):
                    ppl_rows.append({"step": 17339.0, dd.C4_PPL_COLUMN: 40.0, "data": recipe, "params": scale, "seed": seed})
    # a row for another task must be ignored
    eval_rows.append({**eval_rows[0], "task": "mmlu"})
    return pd.DataFrame(eval_rows), pd.DataFrame(ppl_rows)


def test_eval_loader_unpacks_only_macro_rows(tmp_path):
    eval_df, _ = _synthetic_artifacts()
    (tmp_path / "eval_macro_avg.parquet").parent.mkdir(exist_ok=True)
    eval_df.to_parquet(tmp_path / "eval_macro_avg.parquet", index=False)
    loaded = dd.load_eval_macro(tmp_path)
    assert "primary_metric" in loaded.columns and "metrics" not in loaded.columns
    assert len(loaded) == len(eval_df) - 1


def test_scale_constants_recover_n_and_tokens_per_step():
    eval_df, _ = _synthetic_artifacts()
    constants = dd.scale_constants(_unpack(eval_df))
    assert list(constants.index) == ["4M", "150M", "1B"]
    assert constants.loc["4M", "params_n"] == pytest.approx(3.7e6)
    assert constants.loc["1B", "tokens_per_step"] == pytest.approx(1441792.0)


def _unpack(eval_df):
    df = eval_df[eval_df["task"] == dd.MACRO_TASK].drop(columns=["task"]).reset_index(drop=True)
    parsed = [json.loads(m) for m in df["metrics"]]
    for field in ("primary_metric", "correct_prob_per_char"):
        df[field] = [p[field] for p in parsed]
    return df.drop(columns=["metrics"])


def test_build_runs_table_maps_every_metric_with_explicit_metric_name():
    eval_df, ppl_df = _synthetic_artifacts()
    eval_df = _unpack(eval_df)
    for metric in dd.METRICS:
        table = dd.build_runs_table(eval_df, ppl_df, metric=metric)
        validate(table)
        assert set(table["metric_name"]) == {metric}
        assert set(table["intervention_class"]) == {"data"}
        assert set(table["tuning_quality"]) == {dd.TUNING_QUALITY}
        assert table["seed"].isin([0, 1, 2]).all()
        # 3 recipes x 3 scales x 3 seeds at the common final checkpoint
        assert len(table) == 27
        assert (table[table["scale_label"] == "1B"]["step"] == 69369).all()
        assert not table["seed_label"].str.startswith("small").where(table["scale_label"] == "1B", False).any()
    ppl_table = dd.build_runs_table(eval_df, ppl_df, metric="c4_en_bits_per_token")
    expected = np.log2(30.0 / np.log10(3.7e6))
    assert ppl_table[ppl_table["scale_label"] == "4M"]["bpb"].iloc[0] == pytest.approx(expected)
    # perplexity compute is derived as 6 * N * step * tokens_per_step and matches the eval artifact
    eval_table = dd.build_runs_table(eval_df, None, metric="olmes_macro_error")
    assert set(ppl_table["compute"].round(3)) == set(eval_table["compute"].round(3))
    assert eval_table["bpb"].between(0, 1).all()


def test_build_runs_table_rejects_unknown_metric():
    eval_df, ppl_df = _synthetic_artifacts()
    with pytest.raises(ValueError, match="unknown DataDecide metric"):
        dd.build_runs_table(_unpack(eval_df), ppl_df, metric="bpb")


def test_common_checkpoint_requires_every_cell():
    eval_df, _ = _synthetic_artifacts()
    eval_df = _unpack(eval_df)
    # drop the final checkpoint of one 150M cell: the common step must fall back to 1250
    mask = (
        (eval_df["params"] == "150M")
        & (eval_df["data"] == "C4")
        & (eval_df["seed"] == "default")
        & (eval_df["step"] == 37500)
    )
    table = dd.build_runs_table(eval_df[~mask], None, metric="olmes_macro_error")
    assert (table[table["scale_label"] == "150M"]["step"] == 1250).all()


def test_coverage_summary_flags_off_trajectory_scales():
    eval_df, ppl_df = _synthetic_artifacts()
    table = dd.build_runs_table(_unpack(eval_df), ppl_df, metric="olmes_macro_error")
    summary = dd.coverage_summary(table)
    assert summary["n_recipes"] == 3 and summary["n_scales"] == 3
    text = dd.format_coverage(summary)
    assert "150M" in text and "seeds/cell" in text
