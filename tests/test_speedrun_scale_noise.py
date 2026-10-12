"""Tests for the speedrun-scale noise search, the W&B replicate query and the leaderboard robust core."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

srn = importlib.import_module("scripts.run_speedrun_scale_noise")
wq = importlib.import_module("scripts.wandb_replicate_query")
core = importlib.import_module("scripts.run_leaderboard_core")


def _rec(run: str, loss: float, cfg: dict, size: str = "520m") -> dict:
    return {"optimizer": "adamw", "size": size, "budget": "1", "config": cfg, "loss": loss, "run": run}


def test_sweep_repeats_need_distinct_runs_with_identical_configuration() -> None:
    records = [
        _rec("a", 3.00, {"lr": 1e-3, "wd": 0.1}),
        _rec("a", 3.00, {"wd": 0.1, "lr": 1e-3}),  # the same run listed twice is not a repeat
        _rec("b", 3.10, {"lr": 2e-3, "wd": 0.1}),
        _rec("c", 3.20, {"lr": 2e-3, "wd": 0.1}, size="1.2b"),
    ]
    assert srn.repeats(records) == []
    records.append(_rec("d", 3.06, {"lr": 1e-3, "wd": 0.1}))
    found = srn.repeats(records)
    assert len(found) == 1 and found[0]["runs"] == ["a", "d"] and found[0]["df"] == 1
    sd = np.std([3.00, 3.06], ddof=1) / 3.03
    assert found[0]["relative_sd"] == pytest.approx(sd)
    assert srn.pooled(found)["relative_sd"] == pytest.approx(sd)
    assert srn.pooled([]) is None


def test_wandb_config_split_keeps_nested_names_and_drops_seeds_and_identity() -> None:
    cfg = {
        "name": "run-1",
        "trainer": {"seed": 3, "wandb": {"project": "p"}, "checkpointer": {"base_path": "gs://x"}},
        "optimizer": {"name": "adamw", "learning_rate": 1e-3},
        "data_seed": 7,
        "_wandb": {"value": {}},
        "model": {"hidden_dim": {"value": 768, "desc": None}},
    }
    kept, seeds = wq.split_config(cfg)
    assert kept == {"optimizer.name": "adamw", "optimizer.learning_rate": 1e-3, "model.hidden_dim": 768}
    assert seeds == {"trainer.seed": 3, "data_seed": 7}


def test_wandb_grouping_separates_seed_replicates_from_reruns() -> None:
    def run(i: str, name: str, seed: int, value: float, lr: float = 1e-3) -> dict:
        return {"project": "p/q", "id": i, "name": name, "config": {"trainer": {"seed": seed}, "lr": lr},
                "summary": {"eval/paloma/c4_en/bpb": value}}  # fmt: skip

    records = [
        run("1", "llama_520m_x", 0, 1.000),
        run("2", "llama_520m_x-rerun", 1, 1.002),
        run("3", "llama_1_2b_x", 0, 0.950),
        run("4", "llama_1_2b_x-b", 0, 0.951),
        run("5", "llama_520m_y", 0, 1.1, lr=2e-3),
        {"project": "p/q", "id": "6", "name": "llama_7b", "config": {}, "summary": {"eval/paloma/c4_en/bpb": 1.0}},
    ]
    groups = {g["size"]: g for g in wq.group(records)}
    assert set(groups) == {"520m", "1.2b"}
    assert groups["520m"]["kind"] == "seed replicate" and groups["1.2b"]["kind"] == "same-seed rerun"
    est = wq.pooled(list(groups.values()))
    assert est["groups"] == 1 and est["relative_sd"] == pytest.approx(np.std([1.0, 1.002], ddof=1) / 1.001)
    assert wq.pooled([groups["1.2b"]]) is None
    assert wq.metric_of({"x/c4_en/loss": 2.0}) == ("x/c4_en/loss", 2.0)
    assert wq.size_of("sweep-1.2b-24B-adamw") == "1.2b" and wq.size_of("llama_7b") is None


def test_invariant_core_counts_pairs_top_k_and_components() -> None:
    members = ["a", "b", "c", "d"]
    rankings = {"r1": {"a": 1, "b": 2, "c": 3, "d": 4}, "r2": {"a": 1, "b": 3, "c": 2, "d": 4}}
    c = core.invariant_core(rankings, members)
    assert c["invariant_pairs"] == 5 and c["largest_invariant_top_k"] == 1 and c["all_invariant_k"] == [1, 3, 4]
    assert [g["members"] for g in c["groups_by_median_rank"]] == [["a"], ["b", "c"], ["d"]]
    assert ["a", "d"] in c["invariant_pair_list"] and c["top1_share"] == {"a": 1.0}


def test_log_a_least_squares_recovers_a_known_curve() -> None:
    x = np.array([1e18, 4e18, 1.6e19, 6.4e19])
    a, b, c = 50.0, 0.12, 0.8
    y = a * x ** (-b) + c
    forecast, sse = core.least_squares_log_a(x, y, core.start(x, y, 0.7, 1.0))
    assert sse < 1e-12 and forecast == pytest.approx(a * 1e22 ** (-b) + c, abs=1e-6)
