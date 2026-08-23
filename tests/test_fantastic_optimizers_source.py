import json

import pandas as pd
import pytest

from asla.data.schema import validate
from asla.data.sources import fantastic_optimizers as fo


def _write(cache, optimizer, size, ratio, baseline, ablations=None, min_loss=None):
    path = cache / optimizer / size / str(ratio) / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"Baseline": baseline}
    result.update(ablations or {})
    payload = {"result": result, "name": {k: "run" for k in result}}
    if min_loss is not None:
        payload["min_loss"] = min_loss
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def cache(tmp_path):
    for opt, offset in (("adamw", 0.0), ("muon", -0.05), ("nadamw", -0.01), ("soape", -0.03)):
        for size, base in (("130m", 3.5), ("300m", 3.25), ("520m", 3.1), ("1.2b", 2.9)):
            for ratio, drop in ((1, 0.0), (2, -0.1), (4, -0.18), (8, -0.25)):
                _write(tmp_path, opt, size, ratio, base + offset + drop, {"lr=1": base + offset + drop + 0.05, "lr=2": 7.8})
        _write(tmp_path, opt, "130m", 16, 3.5 + offset - 0.3)
        _write(tmp_path, opt, "300m", 16, 3.25 + offset - 0.3)
    _write(tmp_path, "lion", "130m", 1, 3.55, {"lr=1": 3.6})
    _write(tmp_path, "lion", "300m", 1, 3.27, {"lr=1": 3.3})
    _write(tmp_path, "lion", "520m", 1, 3.11, {"lr=1": 3.15})
    _write(tmp_path, "sophia", "130m", 1, None, {"lr=1": 3.6}, min_loss=3.58)
    return tmp_path


def test_load_results_reads_tuned_and_ablation_losses(cache):
    results = fo.load_results(cache)
    row = results[(results.optimizer == "adamw") & (results["size"] == "130m") & (results.chinchilla == 1)].iloc[0]
    assert row["loss_baseline"] == pytest.approx(3.5)
    assert row["loss_min"] == pytest.approx(3.5)  # min over baseline + ablations
    assert row["n_ablations"] == 2
    assert row["loss_ablation_median"] == pytest.approx((3.55 + 7.8) / 2)
    sophia = results[results.optimizer == "sophia"].iloc[0]
    assert pd.isna(sophia["loss_baseline"]) and sophia["loss_min"] == pytest.approx(3.58)


def test_size_ladder_uses_exact_non_embedding_params_and_6nd(cache):
    table = fo.size_ladder(fo.load_results(cache), 1)
    validate(table)
    assert sorted(table["intervention"].unique()) == ["AdamW", "Muon", "NAdamW", "SOAP"]
    assert set(table["metric_name"]) == {fo.METRIC_NAME}
    assert set(table["tuning_quality"]) == {fo.TUNED}
    row = table[(table.intervention == "AdamW") & (table.scale_label == "1.2b")].iloc[0]
    assert row["params_n"] == 1_207_959_552
    assert row["tokens_d"] == pytest.approx(20 * 1_207_959_552)
    assert row["compute"] == pytest.approx(6 * 1_207_959_552 * 20 * 1_207_959_552)
    assert (table["seed"] == 0).all()


def test_data_ladder_and_partial_ladder_and_all_cells(cache):
    results = fo.load_results(cache)
    data = fo.data_ladder(results, "130m")
    assert sorted(data["chinchilla_ratio"].unique()) == [1, 2, 4, 8, 16]
    partial = fo.partial_size_ladder(results, 1, fo.ABLATION_MEDIAN)
    assert "Lion" in set(partial["intervention"]) and "Sophia" not in set(partial["intervention"])
    assert set(partial["tuning_quality"]) == {fo.ABLATION_MEDIAN}
    # the raw grid is not a runs table: 130m@16xC and 520m@1xC share identical 6ND FLOPs
    rows = fo._rows(results, "loss_min", fo.TUNED)
    a = rows[(rows.scale_label == "130m") & (rows.chinchilla_ratio == 16)]["compute"].iloc[0]
    b = rows[(rows.scale_label == "520m") & (rows.chinchilla_ratio == 1)]["compute"].iloc[0]
    assert a == pytest.approx(b)


def test_harvest_writes_every_table_and_prints_coverage(cache, tmp_path, capsys):
    tables = fo.harvest(tmp_path / "out", cache_dir=cache)
    out = capsys.readouterr().out
    assert "no seed replicates" in out
    assert (tmp_path / "out" / "fantastic_optimizers_size_ladder_1xC.parquet").exists()
    assert set(tables) >= {"fantastic_optimizers_size_ladder_8xC", "fantastic_optimizers_data_ladder_300m"}
    assert (tmp_path / "out" / "fantastic_optimizers_cells.csv").exists()
