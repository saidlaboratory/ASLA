import json

import numpy as np
import pytest

import pandas as pd

from asla.cli import _assert_fit_form_available, _crossover_root_message, build_parser
from asla.data.io import save_runs
from asla.data.synthetic import negative_controls_only


def test_demo_trials_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--trials", "0"])


def test_crossover_root_message_does_not_fake_chinchilla_budget():
    msg = _crossover_root_message(None, "a", "b", (1.0, 2.0, 4.0), "chinchilla")
    assert "not reported" in msg


def test_chinchilla_fit_form_errors_when_columns_absent():
    df = pd.DataFrame({"intervention": ["a"]})
    with pytest.raises(SystemExit, match="requires columns"):
        _assert_fit_form_available(df, "chinchilla")


def test_audit_output_includes_reproducibility_metadata(tmp_path, capsys):
    df = negative_controls_only(np.random.default_rng(1), None)
    runs = tmp_path / "runs.parquet"
    save_runs(df, runs)
    out = tmp_path / "audit.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "audit",
            "--runs",
            str(runs),
            "--target",
            "64",
            "--fast",
            "--n-boot",
            "3",
            "--seed",
            "99",
            "--out",
            str(out),
        ]
    )

    assert args.func(args) == 0
    captured = capsys.readouterr()
    assert "audit_metadata" in captured.out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["audit_metadata"]["target"] == 64.0
    assert payload["audit_metadata"]["budgets"] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert payload["audit_metadata"]["n_boot"] == 3
    assert payload["audit_metadata"]["rng_seed"] == 99
    assert payload["audit_metadata"]["rankers"] == ["ensemble_ranker", "projection_ranker", "single_scale_ranker"]
