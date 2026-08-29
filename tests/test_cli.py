import json

import numpy as np
import pandas as pd
import pytest

from asla.cli import _assert_fit_form_available, _crossover_root_message, _resolve_audit_budgets, build_parser
from asla.data.io import save_runs
from asla.data.synthetic import negative_controls_only


def test_demo_trials_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--estimand", "single_design_seed_sensitivity", "--trials", "0"])


def test_paper_facing_commands_require_an_explicit_estimand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--fast"])
    with pytest.raises(SystemExit):
        parser.parse_args(["audit", "--runs", "runs.parquet", "--target", "64"])


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
            "--estimand",
            "single_design_seed_sensitivity",
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
    assert payload["audit_metadata"]["package_version"] == "0.1.0"
    assert len(payload["audit_metadata"]["input_sha256"]) == 64
    assert payload["audit_metadata"]["budget_roles"]["target"] == 64.0
    assert payload["estimand"]["name"] == "single_design_seed_sensitivity"
    assert payload["bootstrap_diagnostics"]
    assert payload["ranker_availability"]["gate_ranker"]["included"] is False


def test_intermediate_budget_is_excluded_from_projection_fit():
    df = negative_controls_only(np.random.default_rng(1), None)
    assert _resolve_audit_budgets(df, 64.0, None, 16.0) == (1.0, 2.0, 4.0, 8.0)


def test_explicit_audit_budget_must_exist():
    df = negative_controls_only(np.random.default_rng(1), None)
    with pytest.raises(SystemExit, match="absent"):
        _resolve_audit_budgets(df, 64.0, [1.0, 2.0, 3.0], None)


def test_audit_includes_gate_only_with_explicit_intermediate_budget(tmp_path, capsys):
    df = negative_controls_only(np.random.default_rng(2), None)
    runs = tmp_path / "runs.parquet"
    save_runs(df, runs)
    parser = build_parser()
    args = parser.parse_args(
        [
            "audit",
            "--runs",
            str(runs),
            "--target",
            "64",
            "--intermediate-budget",
            "16",
            "--estimand",
            "single_design_seed_sensitivity",
            "--n-boot",
            "2",
            "--gate-n-boot",
            "2",
        ]
    )
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_metadata"]["budgets"] == [1.0, 2.0, 4.0, 8.0]
    assert payload["ranker_availability"]["gate_ranker"]["included"] is True
    assert "gate_ranker" in payload["rankers"]


def test_report_renders_markdown_from_audit_json(tmp_path):
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
            "--estimand",
            "single_design_seed_sensitivity",
            "--fast",
            "--n-boot",
            "3",
            "--out",
            str(out),
            "--report",
        ]
    )
    assert args.func(args) == 0
    report_path = out.with_suffix(".md")
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# ASLA audit report" in text
    assert "Measured target ranking" in text
    assert "Extrapolation reliability" in text

    # Standalone report command over the same JSON.
    report2 = tmp_path / "standalone.md"
    args = parser.parse_args(["report", "--audit", str(out), "--out", str(report2)])
    assert args.func(args) == 0
    assert report2.read_text(encoding="utf-8") == text


def test_report_orders_truth_by_mean(tmp_path):
    import json as _json

    from asla.report import render_report

    payload = {
        "truth": {
            "alpha_worst": {"mean": 1.2, "se": 0.01, "n_seeds": 2},
            "zeta_best": {"mean": 0.9, "se": 0.01, "n_seeds": 2},
        },
        "truth_ties": [],
    }
    text = render_report(_json.loads(_json.dumps(payload, sort_keys=True)))
    assert text.index("zeta_best") < text.index("alpha_worst")


def test_resolution_cli_reports_unresolvable_orderings(tmp_path, capsys):
    entries = tmp_path / "entries.json"
    entries.write_text(
        json.dumps({"Muon": 2.748363, "NAdamW": 2.748523, "SOAP": 2.748529, "AdamW": 2.752277}),
        encoding="utf-8",
    )
    out = tmp_path / "resolution.json"
    parser = build_parser()
    args = parser.parse_args(
        ["resolution", "--entries", str(entries), "--sigma", "0.0014312", "--out", str(out)]
    )
    assert args.func(args) == 0
    captured = capsys.readouterr().out
    assert "not resolvable" in captured
    assert "unidentifiable at ANY feasible seed count" in captured
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_entries"] == 4
    assert payload["n_unidentifiable_at_any_budget"] == 1
    assert payload["top_k_fully_resolved"] is False
    assert "sensitivity" in payload and payload["sigma_provenance"]


def test_resolution_cli_rejects_a_non_mapping_entries_file(tmp_path):
    entries = tmp_path / "bad.json"
    entries.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(["resolution", "--entries", str(entries), "--sigma", "0.01"])
    with pytest.raises(SystemExit, match="JSON object"):
        args.func(args)
