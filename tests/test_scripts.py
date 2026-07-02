import subprocess
import sys

import pandas as pd
import pytest

from hpc.train_and_eval import result_from_metrics
from scripts.check_runs_coverage import coverage_report
from scripts.collect_results import collect_results
from scripts.finalize_run_manifest import manifest_to_runs
from scripts.make_run_manifest import _read_budgets, _read_interventions, make_manifest
from scripts.prepare_runs_table import _coerce_runs
from scripts.run_one_manifest_row import build_command, select_row
from hpc.run_training_job import render_site_command, validate_result_json


def test_make_run_manifest_creates_expected_grid():
    interventions = pd.DataFrame(
        {
            "intervention": ["baseline", "candidate"],
            "intervention_class": ["recipe", "recipe"],
        }
    )
    budgets = pd.DataFrame(
        {
            "compute": [1.0, 2.0, 4.0, 64.0],
            "role": ["fit", "fit", "fit", "target"],
        }
    )

    manifest = make_manifest(interventions, budgets, seeds=3)

    assert len(manifest) == 24
    assert set(manifest["status"]) == {"pending"}
    assert manifest["run_id"].is_unique
    assert set(manifest["role"]) == {"fit", "target"}


def test_coverage_report_flags_underpowered_table():
    rows = []
    for intervention in ["a", "b"]:
        for compute in [1.0, 2.0, 4.0, 64.0]:
            rows.append(
                {
                    "intervention": intervention,
                    "intervention_class": "recipe",
                    "compute": compute,
                    "seed": 0,
                    "bpb": 1.0,
                }
            )
    df = pd.DataFrame(rows)

    problems, counts = coverage_report(df, target=64.0, min_seeds=3)

    assert len(counts) == 8
    assert any("need at least 3 interventions" in problem for problem in problems)
    assert any("under-seeded cell" in problem for problem in problems)


def test_manifest_to_runs_rejects_pending_rows_without_bpb():
    manifest = pd.DataFrame(
        {
            "run_id": ["a__c1__s0"],
            "intervention": ["a"],
            "intervention_class": ["recipe"],
            "compute": [1.0],
            "seed": [0],
            "status": ["pending"],
            "bpb": [""],
        }
    )

    try:
        manifest_to_runs(manifest)
    except ValueError as exc:
        assert "not complete" in str(exc)
    else:
        raise AssertionError("pending row was accepted")


def test_manifest_to_runs_exports_canonical_columns():
    manifest = pd.DataFrame(
        {
            "run_id": ["a__c1__s0"],
            "intervention": ["a"],
            "intervention_class": ["recipe"],
            "compute": [1.0],
            "role": ["fit"],
            "seed": [0],
            "status": ["completed"],
            "bpb": [1.2],
            "notes": ["ok"],
        }
    )

    runs = manifest_to_runs(manifest)

    assert runs.columns.tolist() == [
        "intervention",
        "intervention_class",
        "compute",
        "seed",
        "bpb",
        "downstream",
        "params_n",
        "tokens_d",
    ]
    assert runs.loc[0, "bpb"] == 1.2


def test_manifest_to_runs_requires_status_column():
    manifest = pd.DataFrame(
        {
            "intervention": ["a"],
            "intervention_class": ["recipe"],
            "compute": [1.0],
            "seed": [0],
            "bpb": [1.2],
        }
    )

    with pytest.raises(ValueError, match="status"):
        manifest_to_runs(manifest)


def test_prepare_runs_table_drops_noncanonical_columns():
    df = pd.DataFrame(
        {
            "run_id": ["a__c1__s0"],
            "intervention": ["a"],
            "intervention_class": ["recipe"],
            "compute": ["1.0"],
            "seed": ["0"],
            "status": ["completed"],
            "bpb": ["1.2"],
        }
    )

    runs = _coerce_runs(df)

    assert runs.columns.tolist() == [
        "intervention",
        "intervention_class",
        "compute",
        "seed",
        "bpb",
        "downstream",
        "params_n",
        "tokens_d",
    ]
    assert "status" not in runs.columns
    assert "run_id" not in runs.columns


def test_manifest_plan_rejects_duplicate_interventions(tmp_path):
    path = tmp_path / "interventions.csv"
    path.write_text(
        "intervention,intervention_class\n"
        "baseline,recipe\n"
        "baseline,recipe\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unique"):
        _read_interventions(str(path))


def test_manifest_plan_rejects_fit_budget_at_or_above_target(tmp_path):
    path = tmp_path / "budgets.csv"
    path.write_text(
        "compute,role\n"
        "1,fit\n"
        "2,fit\n"
        "4,fit\n"
        "4,target\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unique"):
        _read_budgets(str(path))

    path.write_text(
        "compute,role\n"
        "1,fit\n"
        "2,fit\n"
        "4,fit\n"
        "3,target\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="below"):
        _read_budgets(str(path))


def test_run_one_manifest_row_renders_command():
    manifest = pd.DataFrame(
        {
            "run_id": ["baseline__c1__s0"],
            "intervention": ["baseline"],
            "compute": [1.0],
            "seed": [0],
            "role": ["fit"],
        }
    )
    row = select_row(manifest, row=1, index_base=1)

    command = build_command(
        row,
        "python train.py --recipe {intervention} --compute {compute_g} --seed {seed_int} --out {output_dir}/{run_id}",
        "results/hpc",
        row_index=1,
    )

    assert command == "python train.py --recipe baseline --compute 1 --seed 0 --out results/hpc/baseline__c1__s0"


def test_run_one_manifest_row_requires_core_placeholders():
    row = pd.Series({"run_id": "a", "intervention": "a", "compute": 1.0, "seed": 0})

    with pytest.raises(ValueError, match="placeholders"):
        build_command(row, "python train.py --recipe {intervention}", "results/hpc", row_index=1)


def test_collect_results_merges_json_and_refuses_overwrite(tmp_path):
    manifest = pd.DataFrame(
        {
            "run_id": ["a__c1__s0", "a__c1__s1"],
            "intervention": ["a", "a"],
            "intervention_class": ["recipe", "recipe"],
            "compute": [1.0, 1.0],
            "seed": [0, 1],
            "status": ["pending", "pending"],
            "bpb": [pd.NA, pd.NA],
        }
    )
    result_dir = tmp_path / "results" / "a__c1__s0"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text('{"bpb": 1.23, "status": "completed", "notes": "ok"}', encoding="utf-8")

    updated, collected, missing = collect_results(manifest, tmp_path / "results")

    assert collected == ["a__c1__s0"]
    assert missing == ["a__c1__s1"]
    assert updated.loc[0, "status"] == "completed"
    assert updated.loc[0, "bpb"] == 1.23
    assert updated.loc[0, "notes"] == "ok"

    with pytest.raises(ValueError, match="overwrite"):
        collect_results(updated, tmp_path / "results")


def test_hpc_training_adapter_renders_and_validates_result(tmp_path):
    command = render_site_command(
        "python train.py --recipe {intervention} --result {result_path}",
        {"intervention": "baseline", "result_path": "results/run/result.json"},
    )
    assert command == "python train.py --recipe baseline --result results/run/result.json"

    result_path = tmp_path / "result.json"
    result_path.write_text('{"bpb": 1.234, "status": "completed"}', encoding="utf-8")
    payload = validate_result_json(result_path)
    assert payload["bpb"] == 1.234


def test_site_train_template_uses_entrypoint_contract():
    template = open("hpc/site_train_command.template", encoding="utf-8").read()
    command = render_site_command(
        template,
        {
            "run_id": "baseline__c1__s0",
            "intervention": "baseline",
            "compute": "1",
            "compute_g": "1",
            "seed": "0",
            "seed_int": 0,
            "output_dir": "results/hpc/baseline__c1__s0",
            "result_path": "results/hpc/baseline__c1__s0/result.json",
        },
    )

    assert "hpc/train_and_eval.py" in command
    assert "ASLA_SITE_COMMAND_TEMPLATE" in command
    assert "--run-id baseline__c1__s0" in command
    assert "--intervention baseline" in command
    assert "--result-json results/hpc/baseline__c1__s0/result.json" in command


def test_train_and_eval_writes_result_from_metrics(tmp_path):
    metrics = tmp_path / "metrics.json"
    result = tmp_path / "result.json"
    metrics.write_text('{"bpb": 1.456, "downstream": 0.25}', encoding="utf-8")

    payload = result_from_metrics(metrics, result)

    assert payload["bpb"] == 1.456
    assert result.exists()
    validate_result_json(result)


def test_train_and_eval_cli_runs_site_command(tmp_path):
    template = tmp_path / "site_command.template"
    template.write_text(
        f"{sys.executable} -c \"import pathlib; pathlib.Path('{{metrics_json}}').write_text('{{{{\\\"bpb\\\": 1.789}}}}')\"",
        encoding="utf-8",
    )
    result = tmp_path / "out" / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            "hpc/train_and_eval.py",
            "--run-id",
            "run0",
            "--intervention",
            "baseline",
            "--compute",
            "1",
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
            "--result-json",
            str(result),
            "--command-template-file",
            str(template),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = validate_result_json(result)
    assert payload["bpb"] == 1.789


def test_hpc_training_adapter_rejects_bad_result(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="did not write"):
        validate_result_json(missing)

    bad = tmp_path / "bad.json"
    bad.write_text('{"bpb": "nan", "status": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_result_json(bad)


def test_slurm_template_is_syntax_checked_and_dry_by_default():
    completed = subprocess.run(["bash", "-n", "hpc/slurm_array_template.sh"], check=False)
    assert completed.returncode == 0
    text = open("hpc/slurm_array_template.sh", encoding="utf-8").read()
    initial_args = text.split("RUN_ARGS=(")[1].split(")")[0]
    assert "--execute" not in initial_args
    assert "RUN_ARGS+=(--execute)" in text
    assert "##SBATCH --gres=gpu:1" in text
