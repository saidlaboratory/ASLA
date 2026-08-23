import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hpc.run_training_job import render_site_command, validate_result_json
from hpc.train_and_eval import result_from_metrics
from scripts.check_runs_coverage import coverage_report
from scripts.collect_results import collect_results
from scripts.finalize_run_manifest import manifest_to_runs
from scripts.hpc_preflight import _write_mock_trainer, validate_manifest_array
from scripts.make_run_manifest import _read_budgets, _read_interventions, make_manifest
from scripts.prepare_runs_table import _coerce_runs
from scripts.run_one_manifest_row import build_command, select_row


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


def test_command_values_are_shell_quoted_and_unsafe_run_ids_are_rejected():
    row = pd.Series(
        {
            "run_id": "safe-run",
            "intervention": "candidate; printf unsafe",
            "compute": 1.0,
            "seed": 0,
        }
    )
    command = build_command(
        row,
        "python train.py --run-id {run_id_q} --recipe {intervention_q} "
        "--compute {compute_g_q} --seed {seed_int_q} --out {run_dir_q}",
        "results with spaces",
        row_index=1,
    )
    assert "'candidate; printf unsafe'" in command
    assert "'results with spaces/safe-run'" in command
    unsafe = row.copy()
    unsafe["run_id"] = "../outside"
    with pytest.raises(ValueError, match="unsafe run_id"):
        build_command(
            unsafe,
            "python x.py {run_id_q} {intervention_q} {compute_g_q} {seed_int_q}",
            ".",
            1,
        )


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
    (result_dir / "result.json").write_text(
        '{"bpb": 1.23, "status": "completed", "run_id": "a__c1__s0", '
        '"intervention": "a", "compute": 1.0, "seed": 0, "notes": "ok"}',
        encoding="utf-8",
    )

    updated, collected, missing = collect_results(manifest, tmp_path / "results")

    assert collected == ["a__c1__s0"]
    assert missing == ["a__c1__s1"]
    assert updated.loc[0, "status"] == "completed"
    assert updated.loc[0, "bpb"] == 1.23
    assert updated.loc[0, "notes"] == "ok"

    with pytest.raises(ValueError, match="overwrite"):
        collect_results(updated, tmp_path / "results")


def test_collect_results_rejects_result_manifest_identity_mismatch(tmp_path):
    manifest = pd.DataFrame(
        {
            "run_id": ["a__c1__s0"],
            "intervention": ["a"],
            "compute": [1.0],
            "seed": [0],
            "status": ["pending"],
            "bpb": [pd.NA],
        }
    )
    result_dir = tmp_path / "results" / "a__c1__s0"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        '{"bpb": 1.2, "status": "completed", "run_id": "wrong", '
        '"intervention": "a", "compute": 1.0, "seed": 0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        collect_results(manifest, tmp_path / "results")


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


def test_site_command_template_uses_real_train_env_contract():
    template = open("hpc/site_command.template", encoding="utf-8").read()
    command = render_site_command(
        template,
        {
            "intervention": "baseline",
            "compute": "1",
            "compute_g": "1",
            "seed": "0",
            "seed_int": 0,
            "output_dir": "results/hpc/baseline__c1__s0",
            "metrics_json": "results/hpc/baseline__c1__s0/metrics.json",
        },
    )

    assert "ASLA_REAL_TRAIN_EVAL" in command
    assert "--recipe baseline" in command
    assert "--metrics-json results/hpc/baseline__c1__s0/metrics.json" in command


def test_train_and_eval_writes_result_from_metrics(tmp_path):
    metrics = tmp_path / "metrics.json"
    result = tmp_path / "result.json"
    metrics.write_text('{"bpb": 1.456, "downstream": 0.25}', encoding="utf-8")

    payload = result_from_metrics(metrics, result)

    assert payload["bpb"] == 1.456
    assert result.exists()
    validate_result_json(result)


def test_result_json_requires_an_object_and_numeric_bpb(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        validate_result_json(result)

    result.write_text('{"bpb": {}, "status": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric bpb"):
        validate_result_json(result)


def test_metrics_json_requires_an_object(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        result_from_metrics(metrics, tmp_path / "result.json")


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
    assert payload["run_id"] == "run0"


def test_train_and_eval_refuses_stale_metrics(tmp_path):
    template = tmp_path / "site_command.template"
    template.write_text("true", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics.json").write_text('{"bpb": 9.999}', encoding="utf-8")
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
            str(output),
            "--result-json",
            str(output / "result.json"),
            "--command-template-file",
            str(template),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "existing output" in completed.stderr
    assert not (output / "result.json").exists()


def test_training_adapter_dry_run_retry_preserves_existing_result(tmp_path):
    template = tmp_path / "site_command.template"
    template.write_text("true", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    result = output / "result.json"
    original = '{"bpb": 1.25, "status": "completed"}\n'
    result.write_text(original, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "hpc/run_training_job.py",
            "--run-id",
            "run0",
            "--intervention",
            "baseline",
            "--compute",
            "1",
            "--seed",
            "0",
            "--output-dir",
            str(output),
            "--site-command-template",
            str(template),
            "--overwrite-output",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.read_text(encoding="utf-8") == original


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
    text = Path("hpc/slurm_array_template.sh").read_text(encoding="utf-8")
    initial_args = text.split("RUN_ARGS=(")[1].split(")")[0]
    assert "--execute" not in initial_args
    assert "RUN_ARGS+=(--execute)" in text
    assert "##SBATCH --gres=gpu:1" in text


def test_hpc_preflight_mock_trainer_writes_metrics(tmp_path):
    trainer = tmp_path / "mock_train_eval.py"
    metrics = tmp_path / "metrics.json"
    _write_mock_trainer(trainer)

    completed = subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--recipe",
            "baseline",
            "--compute",
            "1",
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
            "--metrics-json",
            str(metrics),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"bpb": 1.234' in metrics.read_text(encoding="utf-8")


def test_hpc_preflight_rejects_slurm_array_manifest_mismatch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "run_id,intervention,compute,seed,status\n"
        "a0,a,1,0,pending\n"
        "a1,a,1,1,pending\n",
        encoding="utf-8",
    )
    slurm = tmp_path / "job.sh"
    slurm.write_text("#!/bin/bash\n#SBATCH --array=1-3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be 1-2"):
        validate_manifest_array(manifest, slurm)


def test_first_audit_run_design_and_report_on_synthetic_table(tmp_path):
    import importlib

    import numpy as np
    import pandas as pd

    module = importlib.import_module("scripts.run_first_audit")
    rows = []
    rng = np.random.default_rng(0)
    for i, name in enumerate(("a", "b", "c")):
        for label, compute in (("1", 1.0), ("2", 2.0), ("4", 4.0), ("8", 8.0), ("16", 16.0), ("64", 64.0)):
            for seed in range(3):
                rows.append(
                    {
                        "intervention": name,
                        "intervention_class": "data",
                        "compute": compute,
                        "seed": seed,
                        "bpb": 1.0 + 0.05 * i + 0.5 * compute**-0.5 + rng.normal(0, 1e-3),
                        "metric_name": "synthetic",
                        "scale_label": label,
                    }
                )
    df = pd.DataFrame(rows)
    entry = module.run_design(
        df,
        name="synthetic",
        axis="data",
        budgets=(1.0, 2.0, 4.0, 8.0),
        target=64.0,
        intermediate=16.0,
        n_boot_pairwise=1,
        n_boot_single=1,
        n_boot_ensemble=1,
        gate_n_boot=2,
        seed=0,
        include_ensemble=True,
    )
    assert entry["n_pairs"] == 3 and entry["seed_bootstrap_meaningful"]
    assert set(entry["pairwise_decisions"]["rankers"]) == {"projection_ranker", "single_scale_ranker", "ensemble_ranker"}
    assert "gate_ranker" in entry["single_design_seed_sensitivity"]["rankers"]
    assert entry["crossovers"]["projection_vs_target"]["n_flipped_pairs"] == 0
    results = {
        "fast": True, "counts": {"pairwise": 1, "single": 1}, "seed": 0, "inputs": {"x": "0" * 64}, "designs": [entry]
    }
    text = module.render_report(results)
    assert "synthetic" in text and "flipped pairs" in text and "sha256" in text


def test_signal_and_noise_check_helpers_on_synthetic_tables():
    import importlib

    import pandas as pd

    module = importlib.import_module("scripts.run_signal_and_noise_check")
    rows_sn, rows_dd = [], []
    for i, name in enumerate(("A", "B", "C", "D")):
        for label, compute in (("1", 1.0), ("2", 2.0), ("4", 4.0), ("8", 8.0)):
            sn_value = 1.0 + 0.1 * i + 1.0 / compute
            rows_sn.append(
                {"intervention": name, "intervention_class": "data", "compute": compute, "seed": 0, "bpb": sn_value,
                 "metric_name": "bpb_sn", "scale_label": label}
            )
            for seed in range(2):
                # DataDecide table disagrees with S&N about the order of A and B at the largest scale only
                dd_value = sn_value + (0.5 if (name == "A" and compute == 8.0) else 0.0) + 0.001 * seed
                rows_dd.append(
                    {"intervention": name, "intervention_class": "data", "compute": compute, "seed": seed,
                     "bpb": dd_value, "metric_name": "bpt_dd", "scale_label": label}
                )
    sn, dd = pd.DataFrame(rows_sn), pd.DataFrame(rows_dd)
    agreement = module.metric_agreement(sn, dd)
    by_scale = {row["scale_label"]: row for row in agreement}
    assert by_scale["1"]["pairwise_agreement"] == 1.0 and by_scale["1"]["same_winner"] is True
    # the perturbed A row at the largest scale changes both the order and the winner in the DataDecide table
    assert by_scale["8"]["pairwise_agreement"] < 1.0 and by_scale["8"]["same_winner"] is False
    dup = module.duplicate_check(sn, "A", "B")
    assert dup["any_identical"] is False and set(dup["abs_diff"]) == {"1", "2", "4", "8"}
    profile = module.flip_profile(sn, target_label="8")
    assert [row["scale_label"] for row in profile] == ["1", "2", "4"] and all(row["n_flips"] == 0 for row in profile)
    check = module.projection_check(sn, target_label="8")
    assert check["fit_budget_labels"] == ["1", "2", "4"] and check["true_winner"] == "A"
    text = module.render(
        {"inputs": {"x": "0" * 64}, "metric_agreement": agreement, "duplicate_check": dup, "flip_profile": profile,
         "projection_check": check}
    )
    assert "Signal-and-Noise" in text and "sha256" in text


def test_first_audit_derived_analyses():
    import importlib

    import pandas as pd

    module = importlib.import_module("scripts.run_first_audit")
    design = {
        "n_pairs": 6,
        "crossovers": {
            "projection_vs_target": {
                "pairs": [
                    {"a": "a", "b": "b", "true_gap": 0.1, "significant": True},
                    {"a": "a", "b": "c", "true_gap": 0.2, "significant": False},
                    {"a": "b", "b": "d", "true_gap": 0.3, "significant": True},
                ]
            },
            "largest_fit_budget_vs_target": {
                "pairs": [
                    {"a": "a", "b": "b", "true_gap": 0.1, "significant": True},
                    {"a": "c", "b": "d", "true_gap": 0.4, "significant": True},
                ]
            },
        },
    }
    dec = module.decompose_projection_error(design)
    assert dec["crossover_inherited"]["n"] == 1 and dec["fit_error"]["n"] == 2 and dec["single_scale_only"]["n"] == 1
    assert dec["fit_error"]["n_significant"] == 1 and dec["excess_projection_flips"] == 1
    assert dec["fit_error_share_of_projection_flips"] == 2 / 3

    # power: a gap equal to the noise needs ~17 seeds per arm; a gap 10x the noise needs the minimum 2
    assert 15 <= module.seeds_needed(1.0, 1.0) <= 19
    assert module.seeds_needed(10.0, 1.0) == 2
    assert module.seeds_needed(0.0, 1.0) is None

    rows = []
    for name, offset in (("A", 0.0), ("B", 0.01), ("C", 0.03)):
        for label, compute in (("s", 1.0), ("l", 8.0)):
            rows.append(
                {"intervention": name, "intervention_class": "optimizer", "compute": compute, "seed": 0,
                 "bpb": 3.0 + offset * (1 if compute == 1.0 else 0.1), "scale_label": label, "chinchilla_ratio": 1}
            )
    conv = module.optimizer_convergence({"size_ladder_1xC": pd.DataFrame(rows)})
    assert [c["level"] for c in conv] == ["s", "l"]
    assert conv[0]["range_nats"] == pytest.approx(0.03) and conv[1]["range_nats"] == pytest.approx(0.003)
    assert conv[0]["smallest_adjacent_gap_nats"] == pytest.approx(0.01)

    noise_rows = []
    for name in ("A", "B"):
        for seed, value in enumerate((1.0, 1.02, 0.98)):
            noise_rows.append(
                {"intervention": name, "intervention_class": "data", "compute": 1.0, "seed": seed, "bpb": value,
                 "scale_label": "1B", "metric_name": "c4_en_bits_per_token"}
            )
    noise = module.seed_noise_reference(pd.DataFrame(noise_rows), scales=("1B",))
    assert noise[0]["pooled_within_cell_sd"] == pytest.approx(0.02) and noise[0]["seeds_per_cell"] == 3
    power = module.power_analysis(conv, noise)
    assert power["rows"][0]["seeds_to_resolve_range"] >= 2
