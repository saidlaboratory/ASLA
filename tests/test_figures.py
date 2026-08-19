import json
from pathlib import Path

import numpy as np
import pytest

from asla.analysis.fits import project_ranking
from asla.cli import build_parser
from asla.config import AuditConfig
from asla.data.io import save_runs
from asla.data.synthetic import negative_controls_only
from asla.figures import make_figures

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_make_figures_writes_expected_png_and_pdf_outputs(tmp_path: Path) -> None:
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(7), cfg)
    make_figures(
        df,
        tmp_path,
        target=cfg.budgets.target,
        fit_form="compute_power_law",
        budgets=cfg.budgets.fit,
        n_boot=10,
    )
    expected = {
        "scaling_curves",
        "ensemble_disagreement",
        "projected_vs_true",
        "regret_vs_exploration",
        "crossover_frequency_by_class",
    }
    for stem in expected:
        for extension in ("png", "pdf"):
            output = tmp_path / f"{stem}.{extension}"
            assert output.is_file()
            assert output.stat().st_size > 0


def test_figures_cli_writes_provenance_metadata(tmp_path: Path) -> None:
    cfg = AuditConfig()
    runs = tmp_path / "runs.parquet"
    save_runs(negative_controls_only(np.random.default_rng(8), cfg), runs)
    out = tmp_path / "figures"
    args = build_parser().parse_args(
        [
            "figures",
            "--runs",
            str(runs),
            "--out",
            str(out),
            "--target",
            str(cfg.budgets.target),
            "--budgets",
            *(str(value) for value in cfg.budgets.fit),
            "--fast",
            "--n-boot",
            "5",
        ]
    )
    assert args.func(args) == 0
    metadata = json.loads((out / "figure_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fit_form"] == "compute_power_law"
    assert metadata["target"] == cfg.budgets.target
    assert metadata["budgets"] == list(cfg.budgets.fit)
    assert metadata["n_boot"] == 5
    assert metadata["package_version"] == "0.1.0"
    assert metadata["intermediate_budget"] is None
    assert len(metadata["input_sha256"]) == 64


def test_make_figures_holds_out_intermediate_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AuditConfig()
    df = negative_controls_only(np.random.default_rng(7), cfg)
    seen: dict[str, tuple[float, ...]] = {}
    original = project_ranking

    def _capture(data, budgets, target, fit_form="compute_power_law"):
        seen["budgets"] = tuple(float(value) for value in budgets)
        return original(data, budgets, target, fit_form=fit_form)

    monkeypatch.setattr("asla.figures.project_ranking", _capture)
    make_figures(
        df,
        tmp_path,
        target=cfg.budgets.target,
        fit_form="compute_power_law",
        intermediate_budget=cfg.budgets.intermediate,
        n_boot=10,
    )
    assert seen["budgets"] == cfg.budgets.fit
