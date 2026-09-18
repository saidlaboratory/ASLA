"""Tests for the ten-seed noise validation.

Two pin defects found while building it: duplicate evaluation passes being
counted as extra seeds, and a floor cell being treated as a low-noise one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("run_noise_validation", REPO / "scripts" / "run_noise_validation.py")
assert _SPEC is not None and _SPEC.loader is not None
noise = importlib.util.module_from_spec(_SPEC)
sys.modules["run_noise_validation"] = noise
_SPEC.loader.exec_module(noise)


def _record(run: str, step: int, value: float, source: str) -> dict[str, object]:
    return {
        "run": run,
        "size": run.split("-")[1],
        "seed": int(run.rsplit("seed", 1)[1]),
        "step": step,
        "value": value,
        "source_file": source,
    }


def test_deduplicate_keeps_one_pass_per_run_step() -> None:
    """Re-evaluations must not be counted as extra seeds.

    In the real data 19 of 65 (run, step) pairs have more than one result file,
    and they disagree by up to 0.043 in accuracy. Counting both inflates the
    seed count and injects evaluation-rerun variance into a training-seed
    variance estimate.
    """

    records = [
        _record("pythia-410m-seed4", 50000, 0.4473, "a.json"),
        _record("pythia-410m-seed4", 50000, 0.4399, "b.json"),
        _record("pythia-410m-seed5", 50000, 0.4500, "c.json"),
    ]
    kept, report = noise.deduplicate(records)
    assert len(kept) == 2
    assert report["keys_with_duplicates"] == 1
    assert report["max_rerun_spread"] == pytest.approx(0.0074, abs=1e-4)
    # The earliest pass by file name is the one retained.
    assert [r["value"] for r in kept if r["seed"] == 4] == [0.4473]


def test_deduplicate_reports_zero_spread_when_clean() -> None:
    records = [_record("pythia-14m-seed0", 1000, 0.1, "a.json")]
    _, report = noise.deduplicate(records)
    assert report["keys_with_duplicates"] == 0
    assert report["max_rerun_spread"] == 0.0


def test_floor_cell_is_flagged_and_excluded_from_stage_ratios() -> None:
    """A cell at the metric's floor is not a low-noise cell.

    Including 14M at step 1000, whose mean accuracy is 1.9e-05, produced a
    stage ratio of 207x that measured the floor rather than a change in seed
    variance.
    """

    rng = np.random.default_rng(0)
    records = []
    for seed in range(10):
        records.append(_record(f"pythia-14m-seed{seed}", 1000, float(abs(rng.normal(0, 1e-5))), f"{seed}a"))
        records.append(_record(f"pythia-14m-seed{seed}", 50000, 0.11 + float(rng.normal(0, 0.01)), f"{seed}b"))
    report = noise.analyse(records, seed=1, min_seeds=10)
    floor = [c for c in report["per_cell"] if c["step"] == 1000][0]
    assert floor["degenerate_floor_cell"] is True
    assert report["N4_stage_pooling"]["by_size"] == []
    assert report["N4_stage_pooling"]["excluded_degenerate_cells"][0]["step"] == 1000


def test_three_seed_subsample_is_dispersed_and_biased_low() -> None:
    """The N1/N2 machinery must recover the known behaviour of a variance estimate."""

    rng = np.random.default_rng(7)
    records = [_record(f"pythia-160m-seed{s}", 50000, 0.40 + float(rng.normal(0, 0.02)), f"{s}") for s in range(10)]
    report = noise.analyse(records, seed=3, min_seeds=10)
    cell = report["per_cell"][0]
    assert cell["dispersion_p90_over_p10"] > 2.0
    assert cell["median_three_over_ten"] < 1.0


def test_n3_is_underpowered_with_fewer_than_three_sizes() -> None:
    """Two sizes cannot establish a scaling trend; the verdict must say so."""

    rng = np.random.default_rng(11)
    records = []
    for size in ("14m", "410m"):
        for s in range(10):
            records.append(_record(f"pythia-{size}-seed{s}", 50000, 0.3 + float(rng.normal(0, 0.01)), f"{size}{s}"))
    report = noise.analyse(records, seed=5, min_seeds=10)
    assert report["N3_size_scaling"]["n_sizes_available"] == 2
    assert report["N3_size_scaling"]["verdict"] == "UNDERPOWERED"
    assert report["N3_size_scaling"]["spearman_size_vs_sigma"] is None


def test_shipped_results_are_deduplicated_to_ten_seeds() -> None:
    path = REPO / "results" / "noise" / "noise_validation.json"
    if not path.exists():  # pragma: no cover - results not always materialised
        pytest.skip("noise results not present")
    payload = json.loads(path.read_text())
    for cell in payload["analysis"]["per_cell"]:
        assert cell["n_seeds"] == 10, f"{cell['size']} step{cell['step']} has {cell['n_seeds']} seeds"
    assert payload["source"]["weights_downloaded"] is False
