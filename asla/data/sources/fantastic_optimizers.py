"""Harvest the released Fantastic Pretraining Optimizers results into the canonical schema.

Artifact pulled (GitHub, small JSON files only; no checkpoints, no W&B API):
``experiments/optimizer_sweep/Analysis/Results/{optimizer}/{model_size}/{chinchilla}/result.json``
on the ``kaiyue/optimizers`` branch of ``WhenWen/marin`` (the authors' fork of
marin-community/marin), pinned to commit :data:`RESULTS_COMMIT`. The paper
(Wen, Hall, Ma, Liang 2025, arXiv:2509.02046) links this branch and the W&B
project ``marin-community/optimizer-scaling`` as its released artifacts. 150
files, ~1.2 MB total, were present at harvest time (2026-08-23).

What each file holds (verified against the authors' analysis scripts
``speedup_estimation.py`` / ``loss_plotting.py`` and ``utils_simp.py``):

* ``result``: mapping from ``"Baseline"`` (the coordinate-descent optimum found
  in the paper's Phase I/II tuning) and ``"<hyperparameter>=<value>"``
  (one-hyperparameter ablations around that optimum) to the final
  ``eval/paloma/c4_en/loss`` of that run - the C4-EN validation cross-entropy
  in nats per token. Values may be ``null`` when the run was missing.
* ``min_loss``: the minimum loss over all runs in the sweep for that cell. The
  authors' own figures use this value ("the optimal loss achieved at the
  corresponding Chinchilla ratio for each optimizer").
* ``best_config`` / ``approximate_best_config_list``: the tuned hyperparameters.

There is exactly one run per (optimizer, model size, Chinchilla ratio, config):
the artifact contains **no seed replicates**. Every harvested row therefore has
``seed = 0``; seed-bootstrap intervals are degenerate and crossover tests are
reported as untestable for this source. This limitation is structural, not a
harvesting choice.

Compute derivation. Model sizes are Llama-style 130M/300M/520M/1.2B with
non-embedding parameter counts hard-coded in the authors' sweep utilities
(``expected_params``): 134,217,728 / 301,989,888 / 536,870,912 /
1,207,959,552. The Chinchilla-optimal token budget is ``20 * N`` and the ratio
``r`` multiplies it, so ``tokens_d = r * 20 * N`` and ``compute = 6 * N *
tokens_d``. ``params_n = N``.

Tables produced:

* ``size ladder`` per Chinchilla ratio r in {1, 2, 4, 8}: interventions with
  results at every size (AdamW, Muon, NAdamW, SOAP), budgets 130M/300M/520M,
  target 1.2B. This is the trajectory in N at fixed tokens/param.
* ``data ladder`` per model size in {130M, 300M}: the same four optimizers,
  budgets r = 1/2/4/8, target r = 16. This is the trajectory in D at fixed N.
* ``cells.csv`` (descriptive): the raw (optimizer, size, ratio) grid with tuned
  and ablation losses. It is deliberately **not** a runs table: 130M at 16xC and
  520M at 1xC (and 300M at 16xC and 1.2B at 1xC) have identical 6ND FLOPs, so
  the grid violates the one-run-per-(intervention, compute, seed) identity.

Tuning quality. The Baseline is the paper's per-optimizer, per-scale
coordinate-descent optimum, recorded as ``tuning_quality =
"coordinate_descent_tuned"`` (value = ``min_loss``, i.e. best observed run in
the sweep for that cell; ``loss_baseline`` keeps the coordinate-descent
Baseline itself). For the tuning-confound control we also build a
``"single_hp_ablation_median"`` table from the non-null one-hyperparameter
ablations in ``result`` (median loss over ablations). It represents an
optimizer run with one hyperparameter set off its optimum - a concrete,
artifact-backed "under-tuned" condition. A "worst ablation" condition was
considered and dropped: the maximum is dominated by diverged runs (losses of
~7.8 nats or larger), which is a divergence condition, not an under-tuning
one. Ablations are only released where the sweep was run (all optimizers at
ratio 1 for 130M/300M/520M; all ratios at 130M; none at 1.2B except AdamW 1x).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import validate

RESULTS_REPO = "WhenWen/marin"
RESULTS_BRANCH = "kaiyue/optimizers"
RESULTS_COMMIT = "97cfef3590679c41908f5b9432d853b0228aac8c"
RESULTS_PREFIX = "experiments/optimizer_sweep/Analysis/Results"
RAW_BASE = f"https://raw.githubusercontent.com/{RESULTS_REPO}/{RESULTS_COMMIT}"
TREE_API = f"https://api.github.com/repos/{RESULTS_REPO}/git/trees/{RESULTS_COMMIT}?recursive=1"
DEFAULT_CACHE_DIR = Path("data/raw/fantastic_optimizers")
METRIC_NAME = "c4_en_val_loss_nats_per_token"
INTERVENTION_CLASS = "optimizer"

# Non-embedding parameter counts from marin/optimizer_sweep/utils_simp.py (expected_params).
PARAMS_N: dict[str, int] = {
    "130m": 134_217_728,
    "300m": 301_989_888,
    "520m": 536_870_912,
    "1.2b": 1_207_959_552,
}
SIZE_ORDER = ("130m", "300m", "520m", "1.2b")
CHINCHILLA_TOKENS_PER_PARAM = 20.0
OPTIMIZER_NAMES: dict[str, str] = {
    "adamw": "AdamW",
    "nadamw": "NAdamW",
    "mars": "Mars",
    "cautious": "Cautious",
    "lion": "Lion",
    "mini": "Adam-mini",
    "muon": "Muon",
    "scion": "Scion",
    "kron": "Kron",
    "soape": "SOAP",
    "sophia": "Sophia",
}
TUNED = "coordinate_descent_tuned"
ABLATION_MEDIAN = "single_hp_ablation_median"


def _result_paths_from_tree() -> list[str]:
    import urllib.request

    with urllib.request.urlopen(TREE_API, timeout=60) as response:  # noqa: S310 - fixed https URL
        payload = json.load(response)
    return sorted(
        entry["path"]
        for entry in payload.get("tree", [])
        if entry.get("type") == "blob" and entry["path"].startswith(RESULTS_PREFIX) and entry["path"].endswith("result.json")
    )


def ensure_results(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[Path]:
    """Download every ``result.json`` (a few KB each) from the pinned commit if absent."""

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache.glob("*/*/*/result.json"))
    if existing:
        return existing
    import urllib.request

    paths = _result_paths_from_tree()
    if not paths:
        raise RuntimeError("no result.json files found in the pinned Fantastic Optimizers results tree")
    out: list[Path] = []
    for remote in paths:
        relative = remote[len(RESULTS_PREFIX) + 1 :]
        local = cache / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{RAW_BASE}/{remote}", timeout=60) as response:  # noqa: S310
            local.write_bytes(response.read())
        out.append(local)
    print(f"downloaded {len(out)} result.json files ({sum(p.stat().st_size for p in out) / 1e6:.2f} MB)")
    return out


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_results(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Return one row per (optimizer, size, chinchilla) with tuned and ablation losses."""

    records: list[dict[str, Any]] = []
    for path in sorted(Path(cache_dir).glob("*/*/*/result.json")):
        optimizer, size, ratio = path.parts[-4], path.parts[-3], path.parts[-2]
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result") or {}
        baseline = _finite(result.get("Baseline"))
        ablations = [
            value
            for key, raw in result.items()
            if key != "Baseline" and (value := _finite(raw)) is not None
        ]
        min_loss = _finite(payload.get("min_loss"))
        observed = [v for v in ablations + ([baseline] if baseline is not None else []) if v is not None]
        if min_loss is None and observed:
            min_loss = float(min(observed))
        records.append(
            {
                "optimizer": optimizer,
                "size": size,
                "chinchilla": int(ratio),
                "loss_min": min_loss,
                "loss_baseline": baseline,
                "n_ablations": len(ablations),
                "loss_ablation_median": float(np.median(ablations)) if ablations else None,
                "loss_ablation_worst": float(max(ablations)) if ablations else None,
            }
        )
    if not records:
        raise ValueError(f"no result.json files under {cache_dir}")
    return pd.DataFrame.from_records(records).sort_values(["optimizer", "size", "chinchilla"]).reset_index(drop=True)


def _rows(results: pd.DataFrame, value_column: str, tuning_quality: str) -> pd.DataFrame:
    rows = results.dropna(subset=[value_column]).copy()
    params_n = rows["size"].map(PARAMS_N).astype(float)
    tokens_d = rows["chinchilla"].astype(float) * CHINCHILLA_TOKENS_PER_PARAM * params_n
    return pd.DataFrame(
        {
            "intervention": rows["optimizer"].map(lambda key: OPTIMIZER_NAMES.get(key, key)),
            "intervention_class": INTERVENTION_CLASS,
            "compute": 6.0 * params_n * tokens_d,
            "seed": 0,
            "bpb": rows[value_column].astype(float),
            "metric_name": METRIC_NAME,
            "params_n": params_n,
            "tokens_d": tokens_d,
            "tokens_per_param": rows["chinchilla"].astype(float) * CHINCHILLA_TOKENS_PER_PARAM,
            "scale_label": rows["size"].astype(str),
            "chinchilla_ratio": rows["chinchilla"].astype(int),
            "loss_baseline": rows["loss_baseline"].astype(float),
            "n_ablations": rows["n_ablations"].astype(int),
            "tuning_quality": tuning_quality,
        }
    ).reset_index(drop=True)


_VALUE_COLUMNS = {TUNED: "loss_min", ABLATION_MEDIAN: "loss_ablation_median"}


def _complete_ladder(rows: pd.DataFrame, levels: Iterable[Any], column: str) -> pd.DataFrame:
    wanted = list(levels)
    rows = rows[rows[column].isin(wanted)]
    counts = rows.groupby("intervention")[column].nunique()
    complete = sorted(counts[counts == len(wanted)].index.astype(str))
    return rows[rows["intervention"].isin(complete)].reset_index(drop=True)


def size_ladder(results: pd.DataFrame, chinchilla: int, tuning_quality: str = TUNED) -> pd.DataFrame:
    """Budgets 130M/300M/520M and target 1.2B at one Chinchilla ratio (interventions with all four)."""

    rows = _rows(results, _VALUE_COLUMNS[tuning_quality], tuning_quality)
    rows = rows[rows["chinchilla_ratio"] == int(chinchilla)]
    table = _complete_ladder(rows, SIZE_ORDER, "scale_label")
    if table.empty:
        raise ValueError(f"no optimizer has all four sizes at Chinchilla ratio {chinchilla} for {tuning_quality}")
    table = table.sort_values(["compute", "intervention"], kind="mergesort").reset_index(drop=True)
    validate(table)
    return table


def data_ladder(
    results: pd.DataFrame, size: str, tuning_quality: str = TUNED, ratios: Iterable[int] = (1, 2, 4, 8, 16)
) -> pd.DataFrame:
    """Budgets at Chinchilla ratios 1/2/4/8 and target 16 at one model size."""

    rows = _rows(results, _VALUE_COLUMNS[tuning_quality], tuning_quality)
    rows = rows[rows["scale_label"] == size]
    table = _complete_ladder(rows, list(ratios), "chinchilla_ratio")
    if table.empty:
        raise ValueError(f"no optimizer has Chinchilla ratios {list(ratios)} at size {size} for {tuning_quality}")
    table = table.sort_values(["compute", "intervention"], kind="mergesort").reset_index(drop=True)
    validate(table)
    return table


def partial_size_ladder(
    results: pd.DataFrame, chinchilla: int, tuning_quality: str, sizes: Iterable[str] = ("130m", "300m", "520m")
) -> pd.DataFrame:
    """All optimizers present at every listed size for one ratio and tuning condition (no 1.2B target)."""

    rows = _rows(results, _VALUE_COLUMNS[tuning_quality], tuning_quality)
    rows = rows[rows["chinchilla_ratio"] == int(chinchilla)]
    table = _complete_ladder(rows, list(sizes), "scale_label")
    if table.empty:
        raise ValueError(f"no optimizer covers sizes {list(sizes)} at ratio {chinchilla} for {tuning_quality}")
    table = table.sort_values(["compute", "intervention"], kind="mergesort").reset_index(drop=True)
    validate(table)
    return table


def coverage_summary(results: pd.DataFrame) -> dict[str, Any]:
    keys = ["size", "chinchilla"]
    grid = results.pivot_table(index="optimizer", columns=keys, values="loss_min", aggfunc="first")
    ablation_grid = results.pivot_table(index="optimizer", columns=keys, values="n_ablations", aggfunc="first")
    return {
        "n_cells": int(len(results)),
        "n_cells_with_tuned_loss": int(results["loss_min"].notna().sum()),
        "optimizers": sorted(results["optimizer"].unique().tolist()),
        "sizes": [size for size in SIZE_ORDER if size in set(results["size"])],
        "chinchilla_ratios": sorted(int(value) for value in results["chinchilla"].unique()),
        "seeds_per_cell": 1,
        "cells_with_1_2b": sorted(results[results["size"] == "1.2b"]["optimizer"].unique().tolist()),
        "cells_missing_tuned_loss": (
            results[results["loss_min"].isna()][["optimizer", "size", "chinchilla"]].to_dict("records")
        ),
        "loss_grid": grid,
        "ablation_grid": ablation_grid,
    }


def format_coverage(summary: dict[str, Any]) -> str:
    lines = [
        f"Fantastic Optimizers coverage: {summary['n_cells']} (optimizer, size, ratio) cells, "
        f"{summary['n_cells_with_tuned_loss']} with a tuned loss, 1 run per cell (no seed replicates)",
        f"optimizers: {', '.join(summary['optimizers'])}",
        f"sizes: {', '.join(summary['sizes'])}; Chinchilla ratios: {summary['chinchilla_ratios']}",
        f"optimizers with a 1.2B run: {', '.join(summary['cells_with_1_2b'])}",
        "tuned C4-EN validation loss (nats/token) per cell:",
        summary["loss_grid"].round(4).to_string(),
        "number of released one-hyperparameter ablations per cell:",
        summary["ablation_grid"].fillna(0).astype(int).to_string(),
    ]
    if summary["cells_missing_tuned_loss"]:
        lines.append(f"cells without any loss: {summary['cells_missing_tuned_loss']}")
    return "\n".join(lines)


def harvest(out_dir: str | Path = "data", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict[str, pd.DataFrame]:
    """Download (if needed), map, validate, save every ladder table, and print coverage."""

    ensure_results(cache_dir)
    results = load_results(cache_dir)
    out = Path(out_dir)
    tables: dict[str, pd.DataFrame] = {}
    for ratio in (1, 2, 4, 8):
        name = f"fantastic_optimizers_size_ladder_{ratio}xC"
        tables[name] = size_ladder(results, ratio)
    for size in ("130m", "300m"):
        name = f"fantastic_optimizers_data_ladder_{size}"
        tables[name] = data_ladder(results, size)
    for condition in (TUNED, ABLATION_MEDIAN):
        name = f"fantastic_optimizers_partial_1xC_{condition}"
        tables[name] = partial_size_ladder(results, 1, condition)
    for name, table in tables.items():
        save_runs(table, out / f"{name}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "fantastic_optimizers_cells.csv", index=False)
    print(format_coverage(coverage_summary(results)))
    for name, table in tables.items():
        print(
            f"{name}: {len(table)} rows, interventions={sorted(table['intervention'].unique().tolist())}, "
            f"budgets={sorted(table['compute'].unique().tolist())}"
        )
    return tables
