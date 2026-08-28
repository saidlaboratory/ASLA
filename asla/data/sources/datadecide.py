"""Harvest the public DataDecide evaluation tables into the canonical runs schema.

Artifacts pulled (HuggingFace ``datasets`` repos, evaluation tables only; no
model weights are ever downloaded):

* ``allenai/DataDecide-eval-results`` file ``data/macro_avg-00000-of-00001.parquet``
  (~139 MB). One row per (params, data recipe, task, step, seed) with a JSON
  ``metrics`` string holding the OLMES metrics (``primary_metric`` is the
  OLMES ACCURACY with per-task curated normalisation; ``correct_prob_per_char``
  etc. are the continuous likelihood proxies). We keep only
  ``task == "olmes_10_macro_avg"``, the macro average over the 10 OLMES tasks.
  The rows carry ``tokens`` and ``compute`` (FLOPs) columns.
* ``allenai/DataDecide-ppl-results`` file ``data/train-00000-of-00001.parquet``
  (~2 MB). One row per (params, data recipe, seed, step) with validation
  perplexities on 11 Paloma-style validation sets, including
  ``eval/c4_en-validation/Perplexity``. It carries no ``tokens``/``compute``.

Structure of the suite as observed in the artifacts (verified 2026-08-23):
25 data recipes x 14 model scales (4M ... 1B non-embedding parameters) x 3
seeds. Seed labels are ``default``, ``small aux 2``, ``small aux 3`` for every
scale below 1B and ``default``, ``large aux 2``, ``large aux 3`` at 1B. The
perplexity table additionally contains ``small aux 2/3`` rows at 1B that stop
at step 17,339 (a truncated auxiliary run, not the 1B seed replicate); those
rows are dropped. The paper states that non-1B auxiliary seeds are stopped
early at 25% of the 1B compute budget; in the released tables this only
affects 750M (see below).

Compute derivation. DataDecide defines compute as ``FLOPs = 6 * N * D`` with
``N`` the non-embedding parameter count and ``D`` tokens trained. The eval
table stores ``compute`` and ``tokens`` per checkpoint; we recover
``N = compute / (6 * tokens)`` and ``tokens_per_step = tokens / step`` (both
constant per scale in the artifact, asserted at harvest time). For the
perplexity table, which lacks these columns, ``tokens_d = step *
tokens_per_step`` and ``compute = 6 * N * tokens_d`` with the constants taken
from the eval table for the same scale. ``params_n`` is ``N``.

Checkpoint choice. For each scale we take the largest training step at which
every (recipe, seed) cell has a row in the source table, so the three seeds of
each recipe are compared at identical compute. This is the final released
checkpoint for most scales; for 150M, 300M and 530M the last checkpoint common
to all cells is slightly before the end of training (tokens/param 89-97 rather
than 100), and for 750M the released rows stop at ~45 tokens/param, so the 750M
cell sits off the 5xC trajectory the other scales follow. The harvested table
records ``step`` and ``tokens_per_param`` per row so audits can exclude 750M
explicitly (DataDecide's own scaling-law fits also report ``no_750M``
variants).

Metrics. DataDecide's primary metric is downstream accuracy, not bits per
byte. We never coerce silently: every row carries ``metric_name`` and the
``bpb`` column holds the named lower-is-better quantity:

* ``c4_en_bits_per_token`` = log2(C4-EN validation perplexity) from the
  perplexity table. This is a token-level cross-entropy in bits, not bits per
  byte (bytes-per-token for the tokenizer is not released).
* ``olmes_macro_error`` = 1 - OLMES macro-average ACCURACY (``primary_metric``).
  This is the target metric DataDecide's headline decision accuracy is
  defined on; ranking on the error rate is identical to ranking on accuracy.
* ``olmes_macro_correct_prob_per_char_deficit`` = 1 - ``correct_prob_per_char``
  macro average, the continuous proxy DataDecide recommends at small scale.

``tuning_quality`` is set to ``shared_scale_heuristic_hparams`` for every row:
DataDecide uses one hyperparameter configuration per model scale derived from
the OLMo ladder heuristics (Porian et al., 2024) for all 25 recipes, i.e.
hyperparameters were not tuned per recipe. Data recipes are the axis where
per-intervention re-tuning is least expected to matter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import validate

LOGGER = logging.getLogger(__name__)

EVAL_REPO = "allenai/DataDecide-eval-results"
EVAL_FILE = "data/macro_avg-00000-of-00001.parquet"
PPL_REPO = "allenai/DataDecide-ppl-results"
PPL_FILE = "data/train-00000-of-00001.parquet"
SCALING_LAW_FIT_FILE = "data/scaling_law_fit-00000-of-00001.parquet"
MAX_DOWNLOAD_BYTES = 2_000_000_000
DEFAULT_CACHE_DIR = Path("data/raw/datadecide")
LOCAL_NAMES = {
    EVAL_FILE: "eval_macro_avg.parquet",
    PPL_FILE: "ppl_results.parquet",
    SCALING_LAW_FIT_FILE: "eval_scaling_law_fit.parquet",
}

MACRO_TASK = "olmes_10_macro_avg"
C4_PPL_COLUMN = "eval/c4_en-validation/Perplexity"
SEED_MAP: Mapping[str, int] = {
    "default": 0,
    "small aux 2": 1,
    "small aux 3": 2,
    "large aux 2": 1,
    "large aux 3": 2,
}
TARGET_SCALE = "1B"
INTERVENTION_CLASS = "data"
TUNING_QUALITY = "shared_scale_heuristic_hparams"

# Scale labels in ascending parameter order, as released.
SCALE_ORDER = ("4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B")


def _log2(values: pd.Series) -> pd.Series:
    return np.log2(values.astype(float))


def _one_minus(values: pd.Series) -> pd.Series:
    return 1.0 - values.astype(float)


METRICS: dict[str, dict[str, Any]] = {
    "c4_en_bits_per_token": {
        "source": "ppl",
        "field": C4_PPL_COLUMN,
        "transform": _log2,
        "description": "log2 of C4-EN validation perplexity (bits per token; lower is better)",
    },
    "olmes_macro_error": {
        "source": "eval",
        "field": "primary_metric",
        "transform": _one_minus,
        "description": "1 - OLMES 10-task macro-average ACCURACY (lower is better)",
    },
    "olmes_macro_correct_prob_per_char_deficit": {
        "source": "eval",
        "field": "correct_prob_per_char",
        "transform": _one_minus,
        "description": "1 - OLMES macro-average correct_prob_per_char (lower is better)",
    },
}
DEFAULT_METRIC = "c4_en_bits_per_token"


def _local_path(cache_dir: Path, remote_file: str) -> Path:
    return Path(cache_dir) / LOCAL_NAMES[remote_file]


def artifact_sizes() -> dict[str, int | None]:
    """Return the byte size of each needed remote file from the HuggingFace API."""

    from huggingface_hub import HfApi

    api = HfApi()
    sizes: dict[str, int | None] = {}
    for repo, files in ((EVAL_REPO, (EVAL_FILE, SCALING_LAW_FIT_FILE)), (PPL_REPO, (PPL_FILE,))):
        info = api.dataset_info(repo, files_metadata=True)
        by_name = {sibling.rfilename: sibling.size for sibling in info.siblings or []}
        for name in files:
            sizes[f"{repo}:{name}"] = by_name.get(name)
    return sizes


def ensure_artifacts(cache_dir: str | Path = DEFAULT_CACHE_DIR, max_bytes: int = MAX_DOWNLOAD_BYTES) -> dict[str, Path]:
    """Download the evaluation tables (never weights) into ``cache_dir`` if absent.

    Sizes are checked through the HuggingFace API before any transfer and a
    file larger than ``max_bytes`` is refused rather than downloaded.
    """

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    needed = [(EVAL_REPO, EVAL_FILE), (EVAL_REPO, SCALING_LAW_FIT_FILE), (PPL_REPO, PPL_FILE)]
    paths: dict[str, Path] = {}
    missing = [(repo, name) for repo, name in needed if not _local_path(cache, name).exists()]
    if missing:
        from huggingface_hub import hf_hub_download

        sizes = artifact_sizes()
        for repo, name in missing:
            size = sizes.get(f"{repo}:{name}")
            label = "unknown size" if size is None else f"{size / 1e6:.1f} MB"
            print(f"DataDecide artifact {repo}:{name} -> {label}")
            if size is None or size > max_bytes:
                raise RuntimeError(
                    f"refusing to download {repo}:{name} ({label}); limit is {max_bytes / 1e9:.1f} GB "
                    "or the size could not be determined"
                )
            downloaded = hf_hub_download(repo, name, repo_type="dataset")
            target = _local_path(cache, name)
            target.write_bytes(Path(downloaded).read_bytes())
    for repo, name in needed:
        paths[name] = _local_path(cache, name)
    return paths


def load_eval_macro(cache_dir: str | Path = DEFAULT_CACHE_DIR, fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Load OLMES macro-average rows with the requested JSON metric fields unpacked."""

    path = _local_path(Path(cache_dir), EVAL_FILE)
    raw = pd.read_parquet(path, columns=["params", "data", "task", "step", "seed", "tokens", "compute", "metrics"])
    raw = raw[raw["task"] == MACRO_TASK].drop(columns=["task"]).reset_index(drop=True)
    wanted = tuple(fields) if fields else tuple(spec["field"] for spec in METRICS.values() if spec["source"] == "eval")
    parsed = [json.loads(text) for text in raw["metrics"].tolist()]
    for field in wanted:
        raw[field] = [float(entry[field]) for entry in parsed]
    return raw.drop(columns=["metrics"])


def load_ppl(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load the perplexity table."""

    path = _local_path(Path(cache_dir), PPL_FILE)
    df = pd.read_parquet(path)
    return df.drop(columns=[col for col in df.columns if col.startswith("__index")])


def load_published_scaling_law_decision_acc(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Return DataDecide's released per-setup decision accuracies for the OLMES macro average.

    The released ``scaling_law_fit`` table stores ``decision_acc`` (percent) for
    every (task, metric, scaling-law setup). This is a published number read
    from the artifact, never re-derived here.
    """

    path = _local_path(Path(cache_dir), SCALING_LAW_FIT_FILE)
    df = pd.read_parquet(path, columns=["task", "metric", "setup", "decision_acc"])
    df = df[df["task"] == MACRO_TASK]
    return (
        df.groupby(["setup", "metric"], sort=True)["decision_acc"]
        .first()
        .reset_index()
        .sort_values(["setup", "metric"], kind="mergesort")
        .reset_index(drop=True)
    )


def scale_constants(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``params_n`` and ``tokens_per_step`` per scale from the eval table.

    Both quantities are constant within a scale in the released artifact; a
    non-constant value raises rather than being averaged away.
    """

    rows = eval_df[eval_df["step"] > 0][["params", "step", "tokens", "compute"]].drop_duplicates()
    rows = rows.assign(
        tokens_per_step=rows["tokens"].astype(float) / rows["step"].astype(float),
        params_n=rows["compute"].astype(float) / (6.0 * rows["tokens"].astype(float)),
    )
    out: list[dict[str, Any]] = []
    for scale, group in rows.groupby("params", sort=False):
        tps = group["tokens_per_step"].to_numpy(dtype=float)
        n = group["params_n"].to_numpy(dtype=float)
        if not np.allclose(tps, tps[0], rtol=1e-9) or not np.allclose(n, n[0], rtol=1e-6):
            raise ValueError(f"scale {scale}: tokens_per_step or params_n is not constant across checkpoints")
        out.append({"params": str(scale), "tokens_per_step": float(tps[0]), "params_n": float(n[0])})
    table = pd.DataFrame(out).set_index("params")
    order = [scale for scale in SCALE_ORDER if scale in table.index]
    return table.loc[order]


def _filter_seed_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the three seed replicates of every scale and drop truncated auxiliary 1B rows."""

    is_target = df["params"].astype(str) == TARGET_SCALE
    small_aux = df["seed"].astype(str).str.startswith("small")
    large_aux = df["seed"].astype(str).str.startswith("large")
    keep = ~((is_target & small_aux) | (~is_target & large_aux))
    return df[keep].copy()


def select_common_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Return, per scale, the largest step present for every (recipe, seed) cell."""

    n_cells = df.groupby("params")[["data", "seed"]].nunique()
    expected = (n_cells["data"] * n_cells["seed"]).to_dict()
    counts = df.groupby(["params", "step"]).size().reset_index(name="n_rows")
    complete = counts[counts.apply(lambda row: row["n_rows"] == expected[row["params"]], axis=1)]
    if complete.empty:
        raise ValueError("no checkpoint is shared by every recipe and seed at any scale")
    chosen = complete.groupby("params")["step"].max().reset_index(name="step")
    missing = sorted(set(expected) - set(chosen["params"]))
    if missing:
        raise ValueError(f"no common checkpoint for scales: {missing}")
    return chosen


def select_all_complete_checkpoints(df: pd.DataFrame, discard_first_fraction: float = 0.10) -> pd.DataFrame:
    """Return every (scale, step) whose cell is complete, for checkpoint-augmented fitting.

    A checkpoint is usable only when every (recipe, seed) cell has a row at that
    step, so all interventions are compared on identical data. Choshen, Zhang &
    Andreas (arXiv:2410.11840) report that the earliest checkpoints of a run hurt
    scaling-law accuracy and recommend discarding roughly the first 10%;
    ``discard_first_fraction`` implements that rule per scale, measured against
    that scale's maximum step.
    """

    if not 0.0 <= discard_first_fraction < 1.0:
        raise ValueError(f"discard_first_fraction must lie in [0, 1), got {discard_first_fraction}")
    n_cells = df.groupby("params")[["data", "seed"]].nunique()
    expected = (n_cells["data"] * n_cells["seed"]).to_dict()
    counts = df.groupby(["params", "step"]).size().reset_index(name="n_rows")
    complete = counts[counts.apply(lambda row: row["n_rows"] == expected[row["params"]], axis=1)]
    if complete.empty:
        raise ValueError("no checkpoint is shared by every recipe and seed at any scale")
    kept: list[pd.DataFrame] = []
    for scale, group in complete.groupby("params"):
        steps = np.asarray(sorted(group["step"].unique()), dtype=float)
        threshold = steps.max() * discard_first_fraction
        surviving = steps[steps >= threshold]
        kept.append(pd.DataFrame({"params": scale, "step": surviving.astype(int)}))
    return pd.concat(kept, ignore_index=True)


def build_runs_table(
    eval_df: pd.DataFrame,
    ppl_df: pd.DataFrame | None,
    metric: str = DEFAULT_METRIC,
    checkpoints: str = "final",
    discard_first_fraction: float = 0.10,
) -> pd.DataFrame:
    """Map DataDecide tables into the canonical runs schema for one named metric.

    ``checkpoints="final"`` keeps one row per (recipe, scale, seed) at the last
    checkpoint common to every cell - the default, and what every audit to date
    has used. ``checkpoints="all"`` keeps every complete intermediate checkpoint
    as its own row, giving many more fitting points per intervention along the
    same compute ladder. In that mode ``compute`` varies within a scale, so the
    canonical uniqueness key ``(intervention, compute, seed)`` still holds.
    """

    if metric not in METRICS:
        raise ValueError(f"unknown DataDecide metric {metric!r}; choose one of {sorted(METRICS)}")
    spec = METRICS[metric]
    constants = scale_constants(eval_df)
    if spec["source"] == "eval":
        source = eval_df[["params", "data", "seed", "step", "tokens", "compute", spec["field"]]].rename(
            columns={spec["field"]: "value"}
        )
    else:
        if ppl_df is None:
            raise ValueError(f"metric {metric!r} needs the perplexity table")
        source = ppl_df[["params", "data", "seed", "step", spec["field"]]].rename(columns={spec["field"]: "value"})
        source = source.assign(step=source["step"].astype(int))
    source = source[source["step"] > 0]
    source = _filter_seed_replicates(source)
    source = source.dropna(subset=["value"])
    unknown = sorted(set(source["seed"].astype(str)) - set(SEED_MAP))
    if unknown:
        raise ValueError(f"unrecognised DataDecide seed labels: {unknown}")
    if checkpoints == "final":
        chosen = select_common_checkpoints(source)
    elif checkpoints == "all":
        chosen = select_all_complete_checkpoints(source, discard_first_fraction=discard_first_fraction)
    else:
        raise ValueError(f"checkpoints must be 'final' or 'all', got {checkpoints!r}")
    rows = source.merge(chosen, on=["params", "step"], how="inner")
    rows = rows.merge(constants, left_on="params", right_index=True, how="left")
    if rows["params_n"].isna().any():
        raise ValueError("scale constants missing for some scales")
    derived_tokens = rows["step"].astype(float) * rows["tokens_per_step"]
    derived_compute = 6.0 * rows["params_n"] * derived_tokens
    if "compute" in rows.columns:
        if not np.allclose(rows["compute"].astype(float), derived_compute, rtol=1e-6):
            raise ValueError("artifact compute disagrees with 6*N*D derivation")
        if not np.allclose(rows["tokens"].astype(float), derived_tokens, rtol=1e-9):
            raise ValueError("artifact tokens disagree with step * tokens_per_step")
    transform: Callable[[pd.Series], pd.Series] = spec["transform"]
    out = pd.DataFrame(
        {
            "intervention": rows["data"].astype(str),
            "intervention_class": INTERVENTION_CLASS,
            "compute": derived_compute.astype(float),
            "seed": rows["seed"].astype(str).map(SEED_MAP).astype(int),
            "bpb": transform(rows["value"]).astype(float),
            "metric_name": metric,
            "params_n": rows["params_n"].astype(float),
            "tokens_d": derived_tokens.astype(float),
            "tokens_per_param": (derived_tokens / rows["params_n"]).astype(float),
            "scale_label": rows["params"].astype(str),
            "step": rows["step"].astype(int),
            "seed_label": rows["seed"].astype(str),
            "tuning_quality": TUNING_QUALITY,
        }
    )
    order = {scale: i for i, scale in enumerate(SCALE_ORDER)}
    out["_order"] = out["scale_label"].map(order)
    out = out.sort_values(["_order", "intervention", "seed"], kind="mergesort").drop(columns="_order")
    out = out.reset_index(drop=True)
    validate(out)
    return out


def coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise recipes, scales, seeds per cell, and checkpoint positions."""

    cells = df.groupby(["scale_label", "intervention"], sort=False)["seed"].nunique()
    per_scale = df.groupby("scale_label", sort=False).agg(
        compute=("compute", "first"),
        params_n=("params_n", "first"),
        tokens_d=("tokens_d", "first"),
        tokens_per_param=("tokens_per_param", "first"),
        step=("step", "first"),
        recipes=("intervention", "nunique"),
    )
    per_scale["min_seeds_per_cell"] = [int(cells.loc[scale].min()) for scale in per_scale.index]
    per_scale["max_seeds_per_cell"] = [int(cells.loc[scale].max()) for scale in per_scale.index]
    return {
        "metric_name": str(df["metric_name"].iloc[0]),
        "n_rows": int(len(df)),
        "n_recipes": int(df["intervention"].nunique()),
        "recipes": sorted(df["intervention"].unique().tolist()),
        "n_scales": int(df["scale_label"].nunique()),
        "per_scale": per_scale.reset_index().to_dict(orient="records"),
        "off_trajectory_scales": sorted(
            per_scale.index[(per_scale["tokens_per_param"] < 80.0) | (per_scale["tokens_per_param"] > 120.0)].tolist()
        ),
    }


def format_coverage(summary: Mapping[str, Any]) -> str:
    lines = [
        f"DataDecide coverage for metric {summary['metric_name']}: {summary['n_rows']} rows, "
        f"{summary['n_recipes']} recipes, {summary['n_scales']} scales",
        f"{'scale':>6} {'compute':>12} {'params_n':>12} {'tokens_d':>14} {'tok/param':>9} {'step':>6} "
        f"{'recipes':>7} {'seeds/cell':>10}",
    ]
    for row in summary["per_scale"]:
        seeds = f"{row['min_seeds_per_cell']}-{row['max_seeds_per_cell']}"
        if row["min_seeds_per_cell"] == row["max_seeds_per_cell"]:
            seeds = str(row["min_seeds_per_cell"])
        lines.append(
            f"{row['scale_label']:>6} {row['compute']:>12.4e} {row['params_n']:>12.4e} {row['tokens_d']:>14.4e} "
            f"{row['tokens_per_param']:>9.1f} {row['step']:>6d} {row['recipes']:>7d} {seeds:>10}"
        )
    if summary["off_trajectory_scales"]:
        lines.append(
            "off-trajectory scales (tokens/param outside 80-120, i.e. not on the 5xC ladder): "
            + ", ".join(summary["off_trajectory_scales"])
        )
    return "\n".join(lines)


def harvest(
    out_path: str | Path,
    metric: str = DEFAULT_METRIC,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    checkpoints: str = "final",
    discard_first_fraction: float = 0.10,
) -> pd.DataFrame:
    """Download (if needed), map, validate, save, and print a coverage summary."""

    ensure_artifacts(cache_dir)
    eval_df = load_eval_macro(cache_dir)
    ppl_df = load_ppl(cache_dir) if METRICS[metric]["source"] == "ppl" else None
    table = build_runs_table(
        eval_df, ppl_df, metric=metric, checkpoints=checkpoints, discard_first_fraction=discard_first_fraction
    )
    save_runs(table, out_path)
    summary = coverage_summary(table)
    print(format_coverage(summary))
    print(f"wrote {out_path}")
    return table
