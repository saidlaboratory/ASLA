"""Harvest the Signal-and-Noise evaluation release for the DataDecide models.

Artifact pulled (HuggingFace ``datasets`` repo ``allenai/signal-and-noise``,
evaluation results only; no weights): file ``data/core-00000-of-00001.parquet``
(~24 MB, 389,521 rows). Heineman et al. (NeurIPS 2025, arXiv:2508.13144)
evaluate 375 open-weight models on 30+ benchmarks; the ``core`` split includes
``model_type == "datadecide"`` rows for the final checkpoint of every
DataDecide model at nine scales (4M, 20M, 60M, 90M, 150M, 300M, 530M, 750M,
1B) x 25 recipes = 225 models, with ``model_params``, ``model_tokens`` and
``flops`` (= 6 * params * tokens, asserted at harvest) populated.

What we take: ``task == "paloma_c4_en"``, whose ``primary_score`` is the Paloma
C4-EN **bits per byte** (the ``metrics`` payload carries the same value under
``bits_per_byte``; equality is asserted). This is the only public table found
that reports a true bits-per-byte number for the DataDecide suite, so the
``bpb`` column here carries ``metric_name = "c4_en_bits_per_byte_paloma"``.

Limitations recorded in the table:

* One evaluated run per (recipe, scale). For 150M and larger the evaluated run
  is DataDecide's auxiliary seed ``2`` (model names end in ``-5xC-2``); for 4M
  to 90M it is the ``default`` seed. Every row therefore has ``seed = 0`` and
  ``seed_label`` records which DataDecide seed it was. Seed-bootstrap
  intervals are degenerate and crossovers untestable on this table.
* 750M is fully trained here (75B tokens, ~110 tokens/param), unlike the
  released DataDecide eval table where 750M stops early; this table's 750M
  cell is on-trajectory.
* Recipe names are mapped from the ``mix`` column (DataDecide's internal
  recipe identifiers, the same ones used in DataDecide's released
  ``scaling_law_fit`` table) to the display names used by
  :mod:`asla.data.sources.datadecide`, so the two tables share
  ``intervention`` values. ``model_path`` is *not* used for identity: in the
  release the ``dolma17`` and ``dolma-v1-6-and-sources-baseline`` mixes both
  carry ``allenai/DataDecide-dolma1_6plus-*`` paths at some sizes, and
  ``model_revision`` reads ``seed-default`` even for ``-5xC-2`` model names.
  Both raw fields are kept in ``source_model_path`` / ``source_revision``.

The other splits are not harvested: ``random_seeds`` holds 20 OLMo-2 1B seed
runs on QA tasks (no C4-EN loss), and ``datadecide_intermediate`` holds
intermediate-checkpoint OLMES results that duplicate the DataDecide release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asla.data.io import save_runs
from asla.data.schema import validate
from asla.data.sources.datadecide import SCALE_ORDER, TUNING_QUALITY

REPO = "allenai/signal-and-noise"
CORE_FILE = "data/core-00000-of-00001.parquet"
DEFAULT_CACHE_DIR = Path("data/raw/signal_and_noise")
LOCAL_NAME = "core.parquet"
MAX_DOWNLOAD_BYTES = 2_000_000_000
TASK = "paloma_c4_en"
METRIC_NAME = "c4_en_bits_per_byte_paloma"
INTERVENTION_CLASS = "data"

# Signal-and-Noise ``mix`` (DataDecide's internal recipe identifiers, also used in
# DataDecide's released ``scaling_law_fit`` table) -> DataDecide display name.
MIX_TO_RECIPE: Mapping[str, str] = {
    "dolma17": "Dolma1.7",
    "no_code": "Dolma1.7 (no code)",
    "no_math_no_code": "Dolma1.7 (no math, code)",
    "no_reddit": "Dolma1.7 (no Reddit)",
    "no_flan": "Dolma1.7 (no Flan)",
    "dolma-v1-6-and-sources-baseline": "Dolma1.6++",
    "c4": "C4",
    "prox_fineweb_pro": "FineWeb-Pro",
    "fineweb_edu_dedup": "FineWeb-Edu",
    "falcon": "Falcon",
    "falcon_and_cc": "Falcon+CC",
    "falcon_and_cc_eli5_oh_top10p": "Falcon+CC (QC 10%)",
    "falcon_and_cc_eli5_oh_top20p": "Falcon+CC (QC 20%)",
    "falcon_and_cc_og_eli5_oh_top10p": "Falcon+CC (QC Orig 10%)",
    "falcon_and_cc_tulu_qc_top10": "Falcon+CC (QC Tulu 10%)",
    "DCLM-baseline": "DCLM-Baseline",
    "dclm_ft7percentile_fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_ft7percentile_fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm_fw_top3": "DCLM-Baseline (QC FW 3%)",
    "dclm_fw_top10": "DCLM-Baseline (QC FW 10%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "DCLM-Baseline (QC 10%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "DCLM-Baseline (QC 20%)",
    "dolma17-75p-DCLM-baseline-25p": "DCLM-Baseline 25% / Dolma 75%",
    "dolma17-50p-DCLM-baseline-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dolma17-25p-DCLM-baseline-75p": "DCLM-Baseline 75% / Dolma 25%",
}


def _local_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / LOCAL_NAME


def ensure_artifacts(cache_dir: str | Path = DEFAULT_CACHE_DIR, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
    """Download the ``core`` evaluation table if absent, after checking its size."""

    target = _local_path(cache_dir)
    if target.exists():
        return target
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(REPO, files_metadata=True)
    size = next((s.size for s in info.siblings or [] if s.rfilename == CORE_FILE), None)
    label = "unknown size" if size is None else f"{size / 1e6:.1f} MB"
    print(f"Signal-and-Noise artifact {REPO}:{CORE_FILE} -> {label}")
    if size is None or size > max_bytes:
        raise RuntimeError(f"refusing to download {REPO}:{CORE_FILE} ({label})")
    downloaded = hf_hub_download(REPO, CORE_FILE, repo_type="dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(downloaded).read_bytes())
    return target


def load_datadecide_c4(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Return the Paloma C4-EN rows for DataDecide models with the BPB payload unpacked."""

    import pyarrow.parquet as pq

    table = pq.read_table(
        _local_path(cache_dir),
        columns=[
            "task", "model", "model_type", "model_path", "model_revision", "primary_score", "primary_metric",
            "model_params", "model_tokens", "flops", "step", "mix", "size", "metrics",
        ],
        filters=[("task", "==", TASK), ("model_type", "==", "datadecide")],
    )
    df = table.to_pandas()
    df["bits_per_byte"] = [None if m is None else m.get("bits_per_byte") for m in df["metrics"]]
    return df.drop(columns=["metrics"])


def build_runs_table(core: pd.DataFrame) -> pd.DataFrame:
    """Map the Signal-and-Noise DataDecide C4-EN rows into the canonical schema."""

    df = core[(core["task"] == TASK) & (core["model_type"] == "datadecide")].copy()
    if df.empty:
        raise ValueError("no DataDecide paloma_c4_en rows found")
    bpb = pd.to_numeric(df["bits_per_byte"], errors="coerce")
    if bpb.notna().any() and not np.allclose(df.loc[bpb.notna(), "primary_score"], bpb.dropna(), rtol=1e-9):
        raise ValueError("primary_score does not equal metrics['bits_per_byte'] for paloma_c4_en")
    df = df.sort_values("step").groupby("model", sort=False).tail(1)
    unknown = sorted(set(df["mix"].astype(str)) - set(MIX_TO_RECIPE))
    if unknown:
        raise ValueError(f"unmapped DataDecide mix identifiers: {unknown}")
    derived = 6.0 * df["model_params"].astype(float) * df["model_tokens"].astype(float)
    if not np.allclose(df["flops"].astype(float), derived, rtol=1e-6):
        raise ValueError("artifact flops disagree with 6 * params * tokens")
    seed_label = df["model"].str.extract(r"-5xC(?:-(\d+))?$")[0].fillna("default")
    out = pd.DataFrame(
        {
            "intervention": df["mix"].astype(str).map(MIX_TO_RECIPE).astype(str),
            "intervention_class": INTERVENTION_CLASS,
            "compute": df["flops"].astype(float),
            "seed": 0,
            "bpb": df["primary_score"].astype(float),
            "metric_name": METRIC_NAME,
            "params_n": df["model_params"].astype(float),
            "tokens_d": df["model_tokens"].astype(float),
            "tokens_per_param": (df["model_tokens"].astype(float) / df["model_params"].astype(float)),
            "scale_label": df["size"].astype(str),
            "step": df["step"].astype(int),
            "seed_label": seed_label.astype(str),
            "source_model": df["model"].astype(str),
            "source_model_path": df["model_path"].astype(str),
            "source_revision": df["model_revision"].astype(str),
            "tuning_quality": TUNING_QUALITY,
        }
    )
    order = {scale: i for i, scale in enumerate(SCALE_ORDER)}
    out["_order"] = out["scale_label"].map(order)
    out = out.sort_values(["_order", "intervention"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    validate(out)
    return out


def coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    per_scale = df.groupby("scale_label", sort=False).agg(
        compute=("compute", "first"),
        params_n=("params_n", "first"),
        tokens_per_param=("tokens_per_param", "first"),
        recipes=("intervention", "nunique"),
        seed_labels=("seed_label", lambda s: ",".join(sorted(set(s)))),
    )
    return {
        "metric_name": METRIC_NAME,
        "n_rows": int(len(df)),
        "n_recipes": int(df["intervention"].nunique()),
        "n_scales": int(df["scale_label"].nunique()),
        "seeds_per_cell": 1,
        "per_scale": per_scale.reset_index().to_dict(orient="records"),
    }


def format_coverage(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Signal-and-Noise DataDecide coverage for {summary['metric_name']}: {summary['n_rows']} rows, "
        f"{summary['n_recipes']} recipes, {summary['n_scales']} scales, 1 evaluated run per cell (no seed replicates)",
        f"{'scale':>6} {'compute':>12} {'params_n':>12} {'tok/param':>9} {'recipes':>7} {'seed label(s)':>14}",
    ]
    for row in summary["per_scale"]:
        lines.append(
            f"{row['scale_label']:>6} {row['compute']:>12.4e} {row['params_n']:>12.4e} {row['tokens_per_param']:>9.1f} "
            f"{row['recipes']:>7d} {row['seed_labels']:>14}"
        )
    return "\n".join(lines)


def harvest(out_path: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    ensure_artifacts(cache_dir)
    table = build_runs_table(load_datadecide_c4(cache_dir))
    save_runs(table, out_path)
    print(format_coverage(coverage_summary(table)))
    print(f"wrote {out_path}")
    return table
