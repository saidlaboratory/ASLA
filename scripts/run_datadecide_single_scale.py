"""Task 4 of PREDICTIONS_TASK_TRANSFER.md: DataDecide's own single-scale baseline against the scaling-law variants.

Their single-scale code (``single_scale/main.py``, ``single_scale/evaluator.py``)
is run with its functions unmodified: ``TargetFilter`` for the target pairs,
``take_mean_over_seeds`` for the 1B target, ``process_single_combination`` per
small-scale seed, and ``Evaluator`` (binary accuracy). Stated deviations:

* input: the released macro-average accuracy (``olmes_10_macro_avg``) at each
  scale's final common checkpoint, arranged in their transformed-table layout
  (``Raw`` keeps the last raw value), instead of their S3 export, which is not
  released;
* imports: ``yaml``, ``seaborn`` and ``tqdm`` are absent here and stubbed (used
  only for configuration files, plots and progress bars); ``main.py`` reads a
  task list at import that the repository does not contain, so an empty one is
  supplied in a scratch directory (the list is used only by ``preprocess_df``,
  which is not called).

Writes ``results/external/datadecide_single_scale.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "external" / "datadecide_single_scale.json"
RUNS = REPO / "data" / "datadecide_runs_olmes_macro_error.parquet"
SENSITIVITY = REPO / "results" / "external" / "datadecide_sensitivity.json"
MACRO = "olmes_10_macro_avg"
EVALUATOR_CONFIG = {
    "binary_accuracy": True,
    "three_way_accuracy": False,
    "magnitude_correlation": True,
    "pearson_correlation": True,
    "magnitude_ece": False,
    "recall_at_k": False,
    "weighted_pearson_correlation": True,
    "NDCG": True,
}  # the evaluator block of their single_scale/config.yaml


def _stub(name: str, **attrs: Any) -> None:
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


@contextlib.contextmanager
def _import_dir() -> Iterator[None]:
    """A working directory whose ../../all_olmes_rc_tasks.txt exists (empty), for main.py's import."""

    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as root:
        inner = Path(root) / "a" / "b"
        inner.mkdir(parents=True)
        (Path(root) / "all_olmes_rc_tasks.txt").write_text("", encoding="utf-8")
        os.chdir(inner)
        try:
            yield
        finally:
            os.chdir(previous)


def load_theirs(code_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(code_dir / "single_scale"))
    _stub("tqdm", tqdm=lambda iterable=None, *a, **k: iterable)
    _stub("yaml", safe_load=lambda *a, **k: {})
    _stub("seaborn")
    with _import_dir():
        main = importlib.import_module("main")
    evaluator = importlib.import_module("evaluator")
    target_filter = importlib.import_module("target_filter")
    return {"main": main, "Evaluator": evaluator.Evaluator, "TargetFilter": target_filter.TargetFilter}


def their_layout(runs: pd.DataFrame, reverse: dict[str, str]) -> pd.DataFrame:
    """Rows in their 1_*_transformed.csv layout for primary_metric (accuracy = 1 - macro error)."""

    frame = runs.copy()
    frame["group"] = frame["intervention"].map(reverse)
    if frame["group"].isna().any():
        raise ValueError("recipe names do not map to DataDecide group names")
    frame["value"] = 1.0 - frame["bpb"]
    return pd.DataFrame(
        {
            "model": frame["scale_label"],
            "group": frame["group"],
            "seed": frame["seed_label"],
            "metric": "primary_metric",
            "models": frame["scale_label"].map(lambda s: str([s])),
            "compute_latest": frame["compute"].astype(float),
            "token_latest": frame["tokens_d"].astype(float),
            "raw_values": frame["bpb"].map(lambda b: [1.0 - b]),
            "value": frame["value"],
        }
    )


def single_scale_accuracy(
    theirs: dict[str, Any], layout: pd.DataFrame, pairs: list[tuple[str, str]], scale: str, target_seeds: list[str]
) -> dict[str, Any]:
    """Their binary accuracy at one scale: each small-scale seed separately against the seed-mean 1B target."""

    main = theirs["main"]
    primary = main.take_mean_over_seeds(layout[(layout["model"] == "1B") & layout["seed"].isin(target_seeds)])
    metric_df = layout[layout["model"] == scale]
    max_c = float(layout["compute_latest"].max())
    max_tokens = float(layout["token_latest"].max())
    proportion = float(metric_df["compute_latest"].max()) / max_c + 1e-6
    per_seed = {}
    for seed in sorted(metric_df["seed"].unique()):
        res = main.process_single_combination(
            proportion,
            seed,
            scale,
            "primary_metric",
            None,
            "1B",
            pairs,
            metric_df,
            primary,
            theirs["Evaluator"](EVALUATOR_CONFIG),
            max_c,
            max_tokens,
            False,
        )
        per_seed[str(seed)] = 100 * float(res["binary_accuracy"])
    return {"per_seed": per_seed, "mean": float(np.mean(list(per_seed.values())))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args()
    theirs = load_theirs(args.code_dir)
    # single_scale/utils.py shadows scaling_laws/utils, so the recipe-name constants are loaded by path.
    spec = importlib.util.spec_from_file_location(
        "dd_constants_recepies", args.code_dir / "scaling_laws" / "utils" / "constants" / "constants_recepies.py"
    )
    assert spec is not None and spec.loader is not None
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    DATA_NAME_CLEAN = constants.DATA_NAME_CLEAN  # noqa: N806

    commit = subprocess.run(
        ["git", "-C", str(args.code_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    runs = pd.read_parquet(RUNS)
    layout = their_layout(runs, {v: k for k, v in DATA_NAME_CLEAN.items()})
    target_rows = layout[layout["model"] == "1B"].rename(columns={"value": "primary_metric", "compute_latest": "compute"})
    pairs = theirs["TargetFilter"](target_model="1B").apply(target_rows, primary_metric="primary_metric")
    three = sorted(layout[layout["model"] == "1B"]["seed"].unique())
    scales = [
        s
        for s in sorted(layout["model"].unique(), key=lambda s: float(layout[layout["model"] == s]["compute_latest"].max()))
        if s != "1B"
    ]
    by_scale = {
        s: {
            "three_seed_target": single_scale_accuracy(theirs, layout, pairs, s, three),
            "default_seed_target": single_scale_accuracy(theirs, layout, pairs, s, ["default"]),
        }
        for s in scales
    }
    sens = json.loads(SENSITIVITY.read_text())
    macro_rows = [r for r in sens["task_4b"]["tested_rows"]["rows"] if r["task"] == MACRO]
    ss3 = by_scale["750M"]["three_seed_target"]["mean"]
    ssd = by_scale["750M"]["default_seed_target"]["mean"]
    comparisons = {
        r["setup"]: {
            "variant_default_target": r["implemented"],
            "variant_three_seed_target": r["three_seed_mean"],
            "a_as_implemented_minus_their_ss": r["implemented"] - ss3,
            "b_both_three_seed_minus_their_ss": r["three_seed_mean"] - ss3,
            "secondary_both_default_minus_their_ss": r["implemented"] - ssd,
        }
        for r in macro_rows
    }
    exceed = {
        key: sorted(v for v, c in comparisons.items() if c[key] > 0)
        for key in (
            "a_as_implemented_minus_their_ss",
            "b_both_three_seed_minus_their_ss",
            "secondary_both_default_minus_their_ss",
        )
    }
    our_rebuild = sens["task_4b"]["conclusions"]["single_scale_750M"][MACRO]
    out = {
        "code": {
            "repository": "allenai/DataDecide",
            "commit": commit,
            "functions": [
                "single_scale/target_filter.py:TargetFilter.apply",
                "single_scale/main.py:take_mean_over_seeds",
                "single_scale/main.py:process_single_combination",
                "single_scale/evaluator.py:Evaluator (binary_accuracy)",
            ],
        },
        "search": {
            "found": (
                "single_scale/ (main.py, evaluator.py, config.yaml): per-seed binary accuracy against the "
                "three-seed-mean 1B target"
            ),
            "released_outputs": (
                "results/readme.md points to a Google Drive folder (outputs2/, per_task_out/); access was denied in this "
                "environment, so the released outputs were not checked"
            ),
            "not_found": "no single-scale output in the repository or in the Hugging Face release",
        },
        "target_pairs": len(pairs),
        "their_single_scale_macro": by_scale,
        "our_rebuild_750M_three_seed": our_rebuild,
        "comparisons_macro_750M": comparisons,
        "variants_exceeding": exceed,
    }
    out["predictions"] = {
        "P4_flip_persists_3_param_1_step": "3_param-1_step" in exceed["b_both_three_seed_minus_their_ss"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("their SS 750M: three-seed", round(ss3, 2), "default", round(ssd, 2), "| our rebuild", round(our_rebuild, 2))
    print(json.dumps(exceed, indent=1), out["predictions"])
    print({s: round(v["three_seed_target"]["mean"], 2) for s, v in by_scale.items()})


if __name__ == "__main__":
    main()
