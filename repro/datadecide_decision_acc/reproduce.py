"""Recompute DataDecide's released scaling-law decision accuracies from its released predictions.

Standalone: needs only pandas, numpy and pyarrow (plus huggingface_hub to fetch
the files). Inputs are two files from the dataset ``allenai/DataDecide-eval-results``:

* ``data/scaling_law_fit-00000-of-00001.parquet`` --- per (task, metric, setup,
  mix): ``stacked_pred``, ``stacked_y``, ``step_2_pred``, ``step_2_y`` and the
  released ``decision_acc`` (percent, constant within a task/metric/setup).
* ``data/macro_avg-00000-of-00001.parquet`` --- per (params, data, task, step,
  seed): a JSON ``metrics`` field containing ``primary_metric``.

Usage::

    python reproduce.py --out out/                      # fetch from the Hub
    python reproduce.py --local-dir path/ --out out/    # files already downloaded

``--local-dir`` must contain the two parquet files under their Hub file names
(``scaling_law_fit-00000-of-00001.parquet`` and ``macro_avg-00000-of-00001.parquet``),
or pass ``--fit`` and ``--macro`` explicitly.

Decision accuracy is the fraction of the 300 recipe pairs whose predicted order
matches the target order. Every combination of the following is recomputed for
each (task, setup) with metric ``primary_metric``:

* prediction column: ``stacked_pred``, ``step_2_pred``
* target: ``stacked_y``, ``step_2_y``, the 1B ``default``-seed value, the 1B
  mean over the three 1B seeds (``default``, ``large aux 2``, ``large aux 3``),
  all at the final 1B step common to the three seeds
* direction: higher is better, lower is better
* ties (either side): scored 0, 0.5 or 1

Outputs, in ``--out``:

* ``nonmatching_rows.csv`` --- rows whose released ``decision_acc`` differs from
  the recomputation with ``stacked_pred`` against ``stacked_y``, higher is better,
  ties 0.5; with the released value and the recomputation under every convention.
* ``conventions.csv`` --- for each convention, how many rows it reproduces.
* ``target_seed_check.csv`` --- per recipe, ``stacked_y`` against the 1B
  ``default``-seed ``primary_metric`` and the three-seed mean.
* ``summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ID = "allenai/DataDecide-eval-results"
FIT_FILE = "data/scaling_law_fit-00000-of-00001.parquet"
MACRO_FILE = "data/macro_avg-00000-of-00001.parquet"
METRIC = "primary_metric"
TARGET_PARAMS = "1B"
TARGET_SEEDS = ("default", "large aux 2", "large aux 3")
# The eight baselines of section 3.2 / Appendix C, on the default size set.
SETUPS = (
    "3_param-default",
    "2_param-default",
    "5_param-ai2",
    "3_param-1_step",
    "5_param-1_step-ai2",
    "3_param-default-helper_points",
    "3_param-default-step2=0.5",
    "3_param-default-helper_points-step2=0.5",
)
REFERENCE_CONVENTION = ("stacked_pred", "stacked_y", "higher", 0.5)
MATCH_TOLERANCE = 1e-6
# Float equality tolerance for "stacked_y is the 1B default-seed value".
EXACT_TOLERANCE = 1e-12
# Recipe names differ between the two files; they are matched once, on this task,
# by stacked_y equalling the 1B default-seed value (to EXACT_TOLERANCE) for exactly one recipe.
NAME_ANCHOR_TASK = "olmes_10_macro_avg"


def fetch(local_dir: Path | None, fit: Path | None, macro: Path | None) -> tuple[Path, Path]:
    if fit and macro:
        return fit, macro
    if local_dir:
        return local_dir / Path(FIT_FILE).name, local_dir / Path(MACRO_FILE).name
    from huggingface_hub import hf_hub_download

    return (
        Path(hf_hub_download(REPO_ID, FIT_FILE, repo_type="dataset")),
        Path(hf_hub_download(REPO_ID, MACRO_FILE, repo_type="dataset")),
    )


def one_b_targets(macro_path: Path, task: str) -> tuple[pd.Series, pd.Series, int]:
    """1B ``primary_metric`` per recipe: the default seed, and the mean over the three 1B seeds."""

    macro = pd.read_parquet(macro_path, columns=["params", "data", "task", "step", "seed", "metrics"])
    rows = macro[(macro["params"] == TARGET_PARAMS) & (macro["task"] == task) & macro["seed"].isin(TARGET_SEEDS)]
    complete = rows.groupby("step")["seed"].nunique()
    step = int(complete[complete == len(TARGET_SEEDS)].index.max())
    rows = rows[rows["step"] == step].copy()
    rows[METRIC] = [float(json.loads(text)[METRIC]) for text in rows["metrics"]]
    default = rows[rows["seed"] == "default"].set_index("data")[METRIC]
    mean = rows.groupby("data")[METRIC].mean()
    return default, mean, step


def match_names(stacked_y: pd.Series, default: pd.Series) -> dict[str, str]:
    """Map fit-file ``mix`` names to macro-file ``data`` names by exact equality of the 1B default-seed value."""

    mapping = {}
    for mix, value in stacked_y.items():
        hits = default.index[(default - value).abs() < EXACT_TOLERANCE].tolist()
        if len(hits) != 1:
            raise ValueError(f"{mix}: expected one exact match of stacked_y in the 1B default seed, found {hits}")
        mapping[str(mix)] = str(hits[0])
    return mapping


def decision_accuracy(pred: pd.Series, target: pd.Series, higher: bool, tie: float) -> float:
    scores = []
    for a, b in itertools.combinations(sorted(target.index), 2):
        p = np.sign(pred[a] - pred[b]) * (1 if higher else -1)
        t = np.sign(target[a] - target[b])
        scores.append(tie if p == 0 or t == 0 else float(p == t))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--macro", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit_path, macro_path = fetch(args.local_dir, args.fit, args.macro)
    args.out.mkdir(parents=True, exist_ok=True)

    fit = pd.read_parquet(fit_path)
    fit = fit[(fit["metric"] == METRIC) & fit["setup"].isin(SETUPS)]
    tasks = sorted(fit["task"].unique())

    anchor_default, _, _ = one_b_targets(macro_path, NAME_ANCHOR_TASK)
    anchor_y = fit[(fit["task"] == NAME_ANCHOR_TASK) & (fit["setup"] == SETUPS[0])].set_index("mix")["stacked_y"]
    mapping = match_names(anchor_y, anchor_default)

    rows, seed_rows = [], []
    for task in tasks:
        default, mean, step = one_b_targets(macro_path, task)
        anchor = fit[(fit["task"] == task) & (fit["setup"] == SETUPS[0])].set_index("mix")["stacked_y"]
        for mix, name in mapping.items():
            seed_rows.append(
                {
                    "task": task,
                    "mix": mix,
                    "data": name,
                    "one_b_step": step,
                    "stacked_y": float(anchor[mix]),
                    "one_b_default_seed": float(default[name]),
                    "one_b_three_seed_mean": float(mean[name]),
                    "stacked_y_minus_default_seed": float(anchor[mix] - default[name]),
                    "stacked_y_minus_three_seed_mean": float(anchor[mix] - mean[name]),
                }
            )
        for setup in SETUPS:
            block = fit[(fit["task"] == task) & (fit["setup"] == setup)].copy()
            block.index = block["mix"].map(mapping)
            targets = {
                "stacked_y": block["stacked_y"],
                "step_2_y": block["step_2_y"],
                "one_b_default_seed": default,
                "one_b_three_seed_mean": mean,
            }
            row: dict[str, object] = {
                "task": task,
                "setup": setup,
                "released_decision_acc": float(block["decision_acc"].iloc[0]) / 100,
                "stacked_y_equals_3_param_default_stacked_y": bool(
                    np.allclose(block["stacked_y"].sort_index(), anchor.rename(index=mapping).sort_index())
                ),
            }
            for pred_col in ("stacked_pred", "step_2_pred"):
                for target_name, target in targets.items():
                    for higher in (True, False):
                        for tie in (0.0, 0.5, 1.0):
                            key = f"{pred_col}|{target_name}|{'higher' if higher else 'lower'}|tie={tie:g}"
                            row[key] = decision_accuracy(block[pred_col], target, higher, tie)
            rows.append(row)

    table = pd.DataFrame(rows)
    conventions = [c for c in table.columns if "|" in c]
    reference = (
        f"{REFERENCE_CONVENTION[0]}|{REFERENCE_CONVENTION[1]}|{REFERENCE_CONVENTION[2]}|tie={REFERENCE_CONVENTION[3]:g}"
    )
    table["matches_reference_convention"] = (table[reference] - table["released_decision_acc"]).abs() < MATCH_TOLERANCE
    table["matched_by_any_convention"] = [
        any(abs(r[c] - r["released_decision_acc"]) < MATCH_TOLERANCE for c in conventions) for _, r in table.iterrows()
    ]
    nonmatching = table[~table["matches_reference_convention"]]
    nonmatching.to_csv(args.out / "nonmatching_rows.csv", index=False)

    counts = pd.DataFrame(
        [
            {
                "convention": c,
                "rows_matched": int(((table[c] - table["released_decision_acc"]).abs() < MATCH_TOLERANCE).sum()),
            }
            for c in conventions
        ]
    ).sort_values("rows_matched", ascending=False)
    counts.to_csv(args.out / "conventions.csv", index=False)

    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(args.out / "target_seed_check.csv", index=False)

    summary = {
        "inputs": {
            "repo": REPO_ID,
            "fit": FIT_FILE,
            "macro": MACRO_FILE,
            "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
            "macro_sha256": hashlib.sha256(macro_path.read_bytes()).hexdigest(),
            "fit_bytes": fit_path.stat().st_size,
            "macro_bytes": macro_path.stat().st_size,
        },
        "rows": int(len(table)),
        "reference_convention": reference,
        "rows_matching_reference_convention": int(table["matches_reference_convention"].sum()),
        "rows_not_matching_reference_convention": int(len(nonmatching)),
        "rows_matched_by_no_convention": int((~table["matched_by_any_convention"]).sum()),
        "best_single_convention": counts.iloc[0].to_dict(),
        "max_abs_error_reference_convention": float((table[reference] - table["released_decision_acc"]).abs().max()),
        "target_seed_check": {
            "fields": (
                "scaling_law_fit parquet: stacked_y (setup 3_param-default, metric primary_metric) against "
                "macro_avg parquet: metrics['primary_metric'] at params='1B', final step common to seeds "
                "'default', 'large aux 2', 'large aux 3'"
            ),
            "name_mapping": f"exact equality on task {NAME_ANCHOR_TASK}",
            "max_abs_stacked_y_minus_default_seed": float(seeds["stacked_y_minus_default_seed"].abs().max()),
            "tasks_where_stacked_y_equals_default_seed_exactly": sorted(
                t for t, g in seeds.groupby("task") if (g["stacked_y_minus_default_seed"].abs() < EXACT_TOLERANCE).all()
            ),
            "tasks_where_it_does_not": sorted(
                t for t, g in seeds.groupby("task") if not (g["stacked_y_minus_default_seed"].abs() < EXACT_TOLERANCE).all()
            ),
            "max_abs_stacked_y_minus_three_seed_mean": float(seeds["stacked_y_minus_three_seed_mean"].abs().max()),
            "paper_definition": (
                "arXiv:2504.11393v2 section 2.3: 'We define the target-scale winner based on mean downstream "
                "performance over 3 random seeds'"
            ),
        },
        "setups_whose_stacked_y_differs_from_3_param_default": sorted(
            set(table.loc[~table["stacked_y_equals_3_param_default_stacked_y"], "setup"])
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
