"""Task 4 of PREDICTIONS_TASK_REGIME_MAP.md: DataDecide's decision accuracy under other tie orders and targets.

``compute_2_class`` and ``compute_decision_accuracy`` from ``allenai/DataDecide``
are run unmodified. The only stated modifications are the tie order of the
target list and, in 4b, the input rows of the target:

* 4a --- target ties ordered by their default sort, a stable sort, the stable
  order reversed within tie blocks, and 1000 random orders within tie blocks;
* 4b --- the 1B target as the mean over the three 1B seeds, at the final step
  common to them (the paper's section 2.3 definition), passed to their
  function unmodified;
* 4c --- the rows of 5,280 that their own code does not reproduce, with every tie
  order enumerated where there are at most 100,000.

Conclusions are read on the paper's comparison, 8 variants x 11 tasks (primary
metric): C-A, no variant exceeds single-scale ranking at 750M on the macro
average; C-B, the best variant on the macro average and the ordering of the 8
by mean across tasks. Single-scale ranks by each 750M seed and averages
``compute_2_class`` over the seeds, against the same target list.

Usage::

    python scripts/run_datadecide_sensitivity.py --code-dir <clone of allenai/DataDecide>

Writes ``results/external/datadecide_sensitivity.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from itertools import permutations, product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_datadecide_code_check import (  # noqa: E402
    RAW,
    SETUPS_TESTED,
    _install_ladder_stub,
    _install_tqdm_stub,
    their_preprocessing,
)

OUT = REPO / "results" / "external" / "datadecide_sensitivity.json"
MACRO = "olmes_10_macro_avg"
TARGET_SEEDS = ("default", "large aux 2", "large aux 3")
SINGLE_SCALE = "750M"
N_RANDOM = 1000
MAX_ENUMERATION = 100_000
SEED = 20260927


def tie_blocks(values: list[float]) -> list[tuple[int, int]]:
    """(start, end) of each run of equal values, of length at least 2, in an already sorted list."""

    blocks, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            if i - start >= 2:
                blocks.append((start, i))
            start = i
    return blocks


def reorder(names: list[str], blocks: list[tuple[int, int]], orders: list[tuple[int, ...]]) -> list[str]:
    """``names`` with block k's members rearranged by ``orders[k]`` (a permutation of the block's positions)."""

    out = list(names)
    for (start, end), order in zip(blocks, orders):
        members = names[start:end]
        out[start:end] = [members[j] for j in order]
    return out


def stable_target(df: pd.DataFrame, size: str, task: str, metric: str) -> pd.DataFrame:
    """``get_perf_size_simple`` with its sort made stable (the one stated modification in 4a)."""

    from utils.dataloader import get_slice

    _slice = get_slice(df, task=task)
    _slice = _slice[(_slice["size"] == size) & (_slice["task"] == task)]
    _slice = _slice.loc[_slice.groupby("model")["step"].idxmax()]
    return _slice.sort_values(metric, ignore_index=True, kind="stable")


def predicted_lists(results: pd.DataFrame) -> dict[tuple[str, str, str], list[str]]:
    """Exactly as ``compute_decision_accuracy`` builds them."""

    return {
        key: list(group.sort_values("stacked_pred")["mix"])
        for key, group in results.groupby(["task", "metric", "setup"])
        if len(group)
    }


def score(pred: dict[tuple[str, str, str], list[str]], targets: dict[tuple[str, str], list[str]]) -> dict[Any, float]:
    from utils.scaling_laws import compute_2_class

    return {key: 100 * compute_2_class(p, targets[(key[0], key[1])]) for key, p in pred.items()}


def single_scale_lists(clean: dict[str, str]) -> dict[str, list[list[str]]]:
    """Per task, one ascending list of recipes per 750M seed (ties by name), in their group names."""

    from scripts.run_published_comparisons import TASKS, load_task

    reverse = {v: k for k, v in clean.items()}
    out = {}
    for task in TASKS:
        table = load_task(task)
        rows = table[table["params"] == SINGLE_SCALE]
        out[task] = [
            [reverse[d] for d in g.sort_values(["primary_metric", "data"], kind="stable")["data"]]
            for _, g in rows.groupby("seed")
        ]
    return out


def conclusions(
    acc: dict[Any, float], single: dict[str, list[list[str]]], targets: dict[tuple[str, str], list[str]]
) -> dict[str, Any]:
    from utils.scaling_laws import compute_2_class

    base = {
        task: 100 * float(np.mean([compute_2_class(s, targets[(task, "primary_metric")]) for s in lists]))
        for task, lists in single.items()
    }
    variants = {v: {task: acc[(task, "primary_metric", v)] for task in single} for v in SETUPS_TESTED}
    macro = {v: variants[v][MACRO] for v in SETUPS_TESTED}
    mean_across = {v: float(np.mean(list(variants[v].values()))) for v in SETUPS_TESTED}
    return {
        "single_scale_750M": base,
        "C_A_no_variant_exceeds_on_macro": all(macro[v] <= base[MACRO] for v in SETUPS_TESTED),
        "variants_exceeding_on_macro": [v for v in SETUPS_TESTED if macro[v] > base[MACRO]],
        "variant_task_rows_exceeding": int(sum(variants[v][task] > base[task] for v in SETUPS_TESTED for task in single)),
        "C_B_best_on_macro": max(SETUPS_TESTED, key=lambda v: macro[v]),
        "C_B_ordering_by_mean": sorted(SETUPS_TESTED, key=lambda v: -mean_across[v]),
    }


def prediction_tie_orders(
    group: pd.DataFrame, target: list[str], released: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Decision accuracy over orders of tied ``stacked_pred`` values, the target list held at their default.

    Enumerated when there are at most MAX_ENUMERATION orders, otherwise sampled
    MAX_ENUMERATION times.
    """

    from utils.scaling_laws import compute_2_class

    ordered = group.sort_values("stacked_pred", kind="stable")
    names = list(ordered["mix"])
    b = tie_blocks(list(ordered["stacked_pred"]))
    n_orders = math.prod(math.factorial(e - s) for s, e in b) if b else 1
    if n_orders <= MAX_ENUMERATION:
        candidates = product(*[list(permutations(range(e - s))) for s, e in b])
        exhaustive = True
    else:
        candidates = ([tuple(rng.permutation(e - s)) for s, e in b] for _ in range(MAX_ENUMERATION))
        exhaustive = False
    values = {round(100 * compute_2_class(reorder(names, b, list(o)), target), 9) for o in candidates}
    return {
        "prediction_tie_blocks": [e - s for s, e in b],
        "prediction_tie_orders": n_orders,
        "prediction_orders_exhaustive": exhaustive,
        "prediction_tie_values": [min(values), max(values)],
        "some_prediction_tie_order_reproduces": any(abs(v - released) < 1e-6 for v in values),
    }


def three_seed_target_rows(clean: dict[str, str]) -> pd.DataFrame:
    """1B rows in their layout, averaged over the three 1B seeds at the final step common to them, per task."""

    from asla.data.sources.datadecide import select_common_checkpoints

    macro = pd.read_parquet(RAW / "eval_macro_avg.parquet")
    big = macro[(macro["params"] == "1B") & macro["seed"].isin(TARGET_SEEDS)]
    parts = []
    for _, rows in big.groupby("task"):
        chosen = select_common_checkpoints(rows)
        parts.append(rows.merge(chosen, on=["params", "step"]))
    big = pd.concat(parts, ignore_index=True)
    expanded = pd.json_normalize([json.loads(m) for m in big["metrics"]])
    frame = pd.concat([big.drop(columns=["metrics"]).reset_index(drop=True), expanded], axis=1)
    numeric = [c for c in expanded.columns if pd.api.types.is_numeric_dtype(expanded[c])]
    keys = ["params", "data", "task", "step", "chinchilla"]
    mean = frame.groupby(keys, as_index=False)[numeric].mean()
    mean["seed"] = "default"  # a label only; the value is the three-seed mean
    mean["size"] = mean["params"]
    reverse = {v: k for k, v in clean.items()}
    mean["group"] = mean["data"].map(reverse)
    mean["model"] = mean["group"] + "-" + mean["params"].astype(str) + "-" + mean["chinchilla"].astype(str)
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_dir / "scaling_laws"))
    _install_tqdm_stub()
    _install_ladder_stub()
    from utils.constants.constants_recepies import DATA_NAME_CLEAN
    from utils.stats import compute_decision_accuracy, get_perf_size_simple

    commit = subprocess.run(
        ["git", "-C", str(args.code_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    df = their_preprocessing(pd.read_parquet(RAW / "eval_macro_avg.parquet"), DATA_NAME_CLEAN)
    released = pd.read_parquet(RAW / "eval_scaling_law_fit.parquet")
    results = released.drop(columns=["decision_acc"]).copy()
    keys = ["task", "metric", "setup"]
    released_acc = released.groupby(keys)["decision_acc"].first().to_dict()
    their = compute_decision_accuracy(df, results.copy(), "1B").groupby(keys)["decision_acc"].first().to_dict()

    pairs = [(t, m) for t in results["task"].unique() for m in results["metric"].unique()]
    default_targets = {(t, m): list(get_perf_size_simple(df, "1B", t, m)["group"]) for t, m in pairs}
    pred = predicted_lists(results)
    default_acc = score(pred, default_targets)
    gate = max(abs(default_acc[k] - their[k]) for k in their)
    if gate > 1e-9:
        raise RuntimeError(f"per-row path does not reproduce their function (max difference {gate})")

    stable_frames = {(t, m): stable_target(df, "1B", t, m) for t, m in pairs}
    stable_targets = {k: list(f["group"]) for k, f in stable_frames.items()}
    blocks = {k: tie_blocks(list(f[k[1]])) for k, f in stable_frames.items()}
    tied = {k: b for k, b in blocks.items() if b}

    def targets_with(orderer: Callable[[tuple[str, str], list[tuple[int, int]]], list[tuple[int, ...]]]) -> dict:
        return {k: reorder(stable_targets[k], b, orderer(k, b)) if b else default_targets[k] for k, b in blocks.items()}

    stable = {k: stable_targets[k] if blocks[k] else default_targets[k] for k in blocks}
    reversed_ = targets_with(lambda k, b: [tuple(range(e - s - 1, -1, -1)) for s, e in b])
    single = single_scale_lists(DATA_NAME_CLEAN)
    orders = {"default": default_targets, "stable": stable, "reversed": reversed_}
    by_order = {name: score(pred, t) for name, t in orders.items()}
    concl = {name: conclusions(by_order[name], single, t) for name, t in orders.items()}
    rng = np.random.default_rng(SEED)
    row_min = dict(default_acc)
    row_max = dict(default_acc)
    random_concl = []
    for _ in range(N_RANDOM):
        targets = targets_with(lambda k, b: [tuple(rng.permutation(e - s)) for s, e in b])
        acc = dict(default_acc)
        acc.update(score({k: p for k, p in pred.items() if (k[0], k[1]) in tied}, targets))
        for k in acc:
            row_min[k], row_max[k] = min(row_min[k], acc[k]), max(row_max[k], acc[k])
        c = conclusions(acc, single, targets)
        random_concl.append((c["C_A_no_variant_exceeds_on_macro"], c["C_B_best_on_macro"], tuple(c["C_B_ordering_by_mean"])))
    for name in ("stable", "reversed"):
        for k, v in by_order[name].items():
            row_min[k], row_max[k] = min(row_min[k], v), max(row_max[k], v)
    ranges = {k: row_max[k] - row_min[k] for k in row_min}
    tied_pairs = {k: sum((e - s) * (e - s - 1) // 2 for s, e in b) for k, b in blocks.items()}
    bound_ok = all(ranges[k] <= 100 * tied_pairs[(k[0], k[1])] / 300 + 1e-9 for k in ranges)
    tested = [(t, "primary_metric", s) for t in single for s in SETUPS_TESTED]

    # 4b.
    three = three_seed_target_rows(DATA_NAME_CLEAN)
    df3 = pd.concat([df[df["size"] != "1B"], three[three["task"].isin(df["task"].unique())]], ignore_index=True)
    acc3 = compute_decision_accuracy(df3, results.copy(), "1B").groupby(keys)["decision_acc"].first().to_dict()
    targets3 = {(t, m): list(get_perf_size_simple(df3, "1B", t, m)["group"]) for t, m in pairs}
    change = {k: acc3[k] - default_acc[k] for k in default_acc}
    change_tested = {k: change[k] for k in tested}

    # 4c.
    unmatched = [k for k in their if abs(their[k] - released_acc[k]) > 1e-6]
    explained = []
    for k in unmatched:
        b = blocks[(k[0], k[1])]
        n_orders = math.prod(math.factorial(e - s) for s, e in b) if b else 1
        group = results[(results["task"] == k[0]) & (results["metric"] == k[1]) & (results["setup"] == k[2])]
        entry: dict[str, Any] = {
            "task": k[0],
            "metric": k[1],
            "setup": k[2],
            "released": released_acc[k],
            "their_code": their[k],
            "target_tie_blocks": [e - s for s, e in b],
            "tie_orders": n_orders,
            "stacked_pred_ties": int(group["stacked_pred"].duplicated().sum()),
            "stacked_pred_nonfinite": int((~np.isfinite(group["stacked_pred"].astype(float))).sum()),
        }
        if b and n_orders <= MAX_ENUMERATION:
            from utils.scaling_laws import compute_2_class

            values = set()
            for orders_k in product(*[list(permutations(range(e - s))) for s, e in b]):
                values.add(
                    round(100 * compute_2_class(pred[k], reorder(stable_targets[(k[0], k[1])], b, list(orders_k))), 9)
                )
            entry["values_over_all_tie_orders"] = sorted(values)
            entry["some_tie_order_reproduces"] = any(abs(v - released_acc[k]) < 1e-6 for v in values)
        entry.update(prediction_tie_orders(group, default_targets[(k[0], k[1])], released_acc[k], rng))
        explained.append(entry)

    ordering_default = concl["default"]["C_B_ordering_by_mean"]
    out = {
        "code": {"repository": "allenai/DataDecide", "commit": commit},
        "gate_per_row_path_max_difference": gate,
        "task_4a": {
            "target_groups": len(blocks),
            "target_groups_with_ties": len(tied),
            "rows_affected": int(sum((k[0], k[1]) in tied for k in default_acc)),
            "macro_average_has_ties": any(k[0] == MACRO for k in tied),
            "max_row_range_pp": max(ranges.values()),
            "max_tied_pairs": max(tied_pairs.values()),
            "range_within_tied_pair_bound": bound_ok,
            "tested_rows": [
                {
                    "task": k[0],
                    "setup": k[2],
                    "default": default_acc[k],
                    "min": row_min[k],
                    "max": row_max[k],
                    "stable": by_order["stable"][k],
                    "reversed": by_order["reversed"][k],
                }
                for k in tested
                if ranges[k] > 0
            ],
            "conclusions": concl,
            "random_orders": {
                "n": N_RANDOM,
                "C_A_holds_in": int(sum(c[0] for c in random_concl)),
                "best_on_macro": sorted({c[1] for c in random_concl}),
                "orderings_distinct": len({c[2] for c in random_concl}),
                "orderings_equal_to_default": int(sum(list(c[2]) == ordering_default for c in random_concl)),
            },
        },
        "task_4b": {
            "definition": (
                "1B target = mean over seeds default, large aux 2, large aux 3 at the final step common to them per task"
            ),
            "all_rows": {
                "n": len(change),
                "median_abs_change_pp": float(np.median(np.abs(list(change.values())))),
                "max_abs_change_pp": float(np.max(np.abs(list(change.values())))),
                "rows_changed": int(sum(abs(v) > 1e-9 for v in change.values())),
            },
            "tested_rows": {
                "median_abs_change_pp": float(np.median(np.abs(list(change_tested.values())))),
                "max_abs_change_pp": float(np.max(np.abs(list(change_tested.values())))),
                "rows": [
                    {"task": k[0], "setup": k[2], "implemented": default_acc[k], "three_seed_mean": acc3[k]} for k in tested
                ],
            },
            "conclusions": conclusions(acc3, single, targets3),
        },
        "task_4c": {"unmatched_rows": explained},
    }
    c = out["task_4a"]["conclusions"]
    out["predictions"] = {
        "P4a1_macro_invariant": not out["task_4a"]["macro_average_has_ties"],
        "P4a2_range_within_bound_and_4_3": bound_ok and out["task_4a"]["max_row_range_pp"] <= 100 * 4 / 300 + 1e-9,
        "P4a3_conclusions_unchanged": all(
            c[n]["C_A_no_variant_exceeds_on_macro"] == c["default"]["C_A_no_variant_exceeds_on_macro"]
            and c[n]["C_B_best_on_macro"] == c["default"]["C_B_best_on_macro"]
            and c[n]["C_B_ordering_by_mean"] == ordering_default
            for n in c
        )
        and out["task_4a"]["random_orders"]["orderings_equal_to_default"] == N_RANDOM
        and len(out["task_4a"]["random_orders"]["best_on_macro"]) == 1
        and out["task_4a"]["random_orders"]["C_A_holds_in"] in (0, N_RANDOM),
        "P4b1_median_change_at_least_1pp": out["task_4b"]["tested_rows"]["median_abs_change_pp"] >= 1.0,
        "P4b2_C_A_holds": out["task_4b"]["conclusions"]["C_A_no_variant_exceeds_on_macro"],
        "P4c_all_tied_and_reproducible": all(
            e["target_tie_blocks"] and e.get("some_tie_order_reproduces", False) for e in explained
        ),
        "P4c_followup_all_reproduced_by_prediction_tie_order": all(
            e["some_prediction_tie_order_reproduces"] for e in explained
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out["predictions"], indent=1))
    print(json.dumps({k: v for k, v in out["task_4a"].items() if k not in ("tested_rows", "conclusions")}, indent=1))


if __name__ == "__main__":
    main()
