"""Task 0 of PREDICTIONS_TASK_CALIBRATED_INFERENCE.md: are seeds coupled across scales?

DataDecide's seeds control "initialization and data order". If a seed label fixes
the same order or initialization at every size, a recipe's seed deviations at a
small scale and at the target share a component. Single-scale ranking and the
target reference then share noise, which flatters single-scale ranking.

For every pair of scales, the statistic is the correlation, across (recipe,
label), of each seed's deviation from its cell mean at one scale with the same
label's deviation at the other. The null permutes labels within each recipe at
one scale. Labels are matched by name, so below 1B all three labels pair, and
against 1B only ``default`` does.

Writes ``results/target_scoring/seed_coupling.json``.
"""

from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS = {
    "c4_en_bits_per_token": "datadecide_runs.parquet",
    "olmes_macro_error": "datadecide_runs_olmes_macro_error.parquet",
}
N_PERMUTATIONS = 5000
SEED = 20260923
EXCLUDED = ("750M",)  # off the 5xC trajectory, as everywhere else in the project
FOCUS_PAIRS = (("300M", "530M"), ("150M", "530M"), ("300M", "1B"), ("530M", "1B"))


def deviations(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed deviation from its cell mean, indexed by (recipe, label), one column per scale."""

    frame = frame[~frame["scale_label"].isin(EXCLUDED)].copy()
    frame["deviation"] = frame["bpb"] - frame.groupby(["intervention", "scale_label"])["bpb"].transform("mean")
    return frame.pivot_table(index=["intervention", "seed_label"], columns="scale_label", values="deviation")


def coupling(table: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict[str, Any] | None:
    pair = table[[a, b]].dropna()
    recipes = sorted(pair.index.get_level_values(0).unique())
    blocks = [pair.loc[r] for r in recipes]
    labels_per_recipe = {len(block) for block in blocks}
    if len(pair) < 10 or labels_per_recipe == {1}:
        return None
    x = np.concatenate([block[a].to_numpy() for block in blocks])
    y_blocks = [block[b].to_numpy() for block in blocks]
    observed = float(np.corrcoef(x, np.concatenate(y_blocks))[0, 1])
    draws = []
    for _ in range(N_PERMUTATIONS):
        shuffled = np.concatenate([block[rng.permutation(len(block))] for block in y_blocks])
        draws.append(np.corrcoef(x, shuffled)[0, 1])
    draws_array = np.asarray(draws)
    n_labels = max(labels_per_recipe)
    return {
        "rho": observed,
        "p_two_sided": float((np.abs(draws_array) >= abs(observed)).mean()),
        "n_observations": int(len(pair)),
        "labels_matched": n_labels,
        "distinct_permutations_per_recipe": len(list(permutations(range(n_labels)))),
    }


def summarise(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(results.values())
    rhos = np.array([r["rho"] for r in rows])
    ps = np.array([r["p_two_sided"] for r in rows])
    return {
        "n_scale_pairs": len(rows),
        "median_rho": float(np.median(rhos)),
        "mean_rho": float(rhos.mean()),
        "median_p": float(np.median(ps)),
        "fraction_p_below_0_05": float((ps < 0.05).mean()),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {
        "documentation": (
            "arXiv:2504.11393v2: '3 random seeds for initialization and data order'. Labels below 1B are "
            "default / small aux 2 / small aux 3; at 1B default / large aux 2 / large aux 3. The paper does not "
            "say whether a label fixes order or initialization across sizes."
        ),
        "method": "correlation of within-cell seed deviations across (recipe, label); null permutes labels within recipe",
        "n_permutations": N_PERMUTATIONS,
        "metrics": {},
    }
    for metric, file in METRICS.items():
        table = deviations(pd.read_parquet(REPO / "data" / file))
        # Index 0 is ``default`` at every scale; indices 1 and 2 share names below 1B only.
        table.index = table.index.set_levels(table.index.levels[1].astype(str), level=1)
        scales = [s for s in table.columns]
        below = [s for s in scales if s != "1B"]
        all_labels = {f"{a}|{b}": r for a, b in combinations(below, 2) if (r := coupling(table, a, b, rng))}
        default_only = table.xs("default", level=1, drop_level=False)
        focus = {}
        for a, b in FOCUS_PAIRS:
            source = table if "1B" not in (a, b) else default_only
            focus[f"{a}|{b}"] = coupling(source, a, b, rng) if "1B" not in (a, b) else _default_coupling(table, a, b)
        out["metrics"][metric] = {
            "all_labels_below_1B": {"pairs": all_labels, "summary": summarise(all_labels)},
            "focus_pairs": focus,
        }
    c4 = out["metrics"]["c4_en_bits_per_token"]["all_labels_below_1B"]["summary"]
    err = out["metrics"]["olmes_macro_error"]["all_labels_below_1B"]["summary"]
    coupled = any(s["median_p"] < 0.05 and s["median_rho"] > 0 for s in (c4, err))
    out["decision"] = {
        "coupling_detected": coupled,
        "rule": "median permutation p < 0.05 with positive median rho, on either metric",
        "consequence": "seed random effect in the noise model" if coupled else "independent noise across scales",
    }
    path = REPO / "results" / "target_scoring" / "seed_coupling.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({m: v["all_labels_below_1B"]["summary"] for m, v in out["metrics"].items()}, indent=1))
    print(json.dumps({m: v["focus_pairs"] for m, v in out["metrics"].items()}, indent=1))
    print(out["decision"])


def _default_coupling(table: pd.DataFrame, a: str, b: str) -> dict[str, Any] | None:
    """Against 1B only ``default`` shares a label. With one label per recipe there is no within-recipe
    permutation; the deviations still sum to zero within each cell, so the correlation of the default
    seed's deviations across recipes is tested against a sign-flip-free Fisher z interval instead."""

    pair = table.xs("default", level=1)[[a, b]].dropna()
    if len(pair) < 10:
        return None
    rho = float(np.corrcoef(pair[a], pair[b])[0, 1])
    se = 1 / np.sqrt(len(pair) - 3)
    from scipy import stats

    z = float(np.arctanh(rho))
    return {
        "rho": rho,
        "p_two_sided": float(2 * stats.norm.sf(abs(z) / se)),
        "n_observations": int(len(pair)),
        "labels_matched": 1,
        "test": "Fisher z across recipes (default label only)",
    }


if __name__ == "__main__":
    main()
