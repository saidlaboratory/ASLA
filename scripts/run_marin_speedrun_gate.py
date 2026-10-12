"""Task 3 of PREDICTIONS_TASK_REVIEW_RISKS.md: does the Marin speedrun leaderboard support a ladder comparison?

A submission is the run-name prefix before the final "/"; a budget is the size
label after it; budgets are shared when the labels match. Gate: at least 8
submissions sharing the same at least 4 budgets, with C4-EN bits per byte.

Writes ``results/external/marin_speedrun_gate.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "raw" / "marin_speedrun" / "runs.json"
OUT = REPO / "results" / "external" / "marin_speedrun_gate.json"
MIN_SUBMISSIONS, MIN_BUDGETS = 8, 4


def ladders(runs: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        name = run["run_name"]
        if "/" not in name or run.get("eval_paloma_c4_en_bpb") is None:
            continue
        prefix, label = name.rsplit("/", 1)
        out[prefix].add(label.lower())
    return dict(out)


def best_shared(ladder: dict[str, set[str]], min_budgets: int) -> tuple[int, list[str], list[str]]:
    """The largest number of submissions sharing one set of at least ``min_budgets`` budgets."""

    labels = sorted({b for v in ladder.values() for b in v})
    best: tuple[int, list[str], list[str]] = (0, [], [])
    for k in range(min_budgets, len(labels) + 1):
        for combo in combinations(labels, k):
            members = sorted(s for s, v in ladder.items() if set(combo) <= v)
            if len(members) > best[0]:
                best = (len(members), list(combo), members)
    return best


def compare(runs: list[dict[str, Any]], members: list[str], budgets: list[str]) -> dict[str, Any]:
    """Hold out the top shared budget; projection (power law in model FLOPs) vs single-scale at the next rung down."""

    import numpy as np

    from asla.analysis.target_scoring import PairEvidence, decomposition
    from asla.models import FitError, bpb_power_law, fit_power_law

    cells: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        if "/" not in run["run_name"] or run.get("eval_paloma_c4_en_bpb") is None:
            continue
        prefix, label = run["run_name"].rsplit("/", 1)
        if prefix in members and label.lower() in budgets:
            cells[(prefix, label.lower())].append((float(run["model_flops"]), float(run["eval_paloma_c4_en_bpb"])))
    duplicates = {f"{k[0]}|{k[1]}": len(v) for k, v in cells.items() if len(v) > 1}
    mean = {k: (float(np.mean([x for x, _ in v])), float(np.mean([y for _, y in v]))) for k, v in cells.items()}
    order = sorted(budgets, key=lambda b: np.median([mean[(m, b)][0] for m in members]))
    fit_budgets, target, below = order[:-1], order[-1], order[-2]
    truth = {m: mean[(m, target)][1] for m in members}
    single = {m: mean[(m, below)][1] for m in members}
    projected = {}
    for m in members:
        x = np.array([mean[(m, b)][0] for b in fit_budgets])
        y = np.array([mean[(m, b)][1] for b in fit_budgets])
        try:
            projected[m] = float(bpb_power_law(mean[(m, target)][0], *fit_power_law(x, y)))
        except (FitError, RuntimeError, ValueError):
            projected[m] = float("nan")

    def gaps(values: dict[str, float]) -> dict[tuple[str, str], float]:
        return {(a, b): values[a] - values[b] for a, b in combinations(sorted(values), 2)}

    truth_gaps, single_gaps, proj_gaps = gaps(truth), gaps(single), gaps(projected)
    evidence = [
        PairEvidence(left=a, right=b, gap=g, standard_error=float("nan"), welch_df=float("nan"), p_value=float("nan"),
                     determined=False, probability_observed_order_correct=float("nan"))
        for (a, b), g in truth_gaps.items()
    ]  # fmt: skip

    def wrong(pred: dict[tuple[str, str], float]) -> int:
        return sum(np.sign(pred[k]) != np.sign(v) for k, v in truth_gaps.items() if v != 0 and np.isfinite(pred[k]))

    dec = decomposition(proj_gaps, single_gaps, evidence)
    return {
        "fit_budgets": fit_budgets,
        "held_out_target": target,
        "single_scale_rung": below,
        "pairs": len(truth_gaps),
        "single_scale_wrong": int(wrong(single_gaps)),
        "projection_wrong": int(wrong(proj_gaps)),
        "projection_failed_fits": sorted(m for m, v in projected.items() if not np.isfinite(v)),
        "decomposition_projection_vs_single": {k: v for k, v in dec.items() if not k.endswith("_pairs")},
        "duplicate_cells_averaged": duplicates,
        "caveats": [
            "three fitted budgets for a three-parameter power law: an exact fit, so projection is fragile",
            "one run per cell: no seed noise estimate, so no pair is tested for resolvability; descriptive only",
            "submissions are not independent candidates: several are learning-rate variants of one architecture",
            "the target is the observed 1.2B BPB, itself a single run",
        ],
    }


def main() -> None:
    runs = json.loads(RUNS.read_text())
    ladder = ladders(runs)
    by_count: dict[int, int] = defaultdict(int)
    for v in ladder.values():
        by_count[len(v)] += 1
    n, budgets, members = best_shared(ladder, MIN_BUDGETS)
    out = {
        "source": "marin-community/speedrun data/runs.json @ 31fe8028f8caa6e47082c02de095b1fed4f517a8",
        "run_records": len(runs),
        "records_without_a_size_suffix": sum("/" not in r["run_name"] for r in runs),
        "submissions_with_size_ladders": len(ladder),
        "submissions_by_number_of_budgets": dict(sorted(by_count.items())),
        "submissions_with_at_least_4_budgets": sorted(s for s, v in ladder.items() if len(v) >= MIN_BUDGETS),
        "best_shared_set": {"submissions": n, "budgets": budgets, "members": members},
        "gate": f">= {MIN_SUBMISSIONS} submissions sharing the same >= {MIN_BUDGETS} budgets",
        "passes": n >= MIN_SUBMISSIONS,
    }
    if out["passes"]:
        out["comparison"] = compare(runs, members, budgets)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out.get("comparison"), indent=1))


if __name__ == "__main__":
    main()
