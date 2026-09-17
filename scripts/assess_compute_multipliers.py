"""Task 0 gate: is the compute-multipliers ladder public?

The suite would close this project's largest scoping gap. Every decision-accuracy
number we have is measured on DataDecide, which varies *data recipes* --- the
class our own H2 predicts is the scale-stable negative control. compute-multipliers
crosses seven *recipes* (the predicted crossover-prone class) with seven corpora
under one protocol with three seeds per cell, which is the 2x2 the class-dependence
prediction needs.

This script records what is actually retrievable, so the gate verdict rests on a
reproducible artifact rather than on a chat transcript. It downloads nothing but
small metadata files (all under 40 KB) and never touches weights.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "compute_multipliers"

# Budgets the write-up states the campaign ran, for comparison against what the
# public artifact actually carries.
REPORTED_BUDGETS = (1e17, 3.16e17, 1e18, 3.16e18, 1e19)
REPORTED_RUNS = 1397
# A usable ladder needs at least this many distinct fitting budgets below a
# held-out target, with replicate seeds, per the gate in the task prompt.
MIN_FIT_BUDGETS = 3
MIN_SEEDS = 2


def _http_status(url: str) -> int:
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", url],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int(result.stdout.strip() or 0)
    except (subprocess.SubprocessError, ValueError):  # pragma: no cover - network
        return 0


def assess(checkpoints_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(checkpoints_csv)
    budgets = sorted(float(b) for b in frame["budget_flops"].unique())
    seeds = sorted(int(s) for s in frame["seed"].unique())
    recipes = sorted(str(r) for r in frame["recipe"].unique())
    corpora = sorted(str(c) for c in frame["corpus"].unique())

    # Complete cells: a (recipe, corpus, budget) with at least MIN_SEEDS seeds.
    grouped = frame.groupby(["recipe", "corpus", "budget_flops"])["seed"].nunique()
    complete = int((grouped >= MIN_SEEDS).sum())

    has_ladder = len(budgets) >= MIN_FIT_BUDGETS + 1
    return {
        "n_rows_public": int(len(frame)),
        "n_runs_reported_in_campaign": REPORTED_RUNS,
        "fraction_of_campaign_public": len(frame) / REPORTED_RUNS,
        "budgets_public": budgets,
        "budgets_reported": list(REPORTED_BUDGETS),
        "n_budgets_public": len(budgets),
        "n_budgets_reported": len(REPORTED_BUDGETS),
        "recipes": recipes,
        "corpora": corpora,
        "seeds": seeds,
        "n_recipes": len(recipes),
        "n_corpora": len(corpora),
        "n_seeds": len(seeds),
        "cells_with_replicates": complete,
        "metrics_available": [
            column for column in ("native_heldout_nll", "olmes10", "heldout7_mean_nll", "alt8") if column in frame.columns
        ],
        "roles": {str(k): int(v) for k, v in frame["roles"].value_counts().items()},
        "has_usable_ladder": bool(has_ladder),
        "lever_arms_available": ([max(budgets) / b for b in budgets[:-1]] if has_ladder else []),
    }


def metric_agreement(frame: pd.DataFrame) -> dict[str, Any]:
    """What the single public budget CAN answer: do metrics rank candidates alike?

    This needs no ladder --- it compares rankings within one budget --- so it
    survives the gate failure. It is an independent check of the ranking
    inversion the write-up reports between educational-filtered and broad-web
    corpora.
    """

    from scipy import stats

    metrics = {
        "native_heldout_nll": True,
        "olmes10": False,
        "heldout7_mean_nll": True,
        "alt8": False,
    }
    axes = {
        "recipe": (frame[frame["roles"].str.contains("model_axis")], "recipe"),
        "corpus": (frame[frame["roles"].str.contains("data_axis")], "corpus"),
    }

    report: dict[str, Any] = {}
    for axis_name, (subset, key) in axes.items():
        if subset.empty or subset[key].nunique() < 2:
            report[axis_name] = {"n_candidates": int(subset[key].nunique()) if not subset.empty else 0}
            continue
        means = {metric: subset.groupby(key)[metric].mean() for metric in metrics if metric in subset}
        ranks = {metric: series.rank(ascending=metrics[metric]) for metric, series in means.items()}
        winners = {metric: (series.idxmin() if metrics[metric] else series.idxmax()) for metric, series in means.items()}
        names = sorted(ranks)
        pairwise = {}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                pairwise[f"{left}|{right}"] = float(stats.spearmanr(ranks[left], ranks[right]).statistic)
        report[axis_name] = {
            "n_candidates": int(subset[key].nunique()),
            "n_seeds_per_candidate": int(subset.groupby(key)["seed"].nunique().min()),
            "winner_by_metric": {k: str(v) for k, v in winners.items()},
            "n_distinct_winners": len(set(winners.values())),
            "pairwise_rank_spearman": pairwise,
            "min_pairwise_spearman": min(pairwise.values()),
            "max_pairwise_spearman": max(pairwise.values()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "external" / "compute_multipliers_gate.json")
    args = parser.parse_args()

    checkpoints = args.raw / "checkpoints.csv"
    if not checkpoints.exists():
        raise SystemExit(f"{checkpoints} not found; fetch the metadata first (see the module docstring)")

    coverage = assess(checkpoints)
    frame = pd.read_csv(checkpoints)
    agreement = metric_agreement(frame)
    sources = {
        "huggingface_model_repo": _http_status("https://huggingface.co/api/models/j23h67/compute-multipliers-checkpoints"),
        "github_code_release": _http_status("https://api.github.com/repos/Lunar-Society/compute-multipliers"),
        "github_runs_csv": _http_status("https://raw.githubusercontent.com/Lunar-Society/compute-multipliers/main/runs.csv"),
    }

    gate_passes = bool(coverage["has_usable_ladder"])
    payload: dict[str, Any] = {
        "coverage": coverage,
        "metric_agreement_at_top_budget": agreement,
        "source_http_status": sources,
        "gate": {
            "requirement": (
                f"a fit ladder of >={MIN_FIT_BUDGETS} budgets plus a held-out target, "
                f"with >={MIN_SEEDS} seeds, on the recipe axis"
            ),
            "passes": gate_passes,
            "verdict": "PROCEED" if gate_passes else "STOP: only top-budget endpoints are public",
            "reasoning": (
                "The campaign ran five budgets (1e17 to 1e19) with three seeds per cell, which "
                "is exactly the ladder this task needs. The public artifact is the 50 final "
                "checkpoints at the top budget only: every released row sits at 1e19 FLOPs, so "
                "there is no compute ladder to fit and no held-out target above the fit range. "
                "The run record (runs.csv) is cited as living in the code release, but that "
                "repository returns 404 and the Lunar-Society organisation's two public "
                "repositories are unrelated. The per-run 'curve' field in each run.json is a "
                "within-run token-horizon trajectory at fixed model size, not a ladder across "
                "compute budgets, so it cannot substitute."
            ),
            "what_would_unblock": (
                "the released runs.csv covering all 1,397 runs across the five budgets, or any "
                "per-budget table with seeds preserved"
            ),
        },
        "notes": {
            "reportable_flag": (
                "run.json carries experiment_stage=nonreportable_pilot and reportable=false. The "
                "model card documents these as launch-path labels rather than a verdict on the "
                "run; checked against the source rather than assumed."
            ),
            "what_survives_the_gate": (
                "Ranking agreement across metrics needs no ladder, so it is computed here and "
                "reported despite the STOP. It independently reproduces the write-up's ranking "
                "inversion, and locates it on the corpus axis rather than the recipe axis."
            ),
            "weights_not_downloaded": (
                "Only checkpoints.csv (19 KB), audit.jsonl (34 KB) and README.md (14 KB) were "
                "fetched. The 50 safetensors files are model weights and were not retrieved."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
