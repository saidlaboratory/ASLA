"""Harvest the compute-multipliers scaling curves from the write-up's charts.

The Task 0 gate failed because the HuggingFace release carries only the 50
top-budget checkpoints and the cited code release 404s. The compute-scaling data
turned out to be public anyway: the write-up embeds Datawrapper charts, and
Datawrapper exposes each chart's underlying table at
``datawrapper.dwcdn.net/<id>/<version>/dataset.csv``.

Seven charts, all reachable. Three carry OLMES score against compute across all
five budgets with +/-1 sd bands, which recovers per-cell seed dispersion:

* ``djBI7`` --- seven recipes on FineWeb-Edu (the class H2 calls crossover-prone)
* ``lm6Jz`` --- seven corpora under OLMo-2 (the class H2 calls scale-stable)
* ``TrYq2`` --- the 2019/2025 corner 2x2

This is aggregate data, not per-run: each cell is a seed mean with a band, so it
supports ranking and resolvability analysis but not the cell-wise seed bootstrap
our primary estimand uses. That limitation is recorded in the output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "compute_multipliers" / "datawrapper"

# Chart id -> (version, what it plots). Recovered from the write-up's embeds.
CHARTS = {
    "djBI7": (6, "OLMES vs compute, seven recipes on FineWeb-Edu"),
    "lm6Jz": (6, "OLMES vs compute, seven corpora under OLMo-2"),
    "TrYq2": (2, "OLMES vs compute, 2019/2025 corner 2x2"),
    "Cnl63": (2, "7x7 interaction grid (single seed)"),
    "KNa3I": (6, "7x7 interaction grid, alternate scoring"),
    "CzRR5": (2, "recipe compute multipliers at 1e19"),
    "k03qL": (1, "corpus compute multipliers at 1e19"),
}
SCALING_CHARTS = ("djBI7", "lm6Jz")
# A pair counts as resolved when its gap exceeds this many standard errors of
# the difference, matching the two-sigma convention used elsewhere here.
RESOLVE_SIGMAS = 2.0


def fetch(out_dir: Path) -> dict[str, int]:
    """Download each chart's dataset.csv. Returns id -> byte count."""

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for chart, (version, _) in CHARTS.items():
        url = f"https://datawrapper.dwcdn.net/{chart}/{version}/dataset.csv"
        target = out_dir / f"{chart}.csv"
        subprocess.run(["curl", "-sL", url, "-o", str(target)], check=True, timeout=120)
        sizes[chart] = target.stat().st_size
    return sizes


def _candidates(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "compute" and "sd" not in c]


def _sd(frame: pd.DataFrame, name: str, index: int) -> float:
    """Recover the seed standard deviation from the plotted +/-1 sd band."""

    return float((frame[f"{name} +1 sd"].iloc[index] - frame[f"{name} −1 sd"].iloc[index]) / 2)


def seeds_required_curve(frame: pd.DataFrame, seed_counts: tuple[int, ...] = (3, 5, 10, 20, 30)) -> list[dict[str, Any]]:
    """How many endpoint pairs become resolvable at larger seed budgets?

    The bands give the dispersion of a cell estimate at the suite's three seeds,
    so scaling by sqrt(3/n) gives the standard error at ``n`` seeds. The
    threshold is Student-t on 2n-2 degrees of freedom with a Bonferroni
    correction over all pairs --- the same convention as
    :mod:`asla.analysis.abstention`, and for the same reason: sigma is estimated
    from few points, so the Gaussian quantile is anti-conservative here.

    This converts "the axis is underpowered" into a concrete seed budget, which
    is what a request for more data has to specify.
    """

    from scipy.stats import t as student_t

    names = _candidates(frame)
    end = len(frame) - 1
    pairs = list(itertools.combinations(names, 2))
    alpha = 0.05 / len(pairs)

    rows = []
    for n_seeds in seed_counts:
        critical = float(student_t.ppf(1 - alpha / 2, max(2 * n_seeds - 2, 1)))
        resolvable = 0
        for left, right in pairs:
            gap = abs(frame[left].iloc[end] - frame[right].iloc[end])
            se_at_three = float(np.hypot(_sd(frame, left, end), _sd(frame, right, end)))
            se = se_at_three * np.sqrt(3.0 / n_seeds)
            if gap > critical * se:
                resolvable += 1
        rows.append(
            {
                "seeds_per_cell": n_seeds,
                "resolvable_pairs": resolvable,
                "n_pairs": len(pairs),
                "fraction_resolvable": resolvable / len(pairs),
            }
        )
    return rows


def analyse_axis(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    """Ranking reversals and resolvability against the top budget.

    Two counts are reported per fitting budget and they answer different
    questions. The *raw* count is how many pairs change order between that
    budget and the endpoint. The *resolved* count restricts to pairs whose gap
    exceeds two standard errors at BOTH ends, so a reversal cannot be seed noise
    at either side. Reporting only the raw count would attribute noise to
    crossover, which is the distinction this project's estimand exists to make.
    """

    names = _candidates(frame)
    end = len(frame) - 1
    pairs = list(itertools.combinations(names, 2))

    by_budget = []
    for index in range(end):
        raw = 0
        resolved_total = 0
        resolved_reversed = 0
        for left, right in pairs:
            before = frame[left].iloc[index] > frame[right].iloc[index]
            after = frame[left].iloc[end] > frame[right].iloc[end]
            if before != after:
                raw += 1
            gap_before = abs(frame[left].iloc[index] - frame[right].iloc[index])
            gap_after = abs(frame[left].iloc[end] - frame[right].iloc[end])
            se_before = float(np.hypot(_sd(frame, left, index), _sd(frame, right, index)))
            se_after = float(np.hypot(_sd(frame, left, end), _sd(frame, right, end)))
            if gap_before > RESOLVE_SIGMAS * se_before and gap_after > RESOLVE_SIGMAS * se_after:
                resolved_total += 1
                if before != after:
                    resolved_reversed += 1
        by_budget.append(
            {
                "fit_budget": float(frame["compute"].iloc[index]),
                "lever_arm": float(frame["compute"].iloc[end] / frame["compute"].iloc[index]),
                "n_pairs": len(pairs),
                "raw_reversals": raw,
                "raw_reversal_rate": raw / len(pairs),
                "resolved_pairs": resolved_total,
                "resolved_reversals": resolved_reversed,
                "resolved_reversal_rate": (resolved_reversed / resolved_total if resolved_total else None),
            }
        )

    values = np.asarray([float(frame[n].iloc[end]) for n in names])
    sds = np.asarray([_sd(frame, n, end) for n in names])
    endpoint_gaps = [abs(frame[a].iloc[end] - frame[b].iloc[end]) for a, b in pairs]
    se_diffs = [float(np.hypot(_sd(frame, a, end), _sd(frame, b, end))) for a, b in pairs]

    return {
        "axis": label,
        "n_candidates": len(names),
        "candidates": names,
        "budgets": [float(b) for b in frame["compute"]],
        "by_fit_budget": by_budget,
        "seeds_required_curve": seeds_required_curve(frame),
        "endpoint": {
            "spread": float(values.max() - values.min()),
            "median_seed_sd": float(np.median(sds)),
            "spread_in_seed_sds": float((values.max() - values.min()) / np.median(sds)),
            "median_gap": float(np.median(endpoint_gaps)),
            "min_gap": float(np.min(endpoint_gaps)),
            "median_se_of_difference": float(np.median(se_diffs)),
            "pairs_inside_seed_noise": int(sum(1 for g, s in zip(endpoint_gaps, se_diffs) if g < RESOLVE_SIGMAS * s)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--fetch", action="store_true", help="re-download the chart datasets")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "external" / "compute_multipliers_curves.json",
    )
    args = parser.parse_args()

    sizes = (
        fetch(args.raw)
        if args.fetch
        else {chart: (args.raw / f"{chart}.csv").stat().st_size for chart in CHARTS if (args.raw / f"{chart}.csv").exists()}
    )

    recipe = pd.read_csv(args.raw / "djBI7.csv")
    corpus = pd.read_csv(args.raw / "lm6Jz.csv")

    payload: dict[str, Any] = {
        "provenance": {
            "source": "https://www.dwarkesh.com/p/pretraining-progress-is-mostly-data",
            "mechanism": (
                "Datawrapper exposes each embedded chart's underlying table at "
                "datawrapper.dwcdn.net/<id>/<version>/dataset.csv. All seven embeds in the "
                "write-up resolve."
            ),
            "charts": {c: {"version": v, "plots": d} for c, (v, d) in CHARTS.items()},
            "bytes_per_chart": sizes,
            "metric": "OLMES score (higher is better)",
        },
        "limitations": {
            "aggregate_not_per_run": (
                "Each cell is a seed mean with a +/-1 sd band, not per-seed rows. This supports "
                "ranking, reversal and resolvability analysis, but NOT the cell-wise seed "
                "bootstrap our primary estimand uses, so mis-selection rates from this source "
                "are not comparable to the DataDecide numbers."
            ),
            "one_metric": (
                "OLMES only. The metric-conditional result in SELECTION_STABILITY.md shows the "
                "answer can depend on the endpoint, so a single metric is a real restriction."
            ),
            "five_budgets_seven_candidates": (
                "21 pairs per axis. Reversal counts on 21 pairs are coarse, and the resolved subset is smaller still."
            ),
        },
        "recipe_axis": analyse_axis(recipe, "recipe (on FineWeb-Edu)"),
        "corpus_axis": analyse_axis(corpus, "corpus (under OLMo-2)"),
    }

    recipe_end = payload["recipe_axis"]["endpoint"]
    corpus_end = payload["corpus_axis"]["endpoint"]
    payload["class_contrast"] = {
        "raw_reversals_at_max_lever_arm": {
            "recipe": payload["recipe_axis"]["by_fit_budget"][0]["raw_reversals"],
            "corpus": payload["corpus_axis"]["by_fit_budget"][0]["raw_reversals"],
            "n_pairs": payload["recipe_axis"]["by_fit_budget"][0]["n_pairs"],
        },
        "resolved_reversals_total": {
            "recipe": sum(b["resolved_reversals"] for b in payload["recipe_axis"]["by_fit_budget"]),
            "recipe_resolved_pairs": sum(b["resolved_pairs"] for b in payload["recipe_axis"]["by_fit_budget"]),
            "corpus": sum(b["resolved_reversals"] for b in payload["corpus_axis"]["by_fit_budget"]),
            "corpus_resolved_pairs": sum(b["resolved_pairs"] for b in payload["corpus_axis"]["by_fit_budget"]),
        },
        "seeds_needed_for_half_the_pairs": {
            axis: next(
                (
                    row["seeds_per_cell"]
                    for row in payload[f"{axis}_axis"]["seeds_required_curve"]
                    if row["fraction_resolvable"] >= 0.5
                ),
                None,
            )
            for axis in ("recipe", "corpus")
        },
        "endpoint_spread_in_seed_sds": {
            "recipe": recipe_end["spread_in_seed_sds"],
            "corpus": corpus_end["spread_in_seed_sds"],
        },
        "verdict": (
            "UNDERPOWERED on the recipe axis. Raw reversal counts favour H2's direction at the "
            "largest lever arm, but once restricted to pairs resolvable against seed noise at "
            "both ends the recipe axis retains at most one comparison per budget, which cannot "
            "support a rate. The cause is measurable: recipes span "
            f"{recipe_end['spread_in_seed_sds']:.1f} seed standard deviations at the endpoint "
            f"against the corpora's {corpus_end['spread_in_seed_sds']:.1f}, so "
            f"{recipe_end['pairs_inside_seed_noise']} of 21 recipe pairs sit inside the noise "
            f"versus {corpus_end['pairs_inside_seed_noise']} of 21 corpus pairs. Six years of "
            "corpus work simply produced more spread than six years of recipe work, which is "
            "the write-up's own headline and is what makes the recipe axis hard to power at "
            "seven candidates."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["class_contrast"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
