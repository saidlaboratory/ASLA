"""Analysis of the copied-initialisation CRN pilot (PREDICTIONS_TASK_CRN_PILOT.md), run after the 16 runs.

From ``results/crn/pilot/runs/*.json`` (4 learning rates x 4 seed indices,
evaluation loss at every checkpoint):

* per learning-rate pair, the ratio of matched-seed to mismatched-seed gap
  variance, with a bootstrap interval that resamples seed indices jointly
  across pairs;
* the resolvable fraction of the 6 pairs under matched analysis (paired t,
  3 df) and mismatched analysis (the calibrated procedure's raw Welch
  determination), both at alpha = 0.05 with Bonferroni over the pairs;
* all of it at every checkpoint, to show whether any benefit decays;
* the pre-registered decision: median ratio at the final checkpoint <= 0.5
  recommends a full experiment, otherwise the CRN program closes.

Writes ``results/crn/pilot_analysis.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = REPO / "results" / "crn" / "pilot" / "runs"
OUT = REPO / "results" / "crn" / "pilot_analysis.json"
THRESHOLD = 0.5
ALPHA = 0.05
N_BOOT = 4000
SEED = 20260927


Table = dict[tuple[float, int], dict[int, float]]


def load_runs(runs_dir: Path) -> tuple[list[float], list[int], list[int], Table]:
    """(learning-rate multipliers, seed indices, checkpoint steps, {(lr, seed): {step: loss}}); needs the full crossing."""

    table: dict[tuple[float, int], dict[int, float]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        run = json.loads(path.read_text())
        key = (float(run["config"]["lr_multiplier"]), int(run["config"]["seed_index"]))
        table[key] = {int(p["step"]): float(p["eval_loss"]) for p in run["curve"]}
    lrs = sorted({k[0] for k in table})
    seeds = sorted({k[1] for k in table})
    missing = [(m, s) for m in lrs for s in seeds if (m, s) not in table]
    if missing:
        raise SystemExit(f"incomplete crossing, missing runs: {missing}")
    steps = sorted(set.intersection(*(set(v) for v in table.values())))
    return lrs, seeds, steps, table


def pair_ratio(a: np.ndarray, b: np.ndarray, labels: np.ndarray | None = None) -> float:
    """Var(matched gaps) / Var(mismatched gaps) for two recipes' values over the same seed positions.

    ``labels`` (for bootstrap resamples) marks which positions are the same seed; mismatched gaps
    use only position pairs with different labels.
    """

    labels = np.arange(len(a)) if labels is None else labels
    matched = a - b
    mismatched = np.array([a[i] - b[j] for i, j in permutations(range(len(a)), 2) if labels[i] != labels[j]])
    denominator = mismatched.var(ddof=1) if mismatched.size > 1 else float("nan")
    return float(matched.var(ddof=1) / denominator) if denominator > 0 else float("nan")


def resolvable(values: dict[float, np.ndarray]) -> dict[str, Any]:
    """Pairs determined at alpha = 0.05 with Bonferroni: matched (paired t) and mismatched (calibrated Welch)."""

    from asla.analysis.moderation import evidence_from_cells

    pairs = list(combinations(sorted(values), 2))
    level = ALPHA / len(pairs)
    matched = {}
    for a, b in pairs:
        diff = values[a] - values[b]
        sd = diff.std(ddof=1)
        p = 1.0 if sd == 0 else float(2 * stats.t.sf(abs(diff.mean()) / (sd / np.sqrt(diff.size)), diff.size - 1))
        matched[f"{a:g}|{b:g}"] = p <= level
    evidence = evidence_from_cells({f"{m:g}": list(v) for m, v in values.items()}, alpha=ALPHA, multiplicity="bonferroni")
    mismatched = {f"{e.left}|{e.right}": bool(e.determined) for e in evidence}
    return {
        "matched_fraction": float(np.mean(list(matched.values()))),
        "mismatched_fraction": float(np.mean(list(mismatched.values()))),
        "matched": matched,
        "mismatched": mismatched,
        "alpha_per_pair": level,
    }


def at_checkpoint(lrs: list[float], seeds: list[int], table: dict, step: int, rng: np.random.Generator) -> dict[str, Any]:
    values = {m: np.array([table[(m, s)][step] for s in seeds]) for m in lrs}
    pairs = list(combinations(lrs, 2))
    ratios = {f"{a:g}|{b:g}": pair_ratio(values[a], values[b]) for a, b in pairs}
    boot_medians, boot_pairs = [], {k: [] for k in ratios}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(seeds), len(seeds))
        if len(set(pick)) < 2:
            continue
        sample = {f"{a:g}|{b:g}": pair_ratio(values[a][pick], values[b][pick], labels=pick) for a, b in pairs}
        finite = [v for v in sample.values() if np.isfinite(v)]
        if finite:
            boot_medians.append(float(np.median(finite)))
        for k, v in sample.items():
            if np.isfinite(v):
                boot_pairs[k].append(v)
    finite = [v for v in ratios.values() if np.isfinite(v)]
    return {
        "step": step,
        "pair_ratios": ratios,
        "pair_mismatched_variance": {
            f"{a:g}|{b:g}": float(
                np.var([values[a][i] - values[b][j] for i in range(len(seeds)) for j in range(len(seeds)) if i != j], ddof=1)
            )
            for a, b in pairs
        },
        "pair_ratio_ci": {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None for k, v in boot_pairs.items()
        },
        "median_ratio": float(np.median(finite)) if finite else float("nan"),
        "median_ratio_ci": [float(np.percentile(boot_medians, 2.5)), float(np.percentile(boot_medians, 97.5))]
        if boot_medians
        else None,
        "resolvable": resolvable(values),
    }


def analyse(runs_dir: Path, v_nd: float | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """``v_nd``: measured per-run nondeterminism variance; ``context``: device and scale facts to carry along."""

    lrs, seeds, steps, table = load_runs(runs_dir)
    rng = np.random.default_rng(SEED)
    step0 = {m: [table[(m, s)][0] for s in seeds] for m in lrs} if 0 in steps else None
    trained = [s for s in steps if s > 0]
    by_step = [at_checkpoint(lrs, seeds, table, s, rng) for s in trained]
    final = by_step[-1]
    medians = [c["median_ratio"] for c in by_step]
    decision = "recommend a full experiment" if final["median_ratio"] <= THRESHOLD else "close the CRN program"
    out = {
        "preregistration": "PREDICTIONS_TASK_CRN_PILOT.md",
        "learning_rate_multipliers": lrs,
        "seed_indices": seeds,
        "checkpoints": trained,
        "step0_identical_across_learning_rates": None
        if step0 is None
        else all(len(set(np.round(v, 12))) == 1 for v in zip(*step0.values())),
        "by_checkpoint": by_step,
        "final": {
            "step": final["step"],
            "median_ratio": final["median_ratio"],
            "median_ratio_ci": final["median_ratio_ci"],
            "reduction": 1 - final["median_ratio"],
            "resolvable_matched": final["resolvable"]["matched_fraction"],
            "resolvable_mismatched": final["resolvable"]["mismatched_fraction"],
        },
        "decision_rule": f"median ratio at the final checkpoint <= {THRESHOLD}",
        "decision": decision,
        "four_seeds_note": "four seeds detect only large effects; intentional (see the pre-registration)",
    }
    if v_nd is not None:
        median_mismatched = float(np.median(list(final["pair_mismatched_variance"].values())))
        out["nondeterminism"] = {
            "v_nd": v_nd,
            "matched_gap_floor_2_v_nd": 2 * v_nd,
            "floor_over_median_mismatched_variance": 2 * v_nd / median_mismatched if median_mismatched > 0 else None,
            "note": (
                "each matched gap carries two runs' nondeterminism (2 v_nd) that CRN cannot remove, so it biases the "
                "ratio upward, toward concluding CRN does not help"
            ),
        }
    if context:
        out["context"] = context
    first = by_step[0]["median_ratio"]
    out["predictions"] = {
        "PP1_first_checkpoint_at_most_0_2": bool(first <= 0.2),
        "PP2_rises_over_training_spearman_ge_0_6": bool(stats.spearmanr(range(len(medians)), medians).statistic >= 0.6)
        if len(medians) >= 3
        else None,
        "PP3_final_in_0_2_to_0_8": bool(0.2 <= final["median_ratio"] <= 0.8),
        "PP4_more_resolvable_matched": bool(
            final["resolvable"]["matched_fraction"] > final["resolvable"]["mismatched_fraction"]
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--v-nd", type=float, default=None, help="measured per-run nondeterminism variance")
    parser.add_argument("--context-json", type=Path, nargs="*", default=[], help="device/scale records to carry along")
    args = parser.parse_args()
    context = {p.name: json.loads(p.read_text()) for p in args.context_json}
    out = analyse(args.runs, args.v_nd, context)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["final"], indent=1), out["decision"])
    for c in out["by_checkpoint"]:
        print(f"step {c['step']:>6}: median ratio {c['median_ratio']:.3f}")


if __name__ == "__main__":
    main()
