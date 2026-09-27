"""Task 3 of PREDICTIONS_TASK_TRANSFER.md: does a dimensionless map built on one metric predict another's regime?

Each metric's own worlds (its shrunk truth, noise, ladder and seed count) are the
ground truth: the true excess of projection over single-scale at its short,
primary and long designs. A prediction moves a source metric's structure to the
target's measured coordinates at one design and runs 150 worlds on the target's
rung set:

* base: q1 (top-rung cell-mean noise / median target gap), q2 (recipe-specific
  misspecification / that gap), q3 (smallest-rung noise / top-rung noise), with
  the lever arm fixed by the design;
* extended: base plus q5 (floor level) and q6 (decay) of the mean curve, added
  because Task 2 identified curve shape and floor.

Signal-and-Noise bits/byte (8 rungs, one run per cell) serves as a target only:
its ladder lacks the rungs the other metrics' designs fit on.

Writes ``results/target_scoring/dimensionless_transfer.json`` and
``results/target_scoring/figures/dimensionless_transfer.png``.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import transfer_worlds as tw  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "dimensionless_transfer.json"
FIGURE = REPO / "results" / "target_scoring" / "figures" / "dimensionless_transfer.png"
SOURCES = ("c4", "olmes", "cp")
TARGETS = tw.METRICS
VARIANTS = {"base": frozenset(), "extended": frozenset({"iii"})}
N_WORLDS = 150
N_BOOT = 2000
SEED = 20260929


def designs_of(metric: str) -> dict[str, tuple[str, str]]:
    ctx = tw.context(metric)
    return tw.ladder_designs(ctx["scales"], ctx["compute"])


def testable(metric: str, design: tuple[str, str]) -> bool:
    q = tw.own_coordinates(metric, design)
    return bool(q["g"] > 0 and all(np.isfinite(q[k]) for k in ("q1", "q2", "q3")))


def own_world(metric: str, index: int) -> dict[str, float]:
    ctx = tw.context(metric)
    rng = np.random.default_rng([SEED, 1, TARGETS.index(metric), index])
    return tw.world_excess(ctx["mu"], ctx["base"].sigma, ctx, designs_of(metric), rng)


@functools.lru_cache(maxsize=None)
def _moved(source: str, target: str, design_name: str, variant: str) -> tuple[Any, Any, dict[str, Any]]:
    design = designs_of(target)[design_name]
    scales = tw.context(target)["scales"]
    mu, sigma, applied = tw.transform_to(source, tw.own_coordinates(target, design), design, scales, VARIANTS[variant])
    return mu, tw.sigma_dict(sigma), applied


def predicted_world(source: str, target: str, design_name: str, variant: str, index: int) -> float:
    mu, sigma, _ = _moved(source, target, design_name, variant)
    design = designs_of(target)[design_name]
    rng = np.random.default_rng([SEED, 2, SOURCES.index(source), TARGETS.index(target), index])
    ctx = tw.context(source)
    return tw.world_excess(mu, sigma, ctx, {design_name: design}, rng, scales=tw.context(target)["scales"])[design_name]


def summarise(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    if np.all(np.isnan(values)):
        return {"excess_pp": None, "ci": None, "sign": "undefined"}
    mean = float(np.nanmean(values))
    lo, hi = (float(v) for v in np.nanpercentile(np.nanmean(values[boot], axis=1), [2.5, 97.5]))
    return {"excess_pp": mean, "ci": [lo, hi], "sign": "wins" if hi < 0 else ("loses" if lo > 0 else "unresolved")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, N_WORLDS, size=(N_BOOT, N_WORLDS))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        own_jobs = [(m, i) for m in TARGETS for i in range(N_WORLDS)]
        own_rows = list(pool.map(own_world, *zip(*own_jobs), chunksize=25))
        pred_jobs = [
            (s, t, d, v, i)
            for t in TARGETS
            for d, design in designs_of(t).items()
            if testable(t, design)
            for s in SOURCES
            for v in VARIANTS
            for i in range(N_WORLDS)
        ]
        pred_values = list(pool.map(predicted_world, *zip(*pred_jobs), chunksize=25))
    truth: dict[str, dict[str, Any]] = {}
    for m in TARGETS:
        rows = [r for (mm, _), r in zip(own_jobs, own_rows) if mm == m]
        truth[m] = {}
        for d, design in designs_of(m).items():
            entry = summarise(np.array([r[d] for r in rows], dtype=float), boot)
            q = tw.own_coordinates(m, design)
            entry.update(
                design=list(design), coordinates={k: v for k, v in q.items() if k != "normalized_target_deviations"}
            )
            entry["testable"] = testable(m, design)
            truth[m][d] = entry
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for (s, t, d, v, _), value in zip(pred_jobs, pred_values):
        grouped.setdefault((s, t, d, v), []).append(value)
    predictions: dict[str, Any] = {}
    for (s, t, d, v), values in grouped.items():
        entry = summarise(np.array(values, dtype=float), boot)
        applied = _moved(s, t, d, v)[2]
        entry["transform"] = {k: val for k, val in applied.items() if k != "achieved"}
        entry["achieved_coordinates"] = applied["achieved"]
        target_sign = truth[t][d]["sign"]
        entry["target_sign"] = target_sign
        predicted_sign = None if entry["excess_pp"] is None else ("wins" if entry["excess_pp"] < 0 else "loses")
        entry["predicted_sign"] = predicted_sign
        entry["test"] = target_sign in ("wins", "loses") and predicted_sign is not None
        entry["match"] = entry["test"] and predicted_sign == target_sign
        predictions[f"{s}->{t}|{d}|{v}"] = entry

    def rate(variant: str, held_out: bool = True) -> dict[str, Any]:
        tests = [
            (k, e)
            for k, e in predictions.items()
            if k.endswith(f"|{variant}") and e["test"] and (k.split("->")[0] != k.split("->")[1].split("|")[0]) == held_out
        ]
        return {
            "tests": len(tests),
            "matches": sum(e["match"] for _, e in tests),
            "mismatches": [k for k, e in tests if not e["match"]],
        }

    summary = {v: {"held_out": rate(v), "self": rate(v, held_out=False)} for v in VARIANTS}
    out = {
        "design": "each target's own worlds as ground truth; each source moved to the target's coordinates per design",
        "independence": (
            "S&N bits/byte re-evaluates the same DataDecide models as C4 (the user reports 93 to 99.7% pair agreement; "
            "not re-verified this round) and borrows C4's noise; correct-prob and OLMES error share a metric family "
            "and tasks: about two to three independent tests, not four"
        ),
        "sources": list(SOURCES),
        "sn_role": "target only: its 8-rung, one-run ladder cannot host the other metrics' designs",
        "ground_truth": truth,
        "predictions": predictions,
        "summary": summary,
    }
    out["preregistered"] = {
        "P3_1_base_fails": summary["base"]["held_out"]["matches"] < summary["base"]["held_out"]["tests"],
        "P3_2_extended_transfers": summary["extended"]["held_out"]["matches"] == summary["extended"]["held_out"]["tests"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    _figure(truth, predictions)
    for m, rows in truth.items():
        for d, e in rows.items():
            print(
                m,
                d,
                e["design"],
                e["excess_pp"] if e["excess_pp"] is None else round(e["excess_pp"], 2),
                e["ci"] and [round(x, 2) for x in e["ci"]],
                e["sign"],
            )
    for k, e in predictions.items():
        print(
            f"{k:32s} {e['excess_pp'] if e['excess_pp'] is None else round(e['excess_pp'], 2)} "
            f"target {e['target_sign']} match {e['match']}"
        )
    print(json.dumps(summary, indent=1), out["preregistered"])


def _figure(truth: dict[str, Any], predictions: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    markers = {"short": "v", "primary": "o", "long": "^"}
    colors = {"c4": "C0", "olmes": "C3", "cp": "C1", "sn": "C2"}
    for ax, variant in zip(axes, VARIANTS):
        for key, e in predictions.items():
            if not key.endswith(f"|{variant}") or e["excess_pp"] is None:
                continue
            pair, d, _ = key.split("|")
            s, t = pair.split("->")
            own = truth[t][d]["excess_pp"]
            if own is None or s == t:
                continue
            ax.scatter(own, e["excess_pp"], marker=markers[d], color=colors[s], edgecolor="k", s=50)
        lim = ax.get_xlim()
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("target's own true excess (pp)")
        ax.set_title(f"{variant} coordinates")
    axes[0].set_ylabel("predicted excess from source map (pp)")
    for s, c in colors.items():
        if s in SOURCES:
            axes[1].scatter([], [], color=c, edgecolor="k", label=f"source {tw.LABELS[s]}")
    for d, m in markers.items():
        axes[1].scatter([], [], marker=m, color="w", edgecolor="k", label=d)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
