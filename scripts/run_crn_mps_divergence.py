"""PREDICTIONS_TASK_CRN_PILOT_DIVERGENCE.md: full-length identical reruns on MPS against the pilot.

Reruns (seed index 0, 1x) and (seed index 0, 1.5x) for the full pilot schedule
(resumable), then compares each with the original pilot run at every
checkpoint and applies the pre-registered rule.

    nice -n 19 python scripts/run_crn_mps_divergence.py

Writes ``results/crn/pilot_laptop/divergence.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PILOT = REPO / "results" / "crn" / "pilot_laptop"
RERUN = PILOT / "rerun"
OUT = PILOT / "divergence.json"
MEASURED = (1.0, 1.5)
THRESHOLD = 0.10


def fractions(
    finals: dict[tuple[float, int], float], v: dict[float, float], lrs: list[float], seeds: list[int]
) -> dict[str, dict[str, float]]:
    """For each pair involving a measured learning rate: nondeterminism share of the matched-gap variance."""

    out = {}
    for a, b in combinations(lrs, 2):
        if a not in v and b not in v:
            continue
        gaps = np.array([finals[(a, s)] - finals[(b, s)] for s in seeds])
        var = float(gaps.var(ddof=1))
        entry = {"matched_gap_variance": var}
        for m in (a, b):
            if m in v:
                entry[f"2v_{m:g}_fraction"] = 2 * v[m] / var
        entry["decision_fraction"] = (v[a] + v[b]) / var if a in v and b in v else 2 * v.get(a, v.get(b, 0.0)) / var
        out[f"{a:g}|{b:g}"] = entry
    return out


def main() -> None:
    import torch

    torch.set_num_threads(2)
    from pilot.crn import laptop

    manifest = json.loads((PILOT / "manifest.json").read_text())
    for m in MEASURED:
        if not (RERUN / f"lr{m:g}_seed0_rerun.json").exists():
            proc = laptop.run_one(manifest["label"], manifest["device"], manifest["steps"], m, 0, RERUN, ["--tag", "rerun"])
            if proc.returncode != 0:
                raise SystemExit(proc.stderr[-2000:])
    runs = {}
    for path in (PILOT / "runs").glob("*.json"):
        r = json.loads(path.read_text())
        runs[(float(r["config"]["lr_multiplier"]), int(r["config"]["seed_index"]))] = r
    lrs = sorted({k[0] for k in runs})
    seeds = sorted({k[1] for k in runs})
    curves: dict[str, Any] = {}
    v = {}
    for m in MEASURED:
        original = runs[(m, 0)]["curve"]
        rerun = json.loads((RERUN / f"lr{m:g}_seed0_rerun.json").read_text())["curve"]
        assert [p["step"] for p in original] == [p["step"] for p in rerun]
        gaps = [abs(p["eval_loss"] - q["eval_loss"]) for p, q in zip(original, rerun)]
        curves[f"{m:g}"] = {"steps": [p["step"] for p in original], "identical_run_gap": gaps}
        v[m] = gaps[-1] ** 2 / 2
    finals = {k: r["final_eval_loss"] for k, r in runs.items()}
    per_pair = fractions(finals, v, lrs, seeds)
    worst = max(e["decision_fraction"] for e in per_pair.values())
    out = {
        "preregistration": "PREDICTIONS_TASK_CRN_PILOT_DIVERGENCE.md",
        "identical_run_gap_curves": curves,
        "final_identical_run_variance": {f"{m:g}": v[m] for m in MEASURED},
        "variance_note": "one degree of freedom per learning rate; a noisy estimate",
        "pairs": per_pair,
        "max_decision_fraction": worst,
        "decision": "negative stands; close the CRN program"
        if worst < THRESHOLD
        else "inconclusive on MPS at this training length; the unchanged launch stays ready for Colab",
        "pilot_result": {
            "median_ratio": 1.04,
            "ci": [0.29, 2.81],
            "reading": "no large benefit; does not exclude a moderate reduction",
        },
        "exploratory": "the 0.5x/0.75x pair's ratio of 0.075 is one of six pairs; exploratory, no follow-up",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: out[k] for k in ("identical_run_gap_curves", "pairs", "max_decision_fraction", "decision")}, indent=1)
    )


if __name__ == "__main__":
    main()
