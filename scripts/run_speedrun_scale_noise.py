"""PREDICTIONS_TASK_MARIN_NOISE_2.md, Task 1: single-run noise at speedrun scales (130M to 1.2B).

(a) Fantastic Optimizers: the paper (arXiv:2509.02046) reports no seed variance
(searched; it uses an assumed acceptance threshold of 3e-3 in loss). The harvested
sweep (``data/raw/fantastic_optimizers/<optimizer>/<size>/<budget>/result.json``)
is searched for repeated configurations: identical optimizer, size, data budget
and full hyperparameter configuration under distinct W&B run ids. Each run's
configuration is the file's best configuration, with one parameter changed for
an ablation.
(c) If an estimate exists, both-rung determination on Marin and the sign-change
null are rerun with it.

Writes ``results/external/speedrun_scale_noise.json``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "data" / "raw" / "fantastic_optimizers"
OUT = REPO / "results" / "external" / "speedrun_scale_noise.json"


def runs() -> list[dict[str, Any]]:
    out = []
    for path in sorted(RAW.glob("*/*/*/result.json")):
        optimizer, size, budget = path.parts[-4], path.parts[-3], path.parts[-2]
        d = json.loads(path.read_text())
        best = dict(d.get("best_config") or {})
        base = d.get("baseline") or {}
        if base.get("loss") is not None:
            out.append({"optimizer": optimizer, "size": size, "budget": budget, "config": best, "loss": base["loss"],
                        "run": base.get("wandb_id"), "file": str(path.relative_to(REPO)), "role": "baseline"})  # fmt: skip
        for ab in d.get("ablations") or []:
            if ab.get("loss") is None:
                continue
            cfg = dict(best, **{ab["param"]: ab["value"]})
            role = f"ablation {ab['param']}"
            out.append({"optimizer": optimizer, "size": size, "budget": budget, "config": cfg, "loss": ab["loss"],
                        "run": ab.get("wandb_id"), "file": str(path.relative_to(REPO)), "role": role})  # fmt: skip
    return out


def repeats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups with the same optimizer, size, budget and configuration under more than one distinct run id."""

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r["optimizer"], r["size"], r["budget"], json.dumps(r["config"], sort_keys=True))
        groups[key].append(r)
    out = []
    for (opt, size, budget, cfg), members in groups.items():
        by_run: dict[str, float] = {}
        for m in members:
            by_run.setdefault(m["run"], m["loss"])
        if len(by_run) >= 2:
            losses = np.array(list(by_run.values()), dtype=float)
            out.append({
                "optimizer": opt, "size": size, "budget": budget, "config": json.loads(cfg),
                "runs": list(by_run), "losses": losses.tolist(),
                "relative_sd": float(losses.std(ddof=1) / losses.mean()), "df": len(losses) - 1,
            })  # fmt: skip
    return out


def name_check(records: list[dict[str, Any]]) -> dict[str, int]:
    """Run names are the configuration plus a 6-hex hash; a rerun would show one prefix under two hashes."""

    by_id: dict[str, set] = defaultdict(set)
    by_prefix: dict[str, set] = defaultdict(set)
    for r in records:
        if r["run"]:
            by_id[r["run"]].add(r["loss"])
            by_prefix[re.sub(r"-[0-9a-f]{6}$", "", r["run"])].add(r["run"])
    return {
        "ids_listed_more_than_once": sum(sum(1 for x in records if x["run"] == k) > 1 for k in by_id),
        "ids_listed_with_differing_loss": sum(len(v) > 1 for v in by_id.values()),
        "name_prefixes_with_more_than_one_hash": sum(len(v) > 1 for v in by_prefix.values()),
    }


def pooled(groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not groups:
        return None
    ss = sum(g["relative_sd"] ** 2 * g["df"] for g in groups)
    df = sum(g["df"] for g in groups)
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in groups:
        by_size[g["size"]].append(g)
    return {
        "relative_sd": float(np.sqrt(ss / df)),
        "df": int(df),
        "groups": len(groups),
        "by_size": {
            s: {
                "relative_sd": float(np.sqrt(sum(g["relative_sd"] ** 2 * g["df"] for g in v) / sum(g["df"] for g in v))),
                "df": int(sum(g["df"] for g in v)),
            }
            for s, v in by_size.items()
        },  # fmt: skip
    }


def rerun_with(sigma: float) -> dict[str, Any]:
    mn = __import__("scripts.run_marin_noise", fromlist=["marin_both"])
    rig = __import__("scripts.run_marin_rigour", fromlist=["load_cells"])
    members, cells, _ = rig.load_cells()
    return {
        "sigma_rel": sigma,
        "both_rung": mn.marin_both(members, cells, {"speedrun_scale_estimate": sigma})["speedrun_scale_estimate"],
        "sign_pattern_null": mn.sign_pattern(members, cells, sigma),
    }


def main() -> None:
    records = runs()
    found = repeats(records)
    estimate = pooled(found)
    out: dict[str, Any] = {
        "preregistration": "PREDICTIONS_TASK_MARIN_NOISE_2.md",
        "paper": {
            "source": "arXiv:2509.02046 (HTML), searched for seed / random seed / standard deviation / run-to-run",
            "reported_seed_variance": None,
            "note": "no seed variance is reported; the sweep accepts a hyperparameter change only if validation loss "
            "improves by more than 3e-3, an assumed threshold, not a measured noise level",
        },
        "sweep": {
            "result_files": len({r["file"] for r in records}),
            "runs": len({r["run"] for r in records}),
            "records": len(records),
            "sizes": sorted({r["size"] for r in records}),
            "run_name_check": name_check(records),
            "repeated_configurations": found,
            "estimate": estimate,
        },
        "smooth_fit_row_correction": (
            "marin_noise.json's 'smooth fits alone' row used two-parameter log-log fits (2 residual df each), not "
            "three-parameter fits; they do not interpolate. Its two-change pairs come from comparing fitted lines at "
            "misaligned model FLOPs across submissions. Its match to the observed pattern is not evidence of crossing "
            "because the fits inherit the observations' noise."
        ),
    }
    if estimate:
        out["rerun"] = rerun_with(estimate["relative_sd"])
        rr = out["rerun"]["both_rung"]
        out["predictions"] = {
            "PM1_estimate_in_0_1_to_0_4_pct": 0.001 <= estimate["relative_sd"] <= 0.004,
            "PM2_genuine_rate_at_most_10pct": (rr["rate_among_both_determined"] or 0.0) <= 0.10,
            "PM2_most_target_reorderings_not_both_determined": rr["genuine_reorderings"]
            < rr["target_determined_reorderings"] / 2
            if rr["target_determined_reorderings"]
            else True,
        }
    else:
        out["bracket"] = (
            "about 0.05% (borrowed) to under 0.43% (three-parameter bound); "
            "one replicate pair at 520m or 1.2b would settle it"
        )
        out["predictions"] = {"PM3_bracket_stands": True}
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out["sweep"].items() if k != "repeated_configurations"}, indent=1))
    for g in found[:20]:
        print(g["optimizer"], g["size"], g["budget"], g["losses"], round(g["relative_sd"], 5))
    if estimate:
        print(json.dumps(out["rerun"]["both_rung"], indent=1))
        sp = out["rerun"]["sign_pattern_null"]
        print(
            sp["observed_mean_changes"],
            sp["null"]["mean_changes_95"],
            sp["observed_share_two_or_more"],
            sp["null"]["share_two_or_more_95"],
        )
    print(out["predictions"])


if __name__ == "__main__":
    main()
