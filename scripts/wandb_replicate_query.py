"""PREDICTIONS_TASK_MARIN_NOISE_2.md, Task 1(b): find repeated runs at 130M to 1.2B on public W&B.

Runs in one step once an API key exists (``pip install wandb`` first; it is not installed here)::

    WANDB_API_KEY=... python3 scripts/wandb_replicate_query.py
    python3 scripts/wandb_replicate_query.py --dry-run   # prints the query, no network

Projects (with where each name comes from):
- marin-community/marin: 111 of 118 speedrun ``wandb_link``s in data/raw/marin_speedrun/runs.json;
- held/marin, dh2shin2-stanford-university/marin, stanford-mercury/marin: the other 7 links;
- stanford-mercury/optimizer-scaling: the Fantastic Optimizers W&B report (marin docs/reports/index.md);
- marin-community/optimizer-scaling: a follow-up Muon/AdamW report in the same index.

Server-side filter: finished runs whose display name names a speedrun size. Fields: id, name, project,
created_at, config, and every summary key matching C4-EN BPB or loss. Runs are grouped by their
configuration with seed and run-identity keys removed; a group with two or more runs is a repeat.
A group whose seeds differ is a seed replicate; one with identical seeds measures nondeterminism only.

Writes ``results/external/wandb_replicates.json`` and, if a pooled estimate exists, reruns
both-rung determination and the sign-change null with it (``run_speedrun_scale_noise.rerun_with``).
"""

from __future__ import annotations

import argparse
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

OUT = REPO / "results" / "external" / "wandb_replicates.json"

PROJECTS = (
    "marin-community/marin",
    "held/marin",
    "dh2shin2-stanford-university/marin",
    "stanford-mercury/marin",
    "stanford-mercury/optimizer-scaling",
    "marin-community/optimizer-scaling",
)
SIZE_PATTERN = r"(130m|300m|520m|1\.2b|1_2b|1-2b)"
FILTERS = {"$and": [{"state": "finished"}, {"display_name": {"$regex": SIZE_PATTERN}}]}
METRIC_PATTERN = re.compile(r"c4_en.*(bpb|loss)$", re.IGNORECASE)
PREFERRED_METRICS = ("eval/paloma/c4_en/bpb", "eval/paloma/c4_en/loss")
SEED_KEYS = re.compile(r"(^|\.)([a-z_]*seed)$", re.IGNORECASE)
# Run identity: top-level id/name only (nested ``optimizer.name`` is configuration), plus logging and paths anywhere.
IDENTITY_KEYS = re.compile(
    r"^(id|run_id|name|run_name|tags)$|(^|\.)(wandb[^.]*|tracker[^.]*|output_path|checkpoint[^.]*|[^.]*_dir|"
    r"[^.]*_path|timestamp|created_at|git[^.]*|hostname|[^.]*resume[^.]*)$|(^|\.)(wandb|tracker)\.",
    re.IGNORECASE,
)


def size_of(name: str) -> str | None:
    m = re.search(SIZE_PATTERN, name.lower())
    return None if m is None else {"1_2b": "1.2b", "1-2b": "1.2b"}.get(m.group(1), m.group(1))


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and set(v) != {"value", "desc"}:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v["value"] if isinstance(v, dict) else v
    return out


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(configuration with seed and identity keys removed, the seed keys)."""

    flat = {k: v for k, v in flatten(config).items() if not k.startswith("_")}
    seeds = {k: v for k, v in flat.items() if SEED_KEYS.search(k)}
    kept = {k: v for k, v in flat.items() if not SEED_KEYS.search(k) and not IDENTITY_KEYS.search(k)}
    return kept, seeds


def metric_of(summary: dict[str, Any]) -> tuple[str, float] | None:
    for k in PREFERRED_METRICS:
        if isinstance(summary.get(k), (int, float)):
            return k, float(summary[k])
    for k, v in sorted(summary.items()):
        if METRIC_PATTERN.search(k) and isinstance(v, (int, float)):
            return k, float(v)
    return None


def group(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repeated configurations among records {project, id, name, config, summary}."""

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        size, metric = size_of(r["name"]), metric_of(r["summary"])
        if size is None or metric is None:
            continue
        kept, seeds = split_config(r["config"])
        key = (r["project"], size, metric[0], json.dumps(kept, sort_keys=True, default=str))
        groups[key].append({"id": r["id"], "name": r["name"], "seeds": seeds, "value": metric[1]})
    out = []
    for (project, size, metric, _), members in groups.items():
        if len({m["id"] for m in members}) < 2:
            continue
        values = np.array([m["value"] for m in members], dtype=float)
        seed_sets = {json.dumps(m["seeds"], sort_keys=True, default=str) for m in members}
        out.append({
            "project": project, "size": size, "metric": metric, "runs": [m["name"] for m in members],
            "values": values.tolist(), "relative_sd": float(values.std(ddof=1) / values.mean()),
            "df": len(values) - 1, "kind": "seed replicate" if len(seed_sets) > 1 else "same-seed rerun",
        })  # fmt: skip
    return out


def pooled(groups: list[dict[str, Any]], kind: str = "seed replicate") -> dict[str, Any] | None:
    chosen = [g for g in groups if g["kind"] == kind]
    df = sum(g["df"] for g in chosen)
    if not df:
        return None
    by_size: dict[str, dict[str, float]] = {}
    for s in sorted({g["size"] for g in chosen}):
        gs = [g for g in chosen if g["size"] == s]
        d = sum(g["df"] for g in gs)
        by_size[s] = {"relative_sd": float(np.sqrt(sum(g["relative_sd"] ** 2 * g["df"] for g in gs) / d)), "df": d}
    sd = float(np.sqrt(sum(g["relative_sd"] ** 2 * g["df"] for g in chosen) / df))
    return {"kind": kind, "relative_sd": sd, "df": df, "groups": len(chosen), "by_size": by_size}


def fetch(projects: tuple[str, ...]) -> list[dict[str, Any]]:
    import wandb  # lazy: not installed or keyed on this machine

    api = wandb.Api(timeout=60)
    out = []
    for project in projects:
        for run in api.runs(project, filters=FILTERS, per_page=500):
            summary = {k: v for k, v in run.summary._json_dict.items() if METRIC_PATTERN.search(k)}
            out.append({"project": project, "id": run.id, "name": run.name, "created_at": run.created_at,
                        "config": dict(run.config), "summary": summary})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    query = {
        "projects": PROJECTS,
        "filters": FILTERS,
        "fields": ["id", "name", "project", "created_at", "config", f"summary keys matching {METRIC_PATTERN.pattern}"],
        "preferred_metrics": PREFERRED_METRICS,
        "stripped_seed_keys": SEED_KEYS.pattern,
        "stripped_identity_keys": IDENTITY_KEYS.pattern,
        "group_by": ["project", "size", "metric", "stripped config"],
    }
    if args.dry_run:
        print(json.dumps(query, indent=2))
        return
    records = fetch(PROJECTS)
    groups = group(records)
    out: dict[str, Any] = {
        "query": query,
        "runs_fetched": len(records),
        "repeated_groups": groups,
        "estimate_seed": pooled(groups, "seed replicate"),
        "estimate_same_seed": pooled(groups, "same-seed rerun"),
    }
    if out["estimate_seed"]:
        rerun_with = __import__("scripts.run_speedrun_scale_noise", fromlist=["rerun_with"]).rerun_with
        out["rerun"] = rerun_with(out["estimate_seed"]["relative_sd"])
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("query", "repeated_groups")}, indent=1, default=str))


if __name__ == "__main__":
    main()
