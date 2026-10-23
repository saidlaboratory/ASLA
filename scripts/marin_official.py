"""Guards for loading the Marin speedrun board (practice-record instance 11).

The official scaling board at marin-community/speedrun@31fe8028 (App.tsx, useSpeedrunData.ts, scaling.ts) groups
every run whose name contains "/" by folder and ranks a group if any of its runs exceeds 3e20 training hardware
FLOPs. ``hackable_transformer_attn_sink/std_attn`` labels its top rung "1.2b" and was silently dropped by a
loader that skipped unrecognised labels. Two guards stop that recurring:

- ``rung``: every label of an official group must be a recognised rung (with explicit aliases), else ValueError;
- ``check_members``: the loaded member set must equal the official board, apart from omissions that are named
  and justified in ``DOCUMENTED_OMISSIONS``, else ValueError.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

QUALIFY_TRAINING_FLOPS = 3e20
RUNGS = ("130m", "300m", "520m", "1_2b")
ALIASES = {"1.2b": "1_2b"}
OTHER_LADDER = {"75m", "150m"}  # parallel_llama/old_scaling, not on the scaling board
NON_LADDER_GROUPS = {"nano_arch_ablations"}  # leaf names are ablation labels, not rungs
OFFICIAL_COUNT_AT_PIN = 22
# Omissions kept so analyses before results/external/inclusion_recount.json reproduce. Any other difference fails.
DOCUMENTED_OMISSIONS = {
    "hackable_transformer_attn_sink/std_attn": "practice-record instance 11: dropped by rung-label parsing ('1.2b') "
    "from every 21-member analysis before inclusion_recount.json",
}


def group_of(run_name: str) -> str | None:
    parts = [p for p in run_name.split("/") if p]
    return "/".join(parts[:-1]) if len(parts) > 1 else None


def official_groups(runs: list[dict[str, Any]]) -> list[str]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        g = group_of(r["run_name"])
        if g is not None:
            by[g].append(r)
    return sorted(
        g for g, rs in by.items() if any((r.get("training_hardware_flops") or 0) > QUALIFY_TRAINING_FLOPS for r in rs)
    )


def rung(run_name: str, official: set[str]) -> str | None:
    """The run's rung (aliases applied). Raises on an unrecognised label; None for known non-scaling runs."""

    g = group_of(run_name)
    if g is None:
        return None
    label = run_name.rsplit("/", 1)[1].lower()
    label = ALIASES.get(label, label)
    if label in RUNGS:
        return label
    if g in official:
        raise ValueError(f"unrecognised rung label {label!r} in official group {g!r} ({run_name})")
    if label in OTHER_LADDER or g in NON_LADDER_GROUPS:
        return None
    raise ValueError(f"unrecognised rung label {label!r} in {run_name!r}; add it to marin_official.py explicitly")


def check_members(members: list[str], runs: list[dict[str, Any]], allow_documented: bool) -> list[str]:
    """Raise unless ``members`` equals the official board (less documented omissions if allowed)."""

    official = official_groups(runs)
    if len(official) != OFFICIAL_COUNT_AT_PIN:
        raise ValueError(f"official board has {len(official)} groups, expected {OFFICIAL_COUNT_AT_PIN} at the pin")
    missing = sorted(set(official) - set(members))
    extra = sorted(set(members) - set(official))
    undocumented = [m for m in missing if not (allow_documented and m in DOCUMENTED_OMISSIONS)]
    if extra or undocumented:
        raise ValueError(f"member set differs from the official board: missing {undocumented}, not official {extra}")
    return missing
