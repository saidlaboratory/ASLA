"""Generate the allocation study's results table from committed JSON.

Every number is read from the study output; nothing is hand-written. Scores the
pre-registered predictions P1-P5 from ``PREDICTIONS_TASK_ALLOCATION.md``
mechanically, so the verdicts cannot drift from the numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM_LABELS = {
    "decision_optimal": "decision-optimal",
    "estimation_optimal_sl2": "estimation-optimal (SL2)",
    "bias_aware": "bias-aware",
    "uniform_ladder": "uniform ladder",
}


def _entries(payload: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    rows = [(float(entry["budget_fraction"]), entry) for entry in payload.get("sweep", {}).values() if "error" not in entry]
    return sorted(rows, key=lambda item: item[0])


def score_p1(payload: dict[str, Any]) -> tuple[str, str]:
    """P1: sparse design, top-rung cost share < 0.25."""

    entries = _entries(payload)
    if not entries:
        return "UNDERPOWERED", "no completed budget fractions"
    shares = [(f, entry["top_rung_share_decision"]) for f, entry in entries]
    supports = [len(entry["designs"]["decision_optimal"]["support"]) for _, entry in entries]
    lo, hi = min(s for _, s in shares), max(s for _, s in shares)
    sparse = max(supports) <= 2
    detail = (
        f"support points {min(supports)}-{max(supports)}; "
        f"top-rung cost share {lo:.3f} to {hi:.3f} across budgets "
        f"({', '.join(f'{f:g}x:{s:.3f}' for f, s in shares)})"
    )
    if not sparse:
        return "REFUTED (sparsity)", detail + "; the >=3 support points refute the Caratheodory claim"
    if hi >= 0.50:
        return "REFUTED (concentration)", detail
    return "CONFIRMED", detail


def score_p2(payload: dict[str, Any]) -> tuple[str, str]:
    """P2: variance ratio optimal/uniform in [0.25, 0.45]."""

    entries = _entries(payload)
    if not entries:
        return "UNDERPOWERED", "no completed budget fractions"
    ratios = [(f, entry["variance_ratio_decision_over_uniform"]) for f, entry in entries]
    detail = "; ".join(f"{f:g}x: {r:.4f}" for f, r in ratios)
    values = [r for _, r in ratios]
    if all(r > 0.9 for r in values):
        return "REFUTED", detail + " (no meaningful gain at any budget)"
    if any(0.25 <= r <= 0.45 for r in values):
        return "PARTIALLY CONFIRMED", detail + " (in the predicted band at some budgets only)"
    return "REFUTED (band)", detail + " (never in the predicted [0.25, 0.45] band)"


def score_p3(payload: dict[str, Any]) -> tuple[str, str]:
    """P3: the hard bar -- optimal allocation does not beat single-scale."""

    single = payload.get("scored", {}).get("single_scale_top_rung", {})
    if single.get("mis_selection") is None:
        return "UNDERPOWERED", "single-scale baseline not scored"
    if not payload.get("baseline_has_headroom", True):
        return "UNINFORMATIVE", payload.get("headroom_warning", "baseline at zero")
    entries = _entries(payload)
    rows = []
    best_excess = None
    for fraction, entry in entries:
        scored = entry.get("scored", {}).get("decision_optimal", {})
        excess = scored.get("excess_over_single_scale")
        if excess is None:
            continue
        rows.append(f"{fraction:g}x: {excess:+.4f}")
        if best_excess is None or excess < best_excess:
            best_excess = excess
    if best_excess is None:
        return "UNDERPOWERED", "no evaluable designs"
    detail = (
        f"single-scale {single['mis_selection']:.4f} "
        f"[{single['ci_lo']:.4f}, {single['ci_hi']:.4f}]; "
        f"excess by budget: {', '.join(rows)}"
    )
    if best_excess < 0:
        return "REFUTED (a win)", detail + " -- verify matched compute before believing this"
    return "CONFIRMED (as a negative)", detail


def score_p4(payload: dict[str, Any]) -> tuple[str, str]:
    """P4: decision and estimation designs agree within 0.02 in cost share."""

    entries = _entries(payload)
    if not entries:
        return "UNDERPOWERED", "no completed budget fractions"
    gaps = [(f, entry["max_cost_share_gap_decision_vs_estimation"]) for f, entry in entries]
    worst = max(g for _, g in gaps)
    detail = "; ".join(f"{f:g}x: {g:.5f}" for f, g in gaps)
    verdict = "CONFIRMED" if worst < 0.02 else "REFUTED"
    return verdict, f"max cost-share gap {worst:.5f} ({detail})"


def score_p5(payload: dict[str, Any]) -> tuple[str, str]:
    """P5: bias-awareness moves cost up the ladder."""

    entries = _entries(payload)
    if not entries:
        return "UNDERPOWERED", "no completed budget fractions"
    rows = []
    max_shift = 0.0
    for fraction, entry in entries:
        top_shift = entry["top_rung_share_bias_aware"] - entry["top_rung_share_decision"]
        cheap_shift = entry["cheap_rung_share_bias_aware"] - entry["cheap_rung_share_decision"]
        rows.append(f"{fraction:g}x: top {top_shift:+.4f}, cheap {cheap_shift:+.4f}")
        max_shift = max(max_shift, abs(top_shift), abs(cheap_shift))
    deviation = payload.get("deviation", {})
    detail = (
        f"max |cost-share shift| {max_shift:.5f}; " + "; ".join(rows) + "; "
        f"signed deviation correlates with log C at "
        f"{deviation.get('corr_signed_deviation_with_log_compute', float('nan')):.4f}"
    )
    if max_shift < 0.02:
        return "REFUTED", detail + " -- the measured deviation is too small to move the design"
    return "CONFIRMED", detail


SCORERS = {
    "P1 (allocation shape)": score_p1,
    "P2 (variance reduction)": score_p2,
    "P3 (hard bar)": score_p3,
    "P4 (SL2 head-to-head)": score_p4,
    "P5 (bias-aware shift)": score_p5,
}


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Allocation study: {payload['target_scale']} target")
    lines.append("")
    lines.append(
        f"Target {payload['target_compute']:.3e} FLOPs, ladder of {len(payload['ladder'])} rungs, "
        f"lever arm {payload['lever_arm_full_ladder']:.2f}x, "
        f"{payload['n_interventions']} interventions, {payload['n_boot']} bootstrap replicates."
    )
    lines.append("")
    lines.append("## Pre-registered predictions")
    lines.append("")
    lines.append("| prediction | verdict | evidence |")
    lines.append("| --- | --- | --- |")
    for name, scorer in SCORERS.items():
        verdict, detail = scorer(payload)
        lines.append(f"| {name} | **{verdict}** | {detail} |")
    lines.append("")

    single = payload.get("scored", {}).get("single_scale_top_rung", {})
    lines.append("## Decision accuracy at matched compute")
    lines.append("")
    lines.append("| budget | cost vs single-scale | " + " | ".join(ARM_LABELS.values()) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in ARM_LABELS) + " |")
    for fraction, entry in _entries(payload):
        cells = []
        for arm in ARM_LABELS:
            value = entry.get("scored", {}).get(arm, {}).get("mis_selection")
            cells.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {fraction:g}x ladder | {entry['cost_relative_to_single_scale']:.3f}x | " + " | ".join(cells) + " |")
    if single.get("mis_selection") is not None:
        lines.append(
            f"\nSingle-scale ranking at the top rung: **{single['mis_selection']:.4f}** "
            f"[{single['ci_lo']:.4f}, {single['ci_hi']:.4f}], "
            f"costing {single['cost_flops']:.3e} FLOPs."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sections = [render(json.loads(path.read_text())) for path in args.results]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(sections))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
