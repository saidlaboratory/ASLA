"""Figures for the allocation study, generated from committed JSON.

Nothing is hard-coded: every number is read from the study output. Run after
``scripts/run_allocation_study.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_STYLE = {
    "decision_optimal": ("#1f77b4", "o", "decision-optimal"),
    "estimation_optimal_sl2": ("#ff7f0e", "s", "estimation-optimal (SL2)"),
    "bias_aware": ("#2ca02c", "^", "bias-aware"),
    "uniform_ladder": ("#7f7f7f", "d", "uniform ladder"),
}


def _sorted_sweep(payload: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    entries = []
    for entry in payload.get("sweep", {}).values():
        if "error" in entry:
            continue
        entries.append((float(entry["budget_fraction"]), entry))
    return sorted(entries, key=lambda item: item[0])


def figure_allocation_shape(payload: dict[str, Any], out: Path) -> None:
    """Cost share per rung, as the exploration budget grows."""

    entries = _sorted_sweep(payload)
    if not entries:
        return
    ladder = np.asarray(payload["ladder"], dtype=float)
    fig, axes = plt.subplots(1, len(entries), figsize=(3.1 * len(entries), 3.4), sharey=True)
    if len(entries) == 1:
        axes = [axes]
    for ax, (fraction, entry) in zip(axes, entries):
        runs = np.asarray(entry["designs"]["decision_optimal"]["runs"], dtype=float)
        cost = runs * ladder
        share = cost / cost.sum() if cost.sum() > 0 else cost
        ax.bar(np.arange(len(ladder)), share, color="#1f77b4")
        ax.set_title(f"{fraction:g}x ladder budget", fontsize=9)
        ax.set_xticks(np.arange(len(ladder))[::2])
        ax.set_xticklabels([f"{np.log10(c):.0f}" for c in ladder[::2]], fontsize=7)
        ax.set_xlabel("log10 compute", fontsize=8)
    axes[0].set_ylabel("share of exploration cost", fontsize=8)
    fig.suptitle(
        "Decision-optimal allocation shifts up the ladder as the budget grows",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_variance_and_accuracy(payload: dict[str, Any], out: Path) -> None:
    """Variance reduction and mis-selection against exploration budget."""

    entries = _sorted_sweep(payload)
    if not entries:
        return
    fractions = [f for f, _ in entries]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for arm, (color, marker, label) in ARM_STYLE.items():
        values = [entry["variance_factors"].get(arm) for _, entry in entries]
        if any(v is None for v in values):
            continue
        left.plot(fractions, values, marker=marker, color=color, label=label)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("exploration budget (fraction of uniform 3-seed ladder)")
    left.set_ylabel("target-gap variance factor")
    left.set_title("Variance at matched compute", fontsize=10)
    left.legend(fontsize=7)
    left.grid(alpha=0.3)

    single = payload.get("scored", {}).get("single_scale_top_rung", {})
    for arm, (color, marker, label) in ARM_STYLE.items():
        xs, ys = [], []
        for fraction, entry in entries:
            value = entry.get("scored", {}).get(arm, {}).get("mis_selection")
            if value is not None:
                xs.append(fraction)
                ys.append(value)
        if xs:
            right.plot(xs, ys, marker=marker, color=color, label=label)
    if single.get("mis_selection") is not None:
        right.axhline(
            single["mis_selection"],
            color="crimson",
            linestyle="--",
            label=f"single-scale ({single['mis_selection']:.3f})",
        )
    right.set_xscale("log")
    right.set_xlabel("exploration budget (fraction of uniform 3-seed ladder)")
    right.set_ylabel("pairwise mis-selection")
    right.set_title("Decision accuracy at matched compute", fontsize=10)
    right.legend(fontsize=7)
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_decision_vs_estimation(payload: dict[str, Any], out: Path) -> None:
    """The P4 null: the two objectives choose the same design."""

    entries = _sorted_sweep(payload)
    if not entries:
        return
    fractions = [f for f, _ in entries]
    gaps = [entry["max_cost_share_gap_decision_vs_estimation"] for _, entry in entries]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(fractions, gaps, marker="o", color="#9467bd")
    ax.axhline(0.02, color="grey", linestyle=":", label="pre-registered threshold (0.02)")
    ax.set_xscale("log")
    ax.set_xlabel("exploration budget (fraction of uniform 3-seed ladder)")
    ax.set_ylabel("max cost-share difference")
    ax.set_title(
        "Design-for-decision vs design-for-estimation\nchoose the same allocation",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/allocation/allocation_1B.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/allocation/figures"))
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure_allocation_shape(payload, args.out_dir / "allocation_shape.png")
    figure_variance_and_accuracy(payload, args.out_dir / "variance_and_accuracy.png")
    figure_decision_vs_estimation(payload, args.out_dir / "decision_vs_estimation.png")
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
