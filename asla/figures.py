"""Figure generation for runs tables."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from asla.analysis.crossover import detect_crossovers, fitted_crossover_for_pair
from asla.analysis.fits import project_ranking, truth_ranking
from asla.analysis.gate import gate_pick, plain_projection_pick
from asla.config import AuditConfig
from asla.data.schema import validate

LOGGER = logging.getLogger(__name__)


def _save(fig: object, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=160)


def make_figures(df: pd.DataFrame, out_dir: str | Path, target: float | None = None) -> None:
    """Create audit figures and save each as PNG and PDF.

    ``target`` defaults to the largest compute budget in the table; pass it
    explicitly when the table extends beyond the audit's target budget.
    """

    import matplotlib.pyplot as plt

    cfg = AuditConfig()
    validate(df)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_computes = sorted(df["compute"].astype(float).unique())
    if target is None:
        target = all_computes[-1]
    elif not any(np.isclose(c, float(target)) for c in all_computes):
        raise ValueError(f"target budget {target} is not present in the runs table")
    target = float(target)
    computes = [c for c in all_computes if c < target or np.isclose(c, target)]
    if len(computes) < 4:
        LOGGER.warning("skipping projection figures: need at least four compute budgets at or below the target")
        return
    budgets = tuple(c for c in computes if not np.isclose(c, target))

    fig, ax = plt.subplots()
    for name, group in df.groupby("intervention", sort=True):
        means = group.groupby("compute")["bpb"].mean().sort_index()
        ax.plot(means.index, means.values, marker="o", label=name)
    for a, b, _ in detect_crossovers(df, budgets, target):
        root = fitted_crossover_for_pair(df, a, b, budgets)
        if root is not None:
            ax.axvline(root, linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("compute")
    ax.set_ylabel("BPB")
    ax.legend(fontsize="small")
    _save(fig, out, "scaling_curves")
    plt.close(fig)

    projected = project_ranking(df, budgets, target)
    truth = truth_ranking(df, target)
    common = projected.index.intersection(truth.index)
    fig, ax = plt.subplots()
    ax.scatter(projected.loc[common], truth.loc[common])
    for name in common:
        ax.annotate(str(name), (projected.loc[name], truth.loc[name]), fontsize=8)
    ax.set_xlabel("projected target BPB")
    ax.set_ylabel("true target BPB")
    _save(fig, out, "projected_vs_true")
    plt.close(fig)

    regret_plain: list[float] = []
    regret_gate: list[float] = []
    xs: list[float] = []
    truth_series = truth_ranking(df, target)
    for intermediate in computes[1:-1]:
        fit_budgets = tuple(c for c in computes if c < intermediate)
        if len(fit_budgets) < 3:
            continue
        rng = np.random.default_rng(123)
        plain = plain_projection_pick(df, fit_budgets, target)
        gate = gate_pick(df, fit_budgets, target, intermediate, tau=cfg.gate.tau, n_boot=cfg.gate.n_boot, rng=rng)
        xs.append(float(intermediate))
        regret_plain.append(float(truth_series.loc[plain] - truth_series.min()))
        regret_gate.append(float(truth_series.loc[gate] - truth_series.min()))
    if xs:
        fig, ax = plt.subplots()
        ax.plot(xs, regret_plain, marker="o", label="plain")
        ax.plot(xs, regret_gate, marker="o", label="gate")
        ax.set_xscale("log")
        ax.set_xlabel("exploration compute")
        ax.set_ylabel("regret")
        ax.legend()
        _save(fig, out, "regret_vs_exploration")
        plt.close(fig)

    if "intervention_class" not in df.columns:
        LOGGER.warning("skipping crossover frequency: missing intervention_class")
        return
    cross = detect_crossovers(df, budgets, target)
    counts = df[["intervention", "intervention_class"]].drop_duplicates().set_index("intervention")
    class_counts: dict[str, int] = {}
    for a, b, _ in cross:
        for name in (a, b):
            cls = str(counts.loc[name, "intervention_class"])
            class_counts[cls] = class_counts.get(cls, 0) + 1
    fig, ax = plt.subplots()
    if class_counts:
        ax.bar(class_counts.keys(), class_counts.values())
    ax.set_ylabel("crossover pair participation")
    _save(fig, out, "crossover_frequency_by_class")
    plt.close(fig)
