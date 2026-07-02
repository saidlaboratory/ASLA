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


def _ensemble_disagreement_figure(
    df: pd.DataFrame, budgets: tuple[float, ...], target: float, out: Path, plt: object
) -> None:
    """Plot per-family target projections against the measured target band."""

    from asla.analysis.audit import seed_noise_report
    from asla.analysis.ensemble import ensemble_report
    from asla.models import FitError

    if len(budgets) < 4:
        LOGGER.warning("skipping ensemble disagreement figure: needs at least four fitting budgets")
        return
    try:
        band = seed_noise_report(df, target)["noise_band"]
        report = ensemble_report(df, budgets, target, noise_band=band)
    except (FitError, ValueError) as exc:
        LOGGER.warning("skipping ensemble disagreement figure: %s", exc)
        return
    measured = df[np.isclose(df["compute"].astype(float), target)].groupby("intervention")["bpb"].mean()
    names = sorted(report)
    fig, ax = plt.subplots()
    markers = {"power_law": "o", "saturating": "s", "damped_power_law": "^"}
    seen_families: set[str] = set()
    for i, name in enumerate(names):
        entry = report[name]
        for family, info in entry["families"].items():
            label = family if family not in seen_families else None
            seen_families.add(family)
            ax.scatter([i], [info["projection"]], marker=markers.get(family, "x"), color="C0", alpha=0.8, label=label)
        ax.scatter([i], [entry["point"]], marker="_", s=400, color="C3", label="ensemble" if i == 0 else None)
        if name in measured.index and band is not None:
            center = float(measured.loc[name])
            ax.errorbar([i], [center], yerr=[[band], [band]], fmt="*", color="C2", capsize=4,
                        label="measured target ± noise band" if i == 0 else None)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize="small")
    ax.set_ylabel("projected target BPB")
    ax.legend(fontsize="small")
    _save(fig, out, "ensemble_disagreement")
    plt.close(fig)


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

    from asla.analysis.fits import fit_all
    from asla.models import FitError, bpb_power_law

    try:
        fitted_params = fit_all(df, budgets)
    except (FitError, ValueError):
        fitted_params = {}
    fig, ax = plt.subplots()
    extrapolation_grid = np.logspace(np.log10(min(budgets)), np.log10(target), 200)
    for name, group in df.groupby("intervention", sort=True):
        means = group.groupby("compute")["bpb"].mean().sort_index()
        (line,) = ax.plot(means.index, means.values, marker="o", linestyle="none", label=str(name))
        params = fitted_params.get(str(name))
        if params is not None:
            ax.plot(
                extrapolation_grid,
                bpb_power_law(extrapolation_grid, *params),
                linestyle="--",
                alpha=0.6,
                color=line.get_color(),
            )
        target_rows = group[np.isclose(group["compute"].astype(float), target)]
        if not target_rows.empty:
            ax.scatter(
                [target],
                [float(target_rows["bpb"].mean())],
                marker="*",
                s=120,
                color=line.get_color(),
                zorder=5,
            )
    for a, b, _ in detect_crossovers(df, budgets, target):
        root = fitted_crossover_for_pair(df, a, b, budgets)
        if root is not None:
            ax.axvline(root, linestyle="--", alpha=0.4)
    ax.axvline(float(max(budgets)), color="gray", linestyle=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("compute (dashed: fitted extrapolation; star: measured target mean)")
    ax.set_ylabel("BPB")
    ax.legend(fontsize="small")
    _save(fig, out, "scaling_curves")
    plt.close(fig)

    _ensemble_disagreement_figure(df, budgets, target, out, plt)

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
