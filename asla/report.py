"""Render an audit-result JSON into a human-readable markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _interval(entry: dict[str, Any]) -> str:
    point = _fmt(entry.get("point"))
    lo, hi = entry.get("lo"), entry.get("hi")
    if lo is None or hi is None:
        return point
    return f"{point} [{_fmt(lo)}, {_fmt(hi)}]"


def render_report(result: dict[str, Any]) -> str:
    """Return the markdown report for one audit-result payload."""

    lines: list[str] = ["# ASLA audit report", ""]

    meta = result.get("audit_metadata", {})
    if meta:
        lines += ["## Audit configuration", ""]
        for key in sorted(meta):
            lines.append(f"- **{key}**: {_fmt(meta[key])}")
        lines.append("")

    estimand = result.get("estimand", {})
    if estimand:
        lines += ["## Estimand", ""]
        for key in ("name", "replication_unit", "replication_count", "point_interpretation", "uncertainty_scope"):
            if key in estimand:
                lines.append(f"- **{key}**: {_fmt(estimand[key])}")
        assumptions = estimand.get("assumptions", [])
        if assumptions:
            lines.append("- **assumptions**:")
            for assumption in assumptions:
                lines.append(f"  - {assumption}")
        lines.append("")

    truth = result.get("truth", {})
    ties = result.get("truth_ties", [])
    if truth:
        lines += [
            "## Measured target ranking",
            "",
            "| rank | intervention | mean BPB | seed SE | seeds |",
            "|------|--------------|----------|---------|-------|",
        ]
        # JSON serialization sorts keys alphabetically; restore the ranking order.
        ordered = sorted(truth.items(), key=lambda kv: (float(kv[1].get("mean", float("inf"))), kv[0]))
        for rank, (name, row) in enumerate(ordered, start=1):
            lines.append(
                f"| {rank} | {name} | {_fmt(row.get('mean'))} | {_fmt(row.get('se'))} | {row.get('n_seeds', '—')} |"
            )
        lines.append("")
        if ties:
            lines.append(
                f"⚠️ Statistically tied with the winner (within 2 combined SEs): {', '.join(ties)}. "
                "The measured order between these interventions is not settled by this data."
            )
            lines.append("")

    rankers = result.get("rankers", {})
    if rankers:
        lines += [
            "## Decision rules vs. measured truth (95% bootstrap intervals)",
            "",
            "| ranker | mis-selection rate | mean regret | top-1 acc | pairwise acc |",
            "|--------|--------------------|-------------|-----------|--------------|",
        ]
        for name, metrics in rankers.items():
            cells = [
                _interval(metrics.get(key, {})) for key in ("mis_selection_rate", "mean_regret", "top1_acc", "pairwise_acc")
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        availability = result.get("ranker_metric_availability", {})
        for name, entry in availability.items():
            omitted = entry.get("omitted", {})
            if omitted:
                lines.append(f"- **{name} omitted metrics**: " + ", ".join(sorted(omitted)))
        if availability:
            lines.append("")

    ensemble = result.get("ensemble")
    if ensemble:
        lines += [
            "## Extrapolation reliability (ensemble)",
            "",
            "ρ = cross-family disagreement at target ÷ seed-noise band. "
            "ρ ≫ 1 means the fitting data cannot distinguish curve families that "
            "disagree at the target — treat any single-family projection as unreliable.",
            "",
            "| intervention | ρ | ensemble BPB | family weights |",
            "|--------------|---|--------------|----------------|",
        ]
        for name, entry in ensemble.items():
            weights = ", ".join(
                f"{family}: {_fmt(info.get('weight'), 2)}" for family, info in entry.get("families", {}).items()
            )
            lines.append(f"| {name} | {_fmt(entry.get('reliability'), 2)} | {_fmt(entry.get('point'))} | {weights} |")
        lines.append("")

    diagnostics = result.get("fit_diagnostics", {})
    if diagnostics:
        lines += [
            "## Fit diagnostics (fitting budgets only)",
            "",
            "| intervention | R² | RMSE | max |resid| | points | dof |",
            "|--------------|----|------|-------------|--------|-----|",
        ]
        for name, diag in diagnostics.items():
            lines.append(
                f"| {name} | {_fmt(diag.get('r_squared'))} | {_fmt(diag.get('rmse'))} | "
                f"{_fmt(diag.get('max_abs_residual'))} | {diag.get('n_points', '—')} | {diag.get('dof', '—')} |"
            )
        lines.append("")

    fdr = result.get("crossovers_fdr")
    if fdr is not None:
        lines += ["## Projected-vs-measured order disagreements (BH-FDR tested)", ""]
        if not fdr:
            lines.append("None detected.")
        else:
            lines += [
                "| pair | measured gap | p-value | significant |",
                "|------|--------------|---------|-------------|",
            ]
            for row in fdr:
                lines.append(
                    f"| {row['a']} vs {row['b']} | {_fmt(row.get('true_gap'))} | "
                    f"{_fmt(row.get('p_value'))} | {_fmt(row.get('significant'))} |"
                )
        lines.append("")

    noise = result.get("noise", {})
    if noise:
        lines += ["## Seed-noise summary", ""]
        lines.append(f"- noise band: {_fmt(noise.get('noise_band'))} (source: {_fmt(noise.get('noise_band_source'))})")
        lines.append(f"- pooled variance: {_fmt(noise.get('pooled_variance'), 6)}")
        under = result.get("under_seeded_cells", [])
        if under:
            lines.append(f"- ⚠️ {len(under)} under-seeded cell(s) (fewer than 2 seeds); variance there is unestimated:")
            for cell in under[:20]:
                lines.append(
                    f"  - {cell.get('intervention')} @ compute {_fmt(cell.get('compute'), 3)} "
                    f"({cell.get('seed_count')} seed)"
                )
            if len(under) > 20:
                lines.append(f"  - … and {len(under) - 20} more")
        else:
            lines.append("- all cells have at least 2 seeds")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(audit_json: str | Path, out_path: str | Path) -> Path:
    """Read an audit JSON payload and write the markdown report next to it."""

    result = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(result), encoding="utf-8")
    return out
