"""Score the pre-registered abstention predictions A1-A5 from committed JSON.

Verdicts are computed from the numbers, never written by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HEADLINE_DELTA = 0.05
HEADLINE_BUDGET = 3


def _grid_value(payload: dict[str, Any], scale: str, delta: float, budget: int) -> float | None:
    entry = payload.get("abstention_by_scale", {}).get(scale)
    if entry is None:
        return None
    for row in entry["grid"]:
        if row["delta"] == delta and row["seed_budget"] == budget:
            return float(row["abstention_rate"])
    return None


def score_a1(payload: dict[str, Any]) -> tuple[str, str]:
    """A1: parametric widths undercover (bootstrap < 0.85, conformal < 0.70)."""

    coverage = payload.get("coverage", {})
    boot = coverage.get("bootstrap", {})
    conf = coverage.get("conformal", {})
    if not boot.get("n") or not conf.get("n"):
        return "UNDERPOWERED", "coverage not measured"
    boot_centre = coverage.get("bootstrap_centring", {})
    conf_centre = coverage.get("conformal_centring", {})
    detail = (
        f"bootstrap {boot['coverage']:.3f} [{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}] "
        f"(n={boot['n']}, median width {boot['median_width']:.4f}); "
        f"conformal {conf['coverage']:.3f} [{conf['ci_lo']:.3f}, {conf['ci_hi']:.3f}] "
        f"(n={conf['n']}, median width {conf['median_width']:.4f}); "
        f"nominal {coverage['nominal_coverage']:.2f}"
    )
    if boot_centre and conf_centre:
        detail += (
            f". Both methods share the same point projection, so the same centring "
            f"error ({boot_centre['mean_signed_error']:+.4f}, "
            f"{boot_centre['n_overshooting']}/{boot_centre['n']} overshooting); they "
            f"differ only in width. Error-to-width ratio {boot_centre['error_to_width_ratio']:.2f} "
            f"for bootstrap versus {conf_centre['error_to_width_ratio']:.2f} for conformal: "
            f"conformal covers by being {conf['median_width'] / boot['median_width']:.0f}x wider, "
            f"not by being better centred"
        )
    if boot["coverage"] >= 0.88 or conf["coverage"] >= 0.88:
        achieved = [name for name, stats in (("bootstrap", boot), ("conformal", conf)) if stats["coverage"] >= 0.88]
        return f"PARTIALLY REFUTED ({', '.join(achieved)} calibrated)", detail
    if boot["coverage"] < 0.85 and conf["coverage"] < 0.70:
        return "CONFIRMED", detail
    return "PARTIALLY CONFIRMED", detail


def score_a2(payload: dict[str, Any]) -> tuple[str, str]:
    """A2: abstention monotone in budget and delta; >0.85 at delta=.05, 3 seeds."""

    by_scale = payload.get("abstention_by_scale", {})
    if not by_scale:
        return "UNDERPOWERED", "no abstention grid"

    violations: list[str] = []
    for scale, entry in by_scale.items():
        for delta in payload["deltas"]:
            rows = sorted(
                (r for r in entry["grid"] if r["delta"] == delta),
                key=lambda r: r["seed_budget"],
            )
            rates = [r["abstention_rate"] for r in rows]
            if any(b > a + 1e-9 for a, b in zip(rates, rates[1:])):
                violations.append(f"{scale} delta={delta} not decreasing in budget")
        for budget in payload["seed_budgets"]:
            rows = sorted(
                (r for r in entry["grid"] if r["seed_budget"] == budget),
                key=lambda r: -r["delta"],
            )
            rates = [r["abstention_rate"] for r in rows]
            if any(b < a - 1e-9 for a, b in zip(rates, rates[1:])):
                violations.append(f"{scale} budget={budget} not increasing as delta falls")

    headline = _grid_value(payload, payload["target_scale"], HEADLINE_DELTA, HEADLINE_BUDGET)
    rates = [_grid_value(payload, scale, HEADLINE_DELTA, HEADLINE_BUDGET) for scale in sorted(by_scale)]
    present = [r for r in rates if r is not None]
    mean_rate = sum(present) / len(present) if present else float("nan")
    detail = (
        f"at delta={HEADLINE_DELTA}, {HEADLINE_BUDGET} seeds: "
        f"{payload['target_scale']} {headline:.4f}, mean across scales {mean_rate:.4f}; "
        f"monotonicity violations: {len(violations)}"
    )
    if violations:
        detail += " (" + "; ".join(violations[:3]) + ")"
    if violations:
        return "REFUTED (monotonicity)", detail
    if headline is not None and headline < 0.70:
        return "REFUTED (level)", detail
    if headline is not None and headline > 0.85:
        return "CONFIRMED", detail
    return "PARTIALLY CONFIRMED", detail


def score_a3(payload: dict[str, Any]) -> tuple[str, str]:
    """A3: median seeds required > 20 across adjacent pairs."""

    by_scale = payload.get("abstention_by_scale", {})
    medians = {
        scale: entry["median_seeds_required"]
        for scale, entry in by_scale.items()
        if entry.get("median_seeds_required") is not None
    }
    if not medians:
        return "UNDERPOWERED", "no seeds-required estimates"
    never = sum(entry["n_never_resolvable"] for entry in by_scale.values())
    target_median = medians.get(payload["target_scale"])
    values = sorted(medians.values())
    overall = values[len(values) // 2]
    detail = (
        f"median seeds required at {payload['target_scale']}: "
        f"{target_median if target_median is None else f'{target_median:.0f}'}; "
        f"median across scales {overall:.0f}; "
        f"range {min(values):.0f}-{max(values):.0f}; "
        f"{never} adjacent pairs unresolvable within the cap"
    )
    if overall > 20:
        return "CONFIRMED", detail
    if overall <= 10:
        return "REFUTED", detail
    return "PARTIALLY CONFIRMED", detail


def score_a4(payload: dict[str, Any]) -> tuple[str, str]:
    """A4: rule abstains on >60% of the baseline's correct calls."""

    rows = [r for r in payload.get("cost_of_correctness", []) if "error" not in r]
    if not rows:
        return "UNDERPOWERED", "not computed"
    parts = []
    headline = None
    for row in rows:
        fraction = row["fraction_of_correct_calls_abstained"]
        parts.append(
            f"delta={row['delta']}, {row['seed_budget']} seeds: "
            f"{fraction:.3f} of {row['baseline_correct']} correct calls abstained, "
            f"certified error rate {row['certified_error_rate']:.4f}"
        )
        if row["seed_budget"] == HEADLINE_BUDGET:
            headline = fraction
    detail = "; ".join(parts)
    if headline is None:
        return "UNDERPOWERED", detail
    if headline > 0.60:
        return "CONFIRMED", detail
    if headline < 0.30:
        return "REFUTED", detail
    return "PARTIALLY CONFIRMED", detail


def score_a5(payload: dict[str, Any]) -> tuple[str, str]:
    """A5: effective error rate at nominal delta=0.05 exceeds 0.10."""

    coverage = payload.get("coverage", {})
    rows = []
    worst = 0.0
    for method in ("bootstrap", "conformal"):
        stats = coverage.get(method, {})
        if not stats.get("n"):
            continue
        effective = stats["effective_error_at_nominal_delta_0_05"]
        rows.append(f"{method}: nominal 0.05 -> effective {effective:.4f}")
        worst = max(worst, effective)
    if not rows:
        return "UNDERPOWERED", "coverage not measured"
    detail = "; ".join(rows)
    if worst > 0.10:
        return "CONFIRMED", detail + " -- the guarantee is conditional on Task 4's width correction"
    if worst <= 0.07:
        return "REFUTED", detail + " -- parametric widths are adequate for the stated delta"
    return "PARTIALLY CONFIRMED", detail


SCORERS = {
    "A1 (widths undercover)": score_a1,
    "A2 (abstention vs delta and budget)": score_a2,
    "A3 (seeds required)": score_a3,
    "A4 (cost of correctness)": score_a4,
    "A5 (effective delta)": score_a5,
}


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Abstention study: {payload['target_scale']} target")
    lines.append("")
    lines.append(
        f"Target {payload['target_compute']:.3e} FLOPs; "
        f"{len(payload['ladder'])} fitting rungs; {payload['n_boot']} bootstrap replicates. "
        f"Thresholds use Student's t on 2n-2 degrees of freedom, since sigma is estimated."
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

    lines.append("## Abstention rate by delta and seed budget")
    lines.append("")
    budgets = payload["seed_budgets"]
    lines.append("| delta | " + " | ".join(f"{b} seeds" for b in budgets) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in budgets) + " |")
    scale = payload["target_scale"]
    for delta in payload["deltas"]:
        cells = []
        for budget in budgets:
            value = _grid_value(payload, scale, delta, budget)
            cells.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| {delta} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Leaderboard: DataDecide at {scale}, adjacent orderings.")
    lines.append("")

    lines.append("## Abstention across scales at delta=0.05, 3 seeds")
    lines.append("")
    lines.append("| scale | entries | abstention | median seeds required | never resolvable |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, entry in sorted(payload.get("abstention_by_scale", {}).items(), key=lambda kv: kv[1]["compute"]):
        rate = _grid_value(payload, name, HEADLINE_DELTA, HEADLINE_BUDGET)
        median = entry.get("median_seeds_required")
        lines.append(
            f"| {name} | {entry['n_entries']} | "
            f"{'n/a' if rate is None else f'{rate:.3f}'} | "
            f"{'n/a' if median is None else f'{median:.0f}'} | "
            f"{entry['n_never_resolvable']} |"
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
