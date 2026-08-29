"""Generate results/checkpoints/RESULTS.md from checkpoint_study.json.

Every number is read from the JSON; nothing is hand-written. The technique
critique leads because it is the stronger and more transportable finding; the
decision result is secondary and is reported with its statistical caveat
attached rather than as a bare verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "results" / "checkpoints" / "checkpoint_study.json"
OUT = REPO / "results" / "checkpoints" / "RESULTS.md"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * float(x):.2f}%"


def interval(entry: dict[str, Any]) -> str:
    return f"{pct(entry['point'])} [{pct(entry['lo'])}, {pct(entry['hi'])}]"


def render(d: dict[str, Any]) -> str:
    v = d["verdicts"]
    tc = d["technique_critique"]
    primary = d["designs"]["primary"]
    ce = d["correction_effect"]
    c1, c5 = v["C1_beats_plain"], v["C5_correlation_correction_matters"]
    lines: list[str] = [
        "# Checkpoint-augmented fitting: results",
        "",
        f"Generated from `results/checkpoints/checkpoint_study.json` (n_boot={d['counts']['n_boot']}, "
        f"metric `{d['metric']}`, seed {d['seed']}). Predictions were fixed in `PREDICTIONS_TASK_CHECKPOINTS.md` "
        "before implementation.",
        "",
        "## 1. Technique critique (primary result)",
        "",
        tc["context"],
        "",
        "Four properties of the technique, measured here, that change how it must be applied:",
        "",
        "### 1.1 Checkpoints carry far less information than their count suggests",
        "",
        "| metric | ladder points | checkpoints | mean rho | max rho | effective points | deflation |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in tc["per_metric"]:
        lines.append(
            f"| `{m['metric']}` | {m['ladder_points_final_only']} | {m['checkpoint_points']} | "
            f"{m['mean_rho']:.3f} | {m['max_rho']:.3f} | {m['mean_effective_points']:.1f} | "
            f"**{m['information_deflation_factor']:.1f}x** |"
        )
    noop = tc["uniform_weight_noop"]
    worst = max(tc["per_metric"], key=lambda m: m["information_deflation_factor"])
    lines += [
        "",
        f"Serial correlation between checkpoints of the same run reaches rho = {worst['max_rho']:.2f}. Treating "
        f"{worst['checkpoint_points']} checkpoints as {worst['checkpoint_points']} independent observations "
        f"overstates their information by up to **{worst['information_deflation_factor']:.1f}x**.",
        "",
        "### 1.2 The obvious correction is a mathematical no-op",
        "",
        "Down-weighting a correlated run by one constant factor cancels in the least-squares normal equations:",
        "",
        "| weighting | max relative change in fitted parameters |",
        "|---|---|",
    ]
    for constant, change in sorted(noop["max_relative_parameter_change_uniform_weight"].items(), key=lambda kv: float(kv[0])):
        lines.append(f"| uniform, sigma = {constant} | {change:.2e} |")
    for name, change in sorted(noop["max_relative_parameter_change_varying_weight"].items()):
        lines.append(f"| **varying across groups** ({name.replace('_', ' ')}) | **{change:.2e}** |")
    lines += [
        "",
        f"That is, {noop['conclusion']}. A correlation correction must therefore act *between* groups "
        "(here, between "
        "scales), not within a run.",
        "",
        "### 1.3 Real ladders are badly unbalanced, so naive fitting silently reweights them",
        "",
        "| metric | fewest checkpoints at a scale | most | imbalance |",
        "|---|---|---|---|",
    ]
    for m in tc["per_metric"]:
        lines.append(
            f"| `{m['metric']}` | {m['min_checkpoints_at_a_scale']} | {m['max_checkpoints_at_a_scale']} | "
            f"**{m['imbalance_ratio']:.0f}x** |"
        )
    lines += [
        "",
        "Scales contribute checkpoints in proportion to how often they were evaluated, not to how much they "
        "constrain the curve, so an unweighted fit tilts toward whichever scales happen to be densely logged.",
        "",
        "### 1.4 Naive intervals are over-confident by roughly the deflation factor",
        "",
        f"Seed-bootstrap CI width, naive: **{c5['naive_ci_width']:.4f}**; correlation-aware: "
        f"**{c5['corrected_ci_width']:.4f}** "
        f"({c5['corrected_ci_width'] / max(c5['naive_ci_width'], 1e-12):.1f}x wider). The correction changes the "
        f"fit materially: maximum projected-value shift **{ce['max_abs_projection_shift']:.2e}**, and the induced "
        f"ordering differs ({ce['ordering_differs']}).",
        "",
        "*(A first version of this check compared summary mis-selection rates, which coincided at "
        f"{pct(c5['corrected']['point'])} for both weightings, and wrongly read as a no-op. Two different fits can "
        "share a summary rate; the comparison must look at what the correction actually changes.)*",
        "",
        "## 2. Decision result (secondary)",
        "",
        "The decision-level effect is modest and, on the primary design, not statistically separated from the "
        "baseline it improves on. It is reported here with that caveat attached.",
        "",
        "| ranker | pairwise mis-selection | net flips removed vs plain |",
        "|---|---|---|",
    ]
    for name, r in sorted(primary["rankers"].items(), key=lambda kv: kv[1]["mis_selection"]["point"]):
        lines.append(f"| `{name}` | {interval(r['mis_selection'])} | {r['net_removed_vs_plain']:+d} |")
    lines += [
        "",
        "### Pre-registered verdicts",
        "",
        "| prediction | verdict | qualification |",
        "|---|---|---|",
        f"| C1 beats plain projection | **{c1['verdict']}** | point estimate improved {pct(c1['relative_improvement'])} "
        f"relative ({c1['absolute_improvement_points']:.2f} points), exactly at the {25}% threshold, but the CI is "
        f"**not separated from plain** (separated = {c1['separated_from_plain']}): the improvement is not "
        "statistically distinguishable from baseline |",
        f"| C2 does not beat single-scale | **{v['C2_beats_single_scale']['verdict']}** | "
        f"single-scale {interval(v['C2_beats_single_scale']['single_scale'])} versus checkpoint-augmented "
        f"{interval(v['C2_beats_single_scale']['checkpoint_ar1'])} |",
        f"| C3 lever-arm interaction | **{v['C3_lever_arm_interaction']['verdict']}** | the short arm removed "
        f"{v['C3_lever_arm_interaction']['I_short_arm']} flips (net negative), so the ratio is undefined and "
        f"{v['C3_lever_arm_interaction']['n_draws']} bootstrap draws were valid |",
        f"| C4 composition with pooling | **{v['C4_composition_with_pooling']['verdict']}** | "
        f"G_both = {pct(v['C4_composition_with_pooling']['G_both'])} against pooling alone "
        f"{pct(v['C4_composition_with_pooling']['G_pooling'])}; CIs overlap the best single lever |",
        f"| C5 correction matters | **{c5['verdict']}** | see section 1.4 |",
        "",
        "**Natural reading of C3 and C4, explicitly not established at these intervals:** checkpoints add points "
        "along the existing ladder without shortening the lever arm, so they reduce parameter uncertainty but not "
        "extrapolation distance, and buy less than pooling does. The confidence intervals cannot distinguish this "
        "from the alternatives, so it is stated as an interpretation awaiting more seeds, not a result.",
        "",
        f"Mechanism damaged by these results: **{v['mechanism_damaged']}**.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = json.loads(STUDY.read_text(encoding="utf-8"))
    text = render(payload)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
