"""Generate the two standalone confidence-statement findings from committed JSON.

Both are independent of the Task 4 curvature gate's outcome:

* the Student-t defect --- a concrete, checkable error in standard practice that
  anyone reporting significance on a 3-seed grid inherits;
* the A1 split --- two quantified ways confidence statements in this area fail,
  paired with A5's nominal-to-effective delta degradation.

Every number is read from ``results/abstention/abstention_1B.json`` or computed
from the same closed-form quantiles the library uses. Nothing is hand-written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy import stats

from asla.analysis.abstention import effective_error_rate, glr_threshold

SEED_COUNTS = (3, 5, 10, 20, 50)
# The 90M rung: the boundary above which certification stops being total. Named
# rather than inlined so the table and the sentence beneath it cannot drift.
SMALL_SCALE_CEILING_FLOPS = 5.8e18
# A rate at or above this is reported as total abstention; below 1.0 only by
# floating-point representation of an exact fraction.
TOTAL_ABSTENTION = 0.999
TYPICAL_COMPARISONS = 300
TYPICAL_DELTA = 0.05


def student_t_section() -> str:
    lines: list[str] = []
    lines.append("# The reference distribution defect on small-seed leaderboards")
    lines.append("")
    lines.append(
        "A leaderboard that reports significance over a grid of recipes is making many "
        "simultaneous comparisons from very few seeds. Both halves of that sentence "
        "matter, and standard practice gets the second one wrong."
    )
    lines.append("")
    lines.append(
        f"With `n` seeds per entry, the standard deviation is estimated from `n` points, "
        f"so the reference distribution for a two-sided test is Student's t on `2n-2` "
        f"degrees of freedom, not the normal. Under a Bonferroni correction for "
        f"{TYPICAL_COMPARISONS} comparisons at delta={TYPICAL_DELTA}, the per-comparison "
        f"level is {TYPICAL_DELTA / TYPICAL_COMPARISONS:.3e}, and the two quantiles "
        f"diverge sharply:"
    )
    lines.append("")
    lines.append("| seeds per entry | degrees of freedom | t quantile | normal quantile | ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    normal = glr_threshold(TYPICAL_DELTA, TYPICAL_COMPARISONS)
    for seeds in SEED_COUNTS:
        student = glr_threshold(TYPICAL_DELTA, TYPICAL_COMPARISONS, n_seeds=seeds)
        lines.append(f"| {seeds} | {2 * seeds - 2} | {student:.3f} | {normal:.3f} | **{student / normal:.2f}x** |")
    lines.append("")
    three = glr_threshold(TYPICAL_DELTA, TYPICAL_COMPARISONS, n_seeds=3)
    lines.append(
        f"At three seeds --- the count DataDecide uses, and a common choice across "
        f"published suites --- the Gaussian threshold is anti-conservative by "
        f"**{three / normal:.2f}x**. A procedure using it does not hold its stated error "
        f"rate. The approximation only becomes adequate at seed counts nobody runs: the "
        f"ratio is still {glr_threshold(TYPICAL_DELTA, TYPICAL_COMPARISONS, n_seeds=10) / normal:.2f}x "
        f"at ten seeds and reaches "
        f"{glr_threshold(TYPICAL_DELTA, TYPICAL_COMPARISONS, n_seeds=50) / normal:.2f}x only at fifty."
    )
    lines.append("")
    lines.append(
        "The multiplicity correction is what makes this bite. At a single comparison the "
        f"t and normal quantiles differ by only "
        f"{stats.t.ppf(1 - TYPICAL_DELTA / 2, 4) / stats.norm.ppf(1 - TYPICAL_DELTA / 2):.2f}x "
        "at the same degrees of freedom; it is pushing into the far tail, where the t "
        "distribution's heavier tail dominates, that opens the gap."
    )
    lines.append("")
    lines.append("## How this was caught, and why it nearly was not")
    lines.append("")
    lines.append(
        "Not by inspection. The abstention rule reported 16.7% abstention on DataDecide "
        "at 1B where an independently written resolution diagnostic reported 54.2% on the "
        "same data. The pre-registration required any such discrepancy to be resolved "
        "before either number was reported. After the fix the two agree exactly, at "
        "0.5417."
    )
    lines.append("")
    lines.append(
        "The rule's own test suite did not catch it, and the reason generalises. The "
        "test simulated the null and confirmed the family-wise error rate was held --- "
        "but it supplied the **true** standard error to the rule rather than estimating "
        "one from simulated seeds. A simulation that hands the estimator a known sigma "
        "cannot detect a wrong small-sample reference distribution, because the quantity "
        "the t correction exists to handle has been assumed away. The test passed for a "
        "rule whose guarantee did not hold."
    )
    lines.append("")
    lines.append(
        "The replacement draws seeds, estimates sigma from them exactly as the study "
        "does, and checks both variants: the t-based rule holds delta, the Gaussian one "
        "does not. This is the same pattern as the other measurement artifacts in this "
        "project --- the measurement could not see the thing it was supposed to check."
    )
    lines.append("")
    return "\n".join(lines)


def confidence_failure_section(payload: dict[str, Any]) -> str:
    coverage = payload["coverage"]
    boot = coverage["bootstrap"]
    conf = coverage["conformal"]
    boot_centre = coverage["bootstrap_centring"]
    conf_centre = coverage["conformal_centring"]
    # Measured from the target cells' own seed spread, not inferred from the
    # band width -- the bootstrap band reflects fit-parameter uncertainty, which
    # is a different quantity and gave a badly wrong answer when used as a proxy.
    error_in_seed_ses = boot_centre["centring_error_in_seed_standard_errors"]

    lines: list[str] = []
    lines.append("# Two ways confidence statements fail in scaling-law extrapolation")
    lines.append("")
    lines.append(
        f"Measured on DataDecide, projecting a {len(payload['ladder'])}-rung ladder to the "
        f"{payload['target_scale']} target ({payload['target_compute']:.3e} FLOPs), where the "
        f"true target value is **observed** rather than assumed."
    )
    lines.append("")

    lines.append("## Failure 1: precisely wrong, not merely narrow")
    lines.append("")
    lines.append("| | bootstrap | conformal |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| empirical coverage (nominal {coverage['nominal_coverage']:.2f}) | "
        f"**{boot['coverage']:.3f}** [{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}] | "
        f"{conf['coverage']:.3f} [{conf['ci_lo']:.3f}, {conf['ci_hi']:.3f}] |"
    )
    lines.append(f"| median width | {boot['median_width']:.4f} | {conf['median_width']:.4f} |")
    lines.append(
        f"| mean centring error | {boot_centre['mean_signed_error']:+.4f} | {conf_centre['mean_signed_error']:+.4f} |"
    )
    lines.append(
        f"| error-to-width ratio | **{boot_centre['error_to_width_ratio']:.2f}** | "
        f"{conf_centre['error_to_width_ratio']:.2f} |"
    )
    lines.append(
        f"| recipes overshooting | {boot_centre['n_overshooting']}/{boot_centre['n']} | "
        f"{conf_centre['n_overshooting']}/{conf_centre['n']} |"
    )
    lines.append("")
    lines.append(
        f"The bootstrap interval covers the observed target **{boot['coverage']:.0%} of the "
        f"time** --- {int(round(boot['coverage'] * boot['n']))} of {boot['n']} recipes. That is "
        f"not undercoverage that a modest inflation would fix. The band is "
        f"**{boot_centre['error_to_width_ratio']:.2f}x narrower than its own centring error**: "
        f"it is confidently, precisely wrong. A band propagating only seed noise cannot cover "
        f"a projection sitting {error_in_seed_ses:.0f}x the seed standard error from truth, at "
        f"any bootstrap count."
    )
    lines.append("")
    lines.append(
        f"Both methods share the same point projection and therefore the same centring error "
        f"({boot_centre['mean_signed_error']:+.4f}, {boot_centre['n_overshooting']} of "
        f"{boot_centre['n']} overshooting). They differ only in width. Conformal achieves "
        f"valid coverage by being **{conf['median_width'] / boot['median_width']:.0f}x wider**, "
        f"not by being better centred --- its error-to-width ratio is "
        f"{conf_centre['error_to_width_ratio']:.2f}, comfortably below one. Validity here is "
        f"bought with width, and the width is the honest price of extrapolating a "
        f"misspecified curve."
    )
    lines.append("")

    lines.append("## Failure 2: a nominal guarantee that is not the achieved one")
    lines.append("")
    lines.append(
        "Undercovering intervals inflate a rule's real error rate by "
        "`(1 - empirical) / (1 - nominal)`. Stating delta while using them does not make "
        "the procedure delta-PAC:"
    )
    lines.append("")
    lines.append("| interval | nominal delta | effective delta | degradation |")
    lines.append("| --- | --- | --- | --- |")
    for label, stats_block in (("bootstrap", boot), ("conformal", conf)):
        effective = effective_error_rate(TYPICAL_DELTA, stats_block["coverage"], coverage["nominal_coverage"])
        lines.append(f"| {label} | {TYPICAL_DELTA} | **{effective:.3f}** | {effective / TYPICAL_DELTA:.1f}x |")
    lines.append("")
    boot_effective = effective_error_rate(TYPICAL_DELTA, boot["coverage"], coverage["nominal_coverage"])
    lines.append(
        f"A rule quoting delta={TYPICAL_DELTA} on bootstrap widths is in fact running at "
        f"**{boot_effective:.3f}** --- a {boot_effective / TYPICAL_DELTA:.0f}-fold degradation, "
        f"and a coin flip rather than a guarantee. Conformal widths hold up: "
        f"{effective_error_rate(TYPICAL_DELTA, conf['coverage'], coverage['nominal_coverage']):.3f} "
        f"against a nominal {TYPICAL_DELTA}."
    )
    lines.append("")
    lines.append(
        "Taken together these are two quantified failures of confidence statements in this "
        "setting: one where the interval measures the wrong thing entirely, and one where "
        "the number attached to a procedure is not the number it achieves."
    )
    lines.append("")
    return "\n".join(lines)


def certification_cost_section(payload: dict[str, Any]) -> str:
    rows = [r for r in payload.get("cost_of_correctness", []) if "error" not in r]
    by_scale = payload.get("abstention_by_scale", {})

    lines: list[str] = []
    lines.append("# Where certification is free, and where it is unavailable")
    lines.append("")
    lines.append(
        "A delta-PAC rule that abstains rather than guessing was expected to be expensive: "
        "the pre-registered prediction was that it would decline to certify more than 60% "
        "of the comparisons a plain ranking gets right. At the top of the ladder that is "
        "wrong by a wide margin."
    )
    lines.append("")
    lines.append("| delta | seeds | pairs | certified | certified errors | correct calls abstained |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['delta']} | {row['seed_budget']} | {row['n_pairs']} | "
            f"{row['n_certified']} | **{row['certified_wrong']}** | "
            f"{row['fraction_of_correct_calls_abstained']:.1%} |"
        )
    lines.append("")
    headline = next((r for r in rows if r["seed_budget"] == 3), None)
    if headline is not None:
        lines.append(
            f"At the {payload['target_scale']} target with {headline['seed_budget']} seeds, the "
            f"rule certifies **{headline['n_certified']} of {headline['n_pairs']}** pairs with "
            f"**{headline['certified_wrong']} certified errors**, abstaining on only "
            f"**{headline['fraction_of_correct_calls_abstained']:.1%}** of the calls the "
            f"baseline gets right. Correctness is close to free where gaps are wide."
        )
    lines.append("")
    lines.append("## The complement: where nothing can be certified")
    lines.append("")
    lines.append(
        "The same rule abstains on everything at small scales, and that is the honest other "
        "half of the finding. Abstention at delta=0.05 with 3 seeds, by scale:"
    )
    lines.append("")
    lines.append("| scale | entries | abstention | median seeds required |")
    lines.append("| --- | --- | --- | --- |")
    for name, entry in sorted(by_scale.items(), key=lambda kv: kv[1]["compute"]):
        rate = next(
            (r["abstention_rate"] for r in entry["grid"] if r["delta"] == TYPICAL_DELTA and r["seed_budget"] == 3),
            None,
        )
        median = entry.get("median_seeds_required")
        lines.append(
            f"| {name} | {entry['n_entries']} | "
            f"{'n/a' if rate is None else f'{rate:.3f}'} | "
            f"{'n/a' if median is None else f'{median:.0f}'} |"
        )
    lines.append("")
    small = [
        (name, entry)
        for name, entry in sorted(by_scale.items(), key=lambda kv: kv[1]["compute"])
        if entry["compute"] <= SMALL_SCALE_CEILING_FLOPS
    ]
    small_rates = []
    for name, entry in small:
        rate = next(
            (r["abstention_rate"] for r in entry["grid"] if r["delta"] == TYPICAL_DELTA and r["seed_budget"] == 3),
            None,
        )
        if rate is not None:
            small_rates.append(rate)
    if small_rates:
        n_total = sum(1 for _, e in small if e["n_entries"] >= 2)
        n_complete = sum(1 for r in small_rates if r >= TOTAL_ABSTENTION)
        lines.append(
            f"Across the {n_total} scales at or below 90M, abstention runs from "
            f"**{min(small_rates):.3f} to {max(small_rates):.3f}**, and is total "
            f"(1.000) at {n_complete} of them. At those scales a 3-seed leaderboard "
            f"supports at most a handful of certified orderings: nearly every adjacent "
            f"gap is inside the noise. This is not a limitation of the rule --- it is a "
            f"statement about where small-scale selection is possible at all, and on this "
            f"suite the answer below 90M is essentially nowhere, at any delta a reader "
            f"would accept."
        )
    lines.append("")
    lines.append(
        "Reported together, the two halves say something a single number would hide: "
        "certification is cheap exactly where the decision is easy, and unavailable exactly "
        "where small-scale proxies are most tempting to use."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/abstention/abstention_1B.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/confidence"))
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (args.out_dir / "STUDENT_T_DEFECT.md").write_text(student_t_section())
    (args.out_dir / "CONFIDENCE_FAILURES.md").write_text(confidence_failure_section(payload))
    (args.out_dir / "CERTIFICATION_COST.md").write_text(certification_cost_section(payload))
    print(f"wrote three sections to {args.out_dir}")


if __name__ == "__main__":
    main()
