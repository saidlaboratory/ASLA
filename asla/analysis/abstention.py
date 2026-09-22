"""delta-PAC selection with abstention for scaling-law leaderboards.

A procedure that always returns a ranking is making claims it cannot support.
The rule here returns, with probability at least ``1 - delta``, either the
correct ordering of a pair or the answer ABSTAIN.

**Which regime this lives in.** Réda, Tirinzoni and Degenne (arXiv:2111.01479)
show that knowing the scale of the deviation from linearity is necessary *to
exploit the structure* of a misspecified linear bandit --- that is, to beat the
sample complexity available with no structure at all. It is not necessary for
delta-correctness. Our deviation scale is unknown, and its compute-structured
component is near zero (signed residuals correlate with log C at -0.022), so
there is little structure to exploit in the extrapolation direction in any case.

This module therefore takes the **structure-free** route: it certifies orderings
from per-cell empirical evidence and its own calibrated widths, rather than from
a parametric extrapolation it cannot vouch for. The guarantee is weaker than a
structure-exploiting one and it is available, which is the trade this project's
measurements force.

**The guarantee is conditional on width calibration**, and deliberately so: every
function here takes the interval widths it is given and reports what it assumed.
:func:`effective_error_rate` converts measured coverage into the error rate the
rule actually achieves, which is how a nominal ``delta`` is prevented from being
quoted as if it were earned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


class Decision(str, Enum):
    """What the rule returns for one pair."""

    LEFT = "left_better"
    RIGHT = "right_better"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class PairVerdict:
    """A certified (or abstained) comparison of two entries."""

    left: str
    right: str
    decision: Decision
    gap: float
    threshold: float
    statistic: float
    delta: float
    n_comparisons: int
    n_seeds: int | None = None

    @property
    def abstained(self) -> bool:
        return self.decision is Decision.ABSTAIN

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "decision": self.decision.value,
            "gap": self.gap,
            "threshold": self.threshold,
            "statistic": self.statistic,
            "delta": self.delta,
            "n_comparisons": self.n_comparisons,
            "n_seeds": self.n_seeds,
            "reference_distribution": "normal" if self.n_seeds is None else "student_t",
        }


def bonferroni_delta(delta: float, n_comparisons: int) -> float:
    """Split ``delta`` across simultaneous comparisons.

    A leaderboard makes every pairwise claim at once, so the guarantee must be
    family-wise or the stated ``delta`` is not the error rate of the object the
    reader actually consumes (the ordering).
    """

    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1")
    return delta / float(n_comparisons)


def glr_threshold(delta: float, n_comparisons: int = 1, n_seeds: int | None = None) -> float:
    """Return the threshold for a two-sided generalised likelihood ratio test.

    The statistic ``|gap| / se`` is compared against the quantile at
    ``1 - delta/(2 m)``. Rejecting certifies the ordering; failing to reject
    abstains.

    **Use the t quantile whenever sigma is estimated.** Pass ``n_seeds`` and the
    threshold uses Student's t on ``2 n_seeds - 2`` degrees of freedom; omit it
    and the Gaussian quantile is used, which assumes sigma is known exactly.

    This is not a refinement. Leaderboards run 3 seeds, and under a Bonferroni
    correction for 300 comparisons the t quantile at 4 degrees of freedom is
    **13.65 against the normal's 3.76 --- 3.63x larger**. Using the normal there
    would make the rule anti-conservative by at least that factor and the
    delta-PAC guarantee would simply not hold. Four degrees of freedom is the
    equal-variance ceiling; with unequal variances the Welch df is lower and the
    factor larger (the per-pair measurement is in
    ``results/target_scoring/target_scoring.json``). The gap closes slowly: the ratio is 1.76 at
    5 seeds, 1.26 at 10, and only reaches 1.04 by 50.
    """

    per_comparison = bonferroni_delta(delta, n_comparisons)
    if n_seeds is None:
        return float(stats.norm.ppf(1.0 - per_comparison / 2.0))
    if n_seeds < 2:
        raise ValueError("certifying a gap requires at least 2 seeds per entry")
    dof = max(2 * int(n_seeds) - 2, 1)
    return float(stats.t.ppf(1.0 - per_comparison / 2.0, dof))


def certify_pair(
    left: str,
    right: str,
    left_mean: float,
    right_mean: float,
    standard_error: float,
    delta: float,
    n_comparisons: int = 1,
    lower_is_better: bool = True,
    n_seeds: int | None = None,
) -> PairVerdict:
    """Certify an ordering or abstain.

    ``standard_error`` is the standard error of the *difference*; callers are
    responsible for supplying one that reflects whatever uncertainty they intend
    to certify against, including any misspecification inflation. That choice is
    the whole content of the guarantee, which is why it is an explicit argument
    rather than something computed here from a model this module does not trust.
    """

    if standard_error < 0 or not np.isfinite(standard_error):
        raise ValueError("standard error must be finite and non-negative")
    threshold = glr_threshold(delta, n_comparisons, n_seeds)
    gap = float(left_mean - right_mean)
    if standard_error == 0.0:
        statistic = float("inf") if gap != 0.0 else 0.0
    else:
        statistic = abs(gap) / standard_error

    if statistic < threshold:
        decision = Decision.ABSTAIN
    elif (gap < 0) == lower_is_better:
        decision = Decision.LEFT
    else:
        decision = Decision.RIGHT

    return PairVerdict(
        left=left,
        right=right,
        decision=decision,
        gap=gap,
        threshold=threshold,
        statistic=statistic,
        delta=delta,
        n_comparisons=n_comparisons,
        n_seeds=n_seeds,
    )


def certify_leaderboard(
    entries: Mapping[str, float],
    standard_errors: Mapping[str, float],
    delta: float,
    lower_is_better: bool = True,
    adjacent_only: bool = True,
    n_seeds: int | None = None,
) -> list[PairVerdict]:
    """Certify a leaderboard's orderings, abstaining where evidence is short.

    ``adjacent_only`` restricts attention to adjacent pairs in the reported
    order, which is what a leaderboard's reader actually relies on; the
    multiplicity correction still counts all pairs, because the ordering as a
    whole is the claim being made.
    """

    if len(entries) < 2:
        raise ValueError("a leaderboard needs at least two entries")
    ordered = sorted(entries.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    n_entries = len(ordered)
    n_comparisons = n_entries * (n_entries - 1) // 2

    if adjacent_only:
        pairs: Iterable[tuple[tuple[str, float], tuple[str, float]]] = zip(ordered[:-1], ordered[1:])
    else:
        pairs = ((ordered[i], ordered[j]) for i in range(n_entries) for j in range(i + 1, n_entries))

    verdicts = []
    for (left, left_value), (right, right_value) in pairs:
        se_left = float(standard_errors[left])
        se_right = float(standard_errors[right])
        se_difference = float(np.hypot(se_left, se_right))
        verdicts.append(
            certify_pair(
                left,
                right,
                left_value,
                right_value,
                se_difference,
                delta,
                n_comparisons,
                lower_is_better,
                n_seeds,
            )
        )
    return verdicts


def abstention_rate(verdicts: Sequence[PairVerdict]) -> float:
    """Fraction of comparisons on which the rule declined to certify."""

    if not verdicts:
        return float("nan")
    return float(sum(1 for v in verdicts if v.abstained) / len(verdicts))


def seeds_to_resolve(
    gap: float,
    sigma: float,
    delta: float,
    n_comparisons: int = 1,
    power: float = 0.8,
    max_seeds: int = 10_000,
    use_t: bool = True,
) -> int | None:
    """Smallest per-entry seed count that would certify a gap of this size.

    Returns ``None`` when no budget within ``max_seeds`` suffices --- which is
    the honest answer for pairs whose gap is buried in run-to-run noise, and is
    reported rather than silently clipped to the cap.
    """

    if gap <= 0 or not np.isfinite(gap) or sigma <= 0:
        return None
    per_comparison = bonferroni_delta(delta, n_comparisons)
    # Start from the Gaussian solution, then iterate with the t quantile, whose
    # degrees of freedom depend on the answer.
    needed = 2.0 * ((stats.norm.ppf(1 - per_comparison / 2) + stats.norm.ppf(power)) * sigma / gap) ** 2
    needed = max(needed, 2.0)
    if use_t:
        for _ in range(60):
            dof = max(2.0 * needed - 2.0, 1.0)
            critical = stats.t.ppf(1 - per_comparison / 2, dof) + stats.t.ppf(power, dof)
            updated = max(2.0 * (critical * sigma / gap) ** 2, 2.0)
            if abs(updated - needed) < 1e-6:
                needed = updated
                break
            needed = updated
    seeds = int(np.ceil(needed))
    if seeds > max_seeds:
        return None
    return max(seeds, 2)


def effective_error_rate(nominal_delta: float, empirical_coverage: float, nominal_coverage: float) -> float:
    """Convert measured interval coverage into the error rate actually achieved.

    A rule quoting ``delta`` while using intervals that undercover is not a
    ``delta``-PAC rule. If intervals built for ``nominal_coverage`` in fact cover
    only ``empirical_coverage`` of the time, the miss rate is inflated by

        (1 - empirical) / (1 - nominal)

    and the effective error rate is the nominal one scaled by that factor,
    capped at 1. This is the arithmetic that stops a nominal guarantee being
    reported as an earned one.
    """

    if not 0.0 < nominal_delta < 1.0:
        raise ValueError("nominal_delta must be in (0, 1)")
    if not 0.0 <= empirical_coverage <= 1.0:
        raise ValueError("empirical_coverage must be in [0, 1]")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be in (0, 1)")
    inflation = (1.0 - empirical_coverage) / (1.0 - nominal_coverage)
    return float(min(1.0, nominal_delta * inflation))


__all__ = [
    "Decision",
    "PairVerdict",
    "abstention_rate",
    "bonferroni_delta",
    "certify_leaderboard",
    "certify_pair",
    "effective_error_rate",
    "glr_threshold",
    "seeds_to_resolve",
]
