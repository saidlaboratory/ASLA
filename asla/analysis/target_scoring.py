"""Score rankers against a target whose own ordering is uncertain.

The audit's headline rates are computed against the observed target ranking,
treating it as ground truth. It is a three-seed estimate, so some pairs are
ordered by noise. Two rules handle that, and both are reported because they make
different assumptions:

* **determined subset** --- score only on pairs whose target ordering is
  statistically resolved, so no ranker is charged for guessing a coin flip;
* **expected error** --- charge every pair in expectation, weighting by the
  probability that the observed order is the true one.

Neither addresses the deeper problem that observed target means may differ from
*expected* target performance for reasons other than seed noise. Both treat the
target cells as unbiased estimates, and that assumption is stated wherever these
functions are used rather than buried here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairEvidence:
    """The target's own evidence about one pair's ordering."""

    left: str
    right: str
    gap: float
    standard_error: float
    welch_df: float
    p_value: float
    determined: bool
    probability_observed_order_correct: float

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "gap": self.gap,
            "standard_error": self.standard_error,
            "welch_df": self.welch_df,
            "p_value": self.p_value,
            "determined": self.determined,
            "probability_observed_order_correct": self.probability_observed_order_correct,
        }


def bootstrap_two_sided_p(draws: Sequence[float]) -> float:
    """Two-sided bootstrap p-value for a difference being zero, clipped to 1.

    ``2 * min(P(d <= 0), P(d >= 0))`` exceeds 1 when many draws equal zero
    exactly, because both tail proportions then include the ties. A value above
    1 is not a probability; it was shipped once (1.236) before this helper
    existed, which is why every caller now goes through it.
    """

    array = np.asarray(list(draws), dtype=float)
    if array.size == 0:
        return float("nan")
    return float(min(1.0, 2.0 * min((array <= 0).mean(), (array >= 0).mean())))


def welch_pair(
    mean_a: float,
    sd_a: float,
    n_a: int,
    mean_b: float,
    sd_b: float,
    n_b: int,
) -> tuple[float, float, float, float]:
    """Return (gap, standard error, Welch-Satterthwaite df, two-sided p).

    The degrees of freedom are computed per pair rather than assumed. With three
    seeds per cell they range from 2 to 4 and equal 4 only when the two cells
    have equal variance, so asserting 4 everywhere is anti-conservative.
    """

    if n_a < 2 or n_b < 2:
        raise ValueError("Welch's test needs at least two observations per cell")
    va, vb = sd_a**2 / n_a, sd_b**2 / n_b
    se = float(np.sqrt(va + vb))
    if se <= 0:
        return float(mean_a - mean_b), 0.0, float("nan"), 0.0
    df = float((va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1)))
    statistic = abs(mean_a - mean_b) / se
    p = float(2.0 * stats.t.sf(statistic, df))
    return float(mean_a - mean_b), se, df, p


def benjamini_hochberg_threshold(p_values: Sequence[float], q: float) -> float:
    """Largest p passing the BH step-up procedure, or 0 if none do."""

    ordered = np.sort(np.asarray(list(p_values), dtype=float))
    m = ordered.size
    if m == 0:
        return 0.0
    passing = np.nonzero(ordered <= q * np.arange(1, m + 1) / m)[0]
    return float(ordered[passing.max()]) if passing.size else 0.0


def target_evidence(
    means: Mapping[str, float],
    sds: Mapping[str, float],
    counts: Mapping[str, int],
    alpha: float = 0.05,
    multiplicity: str = "bonferroni",
) -> list[PairEvidence]:
    """Per-pair evidence that the target's observed ordering is real.

    ``multiplicity`` is ``bonferroni`` (control the family-wise error rate over
    all pairs), ``bh`` (control the false discovery rate), or ``none``.
    """

    if multiplicity not in ("bonferroni", "bh", "none"):
        raise ValueError(f"unknown multiplicity {multiplicity!r}")
    names = sorted(means)
    pairs = list(combinations(names, 2))
    if not pairs:
        raise ValueError("at least two candidates are required")

    raw = []
    for left, right in pairs:
        gap, se, df, p = welch_pair(means[left], sds[left], counts[left], means[right], sds[right], counts[right])
        raw.append((left, right, gap, se, df, p))

    p_values = [row[5] for row in raw]
    if multiplicity == "bonferroni":
        cutoff = alpha / len(pairs)
    elif multiplicity == "bh":
        cutoff = benjamini_hochberg_threshold(p_values, alpha)
    else:
        cutoff = alpha

    evidence = []
    for left, right, gap, se, df, p in raw:
        # Probability that the observed sign is the true sign, under a Gaussian
        # approximation to the difference of cell means. Used as a weight, not a
        # tail test, which is why the normal rather than the t appears here.
        probability = float(stats.norm.cdf(abs(gap) / se)) if se > 0 else 1.0
        evidence.append(
            PairEvidence(
                left=left,
                right=right,
                gap=gap,
                standard_error=se,
                welch_df=df,
                p_value=p,
                determined=bool(p <= cutoff),
                probability_observed_order_correct=probability,
            )
        )
    return evidence


def score_ranker(
    predictions: Mapping[tuple[str, str], float],
    evidence: Iterable[PairEvidence],
    lower_is_better: bool = True,
) -> dict[str, float]:
    """Score one ranker under observed-target, determined-subset and expected rules.

    ``predictions`` maps an ordered pair to the ranker's predicted difference
    (left minus right); its sign is the predicted ordering.
    """

    observed_errors = 0
    scored = 0
    determined_errors = 0
    determined_total = 0
    expected_errors = 0.0

    for item in evidence:
        key = (item.left, item.right)
        if key not in predictions:
            continue
        predicted = predictions[key]
        if not np.isfinite(predicted) or predicted == 0 or item.gap == 0:
            continue
        scored += 1
        agrees = (predicted < 0) == (item.gap < 0) if lower_is_better else (predicted > 0) == (item.gap > 0)
        if not agrees:
            observed_errors += 1
        if item.determined:
            determined_total += 1
            if not agrees:
                determined_errors += 1
        probability = item.probability_observed_order_correct
        expected_errors += (1.0 - probability) if agrees else probability

    return {
        "n_scored": scored,
        "observed_errors": observed_errors,
        "observed_rate": observed_errors / scored if scored else float("nan"),
        "determined_pairs": determined_total,
        "determined_errors": determined_errors,
        "determined_rate": determined_errors / determined_total if determined_total else float("nan"),
        "expected_errors": expected_errors,
        "expected_rate": expected_errors / scored if scored else float("nan"),
    }


def decomposition(
    left_predictions: Mapping[tuple[str, str], float],
    right_predictions: Mapping[tuple[str, str], float],
    evidence: Iterable[PairEvidence],
    lower_is_better: bool = True,
) -> dict[str, object]:
    """Three-way error accounting between two rankers.

    Named for what it counts, which the earlier "fit error / inherited" framing
    did not: ``introduced`` is where the left ranker is wrong and the right one
    right, ``corrected`` the converse, ``shared`` where both are wrong. Only the
    first two are the discordant cells of a paired comparison.
    """

    introduced: list[tuple[str, str]] = []
    corrected: list[tuple[str, str]] = []
    shared: list[tuple[str, str]] = []
    both_right = 0

    for item in evidence:
        key = (item.left, item.right)
        if key not in left_predictions or key not in right_predictions:
            continue
        a, b = left_predictions[key], right_predictions[key]
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0 or b == 0 or item.gap == 0:
            continue

        def agrees(value: float) -> bool:
            return (value < 0) == (item.gap < 0) if lower_is_better else (value > 0) == (item.gap > 0)

        left_ok, right_ok = agrees(a), agrees(b)
        if left_ok and right_ok:
            both_right += 1
        elif not left_ok and right_ok:
            introduced.append(key)
        elif left_ok and not right_ok:
            corrected.append(key)
        else:
            shared.append(key)

    determined = {(e.left, e.right): e.determined for e in evidence}
    return {
        "introduced": len(introduced),
        "corrected": len(corrected),
        "shared": len(shared),
        "both_correct": both_right,
        "net_introduced": len(introduced) - len(corrected),
        "introduced_pairs": introduced,
        "corrected_pairs": corrected,
        "shared_pairs": shared,
        "shared_determined": sum(1 for k in shared if determined.get(k, False)),
        "shared_undetermined": sum(1 for k in shared if not determined.get(k, True)),
        "corrected_determined": sum(1 for k in corrected if determined.get(k, False)),
        "corrected_undetermined": sum(1 for k in corrected if not determined.get(k, True)),
    }


__all__ = [
    "PairEvidence",
    "bootstrap_two_sided_p",
    "benjamini_hochberg_threshold",
    "decomposition",
    "score_ranker",
    "target_evidence",
    "welch_pair",
]
