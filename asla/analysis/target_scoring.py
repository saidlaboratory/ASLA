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


def expected_charges(
    predictions: Mapping[tuple[str, str], float],
    evidence: Iterable[PairEvidence],
    lower_is_better: bool = True,
) -> dict[tuple[str, str], float]:
    """Per-pair expected-error charge, on exactly the pairs :func:`score_ranker` scores."""

    charges: dict[tuple[str, str], float] = {}
    for item in evidence:
        key = (item.left, item.right)
        predicted = predictions.get(key)
        if predicted is None or not np.isfinite(predicted) or predicted == 0 or item.gap == 0:
            continue
        agrees = (predicted < 0) == (item.gap < 0) if lower_is_better else (predicted > 0) == (item.gap > 0)
        probability = item.probability_observed_order_correct
        charges[key] = (1.0 - probability) if agrees else probability
    return charges


def pair_matrix(kernel: Mapping[tuple[str, str], float]) -> tuple[list[str], np.ndarray]:
    """Symmetric candidate-by-candidate matrix of a pair kernel; NaN where unscored."""

    names = sorted({name for pair in kernel for name in pair})
    index = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(names), len(names)), np.nan)
    for (a, b), value in kernel.items():
        matrix[index[a], index[b]] = matrix[index[b], index[a]] = float(value)
    return names, matrix


def _weighted_pair_mean(matrix: np.ndarray, counts: np.ndarray) -> float:
    weights = np.outer(counts, counts).astype(float)
    np.fill_diagonal(weights, 0.0)
    valid = np.isfinite(matrix) & (weights > 0)
    total = weights[valid].sum()
    return float((weights[valid] * matrix[valid]).sum() / total) if total > 0 else float("nan")


def candidate_bootstrap(kernel: Mapping[tuple[str, str], float], rng: np.random.Generator, n_resamples: int) -> np.ndarray:
    """Bootstrap the mean of a pair kernel by resampling candidates.

    Each replicate draws candidates with replacement and weights every pair of
    *distinct* candidates by the product of their draw counts, so a replicate is
    the statistic at the full candidate count. Scoring only the unique
    candidates drawn, as an earlier version did, evaluates it at about 63% of
    the candidates and overstates the spread.
    """

    _, matrix = pair_matrix(kernel)
    n = matrix.shape[0]
    draws = np.empty(n_resamples)
    for r in range(n_resamples):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        draws[r] = _weighted_pair_mean(matrix, counts)
    return draws[np.isfinite(draws)]


def u_statistic_components(kernel: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Unbiased Hoeffding components of the pair-mean U-statistic.

    ``zeta1 = E[h(1,2) h(1,3)] - theta^2`` and ``zeta2 = E[h(1,2)^2] - theta^2``,
    with ``theta^2`` estimated without bias from products over *disjoint* pairs.
    Centring on the estimated mean instead biases ``zeta1`` low, by about 11% in
    standard-error terms at 25 candidates. Requires every pair to be scored.
    """

    _, matrix = pair_matrix(kernel)
    n = matrix.shape[0]
    off = ~np.eye(n, dtype=bool)
    if n < 4 or not np.all(np.isfinite(matrix[off])):
        raise ValueError("u_statistic_components needs at least four candidates and every pair scored")
    h = np.where(off, matrix, 0.0)
    total = h.sum() / 2.0
    squares = (h**2).sum() / 2.0
    row = h.sum(axis=1)
    shared = float(((row**2).sum() - (h**2).sum()) / 2.0)  # unordered pairs of pairs sharing one candidate
    disjoint = float((total**2 - squares - 2.0 * shared) / 2.0)
    n_pairs = n * (n - 1) / 2.0
    n_shared = n * (n - 1) * (n - 2) / 2.0
    n_disjoint = n_pairs * (n - 2) * (n - 3) / 4.0
    theta_squared = disjoint / n_disjoint
    return {
        "n_candidates": n,
        "mean": float(total / n_pairs),
        "zeta1": float(shared / n_shared - theta_squared),
        "zeta2": float(squares / n_pairs - theta_squared),
    }


def u_statistic_test(kernel: Mapping[tuple[str, str], float], alpha: float = 0.05) -> dict[str, float | bool]:
    """Normal interval and two-sided p for the pair mean, from the unbiased components.

    At the candidate counts and variance ratios seen here this is the most
    accurate candidate-level standard error available: in simulation with a known
    answer it is within a few percent, where the multiplicity-weighted bootstrap
    overstates by about 1.3x and the jackknife by about 1.2x when pair-level
    noise dominates (zeta2 much larger than zeta1).
    """

    components = u_statistic_components(kernel)
    variance = u_statistic_variance(components["zeta1"], components["zeta2"], int(components["n_candidates"]))
    se = float(np.sqrt(max(variance, 0.0)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    mean = components["mean"]
    low, high = mean - z * se, mean + z * se
    return {
        **components,
        "standard_error": se,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
        "p_two_sided": float(2 * stats.norm.sf(abs(mean) / se)) if se > 0 else float(mean == 0),
    }


def u_statistic_variance(zeta1: float, zeta2: float, n: int) -> float:
    """Variance of the pair-mean U-statistic over ``n`` candidates."""

    if n < 2:
        raise ValueError("need at least two candidates")
    return 2.0 / (n * (n - 1)) * (2.0 * (n - 2) * zeta1 + zeta2)


def jackknife_standard_error(kernel: Mapping[tuple[str, str], float]) -> float:
    """Delete-one-candidate jackknife standard error of the pair mean."""

    _, matrix = pair_matrix(kernel)
    n = matrix.shape[0]
    leave_out = []
    for i in range(n):
        counts = np.ones(n, dtype=int)
        counts[i] = 0
        leave_out.append(_weighted_pair_mean(matrix, counts))
    values = np.asarray(leave_out)
    return float(np.sqrt((n - 1) / n * ((values - values.mean()) ** 2).sum()))


def candidates_for_power(
    zeta1: float, zeta2: float, delta: float, power: float = 0.8, alpha: float = 0.05, n_max: int = 100_000
) -> int | None:
    """Smallest candidate count at which a two-sided z test detects ``delta``."""

    z = float(stats.norm.ppf(1 - alpha / 2))
    for n in range(3, n_max + 1):
        se = float(np.sqrt(max(u_statistic_variance(zeta1, zeta2, n), 0.0)))
        if se == 0:
            return n
        achieved = float(stats.norm.cdf(abs(delta) / se - z) + stats.norm.cdf(-abs(delta) / se - z))
        if achieved >= power:
            return n
    return None


__all__ = [
    "PairEvidence",
    "bootstrap_two_sided_p",
    "candidate_bootstrap",
    "candidates_for_power",
    "expected_charges",
    "jackknife_standard_error",
    "pair_matrix",
    "u_statistic_components",
    "u_statistic_test",
    "u_statistic_variance",
    "benjamini_hochberg_threshold",
    "decomposition",
    "score_ranker",
    "target_evidence",
    "welch_pair",
]
