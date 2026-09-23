"""Empirical-Bayes variance moderation for many cells with few seeds.

A cell standard deviation from three seeds is unreliable, and biased low as a
standard deviation. The PolyPythias analysis in this project puts the median
3-seed/10-seed ratio at 0.78. The fix follows Smyth (2004), *Linear Models and
Empirical Bayes Methods for Assessing Differential Expression in Microarray
Experiments*, Statistical Applications in Genetics and Molecular Biology 3(1).

A cell's variance sigma^2 is given a scaled inverse-chi-square prior with d0
degrees of freedom and scale s0^2, and the prior is estimated from all cells at
the same scale. The posterior (moderated) variance is

    s~^2 = (d0 s0^2 + d s^2) / (d0 + d)

and carries d0 + d degrees of freedom. The prior is fitted by moment matching
on log s^2, exactly as limma's ``fitFDist`` does (checked against the limma
source): with e = log s^2 - digamma(d/2) + log(d/2),

    trigamma(d0/2) = var(e) - mean(trigamma(d/2))
    s0^2           = exp(mean(e) + digamma(d0/2) - log(d0/2)).

If the first right-hand side is not positive, the sample variances are no more
dispersed than sampling alone explains. Then d0 is infinite and s0^2 is their
mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import special, stats

from asla.analysis.target_scoring import PairEvidence, benjamini_hochberg_threshold


@dataclass(frozen=True)
class VariancePrior:
    """Estimated prior for one scale: d0 and s0^2, from ``n_cells`` sample variances on ``df`` each."""

    df_prior: float
    var_prior: float
    n_cells: int
    df_residual: float


def trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y > 0 (Newton iteration from limma's ``trigammaInverse``)."""

    if not np.isfinite(x) or x <= 0:
        raise ValueError("trigamma_inverse needs a positive finite argument")
    if x > 1e7:
        return float(1 / np.sqrt(x))
    if x < 1e-6:
        return float(1 / x)
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = float(special.polygamma(1, y))
        dif = tri * (1 - tri / x) / float(special.polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _logmdigamma(x: float | np.ndarray) -> np.ndarray:
    return np.log(x) - special.digamma(x)


def fit_prior(sample_variances: Sequence[float], df: float) -> VariancePrior:
    """Moment estimate of (d0, s0^2) from sample variances that each carry ``df`` degrees of freedom."""

    s2 = np.asarray(sample_variances, dtype=float)
    s2 = s2[np.isfinite(s2) & (s2 >= 0)]
    n = s2.size
    if n < 3:
        raise ValueError("variance moderation needs at least three cells")
    median = float(np.median(s2))
    if median == 0:
        raise ValueError("more than half of the sample variances are zero")
    s2 = np.maximum(s2, 1e-5 * median)
    e = np.log(s2) + _logmdigamma(df / 2)
    excess = float(np.var(e, ddof=1) - special.polygamma(1, df / 2))
    if excess > 0:
        d0 = 2 * trigamma_inverse(excess)
        s0 = float(np.exp(e.mean() - _logmdigamma(d0 / 2)))
    else:
        d0, s0 = float("inf"), float(s2.mean())
    return VariancePrior(df_prior=float(d0), var_prior=s0, n_cells=int(n), df_residual=float(df))


def moderate(sample_variance: float | np.ndarray, prior: VariancePrior) -> np.ndarray:
    """Posterior variance (d0 s0^2 + d s^2)/(d0 + d); the prior variance itself when d0 is infinite."""

    s2 = np.asarray(sample_variance, dtype=float)
    if not np.isfinite(prior.df_prior):
        return np.full_like(s2, prior.var_prior)
    d, d0 = prior.df_residual, prior.df_prior
    return (d0 * prior.var_prior + d * s2) / (d0 + d)


def posterior_df(prior: VariancePrior) -> float:
    return prior.df_residual + prior.df_prior


def homogeneity(groups: Sequence[Sequence[float]]) -> dict[str, float]:
    """Bartlett's and the Brown-Forsythe test of equal variance across recipes at one scale."""

    arrays = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    bartlett = stats.bartlett(*arrays)
    brown_forsythe = stats.levene(*arrays, center="median")
    return {
        "bartlett_statistic": float(bartlett.statistic),
        "bartlett_p": float(bartlett.pvalue),
        "brown_forsythe_statistic": float(brown_forsythe.statistic),
        "brown_forsythe_p": float(brown_forsythe.pvalue),
    }


def moderated_evidence(
    means: Mapping[str, float],
    sample_variances: Mapping[str, float],
    counts: Mapping[str, int],
    prior: VariancePrior,
    alpha: float = 0.05,
    multiplicity: str = "bonferroni",
) -> list[PairEvidence]:
    """Per-pair target evidence with moderated variances.

    The gap's standard error uses each recipe's posterior variance. Its degrees of
    freedom are Welch-Satterthwaite with each variance carrying d + d0, which is
    the natural extension of the moderated t to unequal variances. The
    probability that the observed order is correct is the Student-t CDF at
    those degrees of freedom.
    """

    if multiplicity not in ("bonferroni", "bh", "none"):
        raise ValueError(f"unknown multiplicity {multiplicity!r}")
    names = sorted(means)
    posterior = {name: float(moderate(sample_variances[name], prior)) for name in names}
    df_each = posterior_df(prior)
    raw = []
    for left, right in combinations(names, 2):
        va, vb = posterior[left] / counts[left], posterior[right] / counts[right]
        se = float(np.sqrt(va + vb))
        gap = float(means[left] - means[right])
        if np.isfinite(df_each):
            df = float((va + vb) ** 2 / (va**2 / df_each + vb**2 / df_each))
        else:
            df = float("inf")
        statistic = abs(gap) / se if se > 0 else float("inf")
        p = float(2 * stats.t.sf(statistic, df)) if np.isfinite(df) else float(2 * stats.norm.sf(statistic))
        probability = float(stats.t.cdf(statistic, df)) if np.isfinite(df) else float(stats.norm.cdf(statistic))
        raw.append((left, right, gap, se, df, p, probability))
    p_values = [row[5] for row in raw]
    if multiplicity == "bonferroni":
        cutoff = alpha / len(raw)
    elif multiplicity == "bh":
        cutoff = benjamini_hochberg_threshold(p_values, alpha)
    else:
        cutoff = alpha
    return [
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
        for left, right, gap, se, df, p, probability in raw
    ]


def calibrated_evidence(
    means: Mapping[str, float],
    sample_variances: Mapping[str, float],
    counts: Mapping[str, int],
    prior: VariancePrior,
    alpha: float = 0.05,
    multiplicity: str = "bonferroni",
) -> list[PairEvidence]:
    """Target evidence under the procedure that passed its known-answer tests.

    Determination uses the raw Welch test. Of the three procedures tested under a
    true null on PolyPythias, it is the only one that never exceeded its nominal
    rate while staying close to it; the moderated t with d + d0 degrees of
    freedom was anti-conservative by up to 4.9x at the Bonferroni level. The
    probability that the observed order is correct is the Student-t CDF at the
    moderated standard error with the raw Welch degrees of freedom. That variant
    had the lowest Brier score against held-out seeds (results/target_scoring/
    moderated_variance.json).
    """

    from asla.analysis.target_scoring import target_evidence

    raw = target_evidence(
        means, {name: float(np.sqrt(v)) for name, v in sample_variances.items()}, counts, alpha, multiplicity
    )
    posterior = {name: float(moderate(v, prior)) for name, v in sample_variances.items()}
    out = []
    for item in raw:
        se = float(np.sqrt(posterior[item.left] / counts[item.left] + posterior[item.right] / counts[item.right]))
        if item.gap == 0:
            probability = 0.5
        elif se == 0 or not np.isfinite(item.welch_df):
            probability = 1.0
        else:
            probability = float(stats.t.cdf(abs(item.gap) / se, item.welch_df))
        out.append(
            PairEvidence(
                left=item.left,
                right=item.right,
                gap=item.gap,
                standard_error=se,
                welch_df=item.welch_df,
                p_value=item.p_value,
                determined=item.determined,
                probability_observed_order_correct=probability,
            )
        )
    return out


def evidence_from_cells(values_by_recipe: Mapping[str, Sequence[float]], **kwargs: object) -> list[PairEvidence]:
    """Calibrated evidence from each recipe's seed values at the target scale."""

    means = {name: float(np.mean(v)) for name, v in values_by_recipe.items()}
    s2 = {name: float(np.var(v, ddof=1)) for name, v in values_by_recipe.items()}
    counts = {name: len(v) for name, v in values_by_recipe.items()}
    df = min(counts.values()) - 1
    prior = fit_prior(list(s2.values()), df)
    return calibrated_evidence(means, s2, counts, prior, **kwargs)  # type: ignore[arg-type]


def calibrated_target_evidence(
    means: Mapping[str, float],
    sds: Mapping[str, float],
    counts: Mapping[str, int],
    alpha: float = 0.05,
    multiplicity: str = "bonferroni",
) -> list[PairEvidence]:
    """Drop-in replacement for ``target_evidence`` with the calibrated procedure.

    Same signature and determination; the expected-error weights use moderated
    variances, with the prior fitted across the given recipes at this scale.
    """

    s2 = {name: float(sd) ** 2 for name, sd in sds.items()}
    prior = fit_prior(list(s2.values()), min(counts.values()) - 1)
    return calibrated_evidence(means, s2, counts, prior, alpha, multiplicity)


__all__ = [
    "calibrated_evidence",
    "calibrated_target_evidence",
    "evidence_from_cells",
    "VariancePrior",
    "fit_prior",
    "homogeneity",
    "moderate",
    "moderated_evidence",
    "posterior_df",
    "trigamma_inverse",
]
