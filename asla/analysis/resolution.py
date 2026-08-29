"""Leaderboard resolution limit: which orderings are statistically distinguishable?

Generalises the 1.2B/8xC non-identifiability result into a reusable diagnostic.
Given a leaderboard's entries, their metric values, and an estimate of run-to-run
noise, it returns the smallest gap that is resolvable at a stated seed budget and
multiplicity correction, and reports which adjacent pairs and which top-k
orderings cannot be distinguished from ties.

The motivating measurement: on the Fantastic Optimizers grid at 1.2B with 8x
Chinchilla data, the smallest adjacent gap between tuned optimizers is 7e-6 nats
against a seed standard deviation of ~1.4e-3 nats. Noise is roughly 200x the gap,
so that ordering is not identifiable at any feasible seed count - a leaderboard
ranking those entries is ranking noise.

**The noise estimate is the load-bearing input.** Where a leaderboard reports one
run per entry, sigma cannot be measured from the leaderboard itself and must come
from elsewhere. Transferring it across studies is an assumption, not a
measurement, and every result here carries a sensitivity band over sigma rather
than a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

# A gap this many times smaller than the noise needs so many seeds that no
# realistic budget resolves it; see `minimum_detectable_gap` for the algebra.
UNIDENTIFIABLE_NOISE_TO_GAP = 10.0


@dataclass(frozen=True)
class AdjacentPair:
    """One adjacent pair on a leaderboard and whether its order is resolvable."""

    better: str
    worse: str
    gap: float
    noise_to_gap: float
    seeds_required: int | None
    resolvable_at_budget: bool
    identifiable_at_any_budget: bool


def minimum_detectable_gap(
    n_seeds: int,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_comparisons: int = 1,
) -> float:
    """Smallest mean gap a two-sided Welch test resolves at ``n_seeds`` per entry.

    ``n_comparisons`` applies a Bonferroni correction, which is what a
    leaderboard needs: ranking k entries makes many simultaneous comparisons, and
    testing each at nominal alpha would inflate the family-wise error rate.
    """

    if n_seeds < 2:
        raise ValueError("resolving a gap requires at least 2 seeds per entry")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    corrected_alpha = alpha / max(1, n_comparisons)
    dof = max(2 * n_seeds - 2, 1)
    critical = student_t.ppf(1 - corrected_alpha / 2, dof) + student_t.ppf(power, dof)
    return float(np.sqrt(2.0 / n_seeds) * critical * sigma)


def seeds_required(
    gap: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_comparisons: int = 1,
    max_seeds: int = 1_000_000,
) -> int | None:
    """Seeds per entry needed to resolve ``gap``; ``None`` when beyond ``max_seeds``."""

    if gap <= 0 or sigma <= 0:
        return None
    corrected_alpha = alpha / max(1, n_comparisons)
    n = 2.0 * (norm.ppf(1 - corrected_alpha / 2) + norm.ppf(power)) ** 2 * (sigma / gap) ** 2
    n = max(n, 2.0)
    for _ in range(60):
        dof = max(2.0 * n - 2.0, 1.0)
        critical = student_t.ppf(1 - corrected_alpha / 2, dof) + student_t.ppf(power, dof)
        updated = max(2.0 * critical**2 * (sigma / gap) ** 2, 2.0)
        if abs(updated - n) < 1e-6:
            n = updated
            break
        n = updated
    if n > max_seeds:
        return None
    return int(np.ceil(n))


def resolution_report(
    entries: Mapping[str, float],
    sigma: float,
    seed_budget: int = 3,
    alpha: float = 0.05,
    power: float = 0.8,
    lower_is_better: bool = True,
    top_k: int = 3,
    multiplicity: str = "bonferroni",
) -> dict[str, Any]:
    """Resolution diagnostic for one leaderboard.

    ``entries`` maps entry name to its reported metric value. ``sigma`` is the
    run-to-run standard deviation of that metric on a single entry.
    ``seed_budget`` is how many runs per entry the diagnostic assumes are
    affordable; pairs needing more are reported as unresolved at that budget.
    """

    if len(entries) < 2:
        raise ValueError("a leaderboard needs at least two entries")
    if multiplicity not in ("bonferroni", "none"):
        raise ValueError(f"unknown multiplicity {multiplicity!r}; choose 'bonferroni' or 'none'")
    ordered = sorted(entries.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    n_entries = len(ordered)
    n_pairs = n_entries * (n_entries - 1) // 2
    comparisons = n_pairs if multiplicity == "bonferroni" else 1
    detectable = minimum_detectable_gap(seed_budget, sigma, alpha, power, comparisons)

    pairs: list[AdjacentPair] = []
    for (better, better_value), (worse, worse_value) in zip(ordered[:-1], ordered[1:]):
        gap = abs(worse_value - better_value)
        needed = seeds_required(gap, sigma, alpha, power, comparisons)
        pairs.append(
            AdjacentPair(
                better=str(better),
                worse=str(worse),
                gap=float(gap),
                noise_to_gap=float(sigma / gap) if gap > 0 else float("inf"),
                seeds_required=needed,
                resolvable_at_budget=bool(gap >= detectable),
                identifiable_at_any_budget=bool(gap > 0 and (sigma / gap) <= UNIDENTIFIABLE_NOISE_TO_GAP),
            )
        )
    top = pairs[: max(0, min(top_k - 1, len(pairs)))]
    unresolved_top = [p for p in top if not p.resolvable_at_budget]
    return {
        "n_entries": n_entries,
        "n_adjacent_pairs": len(pairs),
        "sigma": float(sigma),
        "seed_budget": int(seed_budget),
        "multiplicity": multiplicity,
        "n_comparisons_corrected": comparisons,
        "minimum_detectable_gap": detectable,
        "ranking": [name for name, _ in ordered],
        "adjacent_pairs": [p.__dict__ for p in pairs],
        "n_unresolved_at_budget": sum(1 for p in pairs if not p.resolvable_at_budget),
        "fraction_unresolved_at_budget": sum(1 for p in pairs if not p.resolvable_at_budget) / len(pairs),
        "n_unidentifiable_at_any_budget": sum(1 for p in pairs if not p.identifiable_at_any_budget),
        "top_k": int(top_k),
        "top_k_fully_resolved": bool(not unresolved_top),
        "top_k_unresolved_pairs": [f"{p.better} vs {p.worse}" for p in unresolved_top],
        "headline": (
            f"{sum(1 for p in pairs if not p.resolvable_at_budget)} of {len(pairs)} adjacent orderings are not "
            f"resolvable with {seed_budget} seeds per entry"
        ),
    }


def sensitivity(
    entries: Mapping[str, float],
    sigma: float,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    seed_budgets: Iterable[int] = (3, 5, 10),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Resolution across noise multipliers and seed budgets.

    A transferred noise estimate is an assumption; this reports how the verdict
    moves if that estimate is wrong by a factor of two in either direction.
    """

    rows = []
    for multiplier in multipliers:
        for budget in seed_budgets:
            report = resolution_report(entries, sigma * multiplier, seed_budget=budget, **kwargs)
            rows.append(
                {
                    "sigma_multiplier": float(multiplier),
                    "sigma": float(sigma * multiplier),
                    "seed_budget": int(budget),
                    "minimum_detectable_gap": report["minimum_detectable_gap"],
                    "n_unresolved": report["n_unresolved_at_budget"],
                    "fraction_unresolved": report["fraction_unresolved_at_budget"],
                    "top_k_fully_resolved": report["top_k_fully_resolved"],
                }
            )
    return rows


__all__ = [
    "AdjacentPair",
    "UNIDENTIFIABLE_NOISE_TO_GAP",
    "minimum_detectable_gap",
    "resolution_report",
    "seeds_required",
    "sensitivity",
]
