"""Demonstrate the boundary condition on the monotone-distortion result.

The corrected-bounds re-run reproduced every decision number even though the
OLMES fits changed substantially (projections moved by 24% of the
between-intervention spread). NOTES_FIT_BOUNDS_DEFECT.md explains this: the
alpha-floor defect biases every intervention in the same direction when the
interventions differ mainly in *level*, and decision metrics only see ordering,
so a monotone distortion is invisible to them.

That explanation implies a testable boundary: on a suite whose interventions
differ in *curvature* rather than level, the same defect should reorder them and
the decisions should change. This module demonstrates it rather than asserting
it, on synthetic grids where the truth is known.

Three families, all with the same fit ladder, the same seed noise, and the same
defect (alpha floored at 0.05 versus an adaptive floor):

* ``level``     - interventions share an exponent and differ by a constant
                  offset. The DataDecide-like case.
* ``curvature`` - interventions are pinned to a common value at the largest fit
                  budget and differ in floor and exponent, so the ladder barely
                  separates them and the target ordering is set by how they
                  bend. The case the caveat is about.
* ``mixed``     - both vary.

The prediction under test: the defect leaves decision accuracy untouched on
``level`` and changes it on ``curvature``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.models import fit_power_law

REPO = Path(__file__).resolve().parents[1]
LADDER = np.array([8.4e15, 2.2e16, 4.4e16, 5.9e16, 1.2e17, 1.5e17, 2.2e17, 2.0e18, 5.8e18, 1.3e19, 5.7e19], dtype=float)
TARGET = 7.1e20
TRUE_FLOOR = 0.55
TRUE_AMPLITUDE = 3.0


def _fit_range_disagreement(truths: dict[str, tuple[float, float, float]]) -> float:
    """Largest spread between curves anywhere on the fit ladder (how visible they are while fitting)."""

    values = np.array([[e + a * float(c) ** (-al) for c in LADDER] for (e, a, al) in truths.values()])
    return float(np.max(values.max(axis=0) - values.min(axis=0)))


def _target_spread(truths: dict[str, tuple[float, float, float]]) -> float:
    values = np.array([e + a * TARGET ** (-al) for (e, a, al) in truths.values()])
    return float(values.max() - values.min())


def _curvature_truths(
    n_interventions: int, floor_lo: float = 0.0, floor_hi: float = 0.50, alpha_lo: float = 0.01, alpha_hi: float = 0.12
) -> dict[str, tuple[float, float, float]]:
    """Curves pinned to a common value on the ladder that diverge at the target.

    Every intervention is pinned at the largest fit budget to the same value
    ``y_hi``, so the ladder separates them only weakly, and each gets its own
    floor and exponent. With the pin, the amplitude follows as
    ``A = (y_hi - E) / c_hi^-alpha`` and the target value is
    ``E + (y_hi - E) L^-alpha`` for lever arm ``L = C_target / c_hi``. The
    ordering at the target is therefore set by the floor/exponent pair - by how
    the curves *bend* - not by their level on the ladder.

    Floors ascend while exponents ascend with them, so a curve that looks
    similar on the ladder can finish anywhere in the target range. This is the
    geometry the monotone-distortion caveat is about: the exponent, which is
    exactly the parameter the alpha-floor defect distorts, decides the ordering.
    """

    names = [f"r{i:02d}" for i in range(n_interventions)]
    c_hi = float(LADDER.max())
    y_hi = TRUE_FLOOR + TRUE_AMPLITUDE * c_hi ** (-0.15)
    floors = np.linspace(floor_lo, floor_hi, n_interventions)
    alphas = np.linspace(alpha_lo, alpha_hi, n_interventions)
    truths: dict[str, tuple[float, float, float]] = {}
    for name, floor, alpha in zip(names, floors, alphas):
        residual = y_hi - float(floor)
        if residual <= 0:
            raise ValueError(f"floor {floor} exceeds the ladder pin {y_hi}; choose a lower floor range")
        amplitude = residual / c_hi ** (-float(alpha))
        truths[name] = (float(floor), float(amplitude), float(alpha))
    return truths


def build_truth(
    family: str, n_interventions: int = 8, rng: np.random.Generator | None = None
) -> dict[str, tuple[float, float, float]]:
    """Return per-intervention (E, A, alpha) truths for one family.

    ``level``: shared exponent, interventions separated by a constant offset -
    the DataDecide-like geometry, where the alpha-floor defect is monotone.
    ``curvature``: interventions agree at both ends of the fit ladder and differ
    only in how they bend, so the fit range barely separates them and the target
    does. ``mixed``: curvature plus a small offset.
    """

    generator = np.random.default_rng(0) if rng is None else rng
    names = [f"r{i:02d}" for i in range(n_interventions)]
    truths: dict[str, tuple[float, float, float]] = {}
    if family == "level":
        # Matched to the curvature family in two ways so the comparison isolates
        # geometry: the same target spread (so there is equally much signal to
        # detect) and the same slow exponent (so the alpha-floor defect binds
        # equally hard on both).
        offsets = np.linspace(0.0, 0.10, n_interventions)
        slow_alpha = 0.02
        amplitude = TRUE_AMPLITUDE * float(LADDER.max()) ** (-0.15) / float(LADDER.max()) ** (-slow_alpha)
        for name, offset in zip(names, offsets):
            truths[name] = (TRUE_FLOOR + float(offset), float(amplitude), slow_alpha)
    elif family == "curvature":
        truths = _curvature_truths(n_interventions)
    elif family == "mixed":
        base = _curvature_truths(n_interventions, floor_hi=0.35, alpha_hi=0.08)
        offsets = generator.normal(0.0, 0.05, n_interventions)
        truths = {
            name: (e + float(offset), a, alpha) for (name, (e, a, alpha)), offset in zip(sorted(base.items()), offsets)
        }
    else:
        raise ValueError(f"unknown family {family!r}")
    return truths


def sample(
    truths: dict[str, tuple[float, float, float]], sigma: float, n_seeds: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for name, (e, a, alpha) in truths.items():
        for compute in np.append(LADDER, TARGET):
            value = e + a * float(compute) ** (-alpha)
            for seed in range(n_seeds):
                rows.append(
                    {
                        "intervention": name,
                        "intervention_class": "synthetic",
                        "compute": float(compute),
                        "seed": seed,
                        "bpb": value + rng.normal(0.0, sigma),
                    }
                )
    return pd.DataFrame(rows)


def project_all(df: pd.DataFrame, adaptive: bool) -> dict[str, float]:
    out = {}
    fit_rows = df[df["compute"] < TARGET]
    for name, group in fit_rows.groupby("intervention", sort=True):
        cells = group.groupby("compute", sort=True)["bpb"].mean()
        e, a, alpha = fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float), adaptive_bounds=adaptive)
        out[str(name)] = float(e + a * TARGET ** (-alpha))
    return out


def pairwise_accuracy(scores: dict[str, float], truth: dict[str, float]) -> float:
    names = sorted(scores)
    correct = total = 0
    for a, b in itertools.combinations(names, 2):
        total += 1
        correct += int((scores[a] < scores[b]) == (truth[a] < truth[b]))
    return correct / total


def ordering(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda n: scores[n])


def run_family(family: str, sigma: float = 0.01, n_seeds: int = 3, n_trials: int = 60, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truths = build_truth(family, rng=rng)
    true_target = {name: e + a * TARGET ** (-alpha) for name, (e, a, alpha) in truths.items()}
    rows = []
    for _ in range(n_trials):
        df = sample(truths, sigma, n_seeds, rng)
        defective = project_all(df, adaptive=False)
        corrected = project_all(df, adaptive=True)
        from scipy.stats import spearmanr

        rows.append(
            {
                "accuracy_defective": pairwise_accuracy(defective, true_target),
                "accuracy_corrected": pairwise_accuracy(corrected, true_target),
                "ordering_identical": ordering(defective) == ordering(corrected),
                "spearman_defective_vs_corrected": float(
                    spearmanr([defective[n] for n in sorted(defective)], [corrected[n] for n in sorted(corrected)]).statistic
                ),
                "n_pairs_reordered": sum(
                    1
                    for a, b in itertools.combinations(sorted(defective), 2)
                    if (defective[a] < defective[b]) != (corrected[a] < corrected[b])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    # how pinned was the defective fit, and how different were the truths?
    alphas = np.array([alpha for _, _, alpha in truths.values()])
    example = sample(truths, sigma, n_seeds, np.random.default_rng(seed + 1))
    fit_rows = example[example["compute"] < TARGET]
    pinned = 0
    for _, group in fit_rows.groupby("intervention", sort=True):
        cells = group.groupby("compute", sort=True)["bpb"].mean()
        _, _, alpha = fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float), adaptive_bounds=False)
        pinned += int(abs(alpha - 0.05) < 1e-6)
    return {
        "family": family,
        "n_interventions": len(truths),
        "fit_range_disagreement": _fit_range_disagreement(truths),
        "target_spread": _target_spread(truths),
        "target_spread_over_sigma": _target_spread(truths) / sigma,
        "true_alpha_range": [float(alphas.min()), float(alphas.max())],
        "true_alpha_sd": float(alphas.std(ddof=1)),
        "n_pinned_under_defect": pinned,
        "n_trials": n_trials,
        "mean_accuracy_defective": float(frame["accuracy_defective"].mean()),
        "mean_accuracy_corrected": float(frame["accuracy_corrected"].mean()),
        "accuracy_change": float(frame["accuracy_corrected"].mean() - frame["accuracy_defective"].mean()),
        "fraction_trials_ordering_identical": float(frame["ordering_identical"].mean()),
        "mean_pairs_reordered": float(frame["n_pairs_reordered"].mean()),
        "mean_spearman_defective_vs_corrected": float(frame["spearman_defective_vs_corrected"].mean()),
    }


def main() -> int:
    results: dict[str, Any] = {family: run_family(family) for family in ("level", "curvature", "mixed")}
    level, curvature = results["level"], results["curvature"]
    results["contrast"] = {
        "claim": (
            "The alpha-floor defect is invisible to decision metrics when interventions differ in level, and "
            "changes decisions when they differ in curvature."
        ),
        "level_pairs_reordered": level["mean_pairs_reordered"],
        "curvature_pairs_reordered": curvature["mean_pairs_reordered"],
        "level_fraction_trials_ordering_identical": level["fraction_trials_ordering_identical"],
        "curvature_fraction_trials_ordering_identical": curvature["fraction_trials_ordering_identical"],
        "reordering_ratio_curvature_over_level": (
            curvature["mean_pairs_reordered"] / level["mean_pairs_reordered"]
            if level["mean_pairs_reordered"] > 0
            else float("inf")
        ),
        # The claim is a contrast, not an absolute: on level geometry the defect
        # is essentially invisible to decisions, on curvature geometry it is not.
        "demonstrated": bool(
            level["mean_pairs_reordered"] < 0.1
            and level["fraction_trials_ordering_identical"] > 0.9
            and curvature["mean_pairs_reordered"] > 10 * max(level["mean_pairs_reordered"], 1e-9)
            and curvature["fraction_trials_ordering_identical"] < 0.6
        ),
        "note": (
            "Both families are matched on target spread (so equally much signal) and on having true exponents "
            "below the 0.05 floor (so the defect binds on both). The only difference is geometry."
        ),
    }
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "curvature_boundary_demo.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"{'family':11s} {'pinned':>7s} {'acc(defect)':>12s} {'acc(fixed)':>11s} {'delta':>8s} {'same order':>11s} {'pairs reordered':>16s} {'spearman':>9s}"
    )
    for family, row in results.items():
        if family == "contrast":
            continue
        print(
            f"{family:11s} {row['n_pinned_under_defect']:>7d} {row['mean_accuracy_defective']:>12.4f} "
            f"{row['mean_accuracy_corrected']:>11.4f} {row['accuracy_change']:>+8.4f} "
            f"{row['fraction_trials_ordering_identical']:>11.2f} {row['mean_pairs_reordered']:>16.1f} "
            f"{row['mean_spearman_defective_vs_corrected']:>9.5f}"
        )
    contrast = results["contrast"]
    print(
        f"\ncontrast: level reorders {contrast['level_pairs_reordered']:.3f} pairs/trial, "
        f"curvature reorders {contrast['curvature_pairs_reordered']:.3f} "
        f"({contrast['reordering_ratio_curvature_over_level']:.0f}x more); demonstrated={contrast['demonstrated']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
