"""A3: power and ROC of the crossover detector at DataDecide's actual seed count.

The headline claims only 2 of 16 projection flips on C4 are inherited
crossovers. A hostile reading: the detector is underpowered at n=3 seeds, so
"few significant crossovers" could be a power artifact rather than a finding.

Two distinct questions are separated here, because they have different answers:

1. **Flip detection** (no test involved): does the *small-scale order* differ
   from the *target order*? This is a comparison of two sample means with no
   significance test, so it has no "power" in the testing sense - but it does
   have an error rate: with finite seeds a true non-crossing pair can appear
   flipped by noise, and a true crossing pair can appear unflipped. The
   decomposition counts (14 fit error vs 2 inherited) are built from flips,
   not from tests, so this is the error rate that matters for the headline.

2. **Significance labelling** (Welch + BH at the target): given a flipped
   pair, does the measured target gap survive multiplicity correction? This
   is a genuine power question at n=3.

The simulation places two interventions with a known crossover (or none),
samples seed noise at DataDecide's measured level, and asks how often the
detector recovers the truth. Sweeps seeds per cell and true target gap.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import independent_rederivation as ir
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def curve(compute: np.ndarray, e: float, a: float, alpha: float) -> np.ndarray:
    return e + a * np.asarray(compute, dtype=float) ** (-alpha)


def make_pair(
    target_gap: float,
    crossover: bool,
    budgets: np.ndarray,
    target: float,
    ladder_gap: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Two power-law curves with a controlled target gap, crossing or not.

    ``a`` is better (lower) at the target by ``target_gap``. With
    ``crossover`` the ``a`` curve is *worse* than ``b`` at the largest fit
    budget by ``ladder_gap``, so the small-scale order is the opposite of the
    target order (a true crossover between C_max and the target). Without it,
    ``a`` is better at the largest fit budget too, so no crossover exists.

    The ``a`` curve is solved for exactly: two constraints (value at C_max and
    value at the target) determine ``E`` and ``A`` for a chosen ``alpha``,
    which is itself chosen so that ``A > 0``.
    """

    base_e, base_a, base_alpha = 0.60, 3.0, 0.16
    b = {"e": base_e, "a": base_a, "alpha": base_alpha}
    c_max = float(budgets.max())
    b_max = float(curve(np.asarray([c_max]), **b)[0])
    b_target = float(curve(np.asarray([target]), **b)[0])
    a_at_target = b_target - target_gap
    # ``a`` is pinned at the target and at C_max. Crossing means it is *worse*
    # than ``b`` at C_max by ``ladder_gap``; not crossing means it is better
    # there, but by a smaller margin than at the target so the fitted curve
    # still decreases (a power law cannot widen its lead going backwards in
    # compute while keeping A > 0).
    a_at_max = b_max + ladder_gap if crossover else b_max - min(ladder_gap, target_gap * 0.5)
    # Two points (C_max, target) determine E and A for a chosen alpha.
    alpha_a = base_alpha * (1.6 if crossover else 1.0)
    denom = c_max ** (-alpha_a) - target ** (-alpha_a)
    if abs(denom) < 1e-30:
        return None, b
    amp = (a_at_max - a_at_target) / denom
    e_a = a_at_target - amp * target ** (-alpha_a)
    if amp <= 0:
        return None, b
    a_curve: dict[str, Any] | None = {"e": e_a, "a": amp, "alpha": alpha_a}
    assert a_curve is not None
    values_a = curve(budgets, **a_curve)
    values_b = curve(budgets, **b)
    ordered_wrong_on_ladder = bool(values_a[-1] > values_b[-1])
    if ordered_wrong_on_ladder != crossover:
        return None, b
    return a_curve, b


def simulate_once(
    a_curve: dict[str, Any],
    b_curve: dict[str, Any],
    budgets: np.ndarray,
    target: float,
    sigma_small: float,
    sigma_target: float,
    n_seeds: int,
    rng: np.random.Generator,
    q: float = 0.05,
) -> dict[str, Any]:
    """One noisy realisation: the observed flip and the significance label.

    Noise is scale-dependent: DataDecide's seed sd falls by ~12x from 4M to
    1B, so using a single pooled sigma would badly misstate the flip rate.
    """

    def sample(c: dict[str, Any], compute: float, sigma: float) -> np.ndarray:
        return float(curve(np.asarray([compute]), **c)[0]) + rng.normal(0.0, sigma, size=n_seeds)

    largest = float(budgets.max())
    small_a = sample(a_curve, largest, sigma_small)
    small_b = sample(b_curve, largest, sigma_small)
    tgt_a = sample(a_curve, target, sigma_target)
    tgt_b = sample(b_curve, target, sigma_target)
    observed_small_gap = float(np.mean(small_a) - np.mean(small_b))
    observed_target_gap = float(np.mean(tgt_a) - np.mean(tgt_b))
    observed_flip = bool(np.sign(observed_small_gap) != np.sign(observed_target_gap))
    p = ir.welch_p(tgt_a, tgt_b)
    significant = bool(ir.benjamini_hochberg([p], q=q)[0])
    return {"observed_flip": observed_flip, "significant": significant, "p": p}


def power_curve(
    sigma_small: float,
    sigma_target: float,
    target: float,
    budgets: np.ndarray,
    seed_counts: tuple[int, ...],
    target_gaps: tuple[float, ...],
    ladder_gaps: tuple[float, ...],
    n_trials: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for n_seeds, gap, ladder_gap, crossover in itertools.product(seed_counts, target_gaps, ladder_gaps, (True, False)):
        a_curve, b_curve = make_pair(gap, crossover, budgets, target, ladder_gap)
        if a_curve is None:
            continue
        flips = sig = flip_and_sig = 0
        for _ in range(n_trials):
            out = simulate_once(a_curve, b_curve, budgets, target, sigma_small, sigma_target, n_seeds, rng)
            flips += int(out["observed_flip"])
            sig += int(out["significant"])
            flip_and_sig += int(out["observed_flip"] and out["significant"])
        rows.append(
            {
                "n_seeds": n_seeds,
                "target_gap": gap,
                "gap_over_sigma": gap / sigma_target,
                "ladder_gap": ladder_gap,
                "ladder_gap_over_sigma_small": ladder_gap / sigma_small,
                "true_crossover": crossover,
                "n_trials": n_trials,
                "flip_rate": flips / n_trials,
                "significance_rate": sig / n_trials,
                "flip_and_significant_rate": flip_and_sig / n_trials,
            }
        )
    return rows


def real_noise_and_ladder() -> tuple[float, np.ndarray, float, float]:
    """DataDecide's measured seed sd at the target, the real ladder, and the fit-budget sd."""

    df, budgets, target = ir.design_frame("c4_en_bits_per_token")
    tgt = df[np.isclose(df["compute"], target)]
    sds = tgt.groupby("intervention")["bpb"].std(ddof=1)
    sigma = float(np.sqrt((sds**2).mean()))
    fit_rows = df[df["compute"].isin(budgets)]
    fit_sds = fit_rows.groupby(["intervention", "compute"])["bpb"].std(ddof=1)
    sigma_fit = float(np.sqrt((fit_sds**2).mean()))
    return sigma, np.asarray(budgets, dtype=float), float(target), sigma_fit


def observed_gap_distribution() -> dict[str, Any]:
    """How large are the real decomposition gaps relative to target seed noise?"""

    payload = json.loads((REPO / "results" / "first_audit" / "first_audit.json").read_text(encoding="utf-8"))
    design = next(
        d for d in payload["designs"] if d["name"] == "datadecide/c4_en_bits_per_token/full_ladder_4M-300M_gate530M"
    )
    dec = design["projection_error_decomposition"]
    sigma, _, _, _ = real_noise_and_ladder()
    out: dict[str, Any] = {"sigma_target": sigma}
    for key in ("crossover_inherited", "fit_error", "single_scale_only"):
        gaps = [abs(p["true_gap"]) for p in dec[key]["pairs"]]
        out[key] = {
            "n": len(gaps),
            "median_abs_target_gap": float(np.median(gaps)) if gaps else None,
            "median_gap_over_sigma": float(np.median(gaps) / sigma) if gaps else None,
            "min_gap_over_sigma": float(min(gaps) / sigma) if gaps else None,
        }
    return out


def main() -> int:
    sigma, budgets, target, sigma_fit = real_noise_and_ladder()
    gaps = (0.5 * sigma, 1.0 * sigma, 2.0 * sigma, 4.0 * sigma, 8.0 * sigma)
    ladder_gaps = (0.5 * sigma_fit, 1.0 * sigma_fit, 2.0 * sigma_fit)
    rows = power_curve(sigma_fit, sigma, target, budgets, (3, 5, 10, 20), gaps, ladder_gaps, n_trials=2000)
    frame = pd.DataFrame(rows)
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "a3_detector_power.csv", index=False)
    at3 = frame[frame["n_seeds"] == 3]
    summary = {
        "sigma_target_bits_per_token": sigma,
        "sigma_fit_budgets_bits_per_token": sigma_fit,
        "n_fit_budgets": int(len(budgets)),
        "flip_detection_at_n3": at3[at3["true_crossover"]][["gap_over_sigma", "flip_rate"]].to_dict("records"),
        "false_flip_at_n3": at3[~at3["true_crossover"]][["gap_over_sigma", "flip_rate"]].to_dict("records"),
        "significance_power_at_n3": at3[at3["true_crossover"]][["gap_over_sigma", "significance_rate"]].to_dict("records"),
        "observed_gaps": observed_gap_distribution(),
    }
    (destination / "a3_detector_power_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"sigma at target = {sigma:.5f} bits/token; sigma on fit budgets = {sigma_fit:.5f}")
    print("\nAt n=3 seeds (DataDecide's actual count):")
    print(
        frame[frame.n_seeds == 3][
            [
                "true_crossover",
                "gap_over_sigma",
                "ladder_gap_over_sigma_small",
                "flip_rate",
                "significance_rate",
                "flip_and_significant_rate",
            ]
        ].to_string(index=False)
    )
    print("\nObserved real gaps relative to sigma:")
    print(json.dumps(summary["observed_gaps"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
