"""Tests for the semi-synthetic calibration harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asla.analysis.calibration import (
    Truth,
    compare,
    evidence_at_target,
    gaps,
    jackknife_se,
    predict,
    random_truth,
    simulate,
    true_mis_rate,
)

SCALES = {"s1": 1e17, "s2": 1e18, "s3": 1e19, "tgt": 1e20}


def _truth(n: int = 6, sigma: float = 0.01, rho: float = 0.0) -> Truth:
    rows, ckpt = [], []
    for r in range(n):
        for label, compute in SCALES.items():
            mu = 2.0 + 0.1 * r + 3.0 * (compute / 1e17) ** -0.2
            for seed in range(3):
                rows.append(
                    {"intervention": f"r{r}", "scale_label": label, "seed": seed, "step": 100, "compute": compute, "mu": mu}
                )
                for step, factor in ((50, 1.05), (100, 1.0)):
                    ckpt.append(
                        {
                            "intervention": f"r{r}",
                            "scale_label": label,
                            "seed": seed,
                            "step": step,
                            "compute": compute * step / 100,
                            "mu": mu * factor,
                        }
                    )
    final, checkpoints = pd.DataFrame(rows), pd.DataFrame(ckpt)
    final["bpb"], checkpoints["bpb"] = final["mu"], checkpoints["mu"]
    sigma_map = {(f"r{r}", s): sigma for r in range(n) for s in SCALES}
    return Truth(final, checkpoints, sigma_map, dict.fromkeys(SCALES, rho), SCALES["tgt"], "tgt")


def test_zero_noise_reproduces_the_truth() -> None:
    final, ckpt = simulate(_truth(sigma=0.0), np.random.default_rng(0))
    assert np.allclose(final["bpb"], final["mu"]) and np.allclose(ckpt["bpb"], ckpt["mu"])


def test_noise_has_the_stated_scale_and_ar1_correlation() -> None:
    truth = _truth(n=40, sigma=0.02, rho=0.6)
    draws = [simulate(truth, np.random.default_rng(i))[1] for i in range(60)]
    noise = pd.concat([(d["bpb"] - d["mu"]).rename(i) for i, d in enumerate(draws)], axis=1)
    assert float(noise.stack().std()) == pytest.approx(0.02, rel=0.05)
    ckpt = draws[0].sort_values(["intervention", "scale_label", "seed", "step"])
    early = np.concatenate(
        [
            (
                d.sort_values(["intervention", "scale_label", "seed", "step"]).query("step == 50")["bpb"]
                - d.sort_values(["intervention", "scale_label", "seed", "step"]).query("step == 50")["mu"]
            ).to_numpy()
            for d in draws
        ]
    )
    late = np.concatenate(
        [
            (
                d.sort_values(["intervention", "scale_label", "seed", "step"]).query("step == 100")["bpb"]
                - d.sort_values(["intervention", "scale_label", "seed", "step"]).query("step == 100")["mu"]
            ).to_numpy()
            for d in draws
        ]
    )
    assert np.corrcoef(early, late)[0, 1] == pytest.approx(0.6, abs=0.05)
    assert len(ckpt) == len(truth.ckpt)


def test_final_rows_share_the_checkpoint_noise_at_the_same_step() -> None:
    final, ckpt = simulate(_truth(sigma=0.02), np.random.default_rng(1))
    merged = final.merge(ckpt, on=["intervention", "scale_label", "seed", "step"], suffixes=("_f", "_c"))
    assert np.allclose(merged["bpb_f"], merged["bpb_c"])


def test_coupling_correlates_final_noise_across_the_two_scales() -> None:
    truth = _truth(n=60, sigma=0.02)
    a, b = [], []
    for i in range(40):
        final, _ = simulate(truth, np.random.default_rng(i), coupling=("s3", "tgt", 0.5))
        noise = final.assign(e=final["bpb"] - final["mu"]).pivot_table(
            index=["intervention", "seed"], columns="scale_label", values="e"
        )
        a.append(noise["s3"].to_numpy())
        b.append(noise["tgt"].to_numpy())
    assert np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1] == pytest.approx(0.5, abs=0.06)


def test_random_truth_draws_unique_perturbed_candidates() -> None:
    base = _truth()
    params = {f"r{r}": (2.0 + 0.1 * r, 3.0 + r * 0.1, 0.2) for r in range(6)}
    world = random_truth(base, np.random.default_rng(2), 0.5, params)
    names = world.final["intervention"].unique()
    assert len(names) == 6 and len(set(names)) == 6
    assert all(("#" in n) for n in names)
    assert set(world.sigma) == {(n, s) for n in names for s in SCALES}


def _oracle(truth: Truth):
    target = truth.target_truth()

    def ranker(df: pd.DataFrame, budgets: tuple[float, ...], target_compute: float) -> pd.Series:
        return target.loc[sorted(df["intervention"].unique())]

    return ranker


def _top_rung(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> pd.Series:
    rows = df[np.isclose(df["compute"], max(budgets))]
    return rows.groupby("intervention")["bpb"].mean()


def test_an_oracle_ranker_has_zero_true_mis_selection() -> None:
    truth = _truth(sigma=0.05)
    final, ckpt = simulate(truth, np.random.default_rng(3))
    rankers = {"top": _top_rung, "oracle": _oracle(truth)}
    budgets = (SCALES["s1"], SCALES["s2"], SCALES["s3"])
    preds = predict(final, ckpt, rankers, (), budgets, truth.target)
    truth_gaps = gaps(truth.target_truth())
    assert true_mis_rate(preds["oracle"], truth_gaps) == 0.0
    out = compare(preds, evidence_at_target(final, "tgt"), truth_gaps, "top")
    assert out["oracle"].true_difference == pytest.approx(-100 * true_mis_rate(preds["top"], truth_gaps))
    se = jackknife_se(final, ckpt, rankers, (), "oracle", "top", evidence_at_target(final, "tgt"), budgets, truth.target)
    assert np.isfinite(se) and se >= 0
