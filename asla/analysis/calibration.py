"""Semi-synthetic worlds for end-to-end calibration of the selection-rule comparison.

A world keeps DataDecide's candidate structure and scale ladder and replaces
every observation with a known truth plus noise from the moderated noise model.
Because the truth is known, the true difference in mis-selection between two
rankers can be computed against the noiseless target ordering. The full
inference procedure is then run on the noisy observations and checked against
it.

* **Truth.** Per-cell expected values for final runs and checkpoints, with
  per-(recipe, scale) noise standard deviations and a per-scale AR(1)
  autocorrelation for noise along a run's checkpoints.
* **Fixed-candidate worlds** keep the truth and redraw the noise.
* **Random-candidate worlds** draw a new candidate set: recipes with replacement,
  each shifted by a smooth perturbation dE + dA C^-alpha drawn from a normal
  fitted to the recipes' power-law parameters (a smoothed bootstrap over recipes).

Rankers are passed in, never imported, so the harness has no opinion about
which rules exist.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from asla.analysis.moderation import evidence_from_cells, fit_prior, moderate
from asla.analysis.target_scoring import PairEvidence, expected_charges, u_statistic_test

Ranker = Callable[[pd.DataFrame, tuple[float, ...], float], pd.Series]
KEYS = ["intervention", "scale_label", "seed", "step"]


@dataclass(frozen=True)
class Truth:
    """Expected values, noise scales and target for one world."""

    final: pd.DataFrame  # template rows of the final table, with the truth in column "mu"
    ckpt: pd.DataFrame  # template rows of the checkpoint table, with "mu"
    sigma: Mapping[tuple[str, str], float]  # (recipe, scale_label) -> noise sd of one run
    rho_ckpt: Mapping[str, float]  # scale_label -> AR(1) autocorrelation along checkpoints
    target: float
    target_label: str

    def target_truth(self) -> pd.Series:
        rows = self.final[self.final["scale_label"] == self.target_label]
        return rows.groupby("intervention")["mu"].mean()


def _moderated_sigma(final: pd.DataFrame) -> dict[tuple[str, str], float]:
    sigma = {}
    for scale, group in final.groupby("scale_label"):
        cells = group.groupby("intervention")["bpb"]
        s2 = cells.var(ddof=1)
        prior = fit_prior(s2.to_list(), int(cells.count().min()) - 1)
        for recipe, value in s2.items():
            sigma[(str(recipe), str(scale))] = float(np.sqrt(moderate(value, prior)))
    return sigma


def checkpoint_autocorrelation(ckpt: pd.DataFrame) -> dict[str, float]:
    """Lag-1 correlation of a seed's deviation from its cell mean at consecutive checkpoints, per scale."""

    frame = ckpt.copy()
    frame["dev"] = frame["bpb"] - frame.groupby(["intervention", "scale_label", "step"])["bpb"].transform("mean")
    out = {}
    for scale, group in frame.groupby("scale_label"):
        pairs = []
        for _, run in group.sort_values("step").groupby(["intervention", "seed"]):
            values = run["dev"].to_numpy()
            if values.size >= 2:
                pairs.append(np.column_stack((values[:-1], values[1:])))
        if pairs:
            stacked = np.vstack(pairs)
            out[str(scale)] = float(np.clip(np.corrcoef(stacked[:, 0], stacked[:, 1])[0, 1], 0.0, 0.99))
        else:
            out[str(scale)] = 0.0
    return out


def truth_from_tables(
    final: pd.DataFrame,
    ckpt: pd.DataFrame,
    target: float,
    target_label: str,
    rho_ckpt: Mapping[str, float] | None = None,
) -> Truth:
    """Plug-in truth from observed tables: cell means as expected values, moderated sds as noise."""

    final = final.copy()
    ckpt = ckpt.copy()
    final["mu"] = final.groupby(["intervention", "scale_label"])["bpb"].transform("mean")
    ckpt["mu"] = ckpt.groupby(["intervention", "scale_label", "step"])["bpb"].transform("mean")
    return Truth(
        final=final,
        ckpt=ckpt,
        sigma=_moderated_sigma(final),
        rho_ckpt=dict(rho_ckpt) if rho_ckpt is not None else checkpoint_autocorrelation(ckpt),
        target=target,
        target_label=target_label,
    )


def simulate(
    truth: Truth,
    rng: np.random.Generator,
    coupling: tuple[str, str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Noisy final and checkpoint tables. Noise is AR(1) along each run's checkpoints.

    A final-table row takes the noise of the same run's checkpoint at the same
    step when one exists, so the two tables stay consistent. ``coupling`` =
    (scale_a, scale_b, rho) correlates a seed's final noise at the two scales;
    it is used only for the sensitivity arm.
    """

    ckpt = truth.ckpt.sort_values(KEYS).reset_index(drop=True)
    runs = ckpt.groupby(["intervention", "scale_label", "seed"], sort=False).ngroup().to_numpy()
    position = ckpt.groupby(["intervention", "scale_label", "seed"], sort=False).cumcount().to_numpy()
    sd = np.array([truth.sigma[(str(r), str(s))] for r, s in zip(ckpt["intervention"], ckpt["scale_label"])])
    rho = np.array([truth.rho_ckpt.get(str(s), 0.0) for s in ckpt["scale_label"]])
    n_runs, length = int(runs.max()) + 1, int(position.max()) + 1
    z = rng.standard_normal((n_runs, length))
    run_rho = np.zeros(n_runs)
    run_rho[runs] = rho
    e = np.empty_like(z)
    e[:, 0] = z[:, 0]
    for t in range(1, length):
        e[:, t] = run_rho * e[:, t - 1] + np.sqrt(1 - run_rho**2) * z[:, t]
    ckpt["bpb"] = ckpt["mu"].to_numpy() + sd * e[runs, position]

    final = truth.final.copy()
    merged = final[KEYS].merge(ckpt[KEYS + ["bpb"]], on=KEYS, how="left")
    fresh = np.array([truth.sigma[(str(r), str(s))] for r, s in zip(final["intervention"], final["scale_label"])])
    noise = merged["bpb"].to_numpy() - final["mu"].to_numpy()
    missing = np.isnan(noise)
    noise[missing] = fresh[missing] * rng.standard_normal(int(missing.sum()))
    if coupling is not None:
        scale_a, scale_b, strength = coupling
        a = final["scale_label"] == scale_a
        b = final["scale_label"] == scale_b
        index_a = final[a].set_index(["intervention", "seed"]).index
        noise_a = pd.Series(noise[a.to_numpy()] / fresh[a.to_numpy()], index=index_a)
        index_b = final[b].set_index(["intervention", "seed"]).index
        standardized_a = noise_a.reindex(index_b).to_numpy()
        noise[b.to_numpy()] = fresh[b.to_numpy()] * (
            strength * standardized_a + np.sqrt(1 - strength**2) * noise[b.to_numpy()] / fresh[b.to_numpy()]
        )
    final["bpb"] = final["mu"].to_numpy() + noise
    return final, ckpt


def random_truth(base: Truth, rng: np.random.Generator, bandwidth: float, params: Mapping[str, Sequence[float]]) -> Truth:
    """A new candidate set: recipes drawn with replacement, each smoothly perturbed.

    ``params`` maps each recipe to its fitted (E, A, alpha). The perturbation
    dE + A (exp(dlogA) - 1) C^-alpha has (dE, dlogA) normal with the recipes'
    variances of E and log A, scaled by ``bandwidth``.
    """

    recipes = sorted(params)
    e_sd = float(np.std([params[r][0] for r in recipes], ddof=1)) * bandwidth
    loga_sd = float(np.std([np.log(params[r][1]) for r in recipes], ddof=1)) * bandwidth
    chosen = rng.choice(recipes, size=len(recipes), replace=True)
    finals, ckpts, sigma = [], [], {}
    for j, source in enumerate(chosen):
        name = f"{source}#{j}"
        d_e, d_loga = rng.normal(0.0, e_sd), rng.normal(0.0, loga_sd)
        e0, a0, alpha = params[source]

        def shift(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame[frame["intervention"] == source].copy()
            compute = out["compute"].astype(float).to_numpy()
            out["mu"] = out["mu"].to_numpy() + d_e + a0 * (np.exp(d_loga) - 1) * compute ** (-alpha)
            out["intervention"] = name
            return out

        finals.append(shift(base.final))
        ckpts.append(shift(base.ckpt))
        for (recipe, scale), value in base.sigma.items():
            if recipe == source:
                sigma[(name, scale)] = value
    return replace(base, final=pd.concat(finals, ignore_index=True), ckpt=pd.concat(ckpts, ignore_index=True), sigma=sigma)


def gaps(values: pd.Series) -> dict[tuple[str, str], float]:
    names = sorted(values.index)
    return {(a, b): float(values[a] - values[b]) for a, b in combinations(names, 2)}


def evidence_at_target(final: pd.DataFrame, target_label: str) -> list[PairEvidence]:
    rows = final[final["scale_label"] == target_label]
    return evidence_from_cells({str(r): g["bpb"].to_list() for r, g in rows.groupby("intervention")})


def true_mis_rate(pred: Mapping[tuple[str, str], float], truth: Mapping[tuple[str, str], float]) -> float:
    scored = [(p, truth[k]) for k, p in pred.items() if k in truth and p != 0 and truth[k] != 0 and np.isfinite(p)]
    return float(np.mean([np.sign(p) != np.sign(t) for p, t in scored]))


@dataclass
class Comparison:
    estimate: float  # expected-error difference, comparator minus baseline, in points
    true_difference: float  # mis-selection difference against the noiseless target, in points
    u_low: float
    u_high: float
    u_se: float


def predict(
    final: pd.DataFrame,
    ckpt: pd.DataFrame,
    rankers: Mapping[str, Ranker],
    checkpoint_rankers: Sequence[str],
    budgets: tuple[float, ...],
    target: float,
) -> dict[str, dict[tuple[str, str], float]]:
    out = {}
    for name, ranker in rankers.items():
        source = ckpt if name in checkpoint_rankers else final
        out[name] = gaps(ranker(source, budgets, target))
    return out


def compare(
    predictions: Mapping[str, Mapping[tuple[str, str], float]],
    evidence: Sequence[PairEvidence],
    truth_gaps: Mapping[tuple[str, str], float],
    baseline: str,
    with_interval: bool = True,
) -> dict[str, Comparison]:
    base_charges = expected_charges(predictions[baseline], evidence)
    base_mis = true_mis_rate(predictions[baseline], truth_gaps)
    out = {}
    for name, pred in predictions.items():
        if name == baseline:
            continue
        charges = expected_charges(pred, evidence)
        kernel = {pair: 100 * (charges[pair] - base_charges[pair]) for pair in charges if pair in base_charges}
        estimate = float(np.mean(list(kernel.values())))
        truth_diff = 100 * (true_mis_rate(pred, truth_gaps) - base_mis)
        if with_interval:
            test = u_statistic_test(kernel)
            out[name] = Comparison(
                estimate, truth_diff, float(test["ci_low"]), float(test["ci_high"]), float(test["standard_error"])
            )
        else:
            out[name] = Comparison(estimate, truth_diff, float("nan"), float("nan"), float("nan"))
    return out


def jackknife_se(
    final: pd.DataFrame,
    ckpt: pd.DataFrame,
    rankers: Mapping[str, Ranker],
    checkpoint_rankers: Sequence[str],
    comparator: str,
    baseline: str,
    evidence: Sequence[PairEvidence],
    budgets: tuple[float, ...],
    target: float,
) -> float:
    """Delete-one-candidate jackknife that refits the whole ranker without each candidate."""

    names = sorted(final["intervention"].astype(str).unique())
    by_pair = {(e.left, e.right): e for e in evidence}
    values = []
    for drop in names:
        keep_final = final[final["intervention"] != drop]
        keep_ckpt = ckpt[ckpt["intervention"] != drop]
        subset = {comparator: rankers[comparator], baseline: rankers[baseline]}
        preds = predict(keep_final, keep_ckpt, subset, checkpoint_rankers, budgets, target)
        reduced = [by_pair[p] for p in by_pair if drop not in p]
        left = expected_charges(preds[comparator], reduced)
        right = expected_charges(preds[baseline], reduced)
        values.append(100 * float(np.mean([left[p] - right[p] for p in left if p in right])))
    array = np.asarray(values)
    n = array.size
    return float(np.sqrt((n - 1) / n * np.sum((array - array.mean()) ** 2)))


__all__ = [
    "Comparison",
    "Truth",
    "checkpoint_autocorrelation",
    "compare",
    "evidence_at_target",
    "gaps",
    "jackknife_se",
    "predict",
    "random_truth",
    "simulate",
    "true_mis_rate",
    "truth_from_tables",
]
