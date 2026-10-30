"""A parameterised simulator for the fixed-target slope test, built to reconcile two earlier simulators.

Every choice is explicit in a ``spec`` dict, so a chain of specs can move from one simulator to another one
factor at a time:

* ``truth``: ``"specific"`` (each recipe's power law plus its recipe-specific residual, arm B2) or
  ``"full"`` (plus the common-mode residual too, arm B3(1)); ``lam`` scales the recipe-specific part.
* ``candidates``: ``"fixed"`` (DataDecide's 25 recipes as they are) or ``"random"`` (``n`` recipes drawn
  with replacement, each smoothly perturbed as in asla.analysis.calibration.random_truth).
* ``noise``: ``"per_recipe"`` (each recipe's moderated C4 seed SD, interpolated in log compute) or
  ``"by_size"`` (the median relative seed SD at the DataDecide size nearest the rung's model size,
  N = sqrt(C / (6 tpp)), times the true loss); ``tpp`` is ``"datadecide"`` (DataDecide's drift, by
  compute) or a number.
* ``ladder``: log10 compute of the rungs, ascending; ``targets``: target rung indices, or ``"auto"`` for
  every rung with at least five rungs below it; ``admit``: ``"determinacy"`` (a target counts when half its
  pairs are determined in that world) or ``"all"``.
* ``seeds``: seeds per cell; with one seed, target evidence uses the known-variance limit at the true SD.
* ``statistic``: ``"fixed_target"`` (the frozen pooled fixed-target slope) or ``"lever_only"`` (the slope of
  mean true excess on lever arm over a single window with varied targets, as in the confound benchmark).

Rung losses are values at a compute (the fixed-size ladders of DataDecide); every number returned is in
points of mis-selected pairs per decade (log10) of lever arm.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from asla.analysis.budget_ladder import designs
from asla.analysis.fixed_target import fixed_target_slices, jackknife, statistics, target_varied_slices
from asla.analysis.moderation import evidence_from_cells
from asla.models import FitError, bpb_power_law, fit_power_law

BANDWIDTH = 0.5
ADMIT_SHARE = 0.5
MIN_BELOW = 5
_CTX: dict = {}


def context() -> dict:
    if _CTX:
        return _CTX
    import importlib

    import pandas as pd

    arms = importlib.import_module("scripts.run_lever_arm_arms")
    calibration = importlib.import_module("scripts.run_lever_arm_calibration")
    gadre = importlib.import_module("scripts.run_replication_gadre")
    actx, cctx = arms._context(), calibration._context()
    recipes = list(actx["recipes"])
    log_c = np.log10(actx["compute"][actx["scales"]].to_numpy(dtype=float))
    specific = actx["arm_residual"]["B2"]
    common = actx["arm_residual"]["B1"]
    params = {r: tuple(cctx["params"][r]) for r in recipes}
    sigma = cctx["base"].sigma
    frame = pd.read_parquet(__import__("pathlib").Path(gadre.REPO) / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"]
    tpp = frame.groupby("scale_label").agg(c=("compute", "first"), t=("tokens_per_param", "first")).sort_values("c")
    _CTX.update(
        recipes=recipes,
        log_c=log_c,
        specific={r: specific[i] for i, r in enumerate(recipes)},
        common={r: common[i] for i, r in enumerate(recipes)},
        params=params,
        sigma={r: np.array([sigma[(str(r), str(s))] for s in actx["scales"]]) for r in recipes},
        e_sd=float(np.std([params[r][0] for r in recipes], ddof=1)) * BANDWIDTH,
        loga_sd=float(np.std([np.log(params[r][1]) for r in recipes], ddof=1)) * BANDWIDTH,
        rel_sd=gadre.datadecide_relative_sd(),
        tpp_logc=np.log10(tpp["c"].to_numpy(dtype=float)),
        tpp=tpp["t"].to_numpy(dtype=float),
    )
    return _CTX


def _recipes(spec: dict, rng: np.random.Generator) -> list[tuple[str, str, float, float]]:
    c = context()
    if spec["candidates"] == "fixed":
        return [(r, r, 0.0, 0.0) for r in c["recipes"]]
    out = []
    for j, src in enumerate(rng.choice(c["recipes"], size=int(spec["n"]), replace=True)):
        out.append((f"r{j:02d}", str(src), float(rng.normal(0.0, c["e_sd"])), float(rng.normal(0.0, c["loga_sd"]))))
    return out


def truth_curve(spec: dict, src: str, de: float, dloga: float, x: np.ndarray) -> np.ndarray:
    c = context()
    e, a, alpha = c["params"][src]
    compute = 10.0**x
    lam = float(spec.get("lam", 1.0))
    resid = lam * np.interp(x, c["log_c"], c["specific"][src])
    if spec["truth"] == "full":
        resid = resid + np.interp(x, c["log_c"], c["common"][src])
    return e + a * compute ** (-alpha) + resid + de + a * (np.exp(dloga) - 1) * compute ** (-alpha)


def noise_sd(spec: dict, src: str, x: np.ndarray, truth: np.ndarray) -> np.ndarray:
    c = context()
    if spec["noise"] == "per_recipe":
        return np.interp(x, c["log_c"], c["sigma"][src])
    tpp = np.interp(x, c["tpp_logc"], c["tpp"]) if spec["tpp"] == "datadecide" else np.full_like(x, float(spec["tpp"]))
    n_params = np.sqrt(10.0**x / (6.0 * tpp))
    rel, cap = c["rel_sd"], max(c["rel_sd"])
    r = np.array([rel[min(rel, key=lambda m: abs(np.log(m) - np.log(min(n, cap))))] for n in n_params])
    return r * truth


def _lever_only(spec: dict, rng: np.random.Generator) -> dict:
    """The confound benchmark's lever-arm-only variation: one window, targets at several lever arms."""

    rungs = np.asarray(spec["ladder"], dtype=float)
    levers = np.asarray(spec["levers"], dtype=float)
    recipes = _recipes(spec, rng)
    excess = []
    for lever in levers:
        target = rungs[-1] + lever
        projected, single, truth = {}, {}, {}
        for name, src, de, dl in recipes:
            mu = truth_curve(spec, src, de, dl, rungs)
            sd = noise_sd(spec, src, rungs, mu)
            obs = mu + sd * rng.standard_normal((int(spec["seeds"]), mu.size)).mean(axis=0)
            try:
                projected[name] = float(bpb_power_law(10.0**target, *fit_power_law(10.0**rungs, obs)))
            except (FitError, RuntimeError, ValueError):
                projected[name] = float("nan")
            single[name] = float(obs[-1])
            truth[name] = float(truth_curve(spec, src, de, dl, np.array([target]))[0])
        names = [r[0] for r in recipes]

        def mis(pred: dict) -> float:
            w = []
            for a, b in combinations(names, 2):
                p, t = pred[a] - pred[b], truth[a] - truth[b]
                if np.isfinite(p) and p != 0 and t != 0:
                    w.append(np.sign(p) != np.sign(t))
            return float(np.mean(w))

        excess.append(100 * (mis(projected) - mis(single)))
    return {"excess": excess, "true": float(np.polyfit(levers, excess, 1)[0])}


def world(spec: dict, rng: np.random.Generator) -> dict:
    if spec["statistic"] == "lever_only":
        return _lever_only(spec, rng)
    x = np.asarray(spec["ladder"], dtype=float)
    seeds = int(spec["seeds"])
    recipes = _recipes(spec, rng)
    truth, samples = {}, {}
    for name, src, de, dl in recipes:
        mu = truth_curve(spec, src, de, dl, x)
        sd = noise_sd(spec, src, x, mu)
        truth[name] = mu
        samples[name] = (mu, sd, mu[None, :] + sd * rng.standard_normal((seeds, x.size)))
    obs = {n: s[2].mean(axis=0) for n, s in samples.items()}
    targets = list(range(MIN_BELOW, x.size)) if spec["targets"] == "auto" else [int(t) for t in spec["targets"]]
    sds = {t: {n: float(samples[n][1][t] / np.sqrt(seeds)) for n in samples} for t in targets}
    evidence = None
    if seeds >= 2:
        evidence = {t: evidence_from_cells({n: samples[n][2][:, t].tolist() for n in samples}) for t in targets}
    ds = designs(obs, x, sds, targets, truth=truth, evidence_by_target=evidence)
    shares = {d.target: d.determined_share for d in ds}
    admitted = targets if spec.get("admit", "determinacy") == "all" else [t for t in targets if shares.get(t, 0.0) >= ADMIT_SHARE]
    ft = {t: gs for t, gs in fixed_target_slices(ds, admitted).items() if gs}
    if not ft:
        return {"estimate": None, "true": None}
    tv = target_varied_slices(ds)
    labels = [str(i) for i in range(x.size)]
    jk = jackknife(ds, ft, tv, labels)["fixed_target_pooled"]
    true_all = np.array([d.true_excess for d in ds], dtype=float)
    true_det = np.array([d.true_excess_determined if d.true_excess_determined is not None else 0.0 for d in ds])
    tru = statistics(ds, true_all, true_det, ft, tv, labels)["fixed_target_pooled"]
    lever = np.array([d.lever for d in ds])
    in_slices = sorted({i for gs in ft.values() for g in gs for i in g})
    return {
        "estimate": jk["estimate"],
        "se": jk["se"],
        "true": tru,
        "targets": len(ft),
        "lever_range": [float(lever[in_slices].min()), float(lever[in_slices].max())],
        "determined_share": {str(t): shares[t] for t in targets},
    }


__all__ = ["context", "noise_sd", "truth_curve", "world"]
