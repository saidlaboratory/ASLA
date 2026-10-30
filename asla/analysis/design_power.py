"""Power of the fixed-target slope test for a hypothetical design, using DataDecide's structure and noise.

Truth: recipes drawn with replacement from DataDecide's 25 shrunk C4 truth curves (each a power law plus
its residual, interpolated in log10 compute and held at the end values outside DataDecide's range), each
smoothly perturbed as in asla.analysis.calibration.random_truth (dE + A (exp(dlogA) - 1) C^-alpha, with
bandwidth 0.5). Noise: DataDecide C4's relative moderated seed SD at the DataDecide size nearest the
rung's model size (capped at 1B), times the true loss, per seed. Target evidence: the frozen
evidence_from_cells with two or more seeds; its known-variance limit with one seed.

The statistic is the frozen fixed-target pooled slope (asla.analysis.fixed_target) on the designs of
asla.analysis.budget_ladder, at every rung with at least five rungs below it (so at least three 3-rung
windows) whose determined share is at least one half.

Power at a true slope beta is a location shift in two framings:
* fixed-candidate: estimate - (that world's true slope) + beta, against 1.96 x SE;
* random-candidate: estimate - (mean true slope over worlds) + beta, against 1.96 x SE.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from asla.analysis.budget_ladder import designs
from asla.analysis.fixed_target import fixed_target_slices, jackknife, statistics, target_varied_slices
from asla.analysis.moderation import evidence_from_cells

Z = 1.959963984540054
BANDWIDTH = 0.5
ADMIT_SHARE = 0.5
MIN_BELOW = 5
_CTX: dict = {}


def datadecide_context() -> dict:
    if _CTX:
        return _CTX
    import importlib

    arms = importlib.import_module("scripts.run_lever_arm_arms")
    gadre = importlib.import_module("scripts.run_replication_gadre")
    calibration = importlib.import_module("scripts.run_lever_arm_calibration")
    actx, cctx = arms._context(), calibration._context()
    recipes = list(actx["recipes"])
    log_c = np.log10(actx["compute"][actx["scales"]].to_numpy(dtype=float))
    residual = actx["arm_residual"]["B3_1"]
    params = {r: tuple(cctx["params"][r]) for r in recipes}
    _CTX.update(
        recipes=recipes,
        log_c=log_c,
        residual={r: residual[i] for i, r in enumerate(recipes)},
        params=params,
        e_sd=float(np.std([params[r][0] for r in recipes], ddof=1)) * BANDWIDTH,
        loga_sd=float(np.std([np.log(params[r][1]) for r in recipes], ddof=1)) * BANDWIDTH,
        rel_sd=gadre.datadecide_relative_sd(),
    )
    return _CTX


def draw_truth(n: int, log10_flops: Sequence[float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    c = datadecide_context()
    x = np.asarray(log10_flops, dtype=float)
    compute = 10.0**x
    out = {}
    for j, src in enumerate(rng.choice(c["recipes"], size=n, replace=True)):
        e, a, alpha = c["params"][src]
        de, dloga = rng.normal(0.0, c["e_sd"]), rng.normal(0.0, c["loga_sd"])
        base = e + a * compute ** (-alpha) + np.interp(x, c["log_c"], c["residual"][src])
        out[f"r{j:02d}"] = base + de + a * (np.exp(dloga) - 1) * compute ** (-alpha)
    return out


def relative_sd(model_params: Sequence[float]) -> np.ndarray:
    rel = datadecide_context()["rel_sd"]
    cap = max(rel)
    out = []
    for n in model_params:
        nearest = min(rel, key=lambda m: abs(np.log(m) - np.log(min(float(n), cap))))
        out.append(rel[nearest])
    return np.asarray(out)


def one_world(
    n_recipes: int,
    log10_flops: Sequence[float],
    params_of_flops: Callable[[np.ndarray], np.ndarray],
    seeds: int,
    rng: np.random.Generator,
) -> dict | None:
    x = np.asarray(log10_flops, dtype=float)
    truth = draw_truth(n_recipes, x, rng)
    rel = relative_sd(params_of_flops(10.0**x))
    samples = {r: truth[r][None, :] + rel * truth[r] * rng.standard_normal((seeds, x.size)) for r in truth}
    obs = {r: v.mean(axis=0) for r, v in samples.items()}
    candidates = [t for t in range(x.size) if t >= MIN_BELOW]
    if not candidates:
        return None
    sds = {t: {r: float(rel[t] * truth[r][t] / np.sqrt(seeds)) for r in truth} for t in candidates}
    evidence = None
    if seeds >= 2:
        evidence = {t: evidence_from_cells({r: samples[r][:, t].tolist() for r in samples}) for t in candidates}
    ds = designs(obs, x, sds, candidates, truth=truth, evidence_by_target=evidence)
    shares = {d.target: d.determined_share for d in ds}
    admitted = [t for t in candidates if shares.get(t, 0.0) >= ADMIT_SHARE]
    ft = {t: gs for t, gs in fixed_target_slices(ds, admitted).items() if gs}
    if not ft:
        return {"estimate": None}
    tv = target_varied_slices(ds)
    labels = [str(i) for i in range(x.size)]
    jk = jackknife(ds, ft, tv, labels)["fixed_target_pooled"]
    true_all = np.array([d.true_excess for d in ds], dtype=float)
    true_det = np.array([d.true_excess_determined if d.true_excess_determined is not None else 0.0 for d in ds])
    tru = statistics(ds, true_all, true_det, ft, tv, labels)["fixed_target_pooled"]
    return {"estimate": jk["estimate"], "se": jk["se"], "true": tru, "targets": len(ft)}


def summarise(worlds: list[dict | None], betas: Sequence[float] = (11.7, 11.7 / 2)) -> dict:
    rows = [w for w in worlds if w and w.get("estimate") is not None]
    n_all = len(worlds)
    if not rows:
        return {"worlds": n_all, "worlds_with_test": 0, "note": "no admitted target with three windows"}
    est = np.array([r["estimate"] for r in rows])
    se = np.array([r["se"] for r in rows])
    tru = np.array([r["true"] for r in rows])
    # A world whose kernels are all zero has SE 0; a location shift would turn it into a rejection at any
    # slope, which is an artefact of the shift, so such worlds cannot reject.
    usable = se > 0
    # Worlds without an admitted target cannot reject: they count as failures in power.
    frac = len(rows) / n_all
    grid = np.arange(0.0, 150.01, 0.1)
    out = {"worlds": n_all, "worlds_with_test": len(rows)}
    for name, err in (("fixed_candidate", est - tru), ("random_candidate", est - tru.mean())):

        def power(beta: float) -> float:
            return float(np.mean(usable & (err + beta - Z * se > 0))) * frac

        powers = np.array([power(b) for b in grid])
        block = {f"power_at_{b:g}": power(b) for b in betas}
        block["false_positive_rate_positive_at_0"] = power(0.0)
        block["false_positive_rate_two_sided_at_0"] = float(np.mean(usable & (np.abs(err) > Z * se))) * frac
        block["min_slope_80pct_power"] = float(grid[np.argmax(powers >= 0.8)]) if (powers >= 0.8).any() else None
        out[name] = block
    out["median_se"] = float(np.median(se))
    out["share_worlds_se_zero"] = float(np.mean(~usable))
    out["sd_true_slope"] = float(tru.std())
    out["mean_true_slope"] = float(tru.mean())
    return out


__all__ = ["datadecide_context", "draw_truth", "one_world", "relative_sd", "summarise"]
