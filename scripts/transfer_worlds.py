"""Shared machinery for PREDICTIONS_TASK_TRANSFER.md: metric contexts, light worlds, hybrids, coordinates.

A *context* holds one metric's shrunk truth on its own ladder (750M excluded),
per-recipe power-law fits, residual decomposition, noise and seed count. A
*world* draws noise around a truth table and returns the true excess of
projection over single-scale ranking at the requested designs, computed as the
lever-arm test computes it (projection fits the cell means of every rung up to
the top fitted one; single-scale ranks by the top fitted rung's mean; both are
scored against the true target gaps).
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

from asla.analysis.calibration import Truth, gaps, shrink_truth, simulate, true_mis_rate  # noqa: E402

METRICS = ("c4", "olmes", "cp", "sn")
LABELS = {
    "c4": "C4 bits/token",
    "olmes": "OLMES error",
    "cp": "OLMES correct-prob deficit",
    "sn": "S&N C4 bits/byte",
}
VALID = {"c4": (0.0, np.inf), "sn": (0.0, np.inf), "olmes": (0.0, 1.0), "cp": (0.0, 1.0)}
PRIMARY = ("300M", "530M")
GAP_SCALE = "530M"
_CONTEXTS: dict[str, dict[str, Any]] = {}


def _regime() -> Any:
    return importlib.import_module("scripts.run_regime_map")


def load_frame(metric: str) -> pd.DataFrame:
    if metric in ("c4", "olmes"):
        return _regime()._frame(metric)
    path = {
        "cp": REPO / "data" / "datadecide_runs_olmes_correct_prob_per_char.parquet",
        "sn": REPO / "data" / "signal_and_noise_datadecide_c4_bpb.parquet",
    }[metric]
    frame = pd.read_parquet(path)
    return frame[frame["scale_label"] != "750M"].reset_index(drop=True)


def borrowed_sn_truth(frame: pd.DataFrame) -> Truth:
    """S&N has one run per cell: noise is C4 bits/token's, rescaled by the bits-per-byte to bits-per-token ratio."""

    c4 = context("c4")
    c4_mu = c4["base"].final.groupby(["scale_label"])["mu"].mean()
    sn_mean = frame.groupby("scale_label")["bpb"].mean()
    kappa = {s: float(sn_mean[s] / c4_mu[s]) for s in sn_mean.index}
    sigma = {
        (str(r), str(s)): c4["base"].sigma[(str(r), str(s))] * kappa[str(s)]
        for r in frame["intervention"].unique()
        for s in sn_mean.index
    }
    final = frame.copy()
    final["mu"] = final["bpb"]
    target = float(final[final["scale_label"] == "1B"]["compute"].iloc[0])
    truth = Truth(final=final, ckpt=final.copy(), sigma=sigma, rho_ckpt={}, target=target, target_label="1B")
    return shrink_truth(truth, seeds=1)


def context(metric: str) -> dict[str, Any]:
    if metric in _CONTEXTS:
        return _CONTEXTS[metric]
    frame = load_frame(metric)
    base = borrowed_sn_truth(frame) if metric == "sn" else None
    ctx = _regime().build_context(frame, base=base)
    ctx["metric"] = metric
    ctx["frame"] = frame
    ctx["n_seeds"] = int(frame.groupby(["intervention", "scale_label"])["seed"].nunique().min())
    ctx["mu"] = ctx["parametric"] + pd.DataFrame(ctx["residual"], index=ctx["recipes"], columns=ctx["scales"])
    ctx["sigma_table"] = pd.DataFrame(
        [[ctx["base"].sigma[(r, s)] for s in ctx["scales"]] for r in ctx["recipes"]],
        index=ctx["recipes"],
        columns=ctx["scales"],
    )
    ctx["specific_table"] = pd.DataFrame(ctx["specific"], index=ctx["recipes"], columns=ctx["scales"])
    ctx["common_table"] = pd.DataFrame(ctx["common"], index=ctx["recipes"], columns=ctx["scales"])
    ctx["g"] = median_gap(ctx["mu"][GAP_SCALE])
    _CONTEXTS[metric] = ctx
    return ctx


def median_gap(values: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    diff = np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)]
    return float(np.median(diff))


def ladder_designs(scales: list[str], compute: pd.Series) -> dict[str, tuple[str, str]]:
    """primary, and the smallest- and largest-lever-arm designs on this ladder (t >= 3, 2 <= f < t)."""

    order = sorted(scales, key=lambda s: float(compute[s]))
    pairs = [(order[f], order[t]) for t in range(3, len(order)) for f in range(2, t)]
    lever = {p: float(compute[p[1]] / compute[p[0]]) for p in pairs}
    return {"short": min(pairs, key=lever.get), "primary": PRIMARY, "long": max(pairs, key=lever.get)}


def physical(mu: pd.DataFrame, sigma: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Strict and tolerant monotonicity (non-increasing in compute) and the valid range."""

    step = mu.diff(axis=1).iloc[:, 1:]
    tol = sigma.iloc[:, 1:].to_numpy()
    low, high = VALID[metric]
    values = mu.to_numpy()
    return {
        "strict_monotone": bool((step.to_numpy() <= 0).all()),
        "tolerant_monotone": bool((step.to_numpy() <= tol).all()),
        "in_range": bool((((values > low) if np.isinf(high) else (values >= low)) & (values <= high)).all()),
        "max_increase": float(step.to_numpy().max()),
        "max_increase_over_sd": float((step.to_numpy() / tol).max()),
    }


def world_excess(
    mu: pd.DataFrame,
    sigma: Mapping[tuple[str, str], float],
    ctx: dict[str, Any],
    designs: Mapping[str, tuple[str, str]],
    rng: np.random.Generator,
    scales: list[str] | None = None,
) -> dict[str, float]:
    """True excess (points) of projection over single-scale at each design, for one noise draw."""

    from asla.models import FitError, bpb_power_law, fit_power_law

    scales = scales or ctx["scales"]
    template = ctx["base"].final
    template = template[template["scale_label"].isin(scales)]
    lookup = mu.stack()
    final = template.copy()
    final["mu"] = [lookup[(r, s)] for r, s in zip(final["intervention"], final["scale_label"])]
    truth = replace(ctx["base"], final=final, ckpt=final.copy(), sigma=dict(sigma))
    observed, _ = simulate(truth, rng)
    means = observed.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()
    compute = ctx["compute"]
    order = sorted(scales, key=lambda s: float(compute[s]))
    out = {}
    for name, (fit_top, target) in designs.items():
        budgets = order[: order.index(fit_top) + 1]
        x = np.array([float(compute[s]) for s in budgets])
        projected = {}
        for recipe in means.index:
            try:
                params = fit_power_law(x, means.loc[recipe, budgets].to_numpy(dtype=float))
                projected[recipe] = float(bpb_power_law(float(compute[target]), *params))
            except (FitError, RuntimeError, ValueError):
                projected[recipe] = float("nan")
        truth_gaps = gaps(mu[target])
        proj = true_mis_rate(gaps(pd.Series(projected)), truth_gaps)
        single = true_mis_rate(gaps(means[fit_top]), truth_gaps)
        out[name] = 100 * (proj - single)
    return out


# --- Task 2: hybrids -------------------------------------------------------------------------


def noise_profile(ctx: dict[str, Any]) -> tuple[pd.Series, float]:
    """(pi(s), A): the across-scale profile of seed noise in gap units, and its overall RMS."""

    tilde = ctx["sigma_table"] / ctx["g"]
    amplitude = float(np.sqrt(np.mean(tilde.to_numpy() ** 2)))
    return np.sqrt((tilde**2).mean(axis=0)) / amplitude, amplitude


def hybrid(base: str, sources: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truth and per-run sd for a hybrid, in the base metric's units.

    ``sources`` maps each component in (i)-(v) to the metric supplying it; a
    missing component comes from the base.
    """

    src = {k: sources.get(k, base) for k in ("i", "ii", "iii", "iv", "v")}
    # Exploratory split of (iv): the residual's common mode and its recipe-specific part separately.
    src["iv_common"] = sources.get("iv_common", src["iv"])
    src["iv_specific"] = sources.get("iv_specific", src["iv"])
    b = context(base)
    recipes, scales = b["recipes"], b["scales"]

    def norm(metric: str, key: str) -> pd.DataFrame:
        ctx = context(metric)
        table = {"P": ctx["parametric"], "C": ctx["common_table"], "S": ctx["specific_table"]}[key]
        return (table / ctx["g"]).loc[recipes, scales]

    y = norm(src["iii"], "P") + norm(src["iv_common"], "C") + norm(src["iv_specific"], "S")
    dev = y[GAP_SCALE] - y[GAP_SCALE].mean()
    source_mu = context(src["ii"])["mu"].loc[recipes, GAP_SCALE] / context(src["ii"])["g"]
    wanted = np.sort((source_mu - source_mu.mean()).to_numpy())
    ranks = dev.rank(method="first").astype(int).to_numpy() - 1
    shift = pd.Series(wanted[ranks], index=recipes) - dev
    y = y.add(shift, axis=0)
    truth = y * b["g"]
    pi_base, a_base = noise_profile(b)
    pi_src, _ = noise_profile(context(src["i"]))
    _, a_src = noise_profile(context(src["v"]))
    sigma = b["sigma_table"].loc[recipes, scales] * (pi_src[scales] / pi_base[scales]) * (a_src / a_base)
    return truth, sigma


def sigma_dict(sigma: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {(str(r), str(s)): float(sigma.loc[r, s]) for r in sigma.index for s in sigma.columns}


# --- Task 3: coordinates and transforms -----------------------------------------------------


def coordinates(
    ctx: dict[str, Any],
    mu: pd.DataFrame,
    specific: pd.DataFrame,
    sigma: pd.DataFrame,
    design: tuple[str, str],
    scales: list[str],
) -> dict[str, float]:
    compute = ctx["compute"]
    order = sorted(scales, key=lambda s: float(compute[s]))
    fit_top, target = design
    g = median_gap(mu[target])
    upto = order[: order.index(target) + 1]
    mean_curve = mu[order].mean(axis=0)
    return {
        "g": g,
        "q1": float(np.sqrt(np.mean(sigma[fit_top] ** 2)) / np.sqrt(ctx["n_seeds"]) / g),
        "q2": float(np.sqrt(np.mean(specific[upto].to_numpy() ** 2)) / g),
        "q3": float(np.sqrt(np.mean(sigma[order[0]] ** 2)) / np.sqrt(np.mean(sigma[fit_top] ** 2))),
        "lever_arm": float(compute[target] / compute[fit_top]),
        "q5_level": float(mean_curve[target] / g),
        "q6_decay": float((mean_curve[order[0]] - mean_curve[target]) / g),
        "normalized_target_deviations": sorted(((mu[target] - mu[target].mean()) / g).tolist()),
    }


def own_coordinates(metric: str, design: tuple[str, str], scales: list[str] | None = None) -> dict[str, float]:
    ctx = context(metric)
    scales = scales or ctx["scales"]
    return coordinates(ctx, ctx["mu"][scales], ctx["specific_table"][scales], ctx["sigma_table"][scales], design, scales)


def transform_to(
    source: str,
    target: Mapping[str, Any],
    design: tuple[str, str],
    scales: list[str],
    extra: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The source's structure moved to the target's coordinates at one design.

    Always matches q1 (noise at the top fitted rung over the median target gap),
    q2 (recipe-specific misspecification over that gap) and q3 (noise at the
    smallest rung over the top fitted rung). ``extra`` may add "iii" (floor
    level q5 and decay q6 of the mean curve) and "ii" (the target's normalized
    gap distribution at the design's target).
    """

    ctx = context(source)
    compute = ctx["compute"]
    order = sorted(scales, key=lambda s: float(compute[s]))
    fit_top, tgt = design
    base_part = (ctx["parametric"] + ctx["common_table"])[scales]
    specific = ctx["specific_table"][scales]
    lam = 1.0
    for _ in range(50):
        mu = base_part + lam * specific
        q2 = coordinates(ctx, mu, lam * specific, ctx["sigma_table"][scales], design, scales)["q2"]
        if q2 <= 0:
            break
        new = lam * target["q2"] / q2
        if abs(new - lam) < 1e-10:
            break
        lam = new
    mu = base_part + lam * specific
    applied: dict[str, Any] = {"lambda": lam}
    if "iii" in extra:
        g = median_gap(mu[tgt])
        mean_curve = mu[order].mean(axis=0)
        decay = (mean_curve[order[0]] - mean_curve[tgt]) / g
        kappa = target["q6_decay"] / decay if decay != 0 else 1.0
        new_curve = mean_curve[tgt] + kappa * (mean_curve - mean_curve[tgt])
        new_curve = new_curve + (target["q5_level"] * g - new_curve[tgt])
        mu = mu.sub(mean_curve, axis=1).add(new_curve, axis=1)
        applied.update(kappa=float(kappa), level_shift=float(new_curve[tgt] - mean_curve[tgt]))
    if "ii" in extra:
        g = median_gap(mu[tgt])
        dev = mu[tgt] - mu[tgt].mean()
        wanted = np.asarray(target["normalized_target_deviations"]) * g
        ranks = dev.rank(method="first").astype(int).to_numpy() - 1
        mu = mu.add(pd.Series(wanted[ranks], index=mu.index) - dev, axis=0)
    now = coordinates(ctx, mu, lam * specific, ctx["sigma_table"][scales], design, scales)
    small_ratio = target["q3"] / now["q3"]
    beta = -np.log(small_ratio) / np.log(float(compute[order[0]] / compute[fit_top]))
    weights = pd.Series(
        [float(compute[s] / compute[fit_top]) ** (-beta) if compute[s] < compute[fit_top] else 1.0 for s in scales],
        index=scales,
    )
    k = target["q1"] / now["q1"]
    sigma = ctx["sigma_table"][scales] * weights * k
    applied.update(beta=float(beta), k=float(k))
    applied["achieved"] = {
        key: value
        for key, value in coordinates(ctx, mu, lam * specific, sigma, design, scales).items()
        if key != "normalized_target_deviations"
    }
    return mu, sigma, applied
