"""PREDICTIONS_TASK_BREAK.md, Task 3: a certified leaderboard, validated where truth exists.

A pair's order at the target is certified at noise level sigma only if it holds in >= 95% of noise-propagation
draws (primary form, refitted) and agrees between the primary and the alternative form.

Writes ``results/external/certified_leaderboard.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402

OUT = REPO / "results" / "external" / "certified_leaderboard.json"
GRID = (0.0001, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.0043, 0.007, 0.01)
DRAWS_VALIDATE = 400
DRAWS_APPLY = 1000
SEED = 20261022
HOLD = 0.95
NOMINAL = 0.05
DD_FIT = ("150M", "300M", "530M", "750M")
MARIN_FIT = ("130m", "300m", "520m")


def loglog_forecast(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorised two-parameter log-log forecast: x, y (..., k), target (...)."""

    lx, ly = np.log(x), np.log(y)
    mx, my = lx.mean(axis=-1, keepdims=True), ly.mean(axis=-1, keepdims=True)
    slope = ((lx - mx) * (ly - my)).sum(-1) / ((lx - mx) ** 2).sum(-1)
    return np.exp(my[..., 0] + slope * (np.log(target) - mx[..., 0]))


def three_param(points: dict[str, list[dict[str, float]]], target: dict[str, float]) -> dict[str, float]:
    """Patched leaderboard fit in one batch call.

    a x^-b + c is covariant under x -> k x, so each job's x is rescaled to put its target at a common value T0
    (the geometric mean of the targets, so magnitudes stay near the originals) and forecast at T0.
    """

    t0 = float(np.exp(np.mean(np.log(list(target.values())))))
    scaled = {k: [{"x": q["x"] * t0 / target[k], "y": q["y"]} for q in v] for k, v in points.items()}
    res = al.patched_predictions(scaled, (t0,))
    return {k: np.nan if v[0] is None else float(v[0]) for k, v in res.items()}


def certify(draws: np.ndarray, primary: np.ndarray, alternative: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (i, j), i better (lower) than j, certified by the rule. draws: (D, n)."""

    out = []
    for i, j in combinations(range(len(primary)), 2):
        s = np.sign(primary[i] - primary[j])
        if s == 0 or np.sign(alternative[i] - alternative[j]) != s:
            continue
        if np.mean(np.sign(draws[:, i] - draws[:, j]) == s) >= HOLD:
            out.append((i, j) if s < 0 else (j, i))
    return out


def realized(certified: list[tuple[int, int]], truth: np.ndarray) -> dict[str, Any]:
    wrong = sum(truth[i] > truth[j] for i, j in certified)
    return {"certified": len(certified), "wrong": int(wrong), "rate": wrong / len(certified) if certified else None}


def sigma_star(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Smallest grid sigma with realized rate <= nominal, log-linearly interpolated."""

    sig = [float(s) for s in rows]
    rate = [rows[s]["rate"] if rows[s]["rate"] is not None else 0.0 for s in rows]
    for k, (s, r) in enumerate(zip(sig, rate)):
        if r <= NOMINAL:
            if k == 0:
                return {"sigma_star": s, "note": "at or below the smallest grid value"}
            s0, r0 = sig[k - 1], rate[k - 1]
            w = (r0 - NOMINAL) / (r0 - r) if r0 != r else 1.0
            return {"sigma_star": float(np.exp(np.log(s0) + w * (np.log(s) - np.log(s0)))), "bracket": [s0, s]}
    return {"sigma_star": None, "note": "realized rate above nominal at every grid value"}


def noisy(y: np.ndarray, sigma: float, draws: int, rng: np.random.Generator) -> np.ndarray:
    return y[None] * (1 + sigma * rng.standard_normal((draws, *y.shape)))


def dd_validation() -> dict[str, Any]:
    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    recipes = sorted(frame["intervention"].unique())
    x = np.array(
        [[frame[(frame.intervention == r) & (frame.scale_label == s)]["compute"].mean() for s in DD_FIT] for r in recipes]
    )
    target_x = np.array([frame[(frame.intervention == r) & (frame.scale_label == "1B")]["compute"].mean() for r in recipes])
    truth = np.array([frame[(frame.intervention == r) & (frame.scale_label == "1B")]["bpb"].mean() for r in recipes])
    out: dict[str, Any] = {"recipes": len(recipes), "fit_rungs": DD_FIT, "by_seed": {}}
    pooled: dict[str, dict[str, int]] = {f"{s:g}": {"certified": 0, "wrong": 0} for s in GRID}
    for seed in (0, 1, 2):
        y = np.array([[frame[(frame.intervention == r) & (frame.scale_label == s) & (frame.seed == seed)]["bpb"].iloc[0]
                       for s in DD_FIT] for r in recipes])  # fmt: skip
        pts = {r: [{"x": float(x[i, k]), "y": float(y[i, k])} for k in range(len(DD_FIT))] for i, r in enumerate(recipes)}
        tgt = {r: float(target_x[i]) for i, r in enumerate(recipes)}
        fit = three_param(pts, tgt)
        primary = np.array([fit[r] for r in recipes])
        alternative = loglog_forecast(x, y, target_x)
        rows = {}
        rng = np.random.default_rng(SEED + seed)
        for sigma in GRID:
            ys = noisy(y, sigma, DRAWS_VALIDATE, rng)
            jobs = {f"{r}|{d}": [{"x": float(x[i, k]), "y": float(ys[d, i, k])} for k in range(len(DD_FIT))]
                    for d in range(DRAWS_VALIDATE) for i, r in enumerate(recipes)}  # fmt: skip
            res = three_param(jobs, {k: tgt[k.split("|")[0]] for k in jobs})
            draws = np.array([[res[f"{r}|{d}"] for r in recipes] for d in range(DRAWS_VALIDATE)])
            draws = draws[np.all(np.isfinite(draws), axis=1)]
            rows[f"{sigma:g}"] = realized(certify(draws, primary, alternative), truth)
            pooled[f"{sigma:g}"]["certified"] += rows[f"{sigma:g}"]["certified"]
            pooled[f"{sigma:g}"]["wrong"] += rows[f"{sigma:g}"]["wrong"]
        out["by_seed"][str(seed)] = {"rows": rows, "sigma_star": sigma_star(rows)}
    prow = {k: {**v, "rate": v["wrong"] / v["certified"] if v["certified"] else None} for k, v in pooled.items()}
    out["pooled"] = {"rows": prow, "sigma_star": sigma_star(prow)}
    return out


def marin_validation(members: list[str], cells: dict) -> dict[str, Any]:
    x = np.array([[cells[(s, b)]["x"] for b in MARIN_FIT] for s in members])
    y = np.array([[cells[(s, b)]["y"] for b in MARIN_FIT] for s in members])
    target_x = np.array([cells[(s, "1_2b")]["x"] for s in members])
    truth = np.array([cells[(s, "1_2b")]["y"] for s in members])
    primary = loglog_forecast(x, y, target_x)
    pts = {s: [{"x": float(x[i, k]), "y": float(y[i, k])} for k in range(len(MARIN_FIT))] for i, s in enumerate(members)}
    alt = three_param(pts, {s: float(target_x[i]) for i, s in enumerate(members)})
    alternative = np.array([alt[s] for s in members])
    rng = np.random.default_rng(SEED)
    rows = {}
    for sigma in GRID:
        draws = loglog_forecast(x[None], noisy(y, sigma, DRAWS_VALIDATE, rng), target_x[None])
        rows[f"{sigma:g}"] = realized(certify(draws, primary, alternative), truth)
    return {"pairs": len(members) * (len(members) - 1) // 2, "rows": rows, "sigma_star": sigma_star(rows),
            "alternative_failed": [s for s in members if not np.isfinite(alt[s])]}  # fmt: skip


def apply(members: list[str], cells: dict, sigma: float) -> dict[str, Any]:
    rungs = ("130m", "300m", "520m", "1_2b")
    pts = {s: [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in rungs] for s in members}
    primary = np.array([al.patched_forecasts(pts)[s] for s in members])
    x = np.array([[cells[(s, b)]["x"] for b in rungs] for s in members])
    y = np.array([[cells[(s, b)]["y"] for b in rungs] for s in members])
    alternative = loglog_forecast(x, y, np.full(len(members), rig.FORECAST_X))
    rng = np.random.default_rng(SEED)
    ys = noisy(y, sigma, DRAWS_APPLY, rng)
    jobs = {f"{s}|{d}": [{"x": float(x[i, k]), "y": float(ys[d, i, k])} for k in range(4)]
            for d in range(DRAWS_APPLY) for i, s in enumerate(members)}  # fmt: skip
    res = al.patched_predictions(jobs)
    draws = np.array(
        [[np.nan if res[f"{s}|{d}"][0] is None else res[f"{s}|{d}"][0] for s in members] for d in range(DRAWS_APPLY)]
    )
    draws = draws[np.all(np.isfinite(draws), axis=1)]
    cert = certify(draws, primary, alternative)
    above = {s: sorted(members[j] for i, j in cert if members[i] == s) for s in members}
    below = {s: sorted(members[i] for i, j in cert if members[j] == s) for s in members}
    maximal = sorted(s for s in members if not below[s])
    order = sorted(members, key=lambda s: primary[members.index(s)])
    return {
        "sigma": sigma,
        "draws_used": int(len(draws)),
        "certified_pairs": len(cert),
        "pairs": [[members[i], members[j]] for i, j in cert],
        "certified_above_count": {s: len(above[s]) for s in order},
        "certified_below_count": {s: len(below[s]) for s in order},
        "maximal_elements": maximal,
        "top1_certified_above_all": [s for s in members if len(above[s]) == len(members) - 1],
        "fixed_ranking": order,
    }


def main() -> None:
    members, cells, _ = rig.load_cells()
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_BREAK.md", "rule": {
        "noise": f"order holds in >= {HOLD:.0%} of propagation draws (primary form refitted)",
        "form": "primary and alternative form agree", "nominal_false_certification": NOMINAL}}  # fmt: skip
    out["validation_datadecide"] = dd_validation()
    out["validation_marin"] = marin_validation(members, cells)
    applied = {}
    for name, block in (("marin", out["validation_marin"]), ("datadecide", out["validation_datadecide"]["pooled"])):
        s = block["sigma_star"]["sigma_star"]
        if s is not None:
            applied[f"at_sigma_star_{name}"] = apply(members, cells, s)
    out["applied_1e22"] = applied
    dd_s = out["validation_datadecide"]["pooled"]["sigma_star"]["sigma_star"]
    m_s = out["validation_marin"]["sigma_star"]["sigma_star"]
    m_apply = applied.get("at_sigma_star_marin")
    out["predictions"] = {
        "PC1_dd_sigma_star_in_0_05_to_0_3pct": dd_s is not None and 0.0005 <= dd_s <= 0.003,
        "PC2_marin_at_least_dd": m_s is not None and dd_s is not None and m_s >= dd_s,
        "PC3_at_most_120_certified": m_apply is not None and m_apply["certified_pairs"] <= 120,
        "PC3_no_submission_above_all": m_apply is not None and not m_apply["top1_certified_above_all"],
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
