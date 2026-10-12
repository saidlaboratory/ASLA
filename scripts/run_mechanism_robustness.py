"""Task 4 of PREDICTIONS_TASK_REVIEW_RISKS.md: is the lever-arm mechanism attribution robust?

* Noiseless error counts for projection and single-scale on the fixed-candidate
  real shrunk C4 curves, per design. Arm A's projection exactness holds by
  construction: there the truth is the projection's own functional form.
* The variance / common-mode / recipe-specific / interaction attribution of the
  real-curve slope, recomputed under (a) a rank-two common mode, (b) a
  Chinchilla-form base fit E + A / N^alpha + B / D^beta, and (c)
  leave-one-recipe-out. Arms: A = base fit, B1 = A + common mode, B2 = A +
  recipe-specific part, B3 = A + full residual (the real truth). The slope is
  the OLS slope of the mean true excess on log lever arm over the 55 designs;
  common random numbers across variants.

    nice -n 19 python scripts/run_mechanism_robustness.py --workers 2 --resume

Writes ``results/target_scoring/mechanism_robustness.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import transfer_worlds as tw  # noqa: E402

CACHE = REPO / "data" / "cache" / "mechanism_robustness.jsonl"
OUT = REPO / "results" / "target_scoring" / "mechanism_robustness.json"
ARMS = ("A", "B1", "B2", "B3")
N_MAIN, N_LORO = 60, 15
SEED = 20261012
_CACHE: dict[str, Any] = {}


def chinchilla(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per recipe, E + A / N^alpha + B / D^beta fitted to the shrunk curve (N, D per rung from the run table)."""

    per_scale = frame.groupby("scale_label")[["params_n", "tokens_d"]].first().loc[table.columns]
    n = per_scale["params_n"].to_numpy(float) / 1e6
    d = per_scale["tokens_d"].to_numpy(float) / 1e9
    out = {}
    for recipe in table.index:
        y = table.loc[recipe].to_numpy(float)

        def model(_: np.ndarray, e: float, a: float, alpha: float, b: float, beta: float) -> np.ndarray:
            return e + a / n**alpha + b / d**beta

        p0 = [0.8 * y.min(), 1.0, 0.3, 1.0, 0.3]
        lower, upper = [0, 0, 0.01, 0, 0.01], [y.min(), 1e3, 2.0, 1e3, 2.0]
        params, _ = optimize.curve_fit(model, np.zeros_like(y), y, p0=p0, bounds=(lower, upper), maxfev=200000)
        out[recipe] = model(np.zeros_like(y), *params)
    return pd.DataFrame(out, index=table.columns).T


def decomposition(variant: str) -> dict[str, Any]:
    """Arm truth tables and the context to simulate them, for one variant."""

    if variant in _CACHE:
        return _CACHE[variant]
    ctx = tw.context("c4")
    recipes = list(ctx["recipes"])
    if variant.startswith("loro:"):
        recipes = [r for r in recipes if r != variant.split(":", 1)[1]]
    table = ctx["mu"].loc[recipes]
    if variant == "chinchilla":
        base = chinchilla(ctx["frame"], table)
    else:
        base = ctx["parametric"].loc[recipes]
    residual = (table - base).to_numpy()
    u, sv, vt = np.linalg.svd(residual, full_matrices=False)
    rank = 2 if variant == "rank_two" else 1
    common = np.zeros_like(residual)
    for j in range(rank):
        loadings, shape = u[:, j] * sv[j], vt[j]
        common += np.outer(np.full_like(loadings, loadings.mean()), shape)
    frame = lambda m: pd.DataFrame(m, index=recipes, columns=ctx["scales"])  # noqa: E731
    arms = {"A": base, "B1": base + frame(common), "B2": base + frame(residual - common), "B3": table}
    b = ctx["base"]
    keep = b.final["intervention"].isin(recipes)
    sub = dict(
        ctx,
        base=replace(
            b,
            final=b.final[keep].reset_index(drop=True),
            ckpt=b.ckpt[b.ckpt["intervention"].isin(recipes)].reset_index(drop=True),
            sigma={k: v for k, v in b.sigma.items() if k[0] in recipes},
        ),
        recipes=recipes,
    )
    _CACHE[variant] = {"arms": arms, "ctx": sub}
    return _CACHE[variant]


def all_designs() -> dict[str, tuple[str, str]]:
    ctx = tw.context("c4")
    order = ctx["scales"]
    return {f"{order[f]}->{order[t]}": (order[f], order[t]) for t in range(3, len(order)) for f in range(2, t)}


def run_world(variant: str, arm: str, index: int) -> dict[str, Any]:
    dec = decomposition(variant)
    ctx = dec["ctx"]
    rng = np.random.default_rng([SEED, index])  # common random numbers across variants and arms
    excess = tw.world_excess(dec["arms"][arm], ctx["base"].sigma, ctx, all_designs(), rng)
    return {"variant": variant, "arm": arm, "index": index, "excess": excess}


def noiseless_counts() -> dict[str, Any]:
    regime = __import__("scripts.run_regime_map", fromlist=["noiseless_designs"])
    ctx = regime.context("c4")
    rows = regime.noiseless_designs(ctx, regime.truth_table(ctx, "map", 1.0))
    return {
        "designs": rows,
        "projection_wrong_total": int(sum(r["projection_wrong"] for r in rows)),
        "single_wrong_total": int(sum(r["single_wrong"] for r in rows)),
        "designs_where_projection_beats_single": int(sum(r["projection_wrong"] < r["single_wrong"] for r in rows)),
        "arm_a_note": (
            "In arm A the truth is each recipe's fitted E + A C^-alpha, the projection's own functional form, so "
            "projection is exact there by construction (0 wrong pairs at all 55 designs, regime_map.json task_1a); "
            "it is not evidence about real curves"
        ),
    }


def attribution(slopes: dict[str, float]) -> dict[str, float]:
    s3 = slopes["B3"]
    return {
        "variance": slopes["A"] / s3,
        "common_mode": (slopes["B1"] - slopes["A"]) / s3,
        "recipe_specific": (slopes["B2"] - slopes["A"]) / s3,
        "interaction": (s3 - slopes["B1"] - slopes["B2"] + slopes["A"]) / s3,
    }


def _load() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not CACHE.exists():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text().splitlines() if line.strip())
    return {(r["variant"], r["arm"], r["index"]): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    recipes = tw.context("c4")["recipes"]
    variants = {"baseline": N_MAIN, "rank_two": N_MAIN, "chinchilla": N_MAIN}
    variants.update({f"loro:{r}": N_LORO for r in recipes})
    needed_arms = {"baseline": ARMS, "rank_two": ("B1", "B2"), "chinchilla": ("A", "B1", "B2")}
    jobs = [(v, a, i) for v, n in variants.items() for a in needed_arms.get(v, ARMS) for i in range(n)]
    done = _load()
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        for count, row in enumerate(pool.map(run_world, *zip(*todo), chunksize=5) if todo else [], start=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if count % 100 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    cache = _load()
    designs = all_designs()
    lever = np.log([float(tw.context("c4")["compute"][t] / tw.context("c4")["compute"][f]) for f, t in designs.values()])

    def slope(variant: str, arm: str, n: int) -> float:
        source = variant if (variant, arm, 0) in cache else "baseline"  # arms a variant leaves unchanged
        mean = np.nanmean([[cache[(source, arm, i)]["excess"][k] for k in designs] for i in range(n)], axis=0)
        return float(np.polyfit(lever, mean, 1)[0])

    out: dict[str, Any] = {"noiseless": noiseless_counts(), "variants": {}}
    for variant, n in variants.items():
        slopes = {arm: slope(variant, arm, n) for arm in ARMS}
        out["variants"][variant] = {"worlds_per_arm": n, "slopes": slopes, "attribution": attribution(slopes)}
    comps = ("variance", "common_mode", "recipe_specific", "interaction")
    named = {v: out["variants"][v]["attribution"] for v in ("baseline", "rank_two", "chinchilla")}
    loro = [out["variants"][v]["attribution"] for v in variants if v.startswith("loro:")]
    out["ranges"] = {
        c: {
            "baseline": named["baseline"][c],
            "rank_two": named["rank_two"][c],
            "chinchilla": named["chinchilla"][c],
            "loro_min": float(min(a[c] for a in loro)),
            "loro_max": float(max(a[c] for a in loro)),
            "overall": [
                float(min([a[c] for a in loro] + [x[c] for x in named.values()])),
                float(max([a[c] for a in loro] + [x[c] for x in named.values()])),
            ],
        }
        for c in comps
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    nl = out["noiseless"]
    print("noiseless totals: projection", nl["projection_wrong_total"], "single", nl["single_wrong_total"])
    for c, r in out["ranges"].items():
        loro_range = f"[{r['loro_min']:+.2f}, {r['loro_max']:+.2f}]"
        print(f"{c:16s} base {r['baseline']:+.2f} rank2 {r['rank_two']:+.2f} chin {r['chinchilla']:+.2f} loro {loro_range}")


if __name__ == "__main__":
    main()
