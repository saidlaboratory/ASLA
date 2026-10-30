"""Power of the fixed-target slope test under Porian et al. 2024's exact design (before pre-registration).

Design (from data/porian_ladder.csv, fixed by scripts/build_porian_ladder.py): per dataset (RefinedWeb,
OpenWebText2), the admitted FLOP budgets, five recipes, fixed-target slices (target budget and rung
count, at least three windows), targets admitted when at least half their pairs are determined. The
primary statistic pools the slices of both datasets (slice fixed effects); the leave-one-recipe-out
jackknife drops a recipe from both. Intervals use 1.96 (no calibration exists for this suite).

Truth, built without Porian's recipe ordering: per dataset, the mean vertex loss across recipes at each
budget, plus five synthetic recipes whose deviations from it are quadratic in log10 FLOPs, with
coefficients drawn from normals whose SDs equal the between-recipe SDs of the real recipes' fitted
deviation coefficients. Noise per recipe and budget: seed noise (Porian's own seed runs where the vertex
size is at most the largest seeded size, DataDecide C4's relative seed SD otherwise) combined with the
validation-sampling SD.

Power at a true slope beta uses a location shift: in each world, estimate - true value + beta is compared
with 1.96 x SE. This assumes the sampling error of the slope does not depend on its true value.

Writes results/replication/porian_power.json. Cached per world; resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.budget_ladder import designs  # noqa: E402
from asla.analysis.fixed_target import fixed_target_slices, jackknife, statistics, target_varied_slices  # noqa: E402

LADDER = REPO / "data" / "porian_ladder.csv"
SEEDS = REPO / "data" / "porian_seed_noise.csv"
CACHE = REPO / "data" / "cache" / "porian_power.jsonl"
OUT = REPO / "results" / "replication" / "porian_power.json"
N_WORLDS = 1000
SEED = 20261030
Z = 1.959963984540054
ADMIT_SHARE = 0.5
DATASET_OFFSET = {"rw": 0, "owt2": 100}
BETAS = (11.7, 11.7 / 2, 0.0)
UNIT = "points of mis-selected pairs per decade (log10) of lever arm"
_CTX: dict = {}


def context() -> dict:
    if _CTX:
        return _CTX
    gadre = __import__("scripts.run_replication_gadre", fromlist=["datadecide_relative_sd"])
    dd_rel = gadre.datadecide_relative_sd()
    ladder = pd.read_csv(LADDER)
    seeds = pd.read_csv(SEEDS)
    out = {}
    for ds, g in ladder[ladder["admitted"]].groupby("dataset"):
        bins = sorted(g["budget_bin"].unique())
        recipes = sorted(g["recipe"].unique())
        loss = g.pivot(index="recipe", columns="budget_bin", values="vertex_val_loss")[bins]
        vstd = g.pivot(index="recipe", columns="budget_bin", values="vertex_val_loss_std")[bins]
        vN = g.groupby("budget_bin")["vertex_N"].median()[bins]
        sd_seed = seeds[seeds["dataset"] == ds].set_index("N")["relative_sd"]
        rel = []
        for k in bins:
            n = float(vN[k])
            if n <= sd_seed.index.max():
                nearest = min(sd_seed.index, key=lambda m: abs(np.log(m) - np.log(n)))
                rel.append(("porian_seed", float(sd_seed[nearest])))
            else:
                cap = max(dd_rel)
                nearest = min(dd_rel, key=lambda m: abs(np.log(m) - np.log(min(n, cap))))
                rel.append(("datadecide_c4", float(dd_rel[nearest])))
        x = (np.asarray(bins, dtype=float) * np.log10(2) + 15.47)
        mean_curve = loss.mean(axis=0).to_numpy()
        xc = x - x.mean()
        coefs = np.array([np.polyfit(xc, loss.loc[r].to_numpy() - mean_curve, 2) for r in recipes])
        out[ds] = {
            "bins": bins,
            "log10_flops": x.tolist(),
            "recipes": recipes,
            "mean_curve": mean_curve,
            "coef_sd": coefs.std(axis=0, ddof=1),
            "rel_sd": rel,
            "val_std": vstd.median(axis=0).to_numpy(),
            "xc": xc,
        }
    _CTX.update(out)
    return _CTX


def noise_sd(ds_ctx: dict, loss_row: np.ndarray) -> np.ndarray:
    rel = np.array([r for _, r in ds_ctx["rel_sd"]])
    return np.sqrt((rel * loss_row) ** 2 + ds_ctx["val_std"] ** 2)


def world_designs(ds: str, rng: np.random.Generator, truth: dict[str, np.ndarray] | None = None):
    """One synthetic world for a dataset, or the real data when ``truth`` is None and rng is None."""

    c = context()[ds]
    names = [f"r{i}" for i in range(len(c["recipes"]))]
    true = {}
    for name in names:
        a, b, q = rng.normal(0.0, c["coef_sd"][::-1])  # polyfit order is (q, b, a)
        true[name] = c["mean_curve"] + a + b * c["xc"] + q * c["xc"] ** 2
    obs = {n: true[n] + rng.normal(0.0, noise_sd(c, true[n])) for n in names}
    return _designs_for(ds, obs, true)


def _designs_for(ds: str, obs: dict[str, np.ndarray], true: dict[str, np.ndarray] | None):
    c = context()[ds]
    n_b = len(c["bins"])
    candidates = [t for t in range(n_b) if t >= 4]  # at least three 3-rung windows below
    sds = {t: {n: float(noise_sd(c, obs[n])[t]) for n in obs} for t in candidates}
    ds_designs = designs(obs, c["log10_flops"], sds, candidates, truth=true)
    off = DATASET_OFFSET[ds]
    for d in ds_designs:
        d.target += off
        d.start += off
        d.top += off
    return ds_designs


def admitted(ds_designs) -> list[int]:
    shares = {d.target: d.determined_share for d in ds_designs}
    return sorted(t for t, s in shares.items() if s >= ADMIT_SHARE)


def statistic(all_designs, labels):
    targets = admitted(all_designs)
    ft = fixed_target_slices(all_designs, targets)
    ft = {t: gs for t, gs in ft.items() if gs}
    if not ft:
        return None
    tv = target_varied_slices(all_designs)
    jk = jackknife(all_designs, ft, tv, labels)["fixed_target_pooled"]
    true_all = np.array([d.true_excess if d.true_excess is not None else np.nan for d in all_designs])
    true_det = np.array([d.true_excess_determined if d.true_excess_determined is not None else np.nan for d in all_designs])
    truth = statistics(all_designs, np.nan_to_num(true_all), np.nan_to_num(true_det), ft, tv, labels)["fixed_target_pooled"]
    return {
        "estimate": jk["estimate"],
        "se": jk["se"],
        "true": truth,
        "n_designs": int(sum(len(g) for gs in ft.values() for g in gs)),
        "n_targets": len(ft),
    }


LABELS = [str(i) for i in range(300)]


def run_world(index: int) -> dict:
    rng = np.random.default_rng([SEED, index])
    per = {ds: world_designs(ds, rng) for ds in ("rw", "owt2")}
    out = {"index": index}
    out["pooled"] = statistic(per["rw"] + per["owt2"], LABELS)
    for ds in per:
        out[ds] = statistic(per[ds], LABELS)
    return out


def _read() -> list[dict]:
    if not CACHE.exists():
        return []
    return [json.loads(x) for x in CACHE.read_text(encoding="utf-8").splitlines() if x.strip()]


def summarise() -> dict:
    worlds = _read()
    result = {
        "status": "power analysis before pre-registration (PREDICTIONS_TASK_PORIAN.md follows only if adequate)",
        "unit": UNIT,
        "worlds": len(worlds),
        "noise_sources": {ds: [s for s, _ in context()[ds]["rel_sd"]] for ds in ("rw", "owt2")},
        "assumption": "location shift: the slope's sampling error does not depend on its true value",
        "decision": "",
    }
    for key in ("pooled", "rw", "owt2"):
        rows = [w[key] for w in worlds if w.get(key)]
        err = np.array([r["estimate"] - r["true"] for r in rows])
        se = np.array([r["se"] for r in rows])

        def power(beta: float) -> float:
            return float(np.mean(err + beta - Z * se > 0))

        grid = np.arange(0.0, 60.01, 0.1)
        powers = np.array([power(b) for b in grid])
        mds = float(grid[np.argmax(powers >= 0.8)]) if (powers >= 0.8).any() else None
        est = np.array([r["estimate"] for r in rows])
        truths = np.array([r["true"] for r in rows])
        dev = est - truths.mean()  # random-candidate: error against the population slope

        def power_rc(beta: float) -> float:
            return float(np.mean(dev + beta - Z * se > 0))

        powers_rc = np.array([power_rc(b) for b in np.arange(0.0, 120.01, 0.1)])
        mds_rc = float(np.arange(0.0, 120.01, 0.1)[np.argmax(powers_rc >= 0.8)]) if (powers_rc >= 0.8).any() else None
        random_candidate = {
            "description": "error against the population slope (mean true slope over worlds); recipes vary by world",
            "power_at_11_7": power_rc(11.7),
            "power_at_5_85": power_rc(11.7 / 2),
            "false_positive_rate_positive_at_0": power_rc(0.0),
            "false_positive_rate_two_sided_at_0": float(np.mean(np.abs(dev) > Z * se)),
            "min_slope_80pct_power": mds_rc,
            "sd_estimate_across_worlds": float(est.std()),
            "sd_true_slope_across_worlds": float(truths.std()),
        }
        result[key] = {
            "random_candidate": random_candidate,
            "fixed_candidate_description": "error against each world's own true slope (these recipes)",
            "sd_error_fixed_candidate": float(err.std()),
            "worlds_with_statistic": len(rows),
            "power_at_11_7": power(11.7),
            "power_at_5_85": power(11.7 / 2),
            "false_positive_rate_positive_at_0": power(0.0),
            "false_positive_rate_two_sided_at_0": float(np.mean(np.abs(err) > Z * se)),
            "min_slope_80pct_power": mds,
            "median_se": float(np.median(se)),
            "median_designs": float(np.median([r["n_designs"] for r in rows])),
            "median_targets": float(np.median([r["n_targets"] for r in rows])),
            "mean_error": float(err.mean()),
        }
    adequate = result["pooled"]["power_at_11_7"] >= 0.8 or result["pooled"]["random_candidate"]["power_at_11_7"] >= 0.8
    result["decision"] = (
        "power adequate: pre-register" if adequate else "power at +11.7 below 0.8 under both framings: stop before pre-registration"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarise", "time"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="skip cached worlds (always on)")
    args = parser.parse_args(argv)
    if args.stage == "time":
        t = time.time()
        r = run_world(99999)
        print(f"{time.time() - t:.1f}s", {k: (v if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in r.items()})
        return 0
    if args.stage == "run":
        confound = __import__("scripts.run_confound", fromlist=["_run_pool"])
        done = {w["index"] for w in _read()}
        confound._run_pool(run_world, [i for i in range(N_WORLDS) if i not in done], CACHE, min(args.workers, 2))
        return 0
    res = summarise()
    for k in ("pooled", "rw", "owt2"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
