"""Task 5 of PREDICTIONS_TASK_REVIEW_RISKS.md: coverage of the calibrated procedure when the simulator is wrong.

The calibrated procedure for projection vs single-scale (C4, fit to 300M,
target 530M):

* random-candidate estimand: U-statistic SE, interval estimate +/- 2.99 SE;
* fixed-candidate estimand: parametric-bootstrap SE (50 draws under the
  procedure's own Gaussian model), interval estimate +/- 2.24 SE.

Its coverage is measured in worlds that violate the simulator it was calibrated
in: Student-t noise (df 3 to 5), a flattened or steepened noise profile,
misspecification at 0.5x and 1.5x, and candidate spread doubled; plus an
unperturbed control. theta per condition is the mean true difference over that
condition's worlds.

    nice -n 19 python scripts/run_calibration_misspecified.py --workers 2 --resume

Writes ``results/target_scoring/calibration_misspecified.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import (  # noqa: E402
    Truth,
    compare,
    evidence_at_target,
    gaps,
    predict,
    random_truth,
    shrink_truth,
    simulate,
    truth_from_tables,
)

CACHE = REPO / "data" / "cache" / "calibration_misspecified.jsonl"
OUT = REPO / "results" / "target_scoring" / "calibration_misspecified.json"
METRIC, MAX_FIT, TARGET_LABEL = "c4_en_bits_per_token", "300M", "530M"
BASELINE, COMPARATOR = "single_scale", "projection"
K_RANDOM, K_FIXED = 2.9890861649366043, 2.2359097649243167  # results/target_scoring/calibration.json, adopted
CONDITIONS = ("control", "t_noise", "profile_flat", "profile_steep", "misspec_0_5", "misspec_1_5", "spread_2x")
N_RANDOM, N_FIXED, B_WORLD = 200, 80, 50
BANDWIDTH = 0.5
SEED = 20261012
_CONTEXT: dict[str, Any] = {}


class StudentT:
    """A Generator stand-in whose standard normals are unit-variance Student-t draws (df fixed per world)."""

    def __init__(self, rng: np.random.Generator, df: int) -> None:
        self.rng, self.df = rng, df

    def standard_normal(self, size: Any = None) -> np.ndarray:
        return self.rng.standard_t(self.df, size) / np.sqrt(self.df / (self.df - 2))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rng, name)


def context() -> dict[str, Any]:
    if _CONTEXT:
        return _CONTEXT
    from asla.analysis.fits import normalize_budgets
    from asla.analysis.rankers import projection_ranker, single_scale_ranker
    from asla.models import fit_power_law

    built = importlib.import_module("scripts.run_rescoring")._load_checkpoint_builder()(METRIC, MAX_FIT, TARGET_LABEL)
    final, ckpt = built["final_table"], built["ckpt_table"]
    target = float(built["target"])
    base = shrink_truth(truth_from_tables(final, ckpt, target, TARGET_LABEL))
    table = base.final.groupby(["intervention", "compute"])["mu"].first().unstack()
    parametric = {}
    for recipe in table.index:
        x = table.columns.to_numpy(float)
        e, a, alpha = fit_power_law(x, table.loc[recipe].to_numpy(float))
        parametric[str(recipe)] = (e, a, alpha)
    _CONTEXT.update(
        final=final,
        ckpt=ckpt,
        target=target,
        budgets=normalize_budgets(built["budgets"], target=target),
        rankers={BASELINE: single_scale_ranker, COMPARATOR: projection_ranker},
        base=base,
        parametric=parametric,
    )
    return _CONTEXT


def _with_final_mu(base: Truth, mu: pd.Series) -> Truth:
    final = base.final.copy()
    final["mu"] = mu.to_numpy()
    return replace(base, final=final)


def perturbed_base(condition: str) -> Truth:
    ctx = context()
    base: Truth = ctx["base"]
    final = base.final
    if condition.startswith("misspec"):
        lam = 0.5 if condition.endswith("0_5") else 1.5
        p = np.array(
            [
                e + a * c ** (-alpha)
                for (e, a, alpha), c in zip((ctx["parametric"][str(r)] for r in final["intervention"]), final["compute"])
            ]
        )
        return _with_final_mu(base, pd.Series(p + lam * (final["mu"].to_numpy() - p)))
    if condition == "spread_2x":
        centre = final.groupby("scale_label")["mu"].transform("mean")
        return _with_final_mu(base, centre + 2 * (final["mu"] - centre))
    if condition in ("profile_flat", "profile_steep"):
        sigma = pd.Series(base.sigma)
        profile = sigma.groupby(level=1).apply(lambda v: float(np.sqrt(np.mean(v**2))))
        geo = float(np.exp(np.mean(np.log(profile))))
        power = -0.5 if condition == "profile_flat" else 0.5
        scaled = {k: v * (profile[k[1]] / geo) ** power for k, v in base.sigma.items()}
        rms_old = float(np.sqrt(np.mean(np.square(list(base.sigma.values())))))
        rms_new = float(np.sqrt(np.mean(np.square(list(scaled.values())))))
        return replace(base, sigma={k: v * rms_old / rms_new for k, v in scaled.items()})
    return base


def _params(truth: Truth) -> dict[str, Any]:
    from asla.models import fit_power_law

    out = {}
    for recipe, group in truth.final.groupby("intervention"):
        cells = group.groupby("compute")["mu"].mean()
        out[str(recipe)] = fit_power_law(cells.index.to_numpy(float), cells.to_numpy(float))
    return out


def _estimate(truth: Truth, final: pd.DataFrame, ckpt: pd.DataFrame) -> Any:
    ctx = context()
    preds = predict(final, ckpt, ctx["rankers"], (), ctx["budgets"], ctx["target"])
    return compare(preds, evidence_at_target(final, TARGET_LABEL), gaps(truth.target_truth()), BASELINE, with_interval=True)[
        COMPARATOR
    ]


def run_world(condition: str, kind: str, index: int) -> dict[str, Any]:
    ctx = context()
    rng = np.random.default_rng([SEED, CONDITIONS.index(condition), 0 if kind == "fixed" else 1, index])
    base = perturbed_base(condition)
    truth = base if kind == "fixed" else random_truth(base, rng, BANDWIDTH, _params(base))
    noise_rng: Any = StudentT(rng, int(rng.integers(3, 6))) if condition == "t_noise" else rng
    final, ckpt = simulate(truth, noise_rng)
    comp = _estimate(truth, final, ckpt)
    row = {
        "condition": condition,
        "kind": kind,
        "index": index,
        "estimate": comp.estimate,
        "true_difference": comp.true_difference,
        "u_se": comp.u_se,
    }
    if kind == "fixed":
        plug = truth_from_tables(final, ckpt, ctx["target"], TARGET_LABEL, rho_ckpt=ctx["base"].rho_ckpt)
        draws = []
        for _ in range(B_WORLD):
            f, c = simulate(plug, rng)  # the procedure's own Gaussian model
            preds = predict(f, c, ctx["rankers"], (), ctx["budgets"], ctx["target"])
            draws.append(
                compare(
                    preds, evidence_at_target(f, TARGET_LABEL), gaps(plug.target_truth()), BASELINE, with_interval=False
                )[COMPARATOR].estimate
            )
        row["boot_se"] = float(np.std(draws, ddof=1))
    return row


def coverage(rows: list[dict[str, Any]], k: float, se_key: str) -> dict[str, Any]:
    theta = float(np.mean([r["true_difference"] for r in rows]))
    hits = [abs(r["estimate"] - theta) <= k * r[se_key] for r in rows if np.isfinite(r[se_key])]
    n, c = len(hits), int(sum(hits))
    lo, hi = stats.beta.ppf(0.025, c, n - c + 1) if c else 0.0, stats.beta.ppf(0.975, c + 1, n - c) if c < n else 1.0
    return {"theta": theta, "coverage": c / n, "ci": [float(lo), float(hi)], "n_worlds": n, "below_0_90": c / n < 0.90}


def _load() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not CACHE.exists():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text().splitlines() if line.strip())
    return {(r["condition"], r["kind"], r["index"]): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load()
    jobs = [(c, "random", i) for c in CONDITIONS for i in range(N_RANDOM)]
    jobs += [(c, "fixed", i) for c in CONDITIONS for i in range(N_FIXED)]
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        for count, row in enumerate(pool.map(run_world, *zip(*todo), chunksize=5) if todo else [], start=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if count % 100 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    cache = _load()
    out: dict[str, Any] = {
        "procedure": "projection vs single-scale, C4, fit 300M, target 530M; adopted critical values",
        "critical_values": {"random_u": K_RANDOM, "fixed_bootstrap": K_FIXED},
        "conditions": {},
    }
    for condition in CONDITIONS:
        random_rows = [cache[(condition, "random", i)] for i in range(N_RANDOM)]
        fixed_rows = [cache[(condition, "fixed", i)] for i in range(N_FIXED)]
        out["conditions"][condition] = {
            "random_candidate": coverage(random_rows, K_RANDOM, "u_se"),
            "fixed_candidate": coverage(fixed_rows, K_FIXED, "boot_se"),
        }
    out["limits"] = [
        f"{c}: {e} coverage {v[e]['coverage']:.2f}"
        for c, v in out["conditions"].items()
        for e in ("random_candidate", "fixed_candidate")
        if v[e]["below_0_90"]
    ]
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for c, v in out["conditions"].items():
        print(f"{c:15s} random {v['random_candidate']['coverage']:.3f}  fixed {v['fixed_candidate']['coverage']:.3f}")
    print("limits:", out["limits"])


if __name__ == "__main__":
    main()
