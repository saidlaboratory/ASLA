"""PREDICTIONS_TASK_REPLICATION.md: the fixed-target lever-arm effect on Gadre et al. 2024.

Builds data/gadre_scaling_runs.csv from the run files in data/raw/gadre_scaling/models/
(github.com/mlfoundations/scaling, MIT), then runs the frozen fixed-target statistic on the
Chinchilla-1x ladder with the pre-registered no-seed noise model.

Writes results/replication/gadre_fixed_target.json.
"""

from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import _moderated_sigma  # noqa: E402
from asla.analysis.target_scoring import PairEvidence, expected_charges  # noqa: E402
from asla.models import FitError, bpb_power_law, fit_power_law  # noqa: E402

RAW = REPO / "data" / "raw" / "gadre_scaling" / "models"
TABLE = REPO / "data" / "gadre_scaling_runs.csv"
OUT = REPO / "results" / "replication" / "gadre_fixed_target.json"
C4_VAL = "training/eval_data/c4_val/"
RECIPES = ("c4_original", "rpj", "rw_original")
MULTIPLIER = 1.0
WINDOW = 3
ALPHA = 0.05
ADMIT_SHARE = 0.5
Z = 1.959963984540054
T2 = float(stats.t.ppf(0.975, 2))
UNIT = "points of mis-selected pairs per decade (log10) of lever arm"


def build_table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(RAW / "*.json"))):
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        h = run["hyperparameters"]
        c4 = [r for r in run["results"] if any(C4_VAL in v for v in r.get("val_data", []))]
        if len(c4) != 1:
            raise SystemExit(f"{path}: expected one c4_val entry, found {len(c4)}")
        args = run.get("open_lm_args", [])
        rows.append(
            {
                "recipe": run["dataset_name"],
                "params": int(h["params"]),
                "tokens": int(h["tokens"]),
                "chinchilla_multiplier": float(h["chinchilla_multiplier"]),
                "compute": 6.0 * float(h["params"]) * float(h["tokens"]),
                "c4_val_loss": float(c4[0]["loss"]),
                "c4_val_loss_lower_95": float(c4[0]["loss_sequences_lower_95"]),
                "c4_val_tokens": int(c4[0]["tokens"]),
                "seed": args[args.index("--seed") + 1] if "--seed" in args else "",
                "source_file": Path(path).name,
            }
        )
    table = pd.DataFrame(rows).sort_values(["recipe", "chinchilla_multiplier", "params"]).reset_index(drop=True)
    table.to_csv(TABLE, index=False)
    return table


def datadecide_relative_sd() -> dict[float, float]:
    """Median over recipes of moderated seed SD over the cell mean, per DataDecide size (750M excluded)."""

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    frame = frame[frame["scale_label"] != "750M"].reset_index(drop=True)
    sigma = _moderated_sigma(frame)
    means = frame.groupby(["intervention", "scale_label"])["bpb"].mean()
    params = frame.groupby("scale_label")["params_n"].first()
    out = {}
    for scale, n in params.items():
        rel = [sigma[(str(r), str(scale))] / means[(r, scale)] for r in frame["intervention"].unique()]
        out[float(n)] = float(np.median(rel))
    return out


def noise_sd(row: pd.Series, rel_by_params: dict[float, float], scale: float = 1.0) -> float:
    cap = max(rel_by_params)
    nearest = min(rel_by_params, key=lambda n: abs(np.log(min(row["params"], cap)) - np.log(n)))
    seed_sd = rel_by_params[nearest] * row["c4_val_loss"] * scale
    eval_se = (row["c4_val_loss"] - row["c4_val_loss_lower_95"]) / Z
    return float(np.sqrt(seed_sd**2 + eval_se**2))


def evidence(target: pd.DataFrame, sds: dict[str, float]) -> list[PairEvidence]:
    """Known-variance limit of the frozen evidence rule (pre-registered)."""

    names = sorted(target.index)
    m = len(names) * (len(names) - 1) // 2
    crit = float(stats.norm.ppf(1 - ALPHA / (2 * m)))
    out = []
    for a, b in combinations(names, 2):
        gap = float(target[a] - target[b])
        se = float(np.sqrt(sds[a] ** 2 + sds[b] ** 2))
        z = abs(gap) / se
        out.append(
            PairEvidence(
                left=a,
                right=b,
                gap=gap,
                standard_error=se,
                welch_df=float("inf"),
                p_value=float(2 * stats.norm.sf(z)),
                determined=bool(z > crit),
                probability_observed_order_correct=0.5 if gap == 0 else float(stats.norm.cdf(z)),
            )
        )
    return out


def run(noise_scale: float = 1.0) -> dict:
    table = pd.read_csv(TABLE)
    ladder = table[np.isclose(table["chinchilla_multiplier"], MULTIPLIER)]
    sizes = sorted(ladder["params"].unique())
    assert all(set(ladder[ladder["params"] == s]["recipe"]) == set(RECIPES) for s in sizes)
    compute = ladder.groupby("params")["compute"].first().reindex(sizes).to_numpy(dtype=float)
    loss = ladder.pivot(index="recipe", columns="params", values="c4_val_loss")[sizes]
    rel = datadecide_relative_sd()
    t_index = len(sizes) - 1
    target_rows = ladder[ladder["params"] == sizes[t_index]].set_index("recipe")
    sds = {r: noise_sd(target_rows.loc[r], rel, noise_scale) for r in RECIPES}
    ev = evidence(target_rows["c4_val_loss"], sds)
    determined_share = float(np.mean([e.determined for e in ev]))
    designs = []
    for top in range(WINDOW - 1, t_index):
        start = top - WINDOW + 1
        budgets = compute[start : top + 1]
        projected, single = {}, {}
        for r in RECIPES:
            y = loss.loc[r].to_numpy(dtype=float)[start : top + 1]
            try:
                params = fit_power_law(budgets, y)
                projected[r] = float(bpb_power_law(compute[t_index], *params))
            except (FitError, RuntimeError, ValueError):
                projected[r] = float("nan")
            single[r] = float(loss.loc[r].to_numpy(dtype=float)[top])
        gaps = lambda v: {(a, b): v[a] - v[b] for a, b in combinations(sorted(RECIPES), 2)}  # noqa: E731
        left, right = expected_charges(gaps(projected), ev), expected_charges(gaps(single), ev)
        kernel = {p: 100 * (left[p] - right[p]) for p in left if p in right}
        designs.append(
            {
                "window": [int(sizes[start]), int(sizes[top])],
                "lever": float(np.log10(compute[t_index] / compute[top])),
                "kernel": {f"{a}|{b}": v for (a, b), v in kernel.items()},
                "excess": float(np.mean(list(kernel.values()))),
                "fits_failed": [r for r, v in projected.items() if not np.isfinite(v)],
            }
        )
    lever = np.array([d["lever"] for d in designs])

    def slope(drop: str | None) -> float:
        y = np.array([np.mean([v for k, v in d["kernel"].items() if drop is None or drop not in k.split("|")]) for d in designs])
        return float(np.polyfit(lever, y, 1)[0])

    est = slope(None)
    loo = np.array([slope(r) for r in RECIPES])
    n = len(RECIPES)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return {
        "noise_scale": noise_scale,
        "target_params": int(sizes[t_index]),
        "target_noise_sd": sds,
        "determined_share": determined_share,
        "admitted": determined_share >= ADMIT_SHARE,
        "n_designs": len(designs),
        "designs": designs,
        "slope": est,
        "unit": UNIT,
        "se_jackknife": se,
        "leave_one_recipe_out": dict(zip(RECIPES, loo.tolist())),
        "interval_1_96": [est - Z * se, est + Z * se],
        "interval_t2": [est - T2 * se, est + T2 * se],
    }


def main() -> None:
    build_table()
    primary = run(1.0)
    sens = {str(s): run(s) for s in (0.5, 2.0)}
    lo, hi = primary["interval_1_96"]
    if not primary["admitted"]:
        outcome = "not_testable"
    elif lo > 0:
        outcome = "replicates"
    elif hi < 0:
        outcome = "opposite_sign"
    else:
        outcome = "indistinguishable"
    prediction_band = [-10.0, 10.0]
    result = {
        "preregistration": "PREDICTIONS_TASK_REPLICATION.md",
        "suite": "Gadre et al. 2024 (github.com/mlfoundations/scaling, MIT)",
        "recipes": list(RECIPES),
        "metric": "c4_val loss, nats per token, 174M held-out tokens",
        "ladder": "Chinchilla multiplier 1",
        "primary": primary,
        "sensitivity_noise_scale": sens,
        "outcome": outcome,
        "robust_to_t2": bool(primary["interval_t2"][0] > 0) if outcome == "replicates" else None,
        "predictions": {
            "R1_slope_within_minus_10_to_10": bool(prediction_band[0] <= primary["slope"] <= prediction_band[1]),
            "R2_does_not_replicate": outcome != "replicates",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    p = primary
    print(f"determined {p['determined_share']:.2f} admitted {p['admitted']}; designs {p['n_designs']}")
    for d in p["designs"]:
        print(f"  window {d['window']} lever {d['lever']:.2f} excess {d['excess']:+.2f} kernel {d['kernel']}")
    print(f"slope {p['slope']:+.2f} se {p['se_jackknife']:.2f} 1.96 {p['interval_1_96']} t2 {p['interval_t2']}")
    for s, r in sens.items():
        print(f"noise x{s}: determined {r['determined_share']:.2f} slope {r['slope']:+.2f} interval {r['interval_1_96']}")
    print("outcome:", outcome)


if __name__ == "__main__":
    main()
