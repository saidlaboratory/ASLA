"""Derived Porian et al. 2024 tables from the safely loaded export (no pickle is read here).

Inputs (git-ignored, produced by the allow-list loader under a no-network sandbox):
data/raw/porian/export/porian_runs.csv and porian_val_curves.json, from
github.com/formll/resolving-scaling-law-discrepancies (MIT) at the commit in
data/raw/porian/source_commit.txt.

Outputs (committed):
* data/porian_runs.csv: one row per run, scalar settings and the final validation loss;
* data/porian_ladder.csv: for each dataset, recipe and FLOP budget on the shared doubling grid, the
  isoFLOP vertex (the minimum validation loss over model sizes), the size at the vertex, the
  validation-sampling SD there, and whether the budget is admitted.

Rules (fixed before any ranking is computed):
* recipes: the five training-setting variants that share the FLOP grid: base hparams with short
  warmup and Chinchilla decay, with short or long warmup and Kaplan (long cosine) decay, and tuned
  hparams with short or long warmup and constant learning rate;
* a validation point belongs to grid budget k when log10 C is within 0.03 decades of
  15.47 + k log10 2, with C = 6 N B s; the point nearest the grid value is used;
* for the Chinchilla-decay recipe only each run's end-of-run point counts (its intermediate points are
  mid-decay); the other recipes contribute every on-grid point of their long runs, as the paper does;
* the rung loss of a recipe at a budget is the minimum over model sizes (the isoFLOP vertex);
* a budget is admitted when every recipe has at least three sizes there and its minimum is interior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "data" / "raw" / "porian" / "export"
RUNS_OUT = REPO / "data" / "porian_runs.csv"
LADDER_OUT = REPO / "data" / "porian_ladder.csv"
SEED_OUT = REPO / "data" / "porian_seed_noise.csv"
GRID0, STEP, TOL = 15.47, float(np.log10(2)), 0.03
RECIPES = (
    "base|short|chinchilla",
    "base|short|kaplan",
    "base|long|kaplan",
    "tuned|short|const",
    "tuned|long|const",
)


def main() -> None:
    runs = pd.read_csv(EXPORT / "porian_runs.csv")
    curves = json.loads((EXPORT / "porian_val_curves.json").read_text(encoding="utf-8"))
    runs["recipe"] = runs["hparams"] + "|" + runs["warmup"] + "|" + runs["decay"]
    with_path = runs.dropna(subset=["ckpt_path"]).copy()
    with_path["N"] = with_path["ckpt_path"].str.extract(r"params=(\d+)M")[0].astype(float) * 1e6
    size_n = with_path.groupby(["width", "depth"])["N"].first()
    runs = runs.join(size_n, on=["width", "depth"])
    runs.to_csv(RUNS_OUT, index=False)

    points = []
    for _, r in runs[runs["recipe"].isin(RECIPES)].iterrows():
        tokens_per_step = r["bs"] * r["seq_len"]
        losses = curves[str(int(r["run"]))]["val/loss"]
        stds = curves[str(int(r["run"]))]["val/loss_std"]
        for step, loss in losses.items():
            # A Chinchilla-decay run decays to its own budget, so only its end-of-run point represents
            # the recipe; its intermediate points are mid-decay.
            if r["decay"] == "chinchilla" and int(step) != max(int(k) for k in losses):
                continue
            lc = float(np.log10(6 * r["N"] * tokens_per_step * int(step)))
            k = int(round((lc - GRID0) / STEP))
            off = abs(lc - (GRID0 + k * STEP))
            if off <= TOL:
                points.append(
                    {
                        "dataset": r["dataset"],
                        "recipe": r["recipe"],
                        "budget_bin": k,
                        "log10_flops": lc,
                        "N": r["N"],
                        "run": int(r["run"]),
                        "seed": r["seed"],
                        "val_loss": float(loss),
                        "val_loss_std": float(stds.get(step, np.nan)),
                        "off_grid": off,
                    }
                )
    pts = pd.DataFrame(points)
    pts = pts.sort_values("off_grid").drop_duplicates(["dataset", "recipe", "budget_bin", "N"])
    rows = []
    for (ds, rec, k), g in pts.groupby(["dataset", "recipe", "budget_bin"]):
        g = g.sort_values("N")
        i = int(np.argmin(g["val_loss"].to_numpy()))
        best = g.iloc[i]
        rows.append(
            {
                "dataset": ds,
                "recipe": rec,
                "budget_bin": int(k),
                "log10_flops_nominal": GRID0 + k * STEP,
                "vertex_val_loss": float(best["val_loss"]),
                "vertex_N": float(best["N"]),
                "vertex_val_loss_std": float(best["val_loss_std"]),
                "n_sizes": len(g),
                "interior": bool(0 < i < len(g) - 1),
            }
        )
    ladder = pd.DataFrame(rows)
    ok = ladder.groupby(["dataset", "budget_bin"]).apply(
        lambda g: bool(len(g) == len(RECIPES) and g["interior"].all() and (g["n_sizes"] >= 3).all())
    )
    ladder["admitted"] = [bool(ok[(d, k)]) for d, k in zip(ladder["dataset"], ladder["budget_bin"])]
    ladder.sort_values(["dataset", "recipe", "budget_bin"]).to_csv(LADDER_OUT, index=False)
    # Seed noise: the seed|short|const runs (seeds 1-7) at the six smallest sizes; relative SD across seeds
    # of each run's final validation loss, per dataset and size.
    seeds = runs[runs["hparams"] == "seed"]
    seed_noise = (
        seeds.groupby(["dataset", "N"])["val_loss_final"]
        .agg(n_seeds="count", mean="mean", sd=lambda v: float(np.std(v, ddof=1)))
        .reset_index()
    )
    seed_noise["relative_sd"] = seed_noise["sd"] / seed_noise["mean"]
    seed_noise.to_csv(SEED_OUT, index=False)
    print(seed_noise.to_string())
    for ds, g in ladder.groupby("dataset"):
        adm = sorted(g[g["admitted"]]["budget_bin"].unique())
        print(ds, "admitted budget bins:", adm)


if __name__ == "__main__":
    main()
