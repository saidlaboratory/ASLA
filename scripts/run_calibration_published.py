"""Task 4 of PREDICTIONS_TASK_FINAL_CALIBRATION.md: calibrate the published re-tests on their own metrics.

Two families, both from DataDecide's released macro-average evaluations:

* **Correct Prob against accuracy** (13 small scales). A run's accuracy and
  Correct Prob share the run, so noise is bivariate normal. The within-run
  correlation of seed deviations is estimated per scale, the per-recipe sds
  are moderated, and the truth is shrunk per metric and scale.
* **Scaling-law variants against single-scale ranking at 750M** (8 variants).
  The released predictions cannot be regenerated, so they are held fixed; the
  target (1B accuracy, three-seed mean) and the 750M ranking are noisy.

The kernel is the agreement difference with the observed target, as published.
Random-candidate worlds resample recipes. Each drawn recipe gets one coherent
shift, a normal draw times the true between-recipe sd, applied to all of its
quantities, including its released prediction. The true value of a comparison
is its agreement difference against the noiseless target, averaged over worlds.

Writes ``results/external/published_calibration.json``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.moderation import fit_prior, moderate  # noqa: E402
from asla.analysis.target_scoring import jackknife_standard_error, u_statistic_test  # noqa: E402

published = importlib.import_module("scripts.run_published_comparisons")
calibration = importlib.import_module("scripts.run_calibration")

OUT = REPO / "results" / "external" / "published_calibration.json"
METRICS = ("primary_metric", "correct_prob_per_char")
SMALL = published.SMALL_SCALES
N_WORLDS, N_MC = 300, 300
BANDWIDTH = 0.5
LEVEL = 0.95
SEED = 20260924
Z = 1.959963984540054


def build() -> dict[str, Any]:
    table = published.load_task(published.MACRO)
    recipes = sorted(table["data"].unique())
    scales = sorted(table["params"].unique())
    values: dict[tuple[str, str], np.ndarray] = {}  # (metric, scale) -> recipes x seeds
    for metric in METRICS:
        for scale in scales:
            wide = table[table["params"] == scale].pivot_table(index="data", columns="seed", values=metric)
            values[(metric, scale)] = wide.loc[recipes].to_numpy(dtype=float)
    truth, sd, spread, rho = {}, {}, {}, {}
    for scale in scales:
        devs = [values[(m, scale)] - values[(m, scale)].mean(axis=1, keepdims=True) for m in METRICS]
        rho[scale] = float(np.corrcoef(devs[0].ravel(), devs[1].ravel())[0, 1])
        for metric in METRICS:
            v = values[(metric, scale)]
            s2 = v.var(axis=1, ddof=1)
            prior = fit_prior(s2, v.shape[1] - 1)
            sigma = np.sqrt(moderate(s2, prior))
            means = v.mean(axis=1)
            noise_var = sigma**2 / v.shape[1]
            tau2 = max(float(means.var(ddof=1) - noise_var.mean()), 0.0)
            factor = tau2 / (tau2 + noise_var) if tau2 > 0 else np.zeros_like(noise_var)
            truth[(metric, scale)] = means.mean() + factor * (means - means.mean())
            sd[(metric, scale)] = sigma
            spread[(metric, scale)] = float(np.sqrt(tau2))
    fits = pd.read_parquet(published.RAW / "eval_scaling_law_fit.parquet")
    mapping = published._mapping(fits, published.targets(table)["default_seed"])
    rows = fits[(fits["task"] == published.MACRO) & (fits["metric"] == "primary_metric")]
    sl = {}
    for setup in published.SL_VARIANTS:
        block = rows[rows["setup"] == setup].assign(data=lambda f: f["mix"].map(mapping)).set_index("data")
        sl[setup] = block.loc[recipes, "stacked_pred"].to_numpy(dtype=float)
    return {"recipes": recipes, "values": values, "truth": truth, "sd": sd, "spread": spread, "rho": rho, "sl": sl}


def _agree(pred: np.ndarray, target: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    p, t = np.sign(pred[i] - pred[j]), np.sign(target[i] - target[j])
    return np.where((p == 0) | (t == 0), 0.5, (p == t).astype(float))


def world(ctx: dict[str, Any], rng: np.random.Generator, random: bool) -> dict[str, dict[str, Any]]:
    n = len(ctx["recipes"])
    pick = rng.integers(0, n, size=n) if random else np.arange(n)
    shift = rng.normal(0.0, BANDWIDTH, size=n) if random else np.zeros(n)
    seeds = 3

    def truth_of(metric: str, scale: str) -> np.ndarray:
        return ctx["truth"][(metric, scale)][pick] + shift * ctx["spread"][(metric, scale)]

    def draw(scale: str) -> dict[str, np.ndarray]:
        z1, z2 = rng.standard_normal((n, seeds)), rng.standard_normal((n, seeds))
        r = ctx["rho"][scale]
        e = {"primary_metric": z1, "correct_prob_per_char": r * z1 + np.sqrt(1 - r**2) * z2}
        return {m: truth_of(m, scale)[:, None] + ctx["sd"][(m, scale)][pick][:, None] * e[m] for m in METRICS}

    i, j = np.triu_indices(n, k=1)
    target_truth = truth_of("primary_metric", "1B")
    target_obs = draw("1B")["primary_metric"].mean(axis=1)
    out: dict[str, dict[str, Any]] = {}

    def record(label: str, kernel: np.ndarray, truth_kernel: np.ndarray) -> None:
        kernel_map = {(int(a), int(b)): 100 * float(k) for a, b, k in zip(i, j, kernel)}
        test = u_statistic_test(kernel_map)
        out[label] = {
            "estimate": 100 * float(kernel.mean()),
            "true_difference": 100 * float(truth_kernel.mean()),
            "u_se": float(test["standard_error"]),
            "kjk_se": jackknife_standard_error(kernel_map),
        }

    for scale in SMALL:
        cells = draw(scale)
        per_seed = {m: [_agree(cells[m][:, s], target_obs, i, j) for s in range(seeds)] for m in METRICS}
        per_seed_true = {m: [_agree(cells[m][:, s], target_truth, i, j) for s in range(seeds)] for m in METRICS}
        kernel = np.mean(per_seed["correct_prob_per_char"], axis=0) - np.mean(per_seed["primary_metric"], axis=0)
        truth_kernel = np.mean(per_seed_true["correct_prob_per_char"], axis=0) - np.mean(
            per_seed_true["primary_metric"], axis=0
        )
        record(f"correct_prob|{scale}", kernel, truth_kernel)
    single = draw("750M")["primary_metric"]
    base = np.mean([_agree(single[:, s], target_obs, i, j) for s in range(seeds)], axis=0)
    base_true = np.mean([_agree(single[:, s], target_truth, i, j) for s in range(seeds)], axis=0)
    for setup, prediction in ctx["sl"].items():
        pred = prediction[pick] + shift * ctx["spread"][("primary_metric", "1B")]
        record(f"scaling_law|{setup}", _agree(pred, target_obs, i, j) - base, _agree(pred, target_truth, i, j) - base_true)
    return out


def calibrate(
    worlds: list[dict[str, dict[str, Any]]], mc: list[dict[str, dict[str, Any]]], names: list[str]
) -> dict[str, Any]:
    theta = {n: float(np.mean([w[n]["true_difference"] for w in worlds + mc])) for n in names}

    def ratio(dev: float, se: float) -> float:
        return abs(dev) / se if se > 0 else (0.0 if dev == 0 else float("inf"))

    out: dict[str, Any] = {"theta_random": theta}
    for label, field in (("random_u", "u_se"), ("random_jackknife", "kjk_se")):
        stats_all = np.array([ratio(w[n]["estimate"] - theta[n], w[n][field]) for w in worlds for n in names])
        held = {}
        for n in names:
            covered = []
            for parity in (0, 1):
                train = np.array(
                    [
                        ratio(w[m]["estimate"] - theta[m], w[m][field])
                        for k, w in enumerate(worlds)
                        if k % 2 == parity
                        for m in names
                    ]
                )
                k_train = float(np.quantile(train, LEVEL))
                covered += [
                    ratio(w[n]["estimate"] - theta[n], w[n][field]) <= k_train
                    for k, w in enumerate(worlds)
                    if k % 2 != parity
                ]
            held[n] = float(np.mean(covered))
        out[label] = {
            "critical_value": float(np.quantile(stats_all, LEVEL)),
            "coverage_at_1_96": {
                n: float(np.mean([ratio(w[n]["estimate"] - theta[n], w[n][field]) <= Z for w in worlds])) for n in names
            },
            "cross_fit_coverage": held,
            "null_quantiles_of_abs_t": calibration._quantile_grid(stats_all),
            "n_statistics": int(stats_all.size),
        }
    at_least = {
        n: out["random_jackknife"]["cross_fit_coverage"][n] >= out["random_u"]["cross_fit_coverage"][n] for n in names
    }
    out["estimator_rule"] = {"jackknife_at_least_u": at_least, "jackknife_kept": all(at_least.values())}
    out["random_adopted"] = "random_jackknife" if all(at_least.values()) else "random_u"
    return out


def main() -> None:
    ctx = build()
    rng = np.random.default_rng(SEED)
    worlds = [world(ctx, rng, random=True) for _ in range(N_WORLDS)]
    mc = [world(ctx, rng, random=True) for _ in range(N_MC)]
    out: dict[str, Any] = {
        "design": "DataDecide OLMES macro average; random-candidate worlds with shrunk truth and coherent recipe shifts",
        "within_run_correlation_by_scale": ctx["rho"],
        "n_worlds": N_WORLDS,
        "families": {},
    }
    for family, prefix in (("correct_prob_vs_accuracy", "correct_prob|"), ("scaling_law_variants", "scaling_law|")):
        names = sorted(n for n in worlds[0] if n.startswith(prefix))
        out["families"][family] = calibrate(worlds, mc, names)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for family, block in out["families"].items():
        print(
            family,
            block["random_adopted"],
            {k: round(block[k]["critical_value"], 2) for k in ("random_u", "random_jackknife")},
        )
        for label in ("random_u", "random_jackknife"):
            cov = block[label]["cross_fit_coverage"]
            print("   ", label, "held-out", round(min(cov.values()), 3), "-", round(max(cov.values()), 3))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
