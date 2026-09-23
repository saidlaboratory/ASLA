"""Task 1 of PREDICTIONS_TASK_CALIBRATED_INFERENCE.md: moderated variance.

1. Per scale and metric, test homogeneity of variance across recipes, and fit the
   empirical-Bayes prior (d0, s0^2) on the 25 recipes' three-seed variances.
2. Known-answer test on PolyPythias. Tasks within a (size, step) are the units,
   with ten seeds each. Three of the ten are drawn, the same three for every task,
   as in a real experiment. Raw and moderated estimates are then compared with
   the ten-seed standard deviation.
3. Target determination with moderated variances against the raw Welch test, on
   C4 and the accuracy metrics, at the primary target (530M) and at 1B.

Writes ``results/target_scoring/moderated_variance.json``.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asla.analysis.moderation import fit_prior, homogeneity, moderate, moderated_evidence, posterior_df
from asla.analysis.target_scoring import target_evidence

REPO = Path(__file__).resolve().parents[1]
METRICS = {
    "c4_en_bits_per_token": "datadecide_runs.parquet",
    "olmes_macro_error": "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": "datadecide_runs_olmes_correct_prob_per_char.parquet",
}
TARGETS = ("530M", "1B")
SEEDS_PER_CELL = 3
POLY_RAW = REPO / "data" / "raw" / "polypythias"
POLY_METRIC = "acc,none"
POLY_MIN_SEEDS = 10
POLY_SUBSAMPLE = 3
POLY_DRAWS = 2000
POLY_MIN_MEAN = 0.01
SEED = 20260923


def load(metric: str) -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / METRICS[metric])


def scale_priors(frame: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for scale, group in frame.groupby("scale_label"):
        cells = [g["bpb"].to_numpy() for _, g in group.groupby("intervention")]
        s2 = [float(np.var(c, ddof=1)) for c in cells if len(c) == SEEDS_PER_CELL]
        prior = fit_prior(s2, SEEDS_PER_CELL - 1)
        out[str(scale)] = {
            "df_prior": prior.df_prior if np.isfinite(prior.df_prior) else None,
            "df_prior_infinite": not np.isfinite(prior.df_prior),
            "var_prior": prior.var_prior,
            "sd_prior": float(np.sqrt(prior.var_prior)),
            "posterior_df": posterior_df(prior) if np.isfinite(prior.df_prior) else None,
            "n_recipes": prior.n_cells,
            "raw_sd_ratio_max_over_min": float(np.sqrt(max(s2) / min(s2))) if min(s2) > 0 else None,
            **homogeneity(cells),
        }
    return out


def determination(frame: pd.DataFrame, target_label: str) -> dict[str, Any]:
    cells = frame[frame["scale_label"] == target_label].groupby("intervention")["bpb"]
    means = {str(k): float(v) for k, v in cells.mean().items()}
    s2 = {str(k): float(v) for k, v in cells.var(ddof=1).items()}
    counts = {str(k): int(v) for k, v in cells.count().items()}
    prior = fit_prior(list(s2.values()), SEEDS_PER_CELL - 1)
    out: dict[str, Any] = {"prior_df": prior.df_prior if np.isfinite(prior.df_prior) else None}
    for rule in ("bonferroni", "bh"):
        raw = target_evidence(means, {k: np.sqrt(v) for k, v in s2.items()}, counts, 0.05, rule)
        mod = moderated_evidence(means, s2, counts, prior, 0.05, rule)
        out[rule] = {
            "raw_determined_fraction": float(np.mean([e.determined for e in raw])),
            "moderated_determined_fraction": float(np.mean([e.determined for e in mod])),
            "raw_median_df": float(np.median([e.welch_df for e in raw])),
            "moderated_median_df": float(np.median([e.welch_df for e in mod])),
            "pairs_gained": int(sum(m.determined and not r.determined for m, r in zip(mod, raw))),
            "pairs_lost": int(sum(r.determined and not m.determined for m, r in zip(mod, raw))),
        }
    return out


def polypythias_records() -> pd.DataFrame:
    """One value per (run, step, task): the earliest evaluation pass, as in run_noise_validation.py."""

    rows = []
    for path in sorted(glob.glob(str(POLY_RAW / "*.json"))):
        match = re.match(r"pythia-(\w+)-seed(\d+)__step(\d+)__", Path(path).name)
        if not match:
            continue
        try:
            results = json.loads(Path(path).read_text()).get("results", {})
        except json.JSONDecodeError:
            continue
        for task, values in results.items():
            if POLY_METRIC in values:
                rows.append(
                    {
                        "size": match[1],
                        "seed": int(match[2]),
                        "step": int(match[3]),
                        "task": task,
                        "value": float(values[POLY_METRIC]),
                        "file": Path(path).name,
                    }
                )
    frame = pd.DataFrame(rows).sort_values("file")
    return frame.drop_duplicates(["size", "seed", "step", "task"], keep="first")


def polypythias_validation(rng: np.random.Generator) -> dict[str, Any]:
    records = polypythias_records()
    cells = {}
    for (size, step), group in records.groupby(["size", "step"]):
        wide = group.pivot_table(index="task", columns="seed", values="value")
        wide = wide.dropna(thresh=POLY_MIN_SEEDS)
        seeds = [s for s in wide.columns if wide[s].notna().all()]
        wide = wide[seeds]
        wide = wide[(wide.mean(axis=1) > POLY_MIN_MEAN) & (wide.var(axis=1, ddof=1) > 0)]
        if len(seeds) < POLY_MIN_SEEDS or len(wide) < 10:
            continue
        truth = wide.std(axis=1, ddof=1).to_numpy()
        values = wide.to_numpy()
        log_raw, log_mod, ratio_raw, ratio_mod, ratio_c4 = [], [], [], [], []
        for _ in range(POLY_DRAWS):
            chosen = rng.choice(values.shape[1], size=POLY_SUBSAMPLE, replace=False)
            s2 = values[:, chosen].var(axis=1, ddof=1)
            prior = fit_prior(s2, POLY_SUBSAMPLE - 1)
            raw_sd = np.sqrt(np.maximum(s2, 1e-12))
            mod_sd = np.sqrt(moderate(s2, prior))
            log_raw.append(np.log(raw_sd / truth))
            log_mod.append(np.log(mod_sd / truth))
            ratio_raw.append(np.median(raw_sd / truth))
            ratio_mod.append(np.median(mod_sd / truth))
            ratio_c4.append(np.median(raw_sd / 0.8862269254527580 / truth))
        lr, lm = np.concatenate(log_raw), np.concatenate(log_mod)
        cells[f"{size}|{step}"] = {
            "n_tasks": int(len(wide)),
            "n_seeds": len(seeds),
            "rmse_log_sd_raw": float(np.sqrt(np.mean(lr**2))),
            "rmse_log_sd_moderated": float(np.sqrt(np.mean(lm**2))),
            "median_ratio_raw": float(np.median(ratio_raw)),
            "median_ratio_raw_c4_corrected": float(np.median(ratio_c4)),
            "median_ratio_moderated": float(np.median(ratio_mod)),
            "moderation_better": bool(np.mean(lm**2) < np.mean(lr**2)),
        }
    rows = list(cells.values())
    return {
        "source": "EleutherAI/polypythias-evals, all tasks with acc, earliest evaluation pass per (run, step, task)",
        "units": "tasks within a (size, step); the same three seeds drawn for every task in a draw",
        "draws_per_cell": POLY_DRAWS,
        "cells": cells,
        "n_cells": len(rows),
        "n_cells_moderation_better": int(sum(r["moderation_better"] for r in rows)),
        "median_ratio_raw": float(np.median([r["median_ratio_raw"] for r in rows])),
        "median_ratio_moderated": float(np.median([r["median_ratio_moderated"] for r in rows])),
        "median_rmse_log_raw": float(np.median([r["rmse_log_sd_raw"] for r in rows])),
        "median_rmse_log_moderated": float(np.median([r["rmse_log_sd_moderated"] for r in rows])),
    }


NULL_UNITS = 25
NULL_DRAWS = 4000
NULL_LEVELS = (0.05, 0.01, 0.001, 0.05 / 300)


def _null_p_values(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Two-sided p for each unit's a-versus-b difference under three procedures.

    ``raw_welch``: per-unit sample variances, Welch-Satterthwaite df.
    ``moderated_t``: posterior variances, df from d + d0 per variance (pre-registered).
    ``moderated_se_raw_df``: posterior variances for the standard error, but the
    raw Welch df, so no degrees of freedom are credited for the prior.
    """

    from scipy import stats

    n = a.shape[1]
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    gap = np.abs(a.mean(axis=1) - b.mean(axis=1))
    se_raw = np.sqrt(va / n + vb / n)
    df_raw = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
    prior_a, prior_b = fit_prior(va, n - 1), fit_prior(vb, n - 1)
    ma, mb = moderate(va, prior_a) / n, moderate(vb, prior_b) / n
    se_mod = np.sqrt(ma + mb)
    da, db = posterior_df(prior_a), posterior_df(prior_b)
    terms = [m**2 / d for m, d in ((ma, da), (mb, db)) if np.isfinite(d)]
    df_mod = (ma + mb) ** 2 / sum(terms) if terms else np.full_like(gap, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "raw_welch": 2 * stats.t.sf(gap / se_raw, df_raw),
            "moderated_t": 2 * np.where(np.isfinite(df_mod), stats.t.sf(gap / se_mod, df_mod), stats.norm.sf(gap / se_mod)),
            "moderated_se_raw_df": 2 * stats.t.sf(gap / se_mod, df_raw),
        }


def polypythias_null_test(rng: np.random.Generator) -> dict[str, Any]:
    """Type-I error where the null is true by construction.

    Per draw: 25 tasks (the DataDecide cell count) and 6 of the 10 seeds, split
    into two groups of three. Both groups are the same model, so every task's
    true difference is zero. Each group's prior is fitted on its own 25 sample
    variances, as the real prior is fitted on one set of 25 recipes.
    """

    records = polypythias_records()
    procedures = ("raw_welch", "moderated_t", "moderated_se_raw_df")
    hits = {name: dict.fromkeys(NULL_LEVELS, 0) for name in procedures}
    per_cell: dict[str, dict[str, float]] = {}
    n_tests = 0
    for (size, step), group in records.groupby(["size", "step"]):
        wide = group.pivot_table(index="task", columns="seed", values="value").dropna(thresh=POLY_MIN_SEEDS)
        seeds = [c for c in wide.columns if wide[c].notna().all()]
        wide = wide[seeds]
        wide = wide[(wide.mean(axis=1) > POLY_MIN_MEAN) & (wide.var(axis=1, ddof=1) > 0)]
        if len(seeds) < POLY_MIN_SEEDS or len(wide) < NULL_UNITS:
            continue
        values = wide.to_numpy()
        cell_hits = dict.fromkeys(procedures, 0)
        cell_tests = 0
        for _ in range(NULL_DRAWS):
            tasks = rng.choice(values.shape[0], size=NULL_UNITS, replace=False)
            chosen = rng.choice(values.shape[1], size=2 * POLY_SUBSAMPLE, replace=False)
            a = values[np.ix_(tasks, chosen[:POLY_SUBSAMPLE])]
            b = values[np.ix_(tasks, chosen[POLY_SUBSAMPLE:])]
            if min(np.median(a.var(axis=1, ddof=1)), np.median(b.var(axis=1, ddof=1))) <= 0:
                continue
            p_values = _null_p_values(a, b)
            for name in procedures:
                for level in NULL_LEVELS:
                    hits[name][level] += int(np.sum(p_values[name] <= level))
                cell_hits[name] += int(np.sum(p_values[name] <= NULL_LEVELS[1]))
            cell_tests += NULL_UNITS
        n_tests += cell_tests
        per_cell[f"{size}|{step}"] = {name: cell_hits[name] / cell_tests / NULL_LEVELS[1] for name in procedures}
    return {
        "design": "25 tasks per draw, 3 vs 3 seeds of the same model; each group's prior fitted on its own 25 variances",
        "n_tests": n_tests,
        "rates_over_nominal": {
            name: {f"{level:.6g}": hits[name][level] / n_tests / level for level in NULL_LEVELS} for name in procedures
        },
        "per_cell_rate_over_nominal_at_0_01": per_cell,
    }


WEIGHT_DRAWS = 1000
WEIGHT_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001)


def _weights(values: np.ndarray, n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Probability-correct weights under three variants, for every pair of units, and the observed gaps."""

    from scipy import stats

    means, s2 = values.mean(axis=1), values.var(axis=1, ddof=1)
    prior = fit_prior(s2, n - 1)
    mod = moderate(s2, prior)
    i, j = np.triu_indices(values.shape[0], k=1)
    gap = np.abs(means[i] - means[j])
    va, vb = s2[i] / n, s2[j] / n
    se_raw = np.sqrt(va + vb)
    df_raw = (va + vb) ** 2 / (va**2 / (n - 1) + vb**2 / (n - 1))
    ma, mb = mod[i] / n, mod[j] / n
    se_mod = np.sqrt(ma + mb)
    d_each = posterior_df(prior)
    df_mod = (ma + mb) ** 2 / (ma**2 / d_each + mb**2 / d_each) if np.isfinite(d_each) else np.full_like(gap, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = {
            "a_normal_raw_se": stats.norm.cdf(gap / se_raw),
            "b_t_moderated_se_prior_df": np.where(
                np.isfinite(df_mod), stats.t.cdf(gap / se_mod, df_mod), stats.norm.cdf(gap / se_mod)
            ),
            "c_t_moderated_se_raw_df": stats.t.cdf(gap / se_mod, df_raw),
        }
    # A zero standard error with a nonzero gap is a certain order; a zero gap is a coin flip.
    for name, w in weights.items():
        w = np.where(np.isnan(w), np.where(gap > 0, 1.0, 0.5), w)
        weights[name] = np.where(gap == 0, 0.5, w)
    return weights, means[i] - means[j]


def polypythias_weight_calibration(rng: np.random.Generator) -> dict[str, Any]:
    """Known-answer test for expected-error weights: 3 seeds give the weight, 7 held-out seeds the truth."""

    records = polypythias_records()
    names = ("a_normal_raw_se", "b_t_moderated_se_prior_df", "c_t_moderated_se_raw_df")
    squared = {name: 0.0 for name in names}
    bins = {name: np.zeros((len(WEIGHT_BINS) - 1, 2)) for name in names}
    n_pairs = 0
    for (_, _), group in records.groupby(["size", "step"]):
        wide = group.pivot_table(index="task", columns="seed", values="value").dropna(thresh=POLY_MIN_SEEDS)
        seeds = [c for c in wide.columns if wide[c].notna().all()]
        wide = wide[seeds]
        wide = wide[(wide.mean(axis=1) > POLY_MIN_MEAN) & (wide.var(axis=1, ddof=1) > 0)]
        if len(seeds) < POLY_MIN_SEEDS or len(wide) < NULL_UNITS:
            continue
        values = wide.to_numpy()
        for _ in range(WEIGHT_DRAWS):
            tasks = rng.choice(values.shape[0], size=NULL_UNITS, replace=False)
            order = rng.permutation(values.shape[1])
            observed, held_out = values[np.ix_(tasks, order[:POLY_SUBSAMPLE])], values[np.ix_(tasks, order[POLY_SUBSAMPLE:])]
            if np.median(observed.var(axis=1, ddof=1)) <= 0:
                continue
            weights, observed_gap = _weights(observed, POLY_SUBSAMPLE)
            i, j = np.triu_indices(len(tasks), k=1)
            truth = held_out.mean(axis=1)
            correct = (np.sign(observed_gap) == np.sign(truth[i] - truth[j])).astype(float)
            for name in names:
                w = weights[name]
                squared[name] += float(np.sum((w - correct) ** 2))
                index = np.clip(np.digitize(w, WEIGHT_BINS) - 1, 0, len(WEIGHT_BINS) - 2)
                np.add.at(bins[name], (index, 0), 1)
                np.add.at(bins[name], (index, 1), correct)
            n_pairs += correct.size
    brier = {name: squared[name] / n_pairs for name in names}
    curves = {
        name: [
            {
                "bin": [WEIGHT_BINS[k], min(WEIGHT_BINS[k + 1], 1.0)],
                "n": int(bins[name][k, 0]),
                "observed_fraction_correct": float(bins[name][k, 1] / bins[name][k, 0]) if bins[name][k, 0] else None,
            }
            for k in range(len(WEIGHT_BINS) - 1)
        ]
        for name in names
    }
    if not all(np.isfinite(v) for v in brier.values()):
        raise ValueError(f"non-finite Brier score: {brier}")
    best = min(brier, key=lambda name: brier[name])
    return {
        "design": "25 tasks per draw; weights from 3 seeds, true order from the other 7",
        "n_pairs": n_pairs,
        "brier": brier,
        "calibration_curves": curves,
        "adopted": best,
        "caveat": "the held-out order is itself a 7-seed estimate, which pulls close pairs toward chance for every variant",
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {
        "method": (
            "Smyth (2004) empirical-Bayes variance moderation, moment estimation on log s^2 as in limma fitFDist; "
            "prior fitted per scale across the 25 recipes, d = 2"
        ),
        "metrics": {},
    }
    for metric in METRICS:
        frame = load(metric)
        out["metrics"][metric] = {
            "priors_by_scale": scale_priors(frame),
            "determination": {target: determination(frame, target) for target in TARGETS},
        }
    out["polypythias"] = polypythias_validation(rng)
    out["polypythias_null_test"] = polypythias_null_test(rng)
    out["polypythias_weight_calibration"] = polypythias_weight_calibration(rng)
    poly = out["polypythias"]
    c4 = out["metrics"]["c4_en_bits_per_token"]
    err = out["metrics"]["olmes_macro_error"]
    priors_c4 = list(c4["priors_by_scale"].values())
    out["predictions"] = {
        "M1_homogeneity_mostly_not_rejected": {
            "scales_rejected_bartlett": sum(p["bartlett_p"] < 0.05 for p in priors_c4),
            "scales_rejected_brown_forsythe": sum(p["brown_forsythe_p"] < 0.05 for p in priors_c4),
            "n_scales": len(priors_c4),
            "confirmed": sum(p["brown_forsythe_p"] < 0.05 for p in priors_c4) <= len(priors_c4) / 2,
        },
        "M2_moderation_recovers_ten_seed_sigma": {
            "cells_better": poly["n_cells_moderation_better"],
            "n_cells": poly["n_cells"],
            "median_ratio_moderated": poly["median_ratio_moderated"],
            "median_ratio_raw": poly["median_ratio_raw"],
            "confirmed": poly["n_cells_moderation_better"] > poly["n_cells"] / 2
            and abs(poly["median_ratio_moderated"] - 1) <= 0.05,
        },
        "M3_determined_fractions_increase": {
            "c4_530M_bonferroni": [
                c4["determination"]["530M"]["bonferroni"]["raw_determined_fraction"],
                c4["determination"]["530M"]["bonferroni"]["moderated_determined_fraction"],
            ],
            "olmes_error_530M_bonferroni": [
                err["determination"]["530M"]["bonferroni"]["raw_determined_fraction"],
                err["determination"]["530M"]["bonferroni"]["moderated_determined_fraction"],
            ],
            "confirmed": all(
                m["determination"][t]["bonferroni"]["moderated_determined_fraction"]
                >= m["determination"][t]["bonferroni"]["raw_determined_fraction"]
                for m in (c4, err)
                for t in TARGETS
            ),
        },
    }
    path = REPO / "results" / "target_scoring" / "moderated_variance.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["predictions"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
