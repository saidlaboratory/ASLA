"""PREDICTIONS_TASK_MARIN.md, Tasks 1-5: the Marin speedrun comparison made rigorous.

Inputs (git-ignored, checksums in the output): ``data/raw/marin_speedrun/runs.json``
(marin-community/speedrun @ 31fe8028), the leaderboard's own fit code
``website/src/utils/regression.ts`` run unmodified under Node via
``scripts/marin_lm_runner.mjs`` (``ml-levenberg-marquardt`` 5.1.0 in
``data/cache/marin_node``), Delphi's seed replicates
(``data/raw/delphi/lucky-seeds``), and DataDecide's C4 noise.

Writes ``results/external/marin_rigour.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "data" / "raw" / "marin_speedrun"
NODE_DIR = REPO / "data" / "cache" / "marin_node"
OUT = REPO / "results" / "external" / "marin_rigour.json"
BUDGETS = ("130m", "300m", "520m", "1_2b")
FORECAST_X = 1e22
QUALIFY_TRAINING_FLOPS = 3e20
K_CONSERVATIVE = 2.9890861649366043  # results/target_scoring/calibration.json random U, the largest calibrated U value
DESIGNS = {
    "{130m,300m}->520m": (("130m", "300m"), "520m"),
    "{130m,300m}->1_2b": (("130m", "300m"), "1_2b"),
    "{300m,520m}->1_2b": (("300m", "520m"), "1_2b"),
    "{130m,300m,520m}->1_2b": (("130m", "300m", "520m"), "1_2b"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cells() -> tuple[list[str], dict[tuple[str, str], dict[str, float]], dict[str, int]]:
    runs = json.loads((RAW / "runs.json").read_text())
    members = json.loads((REPO / "results" / "external" / "marin_speedrun_gate.json").read_text())["best_shared_set"][
        "members"
    ]
    cells: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    replicate_counts: dict[str, int] = defaultdict(int)
    for run in runs:
        if "/" not in run["run_name"]:
            continue
        prefix, label = run["run_name"].rsplit("/", 1)
        replicate_counts[f"{prefix}|{label.lower()}"] += 1
        if prefix in members and label.lower() in BUDGETS:
            cells[(prefix, label.lower())].append(
                {
                    "x": float(run["model_flops"]),
                    "y": float(run["eval_paloma_c4_en_bpb"]),
                    "train_flops": float(run["training_hardware_flops"]),
                }
            )
    flat = {k: {f: float(np.mean([r[f] for r in v])) for f in ("x", "y", "train_flops")} for k, v in cells.items()}
    repeats = {k: n for k, n in replicate_counts.items() if n > 1}
    return members, flat, repeats


def marin_fit(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Their regression.ts, unmodified, through the Node runner."""

    proc = subprocess.run(
        ["node", str(REPO / "scripts" / "marin_lm_runner.mjs"), str(NODE_DIR / "regression.ts")],
        input=json.dumps({"fits": jobs}),
        capture_output=True,
        text=True,
        check=True,
    )
    return {f["id"]: f for f in json.loads(proc.stdout)["fits"]}


def converged_sse(points: list[tuple[float, float]]) -> float:
    """Residual sum of squares of a converged least-squares fit of the leaderboard's form and bounds."""

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    c0 = min(0.7, y.min() * 0.7)
    slope, intercept = np.polyfit(np.log(x), np.log(y - c0), 1)
    lower, upper = [1e-10, 1e-6, 0.0], [1e30, 2.0, 0.999 * y.min()]
    p0 = np.clip([float(np.exp(intercept)), max(0.01, -slope), c0], lower, upper)
    res = optimize.least_squares(lambda p: p[0] * x ** (-p[1]) + p[2] - y, p0, bounds=(lower, upper), max_nfev=20000)
    return float(np.sum(res.fun**2))


def python_marin(points: list[tuple[float, float]], target: float) -> float:
    """Python reimplementation of computePowerLawWithOffset (same form, bounds and initialisation)."""

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    c0 = min(0.7, y.min() * 0.7)
    slope, intercept = np.polyfit(np.log(x), np.log(y - c0), 1)
    a0, b0 = float(np.exp(intercept)), max(0.01, -slope)
    lower, upper = [1e-10, 1e-6, 0.0], [1e30, 2.0, 0.999 * y.min()]
    p0 = np.clip([a0, b0, c0], lower, upper)
    res = optimize.least_squares(lambda p: p[0] * x ** (-p[1]) + p[2] - y, p0, bounds=(lower, upper), max_nfev=20000)
    a, b, c = res.x
    return float(a * target ** (-b) + c)


def loglog(points: list[tuple[float, float]], target: float) -> float:
    lx, ly = np.log([p[0] for p in points]), np.log([p[1] for p in points])
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(np.exp(intercept + slope * np.log(target)))


def shared_exponent(series: dict[str, list[tuple[float, float]]], targets: dict[str, float]) -> dict[str, float]:
    """log L = a_s - alpha log x with one alpha across submissions."""

    lx = {s: np.log([p[0] for p in v]) for s, v in series.items()}
    ly = {s: np.log([p[1] for p in v]) for s, v in series.items()}
    num = sum(float(((lx[s] - lx[s].mean()) * (ly[s] - ly[s].mean())).sum()) for s in series)
    den = sum(float(((lx[s] - lx[s].mean()) ** 2).sum()) for s in series)
    slope = num / den
    return {s: float(np.exp(ly[s].mean() + slope * (np.log(targets[s]) - lx[s].mean()))) for s in series}


def exact_three(points: list[tuple[float, float]], target: float) -> float:
    from asla.models import FitError, bpb_power_law, fit_power_law

    try:
        return float(
            bpb_power_law(target, *fit_power_law(np.array([p[0] for p in points]), np.array([p[1] for p in points])))
        )
    except (FitError, RuntimeError, ValueError):
        return float("nan")


def gaps(values: dict[str, float]) -> dict[tuple[str, str], float]:
    return {(a, b): values[a] - values[b] for a, b in combinations(sorted(values), 2)}


def wrong(pred: dict[tuple[str, str], float], truth: dict[tuple[str, str], float], pairs: Any = None) -> int:
    keys = truth if pairs is None else pairs
    return int(sum(np.sign(pred[k]) != np.sign(truth[k]) for k in keys if truth[k] != 0 and np.isfinite(pred[k])))


def projections(
    members: list[str], cells: dict, fit_labels: tuple[str, ...], target: str, jobs_out: dict | None = None
) -> dict[str, dict[str, float]]:
    series = {s: [(cells[(s, b)]["x"], cells[(s, b)]["y"]) for b in fit_labels] for s in members}
    tx = {s: cells[(s, target)]["x"] for s in members}
    out: dict[str, dict[str, float]] = {}
    if len(fit_labels) >= 3:
        fits = marin_fit(
            [{"id": s, "points": [{"x": x, "y": y} for x, y in series[s]], "targets": [tx[s]]} for s in members]
        )
        out["i_marin"] = {
            s: (fits[s]["predictions"][0] if fits[s]["predictions"][0] is not None else float("nan")) for s in members
        }
        out["iv_exact_three"] = {s: exact_three(series[s], tx[s]) for s in members}
    out["ii_loglog"] = {s: loglog(series[s], tx[s]) for s in members}
    out["iii_shared_exponent"] = shared_exponent(series, tx)
    return out


def evidence_for(truth: dict[str, float], cv: float, z: float) -> list[Any]:
    from asla.analysis.target_scoring import PairEvidence

    out = []
    for (a, b), g in gaps(truth).items():
        se = cv * float(np.hypot(truth[a], truth[b]))
        p = float(stats.norm.cdf(abs(g) / se)) if se > 0 else 1.0
        out.append(
            PairEvidence(
                left=a, right=b, gap=g, standard_error=se, welch_df=float("inf"),
                p_value=float(2 * stats.norm.sf(abs(g) / se)), determined=bool(abs(g) > z * se),
                probability_observed_order_correct=p,
            )
        )  # fmt: skip
    return out


def borrowed_cvs() -> dict[str, Any]:
    regime = __import__("scripts.run_regime_map", fromlist=["context"])
    c4 = regime.context("c4")
    base = c4["base"]
    mu_1b = base.final[base.final["scale_label"] == "1B"].groupby("intervention")["mu"].first()
    cv_dd = float(np.mean([base.sigma[(r, "1B")] / mu_1b[r] for r in mu_1b.index]))
    delphi = pd.read_parquet(REPO / "data" / "raw" / "delphi" / "lucky-seeds" / "train.parquet")
    held = delphi[delphi["kind"] == "held_out"]
    out = {"datadecide_c4_1B": {"cv": cv_dd, "basis": "moderated per-run sd / cell mean at 1B, mean over 25 recipes"}}
    for c in (1e21, 1e22):
        losses = held[np.isclose(held["C_nom"], c)]["loss"].to_numpy(float)
        out[f"delphi_{c:.0e}"] = {
            "cv": float(losses.std(ddof=1) / losses.mean()),
            "seeds": int(losses.size),
            "losses": losses.tolist(),
            "basis": "sd over seed replicates / mean, held-out loss",
        }
    return out


def u_test(kernel: dict[tuple[str, str], float]) -> dict[str, Any]:
    from asla.analysis.target_scoring import u_statistic_test

    test = u_statistic_test(kernel)
    t = abs(test["mean"]) / test["standard_error"] if test["standard_error"] > 0 else float("inf")
    return {
        "mean_pp": float(test["mean"]),
        "se_pp": float(test["standard_error"]),
        "abs_t": float(t),
        "significant_at_1_96": bool(t > 1.959963984540054),
        "significant_at_2_99": bool(t > K_CONSERVATIVE),
    }


def _convergence(members: list[str], cells: dict, fits: dict[str, Any]) -> dict[str, Any]:
    ratios = {}
    for s in members:
        pts = [(cells[(s, b)]["x"], cells[(s, b)]["y"]) for b in BUDGETS]
        p = fits[s]["params"]
        sse_theirs = float(sum((p["intercept"] * x ** p["slope"] + p["asymptote"] - y) ** 2 for x, y in pts))
        best = converged_sse(pts)
        ratios[s] = sse_theirs / best if best > 0 else float("inf")
    return {
        "sse_theirs_over_converged": ratios,
        "submissions_more_than_1pct_above_converged": int(sum(r > 1.01 for r in ratios.values())),
        "median_ratio": float(np.median(list(ratios.values()))),
        "reading": "the leaderboard's LM fit (200 iterations, damping 1.5) stops short of the least-squares optimum; "
        "its 1e22 forecasts come from unconverged fits. No claim is made about error at 1e22.",
    }


def main() -> None:
    from asla.analysis.target_scoring import decomposition, expected_charges

    members, cells, repeats = load_cells()
    # Task 1.
    full_jobs = [
        {"id": s, "points": [{"x": cells[(s, b)]["x"], "y": cells[(s, b)]["y"]} for b in BUDGETS], "targets": [FORECAST_X]}
        for s in members
    ]
    marin_full = marin_fit(full_jobs)
    published = {s: marin_full[s]["predictions"][0] for s in members}
    python_full = {s: python_marin([(cells[(s, b)]["x"], cells[(s, b)]["y"]) for b in BUDGETS], FORECAST_X) for s in members}
    diffs = [abs(published[s] - python_full[s]) for s in members if published[s] is not None]
    task1 = {
        "procedure": {
            "source": "marin-community/speedrun@31fe8028 website/src/utils/regression.ts, hooks/useSpeedrunData.ts, App.tsx",
            "grouping": "run-name prefix (getScalingGroupName)",
            "x_axis": "model_flops (App.tsx default)",
            "y": "eval_paloma_c4_en_bpb",
            "form": "L = a x^(-b) + c, all of a submission's runs, unweighted",
            "fit": "Levenberg-Marquardt (ml-levenberg-marquardt ^5.0.0, resolved 5.1.0): damping 1.5, gradient "
            "difference 1e-6, 200 iterations, error tolerance 1e-8; a in [1e-10, 1e30], b in [1e-6, 2], c in "
            "[0, 0.999 min y]; c0 = min(0.7, 0.7 min y), a0 and b0 from a log-log regression of y - c0",
            "target": "forecast at x = 1e22 model FLOPs, used to rank submissions",
            "qualification": "a submission ranks only if some run exceeds 3e20 training hardware FLOPs",
            "published_values": "computed in the browser from runs.json; nothing stored elsewhere, so 'published' "
            "means this code's output",
        },
        "regression_ts_sha256": sha256(NODE_DIR / "regression.ts"),
        "runs_json_sha256": sha256(RAW / "runs.json"),
        "forecast_1e22_their_code": published,
        "forecast_1e22_python_reimplementation": python_full,
        "max_abs_difference": float(max(diffs)),
        "rank_spearman_their_vs_python": float(
            stats.spearmanr([published[s] for s in members], [python_full[s] for s in members]).statistic
        ),
        "convergence": _convergence(members, cells, marin_full),
        "qualified": {s: any(cells[(s, b)]["train_flops"] > QUALIFY_TRAINING_FLOPS for b in BUDGETS) for s in members},
        "their_ranking": sorted(members, key=lambda s: published[s]),
    }
    # Task 2.
    truth = {s: cells[(s, "1_2b")]["y"] for s in members}
    single = {s: cells[(s, "520m")]["y"] for s in members}
    truth_gaps, single_gaps = gaps(truth), gaps(single)
    proj = projections(members, cells, ("130m", "300m", "520m"), "1_2b")
    plain_evidence = evidence_for(truth, 1e-9, 1.96)
    task2: dict[str, Any] = {"pairs": len(truth_gaps), "single_scale_wrong": wrong(single_gaps, truth_gaps), "forms": {}}
    for form, values in proj.items():
        pg = gaps(values)
        dec = decomposition(pg, single_gaps, plain_evidence)
        task2["forms"][form] = {
            "projection_wrong": wrong(pg, truth_gaps),
            "failed_fits": sorted(s for s, v in values.items() if not np.isfinite(v)),
            **{k: dec[k] for k in ("introduced", "corrected", "shared", "both_correct")},
        }
    # Task 4.
    cvs = borrowed_cvs()
    z_bonf = float(stats.norm.ppf(1 - 0.025 / len(truth_gaps)))
    task4: dict[str, Any] = {"repeated_runs_any_submission": repeats, "borrowed": cvs, "by_borrowed_sd": {}}
    for name, entry in cvs.items():
        block: dict[str, Any] = {}
        for zname, z in (("z_1_96", 1.96), ("bonferroni", z_bonf)):
            ev = evidence_for(truth, entry["cv"], z)
            determined = {(e.left, e.right) for e in ev if e.determined}
            single_wrong_pairs = {k for k in truth_gaps if np.sign(single_gaps[k]) != np.sign(truth_gaps[k])}
            rows = {
                "determined_fraction": len(determined) / len(ev),
                "single_wrong_on_undetermined": len(single_wrong_pairs - determined),
                "single_wrong_total": len(single_wrong_pairs),
                "single_wrong_determined_only": wrong(single_gaps, truth_gaps, determined),
                "expected_errors_single": float(sum(expected_charges(single_gaps, ev).values())),
                "forms": {},
            }
            for form, values in proj.items():
                pg = gaps(values)
                rows["forms"][form] = {
                    "wrong_determined_only": wrong(pg, truth_gaps, determined),
                    "expected_errors": float(sum(expected_charges(pg, ev).values())),
                }
            block[zname] = rows
        task4["by_borrowed_sd"][name] = block
    # Task 3.
    task3: dict[str, Any] = {"designs": {}}
    for name, (fit_labels, target) in DESIGNS.items():
        p = projections(members, cells, fit_labels, target)
        t_gaps = gaps({s: cells[(s, target)]["y"] for s in members})
        s_gaps = gaps({s: cells[(s, fit_labels[-1])]["y"] for s in members})
        lever = float(np.median([cells[(s, target)]["x"] / cells[(s, fit_labels[-1])]["x"] for s in members]))
        task3["designs"][name] = {
            "lever_arm": lever,
            "single_wrong": wrong(s_gaps, t_gaps),
            "excess_pp": {
                f: 100 * (wrong(gaps(p[f]), t_gaps) - wrong(s_gaps, t_gaps)) / len(t_gaps)
                for f in ("ii_loglog", "iii_shared_exponent")
            },
        }
    levers = np.log([d["lever_arm"] for d in task3["designs"].values()])
    task3["spearman_excess_vs_log_lever"] = {
        f: float(stats.spearmanr(levers, [d["excess_pp"][f] for d in task3["designs"].values()]).statistic)
        for f in ("ii_loglog", "iii_shared_exponent")
    }
    task3["leaderboard_extrapolation"] = {
        "forecast_x": FORECAST_X,
        "median_ratio_1e22_over_1_2b_model_flops": float(np.median([FORECAST_X / cells[(s, "1_2b")]["x"] for s in members])),
        "note": "the leaderboard extrapolates to 1e22, far beyond any design here; no error at 1e22 is claimed",
    }
    # Task 5.
    family = {s: s.split("/")[0] if "/" in s else s for s in members}
    fam_members: dict[str, list[str]] = defaultdict(list)
    for s, f in family.items():
        fam_members[f].append(s)
    within = {k for k in truth_gaps if family[k[0]] == family[k[1]]}
    cross = set(truth_gaps) - within
    largest_cv = max(e["cv"] for e in cvs.values())
    rising = sorted(
        s
        for s in members
        if any(
            cells[(s, b2)]["y"] - cells[(s, b1)]["y"] > 2 * np.sqrt(2) * largest_cv * cells[(s, b2)]["y"]
            for b1, b2 in zip(BUDGETS, BUDGETS[1:])
        )
    )
    nonfinite = sorted(s for s in members if not all(np.isfinite(cells[(s, b)]["y"]) for b in BUDGETS))
    singleton_pairs = [k for k in truth_gaps if len(fam_members[family[k[0]]]) == 1 and len(fam_members[family[k[1]]]) == 1]
    combos = list(product(*[sorted(v) for v in fam_members.values()]))
    rep_rows: dict[str, list[int]] = defaultdict(list)
    for combo in combos:
        keep = set(combo)
        pairs = [k for k in truth_gaps if k[0] in keep and k[1] in keep]
        rep_rows["single"].append(wrong(single_gaps, truth_gaps, pairs))
        for form, values in proj.items():
            rep_rows[form].append(wrong(gaps(values), truth_gaps, pairs))
    task5 = {
        "families": {f: sorted(v) for f, v in fam_members.items()},
        "n_families": len(fam_members),
        "within_family_pairs": len(within),
        "cross_family_pairs": len(cross),
        "errors_within_cross": {
            "single": [wrong(single_gaps, truth_gaps, within), wrong(single_gaps, truth_gaps, cross)],
            **{f: [wrong(gaps(v), truth_gaps, within), wrong(gaps(v), truth_gaps, cross)] for f, v in proj.items()},
        },
        "one_per_family": {
            "combinations": len(combos),
            "pairs_per_combination": len(fam_members) * (len(fam_members) - 1) // 2,
            "errors_mean_min_max": {k: [float(np.mean(v)), int(min(v)), int(max(v))] for k, v in rep_rows.items()},
        },
        "singleton_family_pairs": {
            "pairs": len(singleton_pairs),
            "single": wrong(single_gaps, truth_gaps, singleton_pairs),
            **{f: wrong(gaps(v), truth_gaps, singleton_pairs) for f, v in proj.items()},
        },
        "exclusions": {
            "non_finite_or_failed": nonfinite,
            "missing_budgets": [],
            "rising_beyond_noise": rising,
            "rule": "rise between consecutive budgets > 2 sqrt(2) sigma, sigma = largest borrowed relative SD x BPB",
            "count": len(set(nonfinite) | set(rising)),
        },
        "u_statistic_tests": {
            "note": "no Marin-specific calibration exists; reported under the most conservative earlier calibrated U "
            "critical value (2.99, C4 random-candidate) and under 1.96",
            "forms": {
                form: u_test(
                    {
                        k: 100.0
                        * (
                            float(np.sign(gaps(v)[k]) != np.sign(truth_gaps[k]))
                            - float(np.sign(single_gaps[k]) != np.sign(truth_gaps[k]))
                        )
                        for k in truth_gaps
                    }
                )
                for form, v in proj.items()
            },
        },
    }
    forms = task2["forms"]
    out = {
        "preregistration": "PREDICTIONS_TASK_MARIN.md",
        "members": members,
        "task_1": task1,
        "task_2": task2,
        "task_3": task3,
        "task_4": task4,
        "task_5": task5,
        "predictions": {
            "PT2_projection_loses_under_every_form": all(
                f["projection_wrong"] > task2["single_scale_wrong"] for f in forms.values()
            ),
            "PT2_gap_smallest_under_ii_or_iii": min(forms, key=lambda f: forms[f]["projection_wrong"])
            in ("ii_loglog", "iii_shared_exponent"),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"t1": {k: task1[k] for k in ("max_abs_difference", "rank_spearman_their_vs_python")}, "t2": task2}, indent=1
        )
    )
    print(json.dumps(task4["by_borrowed_sd"], indent=1)[:3000])
    print(json.dumps(task3, indent=1))
    print(
        json.dumps(
            {
                k: task5[k]
                for k in (
                    "n_families",
                    "within_family_pairs",
                    "errors_within_cross",
                    "one_per_family",
                    "exclusions",
                    "u_statistic_tests",
                )
            },
            indent=1,
        )
    )
    print(out["predictions"])


if __name__ == "__main__":
    main()
