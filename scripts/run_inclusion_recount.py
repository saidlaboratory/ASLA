"""PREDICTIONS_TASK_INCLUSION.md: inclusion audit and recount on the official set; selection bias; reordering with
the per-size protocol held fixed; forecast sensitivity to the config step.

Writes ``results/external/inclusion_recount.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_break_diagnosis as bd  # noqa: E402
from scripts import run_certified_leaderboard as cl  # noqa: E402
from scripts import run_leaderboard_core as core  # noqa: E402
from scripts import run_leaderboard_resolution as lr  # noqa: E402
from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_noise_bracket as nb  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402

OUT = REPO / "results" / "external" / "inclusion_recount.json"
HISTORY = REPO / "data" / "raw" / "marin_speedrun" / "runs_history"
RESULTS = REPO / "results" / "external"
RUNGS = lr.RUNGS
STEPS = (("130m", "300m"), ("300m", "520m"), ("520m", "1_2b"))
PROTOCOL = ("batch", "warmup", "schedule", "decay", "min_lr_ratio", "weight_decay", "seq_len", "hardware", "devices")
SIGMA_STAR_PREVIOUS = 0.0058309181591765825
GAP_1E22 = 0.0025607919704123128
DD_1B = lr.DD_1B


def load(name: str) -> Any:
    return json.loads((RESULTS / name).read_text())


# --- Task 1(a) ---------------------------------------------------------------------------------


def entry_audit(groups: list[str]) -> dict[str, Any]:
    rows = [line.split("\t") for line in (HISTORY / "commits.tsv").read_text().splitlines()]
    rows.sort(key=lambda r: r[1])
    present: dict[str, list[str]] = defaultdict(list)
    for sha, date, author, msg in rows:
        runs = json.loads((HISTORY / f"{sha[:10]}.json").read_text())
        names = {"/".join(r["run_name"].split("/")[:-1]) for r in runs if "/" in r["run_name"]}
        for g in groups:
            if g in names:
                present[g].append(sha[:10])
    meta = {sha[:10]: {"date": date, "author": author, "message": msg} for sha, date, author, msg in rows}
    order = [sha[:10] for sha, *_ in rows]
    out = {}
    for g in groups:
        first = present[g][0]
        last_index = order.index(present[g][-1])
        out[g] = {
            "first_commit": first,
            **meta[first],
            "automated": meta[first]["message"].startswith("automated update"),
            "present_at_pin": order[-1] in present[g],
            "removed_and_readded": len(present[g]) != len(order) - order.index(first) and last_index == len(order) - 1,
        }
    return out


# --- Task 1(b) helpers -------------------------------------------------------------------------


def target_reorder(members: list[str], cells: dict, s_fit: float, s_tgt: float) -> dict[str, Any]:
    pairs = set(combinations(sorted(members), 2))
    r = nb.both_rung_two_sigma(members, cells, pairs, s_fit, s_tgt)
    return {
        "determined": r["target_determined"],
        "reordered": r["target_determined_reordered"],
        "rate": r["target_rate"],
        "both_rung_genuine": r["genuine"],
        "both_rung_determined": r["determined_at_both"],
    }


def contiguous_top_k(boot: dict[str, Any]) -> int:
    return int(boot["largest_contiguous_top_k_holding_95"])


# --- Task 1(c) ---------------------------------------------------------------------------------


def selection_bias(cells: dict, members: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    residuals: list[float] = []
    loo_errors: list[float] = []
    for sub in ("muonremez_qwen3_scaling", "prism_berkeley_qwen3_scaling"):
        d = json.loads((bd.SRC / "experiments" / "speedrun" / sub / "selection_summary.json").read_text())
        per: dict[str, Any] = {}
        corrected = {}
        for rung in RUNGS:
            runs = [r for r in d["all_runs"][rung] if r.get("state") == "finished"]
            lrs = np.array([r["learning_rate"] for r in runs], dtype=float)
            y = np.array([r["c4_en_bpb"] for r in runs], dtype=float)
            z = np.log(lrs)
            coef = np.polyfit(z, y, 2)
            grid = np.linspace(z.min(), z.max(), 401)
            fitted_min = float(np.polyval(coef, grid).min())
            yhat = np.polyval(coef, z)
            residuals += list((y - yhat) / yhat)
            e = nb.loo(z, y, "quadratic")
            if e:
                loo_errors += e
            picked_lr = d["selected_runs"][rung]["learning_rate"] if isinstance(d["selected_runs"][rung], dict) else None
            picked = float(y[np.argmin(np.abs(lrs - picked_lr))]) if picked_lr is not None else float(y.min())
            per[rung] = {
                "runs": len(runs),
                "picked_lr": picked_lr,
                "picked_bpb": picked,
                "observed_min": float(y.min()),
                "picked_is_min": bool(np.isclose(picked, y.min())),
                "submitted_bpb": cells[(sub, rung)]["y"],
                "picked_equals_submitted": bool(np.isclose(picked, cells[(sub, rung)]["y"], rtol=1e-6)),
                "fitted_optimum": fitted_min,
                "bias_bpb": picked - fitted_min,
                "bias_relative": (picked - fitted_min) / fitted_min,
                "optimum_inside_range": bool(z.min() < grid[np.argmin(np.polyval(coef, grid))] < z.max()),
            }
            corrected[rung] = fitted_min
        others = [m for m in members if m != sub]
        steps = {}
        for f, t in STEPS:
            db = abs(per[t]["bias_relative"] - per[f]["bias_relative"])
            gaps = [abs(np.log(cells[(sub, r)]["y"]) - np.log(cells[(o, r)]["y"])) for o in others for r in (f, t)]
            steps[f"{f}->{t}"] = {
                "bias_difference_relative": db,
                "median_pair_gap_log": float(np.median(gaps)),
                "smaller_than_gap": bool(db < np.median(gaps)),
            }
        flips = []
        for o in others:
            for r in ("300m", "520m"):
                before = np.sign(cells[(sub, r)]["y"] - cells[(o, r)]["y"])
                after = np.sign(corrected[r] - cells[(o, r)]["y"])
                if before != after:
                    flips.append({"other": o, "rung": r})
        out[sub] = {"by_size": per, "steps": steps, "sign_flips_at_300m_or_520m": flips}
    res = np.asarray(residuals)
    df = 2 * 4 * 2
    out["sweep_noise"] = {
        "residual_sd_relative": float(np.sqrt(np.sum(res**2) / df)),
        "df": df,
        "loo_rms_relative": float(np.sqrt(np.mean(np.square(loo_errors)))),
        "loo_n": len(loo_errors),
    }
    return out


def rerun_substitution(cells: dict, subs: list[str]) -> dict[str, Any]:
    """Exploratory, not pre-registered: board values that differ from the PR's own selected runs, and the
    300m->520m persistent flips recounted with the PR-selected values."""

    runs = {r["run_name"]: r for r in json.loads((rig.RAW / "runs.json").read_text())}
    swapped = {}
    alt = dict(cells)
    for sub in ("muonremez_qwen3_scaling", "prism_berkeley_qwen3_scaling"):
        d = json.loads((bd.SRC / "experiments" / "speedrun" / sub / "selection_summary.json").read_text())
        for rung in RUNGS:
            sel = d["selected_runs"][rung]
            board = runs[f"{sub}/{rung}"]
            if not np.isclose(sel["c4_en_bpb"], board["eval_paloma_c4_en_bpb"], rtol=1e-6):
                swapped[f"{sub}/{rung}"] = {
                    "pr_selected": sel["c4_en_bpb"],
                    "pr_run": sel["run_name"],
                    "board": board["eval_paloma_c4_en_bpb"],
                    "board_wandb": board["wandb_link"],
                    "relative_difference": board["eval_paloma_c4_en_bpb"] / sel["c4_en_bpb"] - 1,
                }
                alt[(sub, rung)] = dict(cells[(sub, rung)], y=float(sel["c4_en_bpb"]))
    family = {s: s.split("/")[0] for s in subs}
    groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    for s in subs:
        groups[tuple(cells[(s, r)]["x"] for r in RUNGS)].append(s)
    genuine = [(a, b) for g in groups.values() for a, b in combinations(sorted(g), 2) if family[a] != family[b]]

    def persistent(c: dict) -> list[tuple[str, str]]:
        out = []
        for a, b in genuine:
            g = np.log([c[(a, r)]["y"] for r in RUNGS]) - np.log([c[(b, r)]["y"] for r in RUNGS])
            sg = [int(np.sign(v)) if abs(v) > st.TAU_PRIMARY else 0 for v in g]
            if int(st.changes_with_tolerance(g[None], st.TAU_PRIMARY)[0]) == 1 and sg[1] and sg[2] and sg[1] != sg[2]:
                out.append((a, b))
        return out

    before, after = persistent(cells), persistent(alt)
    return {
        "label": "exploratory, not pre-registered",
        "swapped": swapped,
        "persistent_flips_board": len(before),
        "persistent_flips_pr_selected": len(after),
        "removed": [list(p) for p in before if p not in after],
        "added": [list(p) for p in after if p not in before],
    }


# --- Task 2(a) ---------------------------------------------------------------------------------


def change_vector(table: dict[str, dict[str, Any]], s: str, f: str, t: str) -> tuple:
    return tuple(json.dumps(bd.change(table[s][f].get(k), table[s][t].get(k)), default=str) for k in PROTOCOL)


def jackknife(kernel: dict[tuple[str, str], float]) -> float | None:
    from asla.analysis.target_scoring import jackknife_standard_error

    names = {n for p in kernel for n in p}
    if len(kernel) < 2 or len(names) < 3:
        return None
    return float(jackknife_standard_error(kernel))


def dd_pairs(f: str, t: str) -> dict[tuple[str, str], float]:
    from asla.analysis.moderation import fit_prior, moderate

    frame = pd.read_parquet(REPO / "data" / "datadecide_runs.parquet")
    fit = frame[frame["scale_label"] == f].groupby("intervention")["bpb"].mean().to_dict()
    tcells = frame[frame["scale_label"] == t].groupby("intervention")["bpb"]
    target, s2, n = tcells.mean().to_dict(), tcells.var(ddof=1).to_dict(), int(tcells.count().min())
    prior = fit_prior(list(s2.values()), n - 1)
    mod = {r: float(moderate(v, prior)) for r, v in s2.items()}
    out = {}
    for a, b in combinations(sorted(target), 2):
        g = target[a] - target[b]
        if abs(g) > al.Z * np.sqrt(mod[a] / n + mod[b] / n):
            out[(a, b)] = 100.0 * float(np.sign(fit[a] - fit[b]) != np.sign(g))
    return out


def class_rates(cells: dict, table: dict, subs: list[str], sig: dict[str, float]) -> dict[str, Any]:
    family = {s: s.split("/")[0] for s in subs}
    out: dict[str, Any] = {}
    for f, t in STEPS:
        groups: dict[tuple[float, float], list[str]] = defaultdict(list)
        for s in subs:
            groups[(cells[(s, f)]["x"], cells[(s, t)]["x"])].append(s)
        genuine = [(a, b) for g in groups.values() for a, b in combinations(sorted(g), 2) if family[a] != family[b]]
        dd = dd_pairs(lr.DD_MATCH[f], lr.DD_MATCH[t])
        dd_rate, dd_se = float(np.mean(list(dd.values()))) / 100, jackknife(dd)
        step: dict[str, Any] = {
            "genuine_pairs": len(genuine),
            "datadecide": {"determined": len(dd), "rate": dd_rate, "jackknife_se": None if dd_se is None else dd_se / 100},
        }
        kernels = {}
        for cls in ("matched", "mismatched"):
            pairs = [
                p
                for p in genuine
                if (change_vector(table, p[0], f, t) == change_vector(table, p[1], f, t)) == (cls == "matched")
            ]
            kern = {}
            for a, b in pairs:
                yt = (cells[(a, t)]["y"], cells[(b, t)]["y"])
                g = yt[0] - yt[1]
                if abs(g) > al.Z * sig[t] * np.hypot(*yt):
                    kern[(a, b)] = 100.0 * float(np.sign(cells[(a, f)]["y"] - cells[(b, f)]["y"]) != np.sign(g))
            kernels[cls] = kern
            rate = float(np.mean(list(kern.values()))) / 100 if kern else None
            se = jackknife(kern)
            row: dict[str, Any] = {
                "pairs": len(pairs),
                "determined": len(kern),
                "reordered": int(sum(v > 0 for v in kern.values())),
                "rate": rate,
                "jackknife_se": None if se is None else se / 100,
                "pair_list": [list(p) for p in pairs],
            }
            if rate is not None:
                tab = [
                    [row["reordered"], row["determined"] - row["reordered"]],
                    [int(round(dd_rate * len(dd))), len(dd) - int(round(dd_rate * len(dd)))],
                ]
                row["fisher_vs_datadecide_one_sided"] = float(stats.fisher_exact(tab, alternative="greater").pvalue)
                if se is not None and dd_se is not None:
                    row["z_vs_datadecide"] = (rate - dd_rate) / np.hypot(se / 100, dd_se / 100) if (se or dd_se) else None
            step[cls] = row
        m, mm = step["matched"], step["mismatched"]
        if m["rate"] is not None and mm["rate"] is not None and m["jackknife_se"] and mm["jackknife_se"]:
            step["z_mismatched_minus_matched"] = (mm["rate"] - m["rate"]) / np.hypot(m["jackknife_se"], mm["jackknife_se"])
        out[f"{f}->{t}"] = step
    return out


def matched_validation(cells: dict, table: dict, subs: list[str], pairs: set) -> dict[str, Any]:
    x = np.array([[cells[(s, b)]["x"] for b in cl.MARIN_FIT] for s in subs])
    y = np.array([[cells[(s, b)]["y"] for b in cl.MARIN_FIT] for s in subs])
    target_x = np.array([cells[(s, "1_2b")]["x"] for s in subs])
    truth = np.array([cells[(s, "1_2b")]["y"] for s in subs])
    primary = cl.loglog_forecast(x, y, target_x)
    pts = {s: [{"x": float(x[i, k]), "y": float(y[i, k])} for k in range(3)] for i, s in enumerate(subs)}
    alt = cl.three_param(pts, {s: float(target_x[i]) for i, s in enumerate(subs)})
    alternative = np.array([alt[s] for s in subs])
    keep = {(subs.index(a), subs.index(b)) for a, b in pairs} | {(subs.index(b), subs.index(a)) for a, b in pairs}
    rng = np.random.default_rng(cl.SEED)
    rows = {}
    for sigma in cl.GRID:
        draws = cl.loglog_forecast(x[None], cl.noisy(y, sigma, cl.DRAWS_VALIDATE, rng), target_x[None])
        cert = [p for p in cl.certify(draws, primary, alternative) if p in keep]
        rows[f"{sigma:g}"] = cl.realized(cert, truth)
    return {"pairs": len(pairs), "rows": rows, "sigma_star": cl.sigma_star(rows)}


# --- Task 2(c) ---------------------------------------------------------------------------------


def sensitivity(cells: dict, table: dict, subs: list[str], fixed: dict[str, float], sig: dict[str, float]) -> dict[str, Any]:
    sd = np.array([sig[r] for r in RUNGS])
    out = {}
    lt = np.log(rig.FORECAST_X)
    for s in subs:
        lx = np.log([cells[(s, r)]["x"] for r in RUNGS])
        ly = np.log([cells[(s, r)]["y"] for r in RUNGS])
        change_at = next((k for k in range(1, 4) if table[s][RUNGS[k]].get("batch") != table[s]["130m"].get("batch")), None)
        W = np.diag(1 / sd**2)

        def coef(X: np.ndarray, xt: np.ndarray, rows: slice | np.ndarray = slice(None)) -> np.ndarray:
            Xs, Ws = X[rows], W[rows][:, rows]
            return xt @ np.linalg.inv(Xs.T @ Ws @ Xs) @ Xs.T @ Ws  # forecast weights on log y[rows]

        c0 = coef(np.column_stack([np.ones(4), lx]), np.array([1, lt]))
        b0 = float(np.exp(c0 @ ly))
        row: dict[str, Any] = {
            "A_three_parameter": fixed[s],
            "B0_loglog": b0,
            "batch_change_rung": None if change_at is None else RUNGS[change_at],
        }
        if change_at is not None:
            step = (np.arange(4) >= change_at).astype(float)
            cb = coef(np.column_stack([np.ones(4), lx, step]), np.array([1, lt, 1]))
            b = float(np.exp(cb @ ly))
            se_log = float(np.sqrt(np.sum((cb - c0) ** 2 * sd**2)))
            row.update(
                {
                    "B_step": b,
                    "B_minus_B0": b - b0,
                    "B_minus_B0_se": se_log * b,
                    "B_minus_A": b - fixed[s],
                    "B_minus_B0_in_gaps": (b - b0) / GAP_1E22,
                }
            )
            post = np.arange(change_at, 4)
            if len(post) >= 2:
                cc = coef(np.column_stack([np.ones(4), lx]), np.array([1, lt]), post)
                c = float(np.exp(cc @ ly[post]))
                row.update({"C_post_only": c, "C_minus_B0": c - b0, "C_minus_B0_in_gaps": (c - b0) / GAP_1E22})
            else:
                row["C_post_only"] = "not identified (one post-change rung)"
            row["cannot_be_separated_from_step"] = bool(abs(b - b0) > GAP_1E22)
        out[s] = row
    marked = [s for s, v in out.items() if v.get("cannot_be_separated_from_step")]
    return {"by_submission": out, "marked": marked, "marked_count": len(marked), "submissions": len(subs)}


def main() -> None:
    sig = load("noise_bracket.json")["e_reruns"]["sigma_by_rung"]
    rows, cells = lr.listing()
    subs = sorted({k[0] for k in cells})
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_INCLUSION.md", "official_set": subs}
    entry = entry_audit(subs)
    out["task1a"] = {
        "pin": "marin-community/speedrun@31fe8028 (main); runs.json sha256 957febab...",
        "display_rule": "scaling track: runs with '/' grouped by folder; a group ranks if any run has > 3e20 training "
        "hardware FLOPs; sorted by the 1e22 forecast (App.tsx, useSpeedrunData.ts, scaling.ts)",
        "entry": entry,
        "non_automated_entries": sorted(s for s, v in entry.items() if not v["automated"]),
        "non_official_included": [],
        "omitted_official": [
            "hackable_transformer_attn_sink/std_attn (top-rung label '1.2b'; absent from every 21-member analysis)"
        ],
        "instance_11": "not recorded: no non-official submission was included (the stated condition)",
    }

    # (b) recount
    sigma_vec = np.array([sig[r] for r in RUNGS])
    pts = {s: [{"x": cells[(s, r)]["x"], "y": cells[(s, r)]["y"]} for r in RUNGS] for s in subs}
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in subs}
    aligned_prev = load("marin_aligned.json")
    reorder = {
        "dd_1b_sd": target_reorder(subs, cells, DD_1B, DD_1B),
        "task1_bound": target_reorder(subs, cells, sig["520m"], sig["1_2b"]),
    }
    dd = lr.dd_reorder("530M", "1B")
    for v in reorder.values():
        v["fisher_vs_datadecide_530M_1B"] = lr.fisher(v, dd)
    prev_named = aligned_prev["task1"]["designs"]["named_rungs"]
    heldout = al.heldout(subs, cells, cells, set(combinations(sorted(subs), 2)), None)
    displayed_res = ref.lm([{"id": s, "points": pts[s], "targets": [rig.FORECAST_X]} for s in subs])
    displayed = {s: displayed_res[s]["reference_predictions"][0] for s in subs}
    fixed = al.patched_forecasts(pts)
    changed = [
        [a, b]
        for a, b in combinations(sorted(subs), 2)
        if np.sign(displayed[a] - displayed[b]) != np.sign(fixed[a] - fixed[b])
    ]
    optimum = st.global_optimum(subs, raw)
    loglog = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in subs}
    form = core.invariant_core({"three_parameter_global_optimum": optimum, "two_parameter_loglog": loglog}, subs)
    boot = lr.bootstrap(pts, sigma_vec, fixed)
    cert = cl.apply(subs, cells, SIGMA_STAR_PREVIOUS)
    prev_fix = aligned_prev["task3"]["displayed_vs_fixed"]
    prev_res = load("leaderboard_resolution.json")["task2a"]["levels"]["task1_bound"]
    prev_cert = load("certified_leaderboard.json")["applied_1e22"]["at_sigma_star_marin"]
    nb_prev = load("noise_bracket.json")["e_reruns"]["both_rung"]["bound"]["named_rungs"]
    out["task1b"] = {
        "reorder_520m_1_2b": {
            "official_22": reorder,
            "previous_21": {
                "dd_1b_sd": {k: prev_named["reorder"]["datadecide_c4_1B"][k] for k in ("determined", "reordered", "rate")},
                "task1_bound": {
                    "determined": nb_prev["target_determined"],
                    "reordered": nb_prev["target_determined_reordered"],
                    "rate": nb_prev["target_rate"],
                },
            },
            "datadecide_530M_1B": dd,
        },
        "heldout_projection": {
            "official_22": {
                "pairs": heldout["pairs"],
                "single_wrong": heldout["single_wrong"],
                "forms": {
                    f: {"wrong": v["wrong"], "u_abs_t": (v["u_test"] or {}).get("abs_t")}
                    for f, v in heldout["forms"].items()
                },
            },
            "previous_21": {
                "single_wrong": prev_named["heldout"]["single_wrong"],
                "forms": {
                    f: {"wrong": v["wrong"], "u_abs_t": (v["u_test"] or {}).get("abs_t")}
                    for f, v in prev_named["heldout"]["forms"].items()
                },
            },
        },
        "fix_effect": {
            "official_22": {
                "changed": len(changed),
                "changed_pairs": changed,
                "kendall": ref.kendall(displayed, fixed),
                "top3_displayed": sorted(subs, key=lambda s: displayed[s])[:3],
                "top3_fixed": sorted(subs, key=lambda s: fixed[s])[:3],
                "std_attn_rank_fixed": sorted(subs, key=lambda s: fixed[s]).index("hackable_transformer_attn_sink/std_attn")
                + 1,
            },
            "previous_21": {
                "changed": prev_fix["count"],
                "kendall": prev_fix["kendall"],
                "top3_displayed": prev_fix["top3_displayed"],
                "top3_fixed": prev_fix["top3_fixed"],
            },
        },
        "form_core": {
            "official_22": {
                "robust": form["invariant_pairs"],
                "pairs": form["pairs"],
                "top_k": form["largest_invariant_top_k"],
                "top1": form["distinct_top1"],
            },
            "previous_21": {
                "robust": aligned_prev["task2"]["two_forms"]["invariant_pairs"],
                "pairs": 210,
                "top_k": aligned_prev["task2"]["two_forms"]["largest_invariant_top_k"],
            },
        },
        "noise_resolution_bound": {
            "official_22": {
                "holding_95": boot["pairs_holding_95"],
                "contiguous_top_k": contiguous_top_k(boot),
                "top1_hold_share": boot["top_k_hold_share"]["1"],
            },
            "previous_21": {
                "holding_95": prev_res["pairs_holding_95"],
                "contiguous_top_k": prev_res["largest_contiguous_top_k_holding_95"],
            },
        },
        "certified_at_0_58pct": {
            "official_22": {
                "certified": cert["certified_pairs"],
                "pairs": cert["pairs"],
                "above_all": cert["top1_certified_above_all"],
            },
            "previous_21": {"certified": prev_cert["certified_pairs"]},
        },
    }
    out["task1c"] = selection_bias(cells, subs)
    out["task1c"]["exploratory_rerun_substitution"] = rerun_substitution(cells, subs)

    audit = bd.audit(subs)
    table = audit["table"]
    out["task2a"] = class_rates(cells, table, subs, sig)
    fo = load("break_diagnosis.json")["task1c"]["reorder"]
    out["task2a"]["fantastic_optimizers_fitted_optima"] = {
        k: {kk: fo[k][kk] for kk in ("determined", "reordered", "rate")} for k in fo
    }
    consistent = [
        s for s in subs if all(len({json.dumps(table[s][r].get(k), default=str) for r in RUNGS}) == 1 for k in PROTOCOL)
    ]
    all_step_matched = set()
    family = {s: s.split("/")[0] for s in subs}
    for a, b in combinations(sorted(subs), 2):
        same_x = all(cells[(a, r)]["x"] == cells[(b, r)]["x"] for r in RUNGS)
        if (
            same_x
            and family[a] != family[b]
            and all(change_vector(table, a, f, t) == change_vector(table, b, f, t) for f, t in STEPS)
        ):
            all_step_matched.add((a, b))
    out["task2b"] = {
        "config_consistent_submissions": consistent,
        "fallback_pairs_matched_at_all_steps": sorted(list(p) for p in all_step_matched),
        "validation": matched_validation(cells, table, subs, all_step_matched) if all_step_matched else None,
        "previous_sigma_star": SIGMA_STAR_PREVIOUS,
    }
    out["task2c"] = sensitivity(cells, table, subs, fixed, sig)

    r_bound = reorder["task1_bound"]
    m = out["task2a"]["300m->520m"]["matched"]
    sel = out["task1c"]
    biases = [
        v["bias_relative"]
        for s in ("muonremez_qwen3_scaling", "prism_berkeley_qwen3_scaling")
        for v in sel[s]["by_size"].values()
    ]
    v = out["task2b"]["validation"]
    out["predictions"] = {
        "PI1_only_muonremez_prism_manual": out["task1a"]["non_automated_entries"]
        == ["muonremez_qwen3_scaling", "prism_berkeley_qwen3_scaling"],
        "PI2_reorder_within_2pp": abs(r_bound["rate"] - nb_prev["target_rate"]) <= 0.02
        and abs(reorder["dd_1b_sd"]["rate"] - prev_named["reorder"]["datadecide_c4_1B"]["rate"]) <= 0.02,
        "PI2_projection_no_better": all(f["wrong"] >= heldout["single_wrong"] for f in heldout["forms"].values()),
        "PI2_fix_changes_top3": set(out["task1b"]["fix_effect"]["official_22"]["top3_displayed"])
        != set(out["task1b"]["fix_effect"]["official_22"]["top3_fixed"]),
        "PI2_core_at_least_190": form["invariant_pairs"] >= 190,
        "PI2_at_most_100_hold": boot["pairs_holding_95"] <= 100,
        "PI2_at_most_20_certified": cert["certified_pairs"] <= 20,
        "PS1_bias_negative_in_range": all(-0.003 <= b <= -0.0002 for b in biases),
        "PS1_no_crossings": all(
            not sel[s]["sign_flips_at_300m_or_520m"] for s in ("muonremez_qwen3_scaling", "prism_berkeley_qwen3_scaling")
        ),
        "PS1_sweep_sd_in_range": 0.0005 <= sel["sweep_noise"]["residual_sd_relative"] <= 0.005,
        "PM1_matched_at_most_5pct": m["rate"] is not None and m["rate"] <= 0.05,
        "PM1_matched_not_different_from_dd": m.get("z_vs_datadecide") is None or abs(m["z_vs_datadecide"]) < 1.96,
        "PM1_mismatched_higher": (out["task2a"]["300m->520m"]["mismatched"]["rate"] or 0) > (m["rate"] or 0),
        "PV0_none_consistent": not consistent,
        "PV1_sigma_star_below_0_58pct": v is not None
        and v["sigma_star"]["sigma_star"] is not None
        and v["sigma_star"]["sigma_star"] < SIGMA_STAR_PREVIOUS,
        "PF1_at_least_half_marked": out["task2c"]["marked_count"] >= len(subs) / 2,
    }
    out["task1b"]["heldout_projection"]["correction"] = (
        "no projection form beats single-scale significantly (no candidate-level U test exceeds the conservative "
        "2.99); the shared exponent has fewer point errors than single-scale: 48 against 52 on the official 22 "
        "(43 against 46 on the 21). Statements that projection never beats single-scale were wrong."
    )
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
