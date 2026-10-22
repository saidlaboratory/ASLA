"""PREDICTIONS_TASK_BREAK.md, Tasks 1 and 2: the 300M->520M break, the GPU/TPU pair, the Delphi forecast bound.

Configs come from marin-community/marin@cdd4e25a8d (and the heads of PRs #4932/#4933 for the two unmerged
submissions), fetched into ``data/raw/marin_speedrun/repo_cdd4e25a8d``.

Writes ``results/external/break_diagnosis.json``.
"""

from __future__ import annotations

import json
import re
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

from scripts import run_leaderboard_resolution as lr  # noqa: E402
from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_noise_bracket as nb  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402

OUT = REPO / "results" / "external" / "break_diagnosis.json"
SRC = REPO / "data" / "raw" / "marin_speedrun" / "repo_cdd4e25a8d"
BRACKET = REPO / "results" / "external" / "noise_bracket.json"
RUNGS = lr.RUNGS
LOGX_RUNG = {"130m": 0, "300m": 1, "520m": 2, "1_2b": 3}
FIELDS = ("tokens", "tokens_per_param", "steps", "batch", "seq_len", "lr", "warmup", "schedule", "decay",
          "min_lr_ratio", "weight_decay", "hardware", "devices", "data_seed", "dataset", "optimizer_keys")  # fmt: skip


# --- (a) config audit --------------------------------------------------------------------------


def first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v: Any = d
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v is not None:
            return v
    return None


def settings(info: dict[str, Any]) -> dict[str, Any]:
    tc, oc = info.get("train_config") or {}, (info.get("train_config") or {}).get("optimizer_config") or {}
    res = info.get("resources") or {}
    hw = first(res, "tpu_type", "accelerator_type") or "/".join(str(first(res, f"device.{k}")) for k in ("type", "variant"))
    return {
        "tokens": info.get("total_tokens"),
        "tokens_per_param": info["total_tokens"] / info["model_size"] if info.get("total_tokens") else None,
        "steps": tc.get("num_train_steps"),
        "batch": tc.get("train_batch_size"),
        "seq_len": first(info, "train_config.train_seq_len", "model_config.seq_len", "model_config.max_seq_len"),
        "lr": first(oc, "learning_rate", "lr", "adam_lr") or tc.get("learning_rate"),
        "warmup": first(oc, "warmup") if oc.get("warmup") is not None else tc.get("warmup"),
        "schedule": oc.get("lr_schedule") or tc.get("lr_schedule"),
        "decay": oc.get("decay") if oc.get("decay") is not None else tc.get("decay"),
        "min_lr_ratio": oc.get("min_lr_ratio") if oc.get("min_lr_ratio") is not None else tc.get("min_lr_ratio"),
        "weight_decay": first(oc, "weight_decay", "adamc_weight_decay", "adam_weight_decay") or tc.get("weight_decay"),
        "hardware": hw,
        "devices": info.get("num_devices"),
        "data_seed": tc.get("data_seed"),
        "dataset": info.get("tokenized_dataset"),
        "optimizer_keys": ",".join(sorted(k for k, v in oc.items() if v is not None)),
        "completed": info.get("run_completion_timestamp"),
        "bpb": info.get("eval/paloma/c4_en/bpb"),
    }


def commits() -> dict[str, str]:
    out = {}
    for name in ("file_commits.tsv", "pr_files.tsv"):
        p = SRC / name
        if p.exists():
            for line in p.read_text().splitlines():
                path, c = line.split("\t")
                out[path] = c
    return out


def audit(subs: list[str]) -> dict[str, Any]:
    runs = {r["run_name"]: r for r in json.loads((rig.RAW / "runs.json").read_text())}
    cm = commits()
    table: dict[str, dict[str, Any]] = {}
    provenance = {}
    for s in subs:
        table[s] = {}
        for rung in RUNGS:
            name = next(n for n in (f"{s}/{rung}", f"{s}/1.2b") if n in runs and (n.endswith(rung) or rung == "1_2b"))
            rel = runs[name]["results_filepath"] + "/speedrun_results.json"
            f = SRC / rel
            if not f.exists():
                table[s][rung] = {"missing": rel}
                continue
            row = settings(json.loads(f.read_text())["runs"][0]["run_info"])
            row["commit"] = cm.get(rel)
            table[s][rung] = row
        top = SRC / "experiments" / "speedrun" / s.split("/")[0]
        text = " ".join(p.read_text(errors="ignore") for p in top.rglob("*.py"))
        lrs = {table[s][r].get("lr") for r in RUNGS}
        provenance[s] = {
            "fantastic_optimizers_derived": bool(
                re.search(r"Kaiyue|Fantastic-Optimizers|Fantastic Optimizers|optimizer_sweep", text)
            ),
            "tuned_per_size": len(lrs) > 1,
            "unmerged_agent_pr": any("PR#" in str(table[s][r].get("commit")) for r in RUNGS),
            "commits": sorted({str(table[s][r].get("commit")) for r in RUNGS}),
        }
    return {"table": table, "provenance": provenance}


def change(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and a:
        return round(b / a, 6)
    return "same" if a == b else f"{a} -> {b}"


def flags(table: dict[str, dict[str, Any]], subsets: dict[str, list[str]]) -> dict[str, Any]:
    out = {}
    for name, subs in subsets.items():
        ch = {f: {s: change(table[s]["300m"].get(f), table[s]["520m"].get(f)) for s in subs} for f in FIELDS}
        out[name] = {
            f: {"changes": v, "flagged": len({json.dumps(x, default=str) for x in v.values()}) > 1} for f, v in ch.items()
        }
    return out


# --- (b) shape tests -------------------------------------------------------------------------


def wls(X: np.ndarray, y: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    W = np.diag(1 / sd**2)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    r = y - X @ beta
    return beta, cov, float(r @ W @ r)


def shape(cells: dict, subs: list[str], pairs: list[tuple[str, str]], sig: dict[str, float]) -> dict[str, Any]:
    sd = np.array([sig[r] for r in RUNGS])
    step = np.array([0, 0, 1, 1.0])
    per_sub = {}
    for s in subs:
        lx = np.log([cells[(s, r)]["x"] for r in RUNGS])
        ly = np.log([cells[(s, r)]["y"] for r in RUNGS])
        beta, cov, _ = wls(np.column_stack([np.ones(4), lx, step]), ly, sd)
        per_sub[s] = {"d": float(beta[2]), "se": float(np.sqrt(cov[2, 2])), "z": float(beta[2] / np.sqrt(cov[2, 2]))}
    ds = np.array([v["d"] for v in per_sub.values()])
    se = float(np.median([v["se"] for v in per_sub.values()]))
    classes: dict[str, list[list[str]]] = defaultdict(list)
    prefer = []
    for a, b in pairs:
        g = np.log([cells[(a, r)]["y"] for r in RUNGS]) - np.log([cells[(b, r)]["y"] for r in RUNGS])
        signs = [int(np.sign(v)) if abs(v) > st.TAU_PRIMARY else 0 for v in g]
        n = int(st.changes_with_tolerance(g[None], st.TAU_PRIMARY)[0])
        across = signs[1] != 0 and signs[2] != 0 and signs[1] != signs[2]
        cls = "reverting" if n >= 2 else "persistent_300_520" if n == 1 and across else "other_single" if n == 1 else "none"
        classes[cls].append([a, b])
        if across:
            lxm = np.log([np.sqrt(cells[(a, r)]["x"] * cells[(b, r)]["x"]) for r in RUNGS])
            gsd = np.sqrt(2) * sd
            _, _, sse_step = wls(np.column_stack([np.ones(4), step]), g, gsd)
            _, _, sse_trend = wls(np.column_stack([np.ones(4), lxm]), g, gsd)
            prefer.append({"pair": [a, b], "class": cls, "sse_step": sse_step, "sse_trend": sse_trend,
                           "prefers": "step" if sse_step < sse_trend else "trend"})  # fmt: skip
    k = sum(p["prefers"] == "step" for p in prefer)
    return {
        "submission_step": per_sub,
        "steps_detected": sum(abs(v["z"]) > 1.96 for v in per_sub.values()),
        "d_spread_sd_over_median_se": float(ds.std(ddof=1) / se),
        "d_mean": float(ds.mean()),
        "flip_classes": {c: {"count": len(v), "pairs": v} for c, v in classes.items()},
        "pairs_changing_300_520": len(prefer),
        "prefer_step": k,
        "binomial_p_two_sided": float(stats.binomtest(k, len(prefer), 0.5).pvalue) if prefer else None,
        "pair_shapes": prefer,
    }


# --- (c) Fantastic Optimizers fitted optima -----------------------------------------------------


def fo_optima(budget: str = "1") -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    batch: dict[str, dict[str, Any]] = defaultdict(dict)
    for path in sorted((REPO / "data" / "raw" / "fantastic_optimizers").glob(f"*/*/{budget}/result.json")):
        opt, size = path.parts[-4], path.parts[-3]
        d = json.loads(path.read_text())
        best, base = d.get("best_config") or {}, d.get("baseline") or {}
        if base.get("loss") is None:
            continue
        batch[opt][size] = {k: best.get(k) for k in ("train_batch_size", "learning_rate", "warmup", "weight_decay")}
        by: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for ab in d.get("ablations") or []:
            if (
                ab.get("loss") is not None
                and isinstance(ab.get("value"), (int, float))
                and not isinstance(ab["value"], bool)
            ):
                by[ab["param"]].append((float(ab["value"]), float(ab["loss"])))
        base_fits, gains = [], []
        for param, rows in by.items():
            if not isinstance(best.get(param), (int, float)) or isinstance(best.get(param), bool):
                continue
            v = np.array([float(best[param])] + [r[0] for r in rows])
            y = np.array([float(base["loss"])] + [r[1] for r in rows])
            keep = y <= y.min() * (1 + nb.NEAR_OPTIMUM)
            if keep.sum() < 3 or not keep[0]:
                continue
            z = nb.coordinate(param, v[keep])
            if z is None or len(np.unique(z)) < 3:
                continue
            coef = np.polyfit(z, y[keep], 2)
            grid = np.linspace(z.min(), z.max(), 201)
            fb = float(np.polyval(coef, z[0]))
            base_fits.append(fb)
            gains.append(min(0.0, float(np.polyval(coef, grid).min()) - fb))
        if len(base_fits) >= 2:
            out[opt][size] = {"optimum": float(np.mean(base_fits) + sum(gains)), "best_run": float(base["loss"]),
                              "slices": len(base_fits)}  # fmt: skip
    return {"optima": out, "best_config_schedule": batch}


def fo_reorder(optima: dict[str, Any], f: str, t: str, sig: dict[str, float]) -> dict[str, Any]:
    opts = sorted(o for o, v in optima.items() if f in v and t in v)
    det = wrong = both = genuine = 0
    flips = []
    for a, b in combinations(opts, 2):
        yf = (optima[a][f]["optimum"], optima[b][f]["optimum"])
        yt = (optima[a][t]["optimum"], optima[b][t]["optimum"])
        gt, gf = yt[0] - yt[1], yf[0] - yf[1]
        if abs(gt) > al.Z * sig[t] * np.hypot(*yt):
            det += 1
            bad = np.sign(gf) != np.sign(gt)
            wrong += int(bad)
            if abs(gf) > al.Z * sig[f] * np.hypot(*yf):
                both += 1
                genuine += int(bad)
            if bad:
                flips.append([a, b])
    return {"optimizers": opts, "pairs": len(opts) * (len(opts) - 1) // 2, "determined": det, "reordered": wrong,
            "rate": wrong / det if det else None, "both_determined": both, "genuine": genuine, "flipped": flips}  # fmt: skip


# --- Task 2 ----------------------------------------------------------------------------------


def gpu_tpu() -> dict[str, Any]:
    def flat(d: dict[str, Any], p: str = "") -> dict[str, Any]:
        o: dict[str, Any] = {}
        for k, v in d.items():
            o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
        return o

    base = SRC / "experiments" / "speedrun" / "parallel_llama" / "old_scaling"
    out: dict[str, Any] = {}
    for size in ("75m", "150m", "300m"):
        g, t = base / "gpu" / size / "speedrun_results.json", base / "tpu" / size / "speedrun_results.json"
        if not (g.exists() and t.exists()):
            out[size] = {"missing": True}
            continue
        a, b = (flat(json.loads(p.read_text())["runs"][0]["run_info"]) for p in (g, t))
        diff = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        out[size] = {
            "differing_fields": diff,
            "data_seed": [a.get("train_config.data_seed"), b.get("train_config.data_seed")],
        }
    texts = {
        p.name: p.read_text(errors="ignore") for p in (SRC / "experiments" / "speedrun" / "parallel_llama").rglob("*.py")
    }
    seeds = {n: re.findall(r".{0,40}seed.{0,40}", t) for n, t in texts.items() if "seed" in t}
    data_seeds = {tuple(v["data_seed"]) for v in out.values() if "data_seed" in v}
    same = all(x[0] == x[1] for x in data_seeds)
    return {
        "by_size": out,
        "seed_mentions_in_scripts": seeds,
        "data_seed_equal_or_both_unset": same,
        "model_seed_recorded": False,
        "relabel": "same-seed hardware divergence (data seed equal or both default; model seed not recorded, so the "
        "code default)"
        if same and not seeds
        else "seeds differ or are set explicitly: see seed_mentions_in_scripts",
    }


def delphi_bound() -> dict[str, Any]:
    fits = {}
    for recipe, table in (("adamh", "delphi-ladder"), ("cautious_adamc", "first-ladder")):
        d = pd.read_parquet(REPO / "data" / "raw" / "delphi" / table / "train.parquet")
        opt, held = d[d["kind"] == "optimum"], d[d["kind"] == "held_out"]
        pts = [{"x": c, "y": float(opt[np.isclose(opt["C_flops"], c)].iloc[0]["loss_star"])} for c in lr.DELPHI_BUDGETS]
        targets = (1e21, 1e22)
        h = 1e-4
        jobs = {"base": pts, **{f"up{i}": [dict(p, y=p["y"] * (1 + h) if j == i else p["y"]) for j, p in enumerate(pts)]
                                for i in range(len(pts))}}  # fmt: skip
        pred = al.patched_predictions(jobs, targets)
        base = np.array(pred["base"], dtype=float)
        sens = np.array(
            [(np.log(np.array(pred[f"up{i}"], dtype=float)) - np.log(base)) / np.log1p(h) for i in range(len(pts))]
        )
        rows: list[dict[str, Any]] = []
        for k, t in enumerate(targets):
            ht = held[np.isclose(held["C_nom"], t)]
            truth, n = float(ht["loss"].mean()), int(len(ht))
            e = float(np.log(base[k] / truth))
            rows.append({"target": f"{t:.0e}", "forecast": float(base[k]), "truth": truth, "n_truth": n, "log_error": e,
                         "sum_s2": float(np.sum(sens[:, k] ** 2))})  # fmt: skip
        q_cons = sum(r["n_truth"] * r["log_error"] ** 2 for r in rows) / len(rows)
        q_eq = sum(r["log_error"] ** 2 / (r["sum_s2"] + 1 / r["n_truth"]) for r in rows) / len(rows)
        fits[recipe] = {"points": pts, "rows": rows, "q_conservative": q_cons, "q_equal_noise": q_eq}
    m = len(fits)
    crit = stats.chi2.ppf(0.05, m) / m
    return {
        "fits": fits,
        "independent_fits": m,
        # each q_j ~ sigma^2 chi2_1, so sum_j q_j ~ sigma^2 chi2_m: sigma_up^2 = sum q / chi2_{0.05, m}
        "conservative_upper_95": float(np.sqrt(sum(f["q_conservative"] for f in fits.values()) / m / crit)),
        "equal_noise_upper_95_not_valid": float(np.sqrt(sum(f["q_equal_noise"] for f in fits.values()) / m / crit)),
        "preregistered_formula_literal": {
            "formula": "sum_j q_j / (chi2_{0.05, m} / m): an extra factor m (a slip in the pre-registration)",
            "conservative": float(np.sqrt(sum(f["q_conservative"] for f in fits.values()) / crit)),
            "equal_noise": float(np.sqrt(sum(f["q_equal_noise"] for f in fits.values()) / crit)),
        },
        "note": "both are upper bounds on noise (misspecification counted as noise); neither can raise the low end",
    }


def main() -> None:
    sig = json.loads(BRACKET.read_text())["e_reruns"]["sigma_by_rung"]
    _, cells = lr.listing()
    subs = sorted({k[0] for k in cells})
    family = {s: s.split("/")[0] for s in subs}
    groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    for s in subs:
        groups[tuple(cells[(s, r)]["x"] for r in RUNGS)].append(s)
    genuine = [(a, b) for g in groups.values() for a, b in combinations(sorted(g), 2) if family[a] != family[b]]
    members = sorted({m for p in genuine for m in p})
    lq = sorted(next(g for g in groups.values() if "adamw_llama_scaling" in g))
    out: dict[str, Any] = {"preregistration": "PREDICTIONS_TASK_BREAK.md", "genuine_pairs": len(genuine), "members": members}
    a = audit(members)
    a["flags"] = flags(a["table"], {"all_genuine_members": members, "llama_qwen": lq})
    out["task1a"] = a
    out["task1b"] = shape(cells, members, genuine, sig)
    fo = fo_optima()
    out["task1c"] = {
        "best_config_schedule": fo["best_config_schedule"],
        "optima": fo["optima"],
        "reorder": {f"{f}->{t}": fo_reorder(fo["optima"], f, t, {**sig, "1.2b": sig["1_2b"]})
                    for f, t in (("130m", "300m"), ("300m", "520m"), ("520m", "1.2b"))},
    }  # fmt: skip
    prov = a["provenance"]
    batch_change = sum(a["flags"]["llama_qwen"]["batch"]["changes"][s] not in (1.0, "same") for s in lq)
    fo_batch = {o: (v.get("300m", {}).get("train_batch_size"), v.get("520m", {}).get("train_batch_size"))
                for o, v in fo["best_config_schedule"].items()}  # fmt: skip
    out["task1c"]["fo_batch_300m_520m"] = fo_batch
    out["task1c"]["independence"] = (
        "not independent in config schedule: speedrun submissions citing Fantastic Optimizers copy its per-size configs "
        f"({sum(p['fantastic_optimizers_derived'] for p in prov.values())} of {len(prov)} submissions cite it); "
        "independent in run noise (separate runs)"
    )
    out["task2a"] = gpu_tpu()
    out["task2b"] = delphi_bound()
    b, r = out["task1b"], out["task1c"]["reorder"]["300m->520m"]
    out["predictions"] = {
        "PA1_batch_changes_in_8_of_10_llama_qwen": batch_change >= 8,
        "PA1_tuned_per_size_in_8_of_10": sum(prov[s]["tuned_per_size"] for s in lq) >= 8,
        "PA2_some_setting_flagged": any(v["flagged"] for v in a["flags"]["all_genuine_members"].values()),
        "PA3_fo_reorder_at_least_15pct": (r["rate"] or 0) >= 0.15,
        "PA4_step_preferred_in_most": b["prefer_step"] > b["pairs_changing_300_520"] / 2,
        "PA4_at_least_3_steps_detected": b["steps_detected"] >= 3,
        "PG1_relabelled": out["task2a"]["data_seed_equal_or_both_unset"] and not out["task2a"]["seed_mentions_in_scripts"],
        "PG2_conservative_above_0_4pct": out["task2b"]["conservative_upper_95"] > 0.004,
        "PG2_equal_noise_in_0_03_to_0_2pct": 0.0003 <= out["task2b"]["equal_noise_upper_95_not_valid"] <= 0.002,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
