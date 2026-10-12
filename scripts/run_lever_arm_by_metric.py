"""Task 1 of PREDICTIONS_TASK_REVIEW_RISKS.md: the calibrated lever-arm test on every full-ladder metric.

The test (``asla.analysis.lever_arm``): Spearman rho of log lever arm against
projection's per-design mean excess over single-scale, leave-one-recipe-out
Fisher-z standard error. It is recalibrated per metric exactly as for C4
(``scripts/run_lever_arm_calibration.py``): random-candidate worlds from that
metric's shrunk truth (bandwidth 0.5) give the critical value k from coverage of
the true rho; replicate-only null worlds give p. Each world also scores the
design subsets that exclude chance-level top fitted rungs, so the exclusion is
calibrated in the same worlds.

C4 bits per token reuses ``results/target_scoring/lever_arm_calibration.json``
(no chance rungs). Signal and Noise (one run per cell) is reported descriptively.

    nice -n 19 python scripts/run_lever_arm_by_metric.py --workers 2 --resume

Writes ``results/target_scoring/lever_arm_by_metric.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from asla.analysis.calibration import random_truth, shrink_truth, simulate, truth_from_tables  # noqa: E402
from asla.analysis.lever_arm import designs, loro_test  # noqa: E402

CACHE = REPO / "data" / "cache" / "lever_arm_by_metric.jsonl"
OUT = REPO / "results" / "target_scoring" / "lever_arm_by_metric.json"
METRICS = {
    "olmes": "olmes_macro_error",
    "cp": "olmes_macro_correct_prob_per_char_deficit",
    "margin": "olmes_macro_margin_per_char_deficit",
    "norm_cp": "olmes_macro_norm_correct_prob_per_char_deficit",
}
EXCLUDED_METRICS = {
    "total_prob": "1 - total_prob_per_char is not positive on some cells (total prob exceeds 1), so no power law fits"
}
N_MAIN, N_NULL = 150, 1000
BANDWIDTH = 0.5
LEVEL = 0.95
CHANCE_ACCURACY = 0.348
CHANCE_MARGIN = 0.01
SEED = 20261012
_CONTEXT: dict[str, dict[str, Any]] = {}


def load_frame(metric: str) -> pd.DataFrame:
    build = importlib.import_module("scripts.run_rescoring")._load_checkpoint_builder()
    frame = build(METRICS[metric], "530M", "1B")["final_table"]
    return frame[frame["scale_label"] != "750M"].reset_index(drop=True)


def chance_rungs(margin: float = 0.0) -> list[str]:
    """Rungs whose recipe-mean OLMES macro accuracy is at or below chance (+ margin)."""

    frame = load_frame("olmes")
    accuracy = 1 - frame.groupby("scale_label")["bpb"].mean()
    return sorted(str(s) for s, a in accuracy.items() if a <= CHANCE_ACCURACY + margin)


def context(metric: str) -> dict[str, Any]:
    if metric not in _CONTEXT:
        from asla.models import fit_power_law

        frame = load_frame(metric)
        target = float(frame[frame["scale_label"] == "1B"]["compute"].iloc[0])
        base = shrink_truth(truth_from_tables(frame, frame.copy(), target, "1B", rho_ckpt={}))
        params = {}
        for recipe, group in base.final.groupby("intervention"):
            cells = group.groupby("compute")["mu"].mean()
            params[str(recipe)] = fit_power_law(cells.index.to_numpy(dtype=float), cells.to_numpy(dtype=float))
        _CONTEXT[metric] = {
            "frame": frame,
            "base": base,
            "params": params,
            "subsets": {"strict": chance_rungs(0.0), "margin": chance_rungs(CHANCE_MARGIN)},
        }
    return _CONTEXT[metric]


def subset_index(all_designs: list[Any], excluded: list[str]) -> list[int]:
    return [i for i, d in enumerate(all_designs) if d.max_fit_label not in excluded]


def _tests(world_designs: list[Any], ctx: dict[str, Any], replicate_only: bool) -> dict[str, Any]:
    out = {}
    sets = {"all": list(range(len(world_designs)))}
    sets.update({k: subset_index(world_designs, v) for k, v in ctx["subsets"].items()})
    for name, idx in sets.items():
        chosen = [world_designs[i] for i in idx]
        if len(chosen) < 4:
            continue
        null = loro_test(chosen, replicate=True)
        entry = {"null_fisher_z": null["fisher_z"], "null_se": null["se_fisher_z"], "null_p": null["p_two_sided"]}
        if not replicate_only:
            main = loro_test(chosen)
            entry.update(
                rho=main["rho"],
                fisher_z=main["fisher_z"],
                se=main["se_fisher_z"],
                true_excess=[d.true_excess for d in chosen],
                lever_arm=[d.lever_arm for d in chosen],
            )
        out[name] = entry
    return out


def run_world(metric: str, kind: str, index: int) -> dict[str, Any]:
    ctx = context(metric)
    rng = np.random.default_rng([SEED, list(METRICS).index(metric), 0 if kind == "main" else 1, index])
    truth = random_truth(ctx["base"], rng, BANDWIDTH, ctx["params"])
    observed, _ = simulate(truth, rng)
    replicate, _ = simulate(truth, rng)
    if kind == "main":
        targets = {label: g.groupby("intervention")["mu"].first() for label, g in truth.final.groupby("scale_label")}
        world_designs = designs(observed, targets, replicate)
    else:
        world_designs = designs(observed, replicate=replicate, with_projection=False)
    return {"metric": metric, "kind": kind, "index": index, "sets": _tests(world_designs, ctx, kind == "null")}


def summarise_set(rows: list[dict[str, Any]], nulls: list[dict[str, Any]], real: dict[str, Any]) -> dict[str, Any]:
    log_lever = np.log(rows[0]["lever_arm"])
    mean_true = np.mean([r["true_excess"] for r in rows], axis=0)
    rho_true = float(stats.spearmanr(log_lever, mean_true).statistic)
    z_true = float(np.arctanh(np.clip(rho_true, -0.999999, 0.999999)))
    t_main = np.array([abs(r["fisher_z"] - z_true) / r["se"] if r["se"] > 0 else np.inf for r in rows])
    t_null = np.array([abs(r["null_fisher_z"]) / r["null_se"] for r in nulls if r["null_se"] > 0])
    k = float(np.quantile(t_main, LEVEL))
    abs_t = abs(real["fisher_z"]) / real["se_fisher_z"]
    ci = [
        float(np.tanh(real["fisher_z"] - k * real["se_fisher_z"])),
        float(np.tanh(real["fisher_z"] + k * real["se_fisher_z"])),
    ]
    p_null = max(float(np.mean(t_null >= abs_t)), 1 / len(t_null))
    verdict = "holds" if ci[0] > 0 else ("reverses" if ci[1] < 0 else "inconclusive")
    return {
        "n_designs": len(log_lever),
        "n_worlds": len(rows),
        "n_null_statistics": len(t_null),
        "rho_true_in_worlds": rho_true,
        "calibrated_critical_value": k,
        "coverage_at_1_96": float(np.mean(t_main <= 1.959963984540054)),
        "real_rho": real["rho"],
        "real_ci_calibrated": ci,
        "real_p_against_null": p_null,
        "verdict": verdict,
    }


def sn_descriptive() -> dict[str, Any]:
    """S&N: one run per cell, so the observed 1B order is the target; LORO jackknife on Fisher z, no calibration."""

    from asla.models import FitError, bpb_power_law, fit_power_law

    frame = pd.read_parquet(REPO / "data" / "signal_and_noise_datadecide_c4_bpb.parquet")
    frame = frame[frame["scale_label"] != "750M"]
    table = frame.groupby(["intervention", "scale_label"])["bpb"].mean().unstack()
    compute = frame.groupby("scale_label")["compute"].first().sort_values()
    order = list(compute.index)
    names = sorted(table.index)
    rows = []
    for t in range(3, len(order)):
        for f in range(2, t):
            x = compute.iloc[: f + 1].to_numpy(float)
            proj = {}
            for n in names:
                try:
                    proj[n] = float(
                        bpb_power_law(
                            float(compute.iloc[t]), *fit_power_law(x, table.loc[n, order[: f + 1]].to_numpy(float))
                        )
                    )
                except (FitError, RuntimeError, ValueError):
                    proj[n] = float("nan")
            rows.append(
                (float(compute.iloc[t] / compute.iloc[f]), proj, table[order[f]].to_dict(), table[order[t]].to_dict())
            )

    def excess(drop: str | None) -> np.ndarray:
        out = []
        for _, proj, single, target in rows:
            pairs = [(a, b) for a, b in combinations(names, 2) if drop not in (a, b)]
            wrong_p = np.mean([np.sign(proj[a] - proj[b]) != np.sign(target[a] - target[b]) for a, b in pairs])
            wrong_s = np.mean([np.sign(single[a] - single[b]) != np.sign(target[a] - target[b]) for a, b in pairs])
            out.append(100 * (wrong_p - wrong_s))
        return np.array(out)

    log_lever = np.log([r[0] for r in rows])
    rho = float(stats.spearmanr(log_lever, excess(None)).statistic)
    loo = np.array([stats.spearmanr(log_lever, excess(n)).statistic for n in names])
    z = np.arctanh(np.clip(loo, -0.999999, 0.999999))
    se = float(np.sqrt((len(names) - 1) / len(names) * np.sum((z - z.mean()) ** 2)))
    fz = float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
    return {
        "n_designs": len(rows),
        "rho": rho,
        "loro_ci_1_96": [float(np.tanh(fz - 1.96 * se)), float(np.tanh(fz + 1.96 * se))],
        "note": "one run per cell: the observed 1B order is the target, no noise model, no calibration; descriptive",
    }


def _load() -> dict[tuple[str, str, int], dict[str, Any]]:
    if not CACHE.exists():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text().splitlines() if line.strip())
    return {(r["metric"], r["kind"], r["index"]): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load()
    jobs = [(m, "main", i) for m in METRICS for i in range(N_MAIN)] + [
        (m, "null", i) for m in METRICS for i in range(N_NULL)
    ]
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
        for count, row in enumerate(pool.map(run_world, *zip(*todo), chunksize=10) if todo else [], start=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if count % 200 == 0:
                print(f"  {count}/{len(todo)}", flush=True)
    cache = _load()
    out: dict[str, Any] = {"metrics": {}}
    for metric in METRICS:
        ctx = context(metric)
        real_designs = designs(ctx["frame"])
        real_sets = {"all": real_designs}
        real_sets.update({k: [real_designs[i] for i in subset_index(real_designs, v)] for k, v in ctx["subsets"].items()})
        block: dict[str, Any] = {"chance_rungs": ctx["subsets"]}
        for name, chosen in real_sets.items():
            mains = [
                cache[(metric, "main", i)]["sets"][name] for i in range(N_MAIN) if name in cache[(metric, "main", i)]["sets"]
            ]
            nulls = [
                cache[(metric, "null", i)]["sets"][name] for i in range(N_NULL) if name in cache[(metric, "null", i)]["sets"]
            ]
            nulls += [{k: m[k] for k in ("null_fisher_z", "null_se", "null_p")} for m in mains]
            if len(chosen) < 4 or not mains:
                block[name] = {"n_designs": len(chosen), "verdict": "too few designs"}
                continue
            block[name] = summarise_set(mains, nulls, loro_test(chosen))
        out["metrics"][metric] = block
    c4 = json.loads((REPO / "results" / "target_scoring" / "lever_arm_calibration.json").read_text())
    out["metrics"]["c4"] = {
        "chance_rungs": "none (bits per token has no chance level)",
        "all": {
            "source": "results/target_scoring/lever_arm_calibration.json (300 main worlds, 8,000 null worlds)",
            "real_rho": c4["real_data"]["rho"],
            "real_ci_calibrated": c4["real_data"]["ci_calibrated"],
            "calibrated_critical_value": c4["summary"]["calibrated_critical_value"],
            "verdict": "holds" if c4["real_data"]["ci_calibrated"][0] > 0 else "inconclusive",
        },
    }
    out["signal_and_noise"] = sn_descriptive()
    out["excluded_metrics"] = EXCLUDED_METRICS
    out["existing_claim_tested_on"] = "C4 bits per token (lever_arm_calibration.json design field)"
    m = out["metrics"]
    out["predictions"] = {
        "P1a_holds_on_c4": m["c4"]["all"]["verdict"] == "holds",
        "P1b_cp_fails_or_reverses_with_chance_rungs": m["cp"]["all"].get("verdict") in ("inconclusive", "reverses"),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for metric, block in m.items():
        for name in ("all", "strict", "margin"):
            b = block.get(name)
            if isinstance(b, dict) and "real_rho" in b:
                ci = [round(x, 2) for x in b["real_ci_calibrated"]]
                print(f"{metric:10s} {name:6s} rho={b['real_rho']:+.2f} CI={ci} {b['verdict']}")
    print("S&N", out["signal_and_noise"])
    print(out["predictions"])


if __name__ == "__main__":
    main()
