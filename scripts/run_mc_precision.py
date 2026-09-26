"""Task 3 of PREDICTIONS_TASK_LEVER_GAP.md: Monte Carlo precision of fixed-candidate p-values.

A reported fixed-candidate p is calibrated_p(|estimate| / SE, null), and it has
two Monte Carlo error sources:

* (a) SE comes from a parametric bootstrap of the real data. Here it is redone
  with 4000 draws per metric, in 8 independent batches, so the batch means give
  its Monte Carlo error. That error is propagated to p numerically.
* (b) The null distribution comes from a finite number of simulated statistics,
  which gives a binomial error on the tail fraction.

For each p: both errors, the draws each source would need to bring its error
below the unit of the digit currently reported, and the precision the current
draws actually support.

Writes ``results/target_scoring/mc_precision.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = REPO / "data" / "cache" / "mc_precision.jsonl"
OUT = REPO / "results" / "target_scoring" / "mc_precision.json"
FILES = {
    "c4_en_bits_per_token": "calibration.json",
    "olmes_macro_error": "calibration_olmes_macro_error.json",
}
N_BATCHES, BATCH = 8, 500
SEED = 20260926


def batch(metric: str, index: int) -> dict[str, Any]:
    """One batch of parametric bootstrap draws on the real data (worker process)."""

    os.environ["ASLA_CAL_METRIC"] = metric
    calibration = importlib.import_module("scripts.run_calibration")
    calibration.METRIC = metric
    calibration._CONTEXT.clear()
    ctx = calibration._context()
    plug = calibration.truth_from_tables(
        ctx["final"], ctx["ckpt"], ctx["target"], calibration.TARGET_LABEL, rho_ckpt=ctx["base"].rho_ckpt
    )
    plug_gaps = calibration.gaps(plug.target_truth())
    rng = np.random.default_rng([SEED, list(FILES).index(metric), index])
    draws: dict[str, list[float]] = {}
    for _ in range(BATCH):
        final, ckpt = calibration.simulate(plug, rng)
        preds = calibration.predict(
            final, ckpt, ctx["rankers"], calibration.CHECKPOINT_RANKERS, ctx["budgets"], ctx["target"]
        )
        evidence = calibration.evidence_at_target(final, calibration.TARGET_LABEL)
        for name, comp in calibration.compare(preds, evidence, plug_gaps, calibration.BASELINE, with_interval=False).items():
            draws.setdefault(name, []).append(comp.estimate)
    return {"metric": metric, "index": index, "draws": draws}


def _load() -> dict[tuple[str, int], dict[str, Any]]:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["metric"], row["index"])] = row
    return done


def _round_up(value: float) -> str:
    """One significant figure, rounded up, so a floor is never reported tighter than it is."""

    exponent = math.floor(math.log10(value))
    mantissa = math.ceil(round(value / 10**exponent, 9))
    return f"{mantissa * 10**exponent:.{max(0, -exponent)}f}"


def decimals_supported(mc_se: float) -> int:
    """Decimal places whose last digit exceeds the Monte Carlo SE."""

    return max(0, math.floor(-math.log10(mc_se))) if mc_se > 0 else 6


def summarise(metric: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from asla.provenance import Source

    calibration = importlib.import_module("scripts.run_calibration")
    source = Source.load(f"target_scoring/{FILES[metric]}")
    null = source.get("calibrated.fixed_bootstrap")
    reported = source.get("real_data.comparisons")
    n_stat = int(null["n_statistics"])
    out = {}
    for name in reported:
        if name.startswith("single_scale_"):
            continue
        estimate = float(reported[name]["estimate_pp"])
        batches = [np.asarray(r["draws"][name]) for r in rows]
        pooled = np.concatenate(batches)
        se = float(pooled.std(ddof=1))
        batch_se = np.array([b.std(ddof=1) for b in batches])
        se_mc = float(batch_se.std(ddof=1) / np.sqrt(len(batches)))

        def p_at(s: float) -> float:
            return calibration.calibrated_p(abs(estimate) / s, null["null_quantiles_of_abs_t"], n_stat) if s > 0 else 0.0

        p = p_at(se)
        a_err = abs(p_at(se + se_mc) - p_at(max(se - se_mc, 1e-12))) / 2
        at_floor = p <= 1.0 / n_stat + 1e-15
        b_err = math.sqrt(p * (1 - p) / n_stat) if not at_floor else 1.0 / n_stat
        total = math.hypot(a_err, b_err)
        previous = float(reported[name]["calibrated_fixed_candidate"]["p_two_sided"] or 0)
        unit = 0.001  # the previously reported fixed-candidate p had three decimals
        draws_a = int(len(pooled) * (a_err / unit) ** 2) + 1 if a_err > 0 else len(pooled)
        stats_b = int(p * (1 - p) / unit**2) + 1 if not at_floor else None
        places = decimals_supported(total)
        out[name] = {
            "estimate_pp": estimate,
            "se_pp": se,
            "bootstrap_draws": int(len(pooled)),
            "p": p,
            "previous_reported_p": previous,
            "mc_se_from_bootstrap": a_err,
            "mc_se_from_null": b_err,
            "mc_se_total": total,
            "p_at_floor": at_floor,
            "null_statistics": n_stat,
            "draws_needed_for_third_decimal": draws_a,
            "null_statistics_needed_for_third_decimal": stats_b,
            "decimals_supported": places,
            "stable_report": (
                f"< {_round_up(1.0 / n_stat)}"
                if at_floor
                else (f"{p:.{places}f}" if places >= 1 and round(p, places) > 0 else f"{p:.1g}")
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    done = _load()
    for metric in FILES:
        todo = [i for i in range(N_BATCHES) if (metric, i) not in done]
        if not todo:
            continue
        with ProcessPoolExecutor(max_workers=args.workers) as pool, CACHE.open("a", encoding="utf-8") as sink:
            for row in pool.map(batch, [metric] * len(todo), todo):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(f"  {metric} batch {row['index']} done", flush=True)
    cache = _load()
    out = {
        "method": "4000 parametric bootstrap draws per metric in 8 batches; MC error of SE by batch means, "
        "propagated to p numerically; null-distribution error binomial on the tail fraction",
        "metrics": {m: summarise(m, [cache[(m, i)] for i in range(N_BATCHES)]) for m in FILES},
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for metric, rows in out["metrics"].items():
        for name, r in rows.items():
            print(
                f"{metric[:6]} {name:22s} p={r['p']:.4f} (was {r['previous_reported_p']:.4f}) "
                f"mcA={r['mc_se_from_bootstrap']:.4f} mcB={r['mc_se_from_null']:.4f} -> {r['stable_report']}"
            )


if __name__ == "__main__":
    main()
