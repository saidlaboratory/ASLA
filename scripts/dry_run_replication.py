"""Dry run of the replication protocol on simulator outputs (PREDICTIONS_REPLICATION_PROTOCOL.md). Trains nothing.

For each of WORLDS simulated worlds of the frozen design (corrected recipe model, 10 recipes, the eight sizes,
3 seeds, multiplier 1; the spec of results/replication/untuned_spec.json), this script:

1. draws the world's cells exactly as asla.analysis.sim_design.world does (same RNG order);
2. writes them as a runs table in the protocol's schema, under the ten selected recipe ids;
3. runs scripts/analyze_replication.py on that table (pins checked, nominal 6ND compute from ladder.json);
4. checks the entry point's code path against the simulator's own estimate on the same draw, by running the
   entry's analyse() at the simulator's compute grid.

Per-world results are checkpointed to a JSONL file, so an interrupted run resumes.
Run: nice -n 19 .venv/bin/python scripts/dry_run_replication.py
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from asla.analysis import sim_design  # noqa: E402

WORLDS = 150
SEED = 20261031
OUT = REPO / "results" / "replication" / "protocol_dry_run.json"
CHECKPOINT = REPO / "results" / "replication" / "protocol_dry_run_worlds.jsonl"
Z = 1.959963984540054


def exact_ladder() -> list[float]:
    """log10 of 6 N D at the protocol's sizes, exactly as the entry point computes compute (protocol v2)."""

    ladder = json.loads((REPO / "replication_protocol" / "ladder.json").read_text(encoding="utf-8"))
    return [float(np.log10(6.0 * s["params_n"] * s["tokens_d"])) for s in ladder["sizes"]]


def spec() -> dict:
    cr = importlib.import_module("scripts.run_curve_round")
    lad = exact_ladder()
    floor = float(np.log10(6.0 * 151898880 * 14745600000))  # 150M, the target floor
    targets = [t for t in range(5, len(lad)) if lad[t] >= floor - 1e-12]
    return cr.base(n=10, ladder=lad, targets=targets, admit="determinacy")


def draw(sp: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The world's seed-level cells, in sim_design.world's RNG order."""

    x = np.asarray(sp["ladder"], dtype=float)
    cells = {}
    for name, src, de, dl in sim_design._recipes(sp, rng):
        mu = sim_design.truth_curve(sp, src, de, dl, x)
        sd = sim_design.noise_sd(sp, src, x, mu)
        cells[name] = mu[None, :] + sd * rng.standard_normal((int(sp["seeds"]), x.size))
    return cells


def runs_table(cells: dict[str, np.ndarray]) -> pd.DataFrame:
    ladder = json.loads((REPO / "replication_protocol" / "ladder.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in json.loads((REPO / "replication_protocol" / "recipes.json").read_text(encoding="utf-8"))["selected"]]
    rows = []
    if len(ids) != len(cells):
        raise ValueError("recipe count differs from the protocol's selection")
    for rid, (name, c) in zip(ids, sorted(cells.items())):
        for j, s in enumerate(ladder["sizes"]):
            for k, seed in enumerate(ladder["seeds"]):
                rows.append(
                    {
                        "recipe": rid,
                        "simulated_source": name,
                        "size": s["size"],
                        "seed": seed,
                        "params_n": s["params_n"],
                        "tokens_d": s["tokens_d"],
                        "c4_en_bits_per_token": float(c[k, j]),
                    }
                )
    return pd.DataFrame(rows)


def one_world(sp: dict, w: int, entry) -> dict:
    rng_a = np.random.default_rng([SEED, w])
    rng_b = np.random.default_rng([SEED, w])
    sim = sim_design.world(sp, rng_a)
    cells = draw(sp, rng_b)
    table = runs_table(cells)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "runs.csv"
        table.to_csv(path, index=False)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            res = entry.main([str(path)])
    grid = table.copy()
    grid_frame = pd.DataFrame(
        {
            "intervention": grid["recipe"],
            "scale_label": grid["size"],
            "seed": grid["seed"],
            "compute": grid["size"].map(dict(zip([s["size"] for s in json.loads((REPO / "replication_protocol" / "ladder.json").read_text())["sizes"]], 10.0 ** np.asarray(sp["ladder"])))),
            "bpb": grid["c4_en_bits_per_token"],
        }
    )
    same = entry.analyse(grid_frame)
    return {
        "world": w,
        "entry_estimate": res["estimate"],
        "entry_se": res["se"],
        "entry_verdict": res["verdict"],
        "entry_targets": res["targets_admitted"],
        "entry_within_range": res["secondary"]["within_recipe_draw_range"],
        "sim_estimate": sim.get("estimate"),
        "sim_se": sim.get("se"),
        "sim_true": sim.get("true"),
        "codepath_estimate_at_sim_grid": same["estimate"],
        "codepath_se_at_sim_grid": same["se"],
    }


def main() -> dict:
    entry = importlib.import_module("scripts.analyze_replication")
    sp = spec()
    done = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["world"]] = r
    with CHECKPOINT.open("a", encoding="utf-8") as fh:
        for w in range(WORLDS):
            if w in done:
                continue
            r = one_world(sp, w, entry)
            done[w] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(w, r["entry_verdict"], round(r["entry_estimate"] or float("nan"), 2), flush=True)
    rows = [done[w] for w in range(WORLDS)]

    def close(a, b):
        return a is None and b is None or (a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b)))

    match = [close(r["codepath_estimate_at_sim_grid"], r["sim_estimate"]) and close(r["codepath_se_at_sim_grid"], r["sim_se"]) for r in rows]
    est = np.array([r["entry_estimate"] for r in rows if r["entry_estimate"] is not None], dtype=float)
    diff = np.array([r["entry_estimate"] - r["sim_estimate"] for r in rows if r["entry_estimate"] is not None and r["sim_estimate"] is not None])
    verdicts = [r["entry_verdict"].split(":")[0] for r in rows]
    out = {
        "preregistration": "PREDICTIONS_REPLICATION_PROTOCOL.md",
        "note": "simulated outputs only; nothing trained",
        "worlds": WORLDS,
        "spec": sp,
        "codepath_matches_simulator_all_worlds": bool(all(match)),
        "nominal_compute_vs_simulator_grid_max_abs_estimate_diff": float(np.max(np.abs(diff))) if diff.size else None,
        "worlds_entry_differs_from_simulator_beyond_1e-9": [
            {"world": r["world"], "difference": float(r["entry_estimate"] - r["sim_estimate"])}
            for r in rows
            if r["entry_estimate"] is not None and r["sim_estimate"] is not None and abs(r["entry_estimate"] - r["sim_estimate"]) > 1e-9
        ],
        "fp_note": "the entry point computes compute as 6ND and the simulator as 10**log10(6ND); they differ by at most 3.6e-15 relative, and in a rare world the frozen power-law fit lands differently under that perturbation",
        "verdict_disagreements_nominal_vs_simulator_grid": [
            r["world"] for r in rows
            if r["sim_estimate"] is not None and r["entry_estimate"] is not None
            and (r["sim_estimate"] - Z * r["sim_se"] > 0) != (r["entry_estimate"] - Z * r["entry_se"] > 0)
        ],
        "grid_note": "protocol v2: the simulator's ladder is exact log10(6ND) from ladder.json, the same compute the entry point uses (v1 used log10 compute rounded to 3 decimals)",
        "true_slope_spread_2_5_97_5": [float(np.quantile([r["sim_true"] for r in rows if r["sim_true"] is not None], q)) for q in (0.025, 0.975)],
        "true_slope_mc_interval": [
            float(np.mean(t := [r["sim_true"] for r in rows if r["sim_true"] is not None]) - Z * np.std(t, ddof=1) / np.sqrt(len(t))),
            float(np.mean(t) + Z * np.std(t, ddof=1) / np.sqrt(len(t))),
        ],
        "verdict_shares": {v: verdicts.count(v) / WORLDS for v in sorted(set(verdicts))},
        "replicates_share_mc_se": float(np.sqrt((verdicts.count("replicates") / WORLDS) * (1 - verdicts.count("replicates") / WORLDS) / WORLDS)),
        "mean_estimate": float(est.mean()) if est.size else None,
        "mean_true": float(np.mean([r["sim_true"] for r in rows if r["sim_true"] is not None])),
        "share_within_recipe_draw_range": float(np.mean([bool(r["entry_within_range"]) for r in rows if r["entry_within_range"] is not None])),
        "untuned_spec_random_candidate_power": 0.9066666666666666,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "spec"}, indent=2))
    return out


if __name__ == "__main__":
    main()
