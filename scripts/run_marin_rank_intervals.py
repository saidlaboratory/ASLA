"""Each official submission's rank interval at 1e22, for the paper's Marin figure (a readout, not a new analysis).

Regenerates the frozen noise-propagation draws of ``inclusion_recount.json`` (official 22, per-rung Task-1 bound,
corrected solver, the same seed and draw count) and fails unless they reproduce its frozen count of pairs holding
in >= 95% of draws. Adds the rank under each of the two converged forms (three-parameter optimum, log-log).

Writes ``results/external/marin_rank_intervals.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_leaderboard_resolution as lr  # noqa: E402
from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402

OUT = REPO / "results" / "external" / "marin_rank_intervals.json"
RESULTS = REPO / "results" / "external"
LEVEL = 0.95  # central interval of ranks; also the pairs-holding level of run_leaderboard_resolution


def main() -> None:
    sig = json.loads((RESULTS / "noise_bracket.json").read_text())["e_reruns"]["sigma_by_rung"]
    frozen = json.loads((RESULTS / "inclusion_recount.json").read_text())["task1b"]["noise_resolution_bound"]["official_22"]
    _, cells = lr.listing()
    members = sorted({k[0] for k in cells})
    pts = {s: [{"x": cells[(s, r)]["x"], "y": cells[(s, r)]["y"]} for r in lr.RUNGS] for s in members}
    fixed = al.patched_forecasts(pts)
    sigma = np.array([sig[r] for r in lr.RUNGS])
    # the draws of run_leaderboard_resolution.bootstrap, reproduced exactly
    rng = np.random.default_rng(lr.SEED)
    noise = 1 + sigma[None, None, :] * rng.standard_normal((lr.DRAWS, len(members), len(lr.RUNGS)))
    jobs = {
        f"{s}|{i}": [{"x": q["x"], "y": q["y"] * noise[i, j, r]} for r, q in enumerate(pts[s])]
        for i in range(lr.DRAWS)
        for j, s in enumerate(members)
    }
    res = al.patched_predictions(jobs)
    f = np.array([[np.nan if res[f"{s}|{i}"][0] is None else res[f"{s}|{i}"][0] for s in members] for i in range(lr.DRAWS)])
    f = f[np.all(np.isfinite(f), axis=1)]
    holding = sum(
        np.mean(np.sign(f[:, members.index(a)] - f[:, members.index(b)]) == np.sign(fixed[a] - fixed[b])) >= LEVEL
        for a, b in combinations(members, 2)
    )
    if holding != frozen["holding_95"]:
        raise SystemExit(f"draws do not reproduce the frozen result: {holding} pairs hold, frozen {frozen['holding_95']}")
    ranks = np.argsort(np.argsort(f, axis=1), axis=1) + 1  # 1 = best (lowest BPB)
    raw = {s: [(q["x"], q["y"]) for q in pts[s]] for s in members}
    optimum = st.global_optimum(members, raw)
    loglog = {s: rig.loglog(raw[s], rig.FORECAST_X) for s in members}

    def rank_of(values: dict[str, float]) -> dict[str, int]:
        order = sorted(members, key=lambda s: values[s])
        return {s: order.index(s) + 1 for s in members}

    r_fixed, r_opt, r_log = rank_of(fixed), rank_of(optimum), rank_of(loglog)
    rows = []
    for j, s in enumerate(sorted(members, key=lambda m: fixed[m])):
        k = members.index(s)
        rows.append(
            {
                "submission": s,
                "rank_fixed": r_fixed[s],
                "rank_three_parameter_optimum": r_opt[s],
                "rank_loglog": r_log[s],
                "noise_rank_2_5": float(np.percentile(ranks[:, k], 100 * (1 - LEVEL) / 2)),
                "noise_rank_97_5": float(np.percentile(ranks[:, k], 100 * (1 + LEVEL) / 2)),
                "noise_rank_median": float(np.median(ranks[:, k])),
                "top1_share": float(np.mean(ranks[:, k] == 1)),
            }
        )
    out = {
        "source": "inclusion_recount.json noise_resolution_bound (official 22), regenerated with the same seed",
        "reproduces_frozen_pairs_holding": int(holding),
        "draws_used": int(len(f)),
        "forecast_x": rig.FORECAST_X,
        "noise_rank_interval": "2.5th to 97.5th percentile of the rank over draws",
        "interval_level": LEVEL,
        "hold_level": LEVEL,
        "sigma_by_rung": sig,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
