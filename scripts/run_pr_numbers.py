"""Numbers for the upstream speedrun PR (branch fix-power-law-convergence in data/cache/speedrun_upstream).

Computed with the project's machinery on the official 22 and cross-checked against the PR's own reproduction
script (website/scripts/check-power-law-fit.mjs), which uses an independent least-squares reference.

Writes ``results/external/pr_numbers.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_leaderboard_resolution as lr  # noqa: E402
from scripts import run_marin_aligned as al  # noqa: E402
from scripts import run_marin_reference as ref  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402
from scripts import run_sign_ties as st  # noqa: E402

OUT = REPO / "results" / "external" / "pr_numbers.json"
UPSTREAM = REPO / "data" / "cache" / "speedrun_upstream"


def repro() -> dict[str, Any] | None:
    website = UPSTREAM / "website"
    if not (website / "scripts" / "check-power-law-fit.mjs").exists():
        return None
    link = website / "node_modules"
    made = not link.exists()
    if made:
        link.symlink_to(rig.NODE_DIR / "node_modules")
    try:
        proc = subprocess.run(
            ["node", "scripts/check-power-law-fit.mjs", str(rig.RAW / "runs.json"), "--json"],
            cwd=website,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        if made:
            link.unlink()
    return json.loads(proc.stdout)


def main() -> None:
    _, cells = lr.listing()
    subs = sorted({k[0] for k in cells})
    pts = {s: [{"x": cells[(s, r)]["x"], "y": cells[(s, r)]["y"]} for r in lr.RUNGS] for s in subs}
    optimum = st.global_optimum(subs, {s: [(q["x"], q["y"]) for q in pts[s]] for s in subs})
    res = ref.lm([{"id": s, "points": pts[s], "targets": [rig.FORECAST_X]} for s in subs])
    displayed = {s: res[s]["reference_predictions"][0] for s in subs}
    fixed = al.patched_forecasts(pts)
    dep_before = np.array([abs(displayed[s] - optimum[s]) for s in subs])
    dep_after = np.array([abs(fixed[s] - optimum[s]) for s in subs])
    multistart = {
        "x_over_1e18_error_tolerance_1e-15": st.reach(
            st.lm_starts(pts, subs, st.SCALE, error_tolerance=1e-15), optimum, subs
        ),
        "x_over_1e18_y_times_100": st.reach(st.lm_starts(pts, subs, st.SCALE, y_scale=100.0), optimum, subs),
    }
    changed = [
        [a, b] for a, b in combinations(subs, 2) if np.sign(displayed[a] - displayed[b]) != np.sign(fixed[a] - fixed[b])
    ]
    out: dict[str, Any] = {
        "source": "marin-community/speedrun@31fe8028 data/runs.json (the displayed Scaling leaderboard, 22 submissions)",
        "branch": "fix-power-law-convergence in data/cache/speedrun_upstream (local only; no PR opened)",
        "optimum": "global least-squares optimum of a x^-b + c with the leaderboard's bounds (18 starts, log a, "
        "tolerances 1e-15)",
        "current_code_default_start": {
            "max_departure_bpb": float(dep_before.max()),
            "median_departure_bpb": float(np.median(dep_before)),
        },
        "fixed_max_departure_bpb": float(dep_after.max()),
        "multistart_13_starts_cap_200": {
            k: {
                kk: v[kk]
                for kk in (
                    "fits",
                    "fits_reaching_optimum",
                    "max_departure_bpb",
                    "submissions_reaching_optimum_from_every_start",
                )
            }
            for k, v in multistart.items()
        },
        "pairs_changed": len(changed),
        "pairs": len(subs) * (len(subs) - 1) // 2,
        "kendall_tau": ref.kendall(displayed, fixed),
        "top5_displayed": sorted(subs, key=lambda s: displayed[s])[:5],
        "top5_fixed": sorted(subs, key=lambda s: fixed[s])[:5],
        "changed_pairs": changed,
    }
    js = repro()
    if js is not None:
        s = js["summary"]
        out["reproduction_script_cross_check"] = {
            "summary": s,
            "agrees": bool(
                s["submissions"] == len(subs)
                and s["pairsChanged"] == len(changed)
                and s["top5Before"] == out["top5_displayed"]
                and s["top5After"] == out["top5_fixed"]
                and abs(s["maxDepartureBefore"] - out["current_code_default_start"]["max_departure_bpb"]) < 1e-6
                and abs(s["medianDepartureBefore"] - out["current_code_default_start"]["median_departure_bpb"]) < 1e-6
            ),
            "max_abs_optimum_difference_js_vs_python": max(abs(r["optimum"] - optimum[r["name"]]) for r in js["rows"]),
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
