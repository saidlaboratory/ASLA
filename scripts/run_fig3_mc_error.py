"""Monte Carlo error of every quantity Figure 3 (fig_flexibility) plots.

Figure 3 plots, for each ranker on the 300M-to-1B design, the observed
mis-selection rate (a point computed once on the observed data, so it has no Monte
Carlo error) and the 2.5% and 97.5% percentiles of a seed bootstrap with
``n_boot`` draws, plus single-scale ranking's interval as a band.

The draws themselves were not stored. The Monte Carlo standard error of a sample
p-quantile from B draws is sqrt(p (1 - p) / B) / f(q_p). With f taken as the normal
density whose 95% interval matches the reported one, f(q_p) = phi(z_p) / sd and
sd = (hi - lo) / (2 z_0.975), so the standard error is sqrt(p (1 - p) / B) sd /
phi(z_p). This is an approximation: the bootstrap distribution is discrete in
steps of one pair in 300 and need not be normal; the step size is reported too.

The text of the paper quotes only the points. The question this answers is
whether the interval ends' Monte Carlo error could change any number or
statement there.

Writes ``results/task_b/fig3_mc_error.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results" / "task_b" / "task_b.json"
OUT = REPO / "results" / "task_b" / "fig3_mc_error.json"
DESIGN = "primary_4M-300M_target1B"
PLOTTED = (
    "single_scale",
    "shared_exponent",
    "shrinkage_1",
    "shrinkage_0.8",
    "shrinkage_0.6",
    "shrinkage_0.4",
    "shrinkage_0.2",
    "plain_projection",
    "ensemble",
)


def main() -> None:
    task = json.loads(SOURCE.read_text(encoding="utf-8"))
    design = task["designs"][DESIGN]
    rankers = design["rankers"]
    n_pairs = int(design["n_pairs"])
    out = {}
    for name in PLOTTED:
        ms = rankers[name]["mis_selection"]
        level = float(ms["confidence_level"])
        b = int(ms["n_boot"])
        p = (1 - level) / 2
        z = float(stats.norm.ppf(1 - p))
        sd = (ms["hi"] - ms["lo"]) / (2 * z)
        se_end = float(np.sqrt(p * (1 - p) / b) * sd / stats.norm.pdf(z))
        out[name] = {
            "point": ms["point"],
            "point_mc_se": 0.0,
            "lo": ms["lo"],
            "hi": ms["hi"],
            "n_boot": b,
            "interval_end_mc_se": se_end,
            "interval_end_mc_se_points": 100 * se_end,
        }
    worst = max(v["interval_end_mc_se_points"] for v in out.values())
    record = {
        "design": DESIGN,
        "method": "normal-approximation standard error of a sample quantile; points are computed once and exact",
        "step_points": 100 / n_pairs,
        "rankers": out,
        "max_interval_end_mc_se_points": worst,
        "text_quotes_interval_ends": False,
        "rerun": False,
        "reason": (
            "The text quotes only the points, which have no Monte Carlo error; the interval ends appear only in the "
            "figure and no sentence depends on them, so their Monte Carlo error cannot change a number or statement."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"max interval-end MC SE {worst:.2f} points; step {100 / n_pairs:.2f} points")


if __name__ == "__main__":
    main()
