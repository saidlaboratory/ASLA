"""Sweep gate tau and race beta to trace the compute-regret Pareto frontier.

Writes a tidy CSV of (rule, parameter, mean_regret, wrong_pick_rate,
mean_compute) rows and, when matplotlib is installed, a
``pareto_regret_cost`` figure.

Example:
    python scripts/run_pareto_study.py --scenario noise_close_call --fast \
        --out results/pareto
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from asla.analysis.racing import monte_carlo_selection
from asla.config import AuditConfig
from asla.data.synthetic import SCENARIOS

DEFAULT_TAUS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def run_study(
    scenario: str,
    n_trials: int,
    n_boot: int,
    seed: int,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    betas: tuple[float, ...] = DEFAULT_BETAS,
) -> list[dict[str, object]]:
    cfg = AuditConfig()
    cfg = replace(cfg, gate=replace(cfg.gate, n_boot=n_boot))
    scenario_fn = SCENARIOS[scenario]
    rows: list[dict[str, object]] = []

    base = monte_carlo_selection(scenario_fn, cfg, n_trials=n_trials, rng=np.random.default_rng(seed))
    for rule in ("plain", "largest"):
        rows.append({"rule": rule, "parameter": "", **base[rule]})
    rows.append({"rule": "gate", "parameter": f"tau={cfg.gate.tau}", **base["gate"]})
    rows.append({"rule": "race", "parameter": "beta=1.0", **base["race"]})

    for tau in taus:
        if np.isclose(tau, cfg.gate.tau):
            continue
        result = monte_carlo_selection(scenario_fn, cfg, n_trials=n_trials, rng=np.random.default_rng(seed), tau=tau)
        rows.append({"rule": "gate", "parameter": f"tau={tau}", **result["gate"]})
    for beta in betas:
        if np.isclose(beta, 1.0):
            continue
        result = monte_carlo_selection(scenario_fn, cfg, n_trials=n_trials, rng=np.random.default_rng(seed), beta=beta)
        rows.append({"rule": "race", "parameter": f"beta={beta}", **result["race"]})
    return rows


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pareto_regret_cost.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rule", "parameter", "mean_regret", "wrong_pick_rate", "mean_compute"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    fig, ax = plt.subplots()
    markers = {"plain": "s", "largest": "D", "gate": "o", "race": "^"}
    for rule in ("plain", "largest", "gate", "race"):
        rule_rows = [r for r in rows if r["rule"] == rule]
        xs = [float(r["mean_compute"]) for r in rule_rows]
        ys = [float(r["mean_regret"]) for r in rule_rows]
        ax.scatter(xs, ys, marker=markers[rule], label=rule)
        for r, x, y in zip(rule_rows, xs, ys):
            if r["parameter"]:
                ax.annotate(str(r["parameter"]), (x, y), fontsize=7, alpha=0.8)
    ax.set_xlabel("mean compute spent (budget units x runs)")
    ax.set_ylabel("mean regret at target")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_regret_cost.{ext}", bbox_inches="tight", dpi=160)
    print(f"wrote {out_dir / 'pareto_regret_cost.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="noise_close_call")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fast", action="store_true", help="Small counts for a quick smoke run.")
    parser.add_argument("--out", default="results/pareto")
    args = parser.parse_args()
    n_trials = 20 if args.fast else args.trials
    n_boot = 60 if args.fast else args.n_boot
    rows = run_study(args.scenario, n_trials, n_boot, args.seed)
    write_outputs(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
