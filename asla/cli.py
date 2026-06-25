"""Command-line interface for the algorithm-selection audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asla.analysis.crossover import detect_crossovers, fitted_crossover_for_pair
from asla.analysis.fits import normalize_budgets, project_ranking, truth_ranking
from asla.analysis.gate import decision_report, monte_carlo
from asla.analysis.metrics import decision_metrics
from asla.config import AuditConfig
from asla.data.harvest import discover, harvest
from asla.data.io import load_runs, save_runs
from asla.data.schema import SchemaError, validate
from asla.data.synthetic import SCENARIOS, negative_controls_only, true_ranking as synthetic_true_ranking
from asla.figures import make_figures


def decision_report_against_truth(projected: pd.Series, truth: pd.Series, k: int) -> dict[str, float]:
    """Return decision metrics for an already computed projected ranking and truth ranking."""

    return decision_metrics(projected, truth, k=k)


def _demo(args: argparse.Namespace) -> int:
    cfg = AuditConfig()
    rng = np.random.default_rng(args.seed)
    df = SCENARIOS[args.scenario](rng, cfg)
    projected = project_ranking(df, cfg.budgets.fit, cfg.budgets.target)
    measured_truth = truth_ranking(df, cfg.budgets.target)
    truth = synthetic_true_ranking(df, cfg.budgets.target)
    metrics = decision_report(df, cfg.budgets.fit, cfg.budgets.target, k=2)
    metrics_vs_noiseless = decision_report_against_truth(projected, truth, k=2)
    print(f"Scenario: {args.scenario}")
    print("Projected ranking at target:")
    print(projected.to_string())
    print("Measured true ranking at target:")
    print(measured_truth.to_string())
    print("Noiseless synthetic true ranking at target:")
    print(truth.to_string())
    print("Decision metrics vs measured target rows:")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Decision metrics vs noiseless synthetic truth:")
    print(json.dumps(metrics_vs_noiseless, indent=2, sort_keys=True))
    projected_winner = str(projected.index[0])
    true_winner = str(truth.index[0])
    if projected_winner != true_winner:
        regret = float(truth.loc[projected_winner] - truth.loc[true_winner])
        print(f"Wrong winner: {projected_winner}; true winner: {true_winner}; regret={regret:.6f}")
    else:
        print(f"Winner is correct: {projected_winner}; regret=0.000000")

    crossovers = detect_crossovers(df, cfg.budgets.fit, cfg.budgets.target)
    print("Detected crossovers:")
    if crossovers:
        for a, b, gap in crossovers:
            root = fitted_crossover_for_pair(df, a, b, cfg.budgets.fit)
            root_msg = "fit is blind" if root is None else f"fitted crossover budget={root:.3f}"
            print(f"- {a} vs {b}: true_gap={gap:.6f}; {root_msg}")
    else:
        print("- none")

    neg = negative_controls_only(np.random.default_rng(args.seed + 1), cfg)
    neg_cross = detect_crossovers(neg, cfg.budgets.fit, cfg.budgets.target)
    print(f"Negative-control crossovers detected: {len(neg_cross)}")

    mc = monte_carlo(SCENARIOS[args.scenario], cfg, n_trials=args.trials, rng=np.random.default_rng(args.seed + 2))
    print("Monte Carlo gate comparison:")
    print(json.dumps(mc, indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        df = load_runs(args.runs)
        validate(df)
    except SchemaError as exc:
        print(str(exc))
        return 1
    print(f"valid runs table: {args.runs}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    df = load_runs(args.runs)
    computes = sorted(df["compute"].astype(float).unique())
    budgets = normalize_budgets(tuple(c for c in computes if c < args.target), target=args.target)
    if len(budgets) < 3:
        raise SystemExit(f"need at least 3 distinct pre-target fitting budgets; found {len(budgets)}")
    projected = project_ranking(df, budgets, args.target)
    truth = truth_ranking(df, args.target)
    metrics = decision_report(df, budgets, args.target, k=min(3, len(projected)))
    crossovers = detect_crossovers(df, budgets, args.target)
    result = {
        "projected": projected.to_dict(),
        "truth": truth.to_dict(),
        "metrics": metrics,
        "crossovers": crossovers,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _harvest(args: argparse.Namespace) -> int:
    if args.discover:
        discover(args.entity_project)
        return 0
    if not args.field_map:
        raise SystemExit("--field-map is required unless --discover is used")
    field_map = json.loads(Path(args.field_map).read_text(encoding="utf-8"))
    harvest(args.entity_project, field_map, args.out)
    return 0


def _figures(args: argparse.Namespace) -> int:
    df = load_runs(args.runs)
    make_figures(df, args.out)
    print(f"wrote figures to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="asla")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--scenario", choices=["clean_crossover", "saturation_crossover", "noise_close_call"], default="clean_crossover")
    demo.add_argument("--seed", type=int, default=1729)
    demo.add_argument("--trials", type=int, default=30)
    demo.set_defaults(func=_demo)

    val = sub.add_parser("validate")
    val.add_argument("--runs", required=True)
    val.set_defaults(func=_validate)

    audit = sub.add_parser("audit")
    audit.add_argument("--runs", required=True)
    audit.add_argument("--target", type=float, required=True)
    audit.add_argument("--out")
    audit.set_defaults(func=_audit)

    harv = sub.add_parser("harvest")
    harv.add_argument("--discover", action="store_true")
    harv.add_argument("--entity-project", required=True)
    harv.add_argument("--field-map")
    harv.add_argument("--out", default="runs.parquet")
    harv.set_defaults(func=_harvest)

    figs = sub.add_parser("figures")
    figs.add_argument("--runs", required=True)
    figs.add_argument("--out", required=True)
    figs.set_defaults(func=_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
