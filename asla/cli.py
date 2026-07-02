"""Command-line interface for the algorithm-selection audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from asla.analysis.audit import audit_with_ci
from asla.analysis.crossover import detect_crossovers, fitted_crossover_for_pair
from asla.analysis.fits import normalize_budgets, project_ranking, truth_ranking
from asla.analysis.gate import monte_carlo
from asla.analysis.metrics import decision_metrics
from asla.analysis.rankers import Ranker, make_projection_ranker, single_scale_ranker
from asla.config import AuditConfig
from asla.data.harvest import discover, harvest
from asla.data.io import load_runs
from asla.data.schema import SchemaError, validate
from asla.data.synthetic import SCENARIOS, negative_controls_only, true_ranking as synthetic_true_ranking
from asla.figures import make_figures


def decision_report_against_truth(projected: pd.Series, truth: pd.Series, k: int) -> dict[str, float]:
    """Return decision metrics for an already computed projected ranking and truth ranking."""

    return decision_metrics(projected, truth, k=k)


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _runtime_config(args: argparse.Namespace) -> tuple[AuditConfig, int, int]:
    """Return config plus bootstrap/trial counts for CLI execution."""

    cfg = AuditConfig()
    n_boot = cfg.counts.fast_n_boot if getattr(args, "fast", False) else cfg.counts.n_boot
    n_trials = cfg.counts.fast_n_trials if getattr(args, "fast", False) else cfg.counts.n_trials
    if getattr(args, "n_boot", None) is not None:
        n_boot = int(args.n_boot)
    if getattr(args, "trials", None) is not None:
        n_trials = int(args.trials)
    cfg = replace(cfg, gate=replace(cfg.gate, n_boot=n_boot))
    return cfg, n_boot, n_trials


def _default_rankers(fit_form: str) -> dict[str, Ranker]:
    return {
        "projection_ranker": make_projection_ranker(fit_form),  # type: ignore[arg-type]
        "single_scale_ranker": single_scale_ranker,
    }


def _assert_fit_form_available(df: pd.DataFrame, fit_form: str) -> None:
    """Raise a clean CLI error when a fit form's required columns are absent."""

    if fit_form == "chinchilla":
        missing = [col for col in ("params_n", "tokens_d") if col not in df.columns]
        if missing:
            raise SystemExit(f"fit-form chinchilla requires columns: {missing}")


def _print_interval_report(results: dict[str, object]) -> None:
    rankers = results["rankers"]
    assert isinstance(rankers, dict)
    print("Ranker metrics with 95% bootstrap intervals:")
    for ranker_name, metrics in rankers.items():
        print(f"{ranker_name}:")
        assert isinstance(metrics, dict)
        for metric_name in ("mis_selection_rate", "mean_regret", "top1_acc", "pairwise_acc"):
            if metric_name not in metrics:
                continue
            interval = metrics[metric_name]
            assert isinstance(interval, dict)
            print(
                f"  {metric_name}: {interval['point']:.6f} "
                f"[{interval['lo']:.6f}, {interval['hi']:.6f}]"
            )


def _crossover_root_message(df: pd.DataFrame, a: str, b: str, budgets: tuple[float, ...], fit_form: str) -> str:
    """Return a human-readable crossover-budget message for the selected fit form."""

    if fit_form != "compute_power_law":
        return "naive compute-power-law crossover budget not reported for this fit form"
    root = fitted_crossover_for_pair(df, a, b, budgets)
    return "fit is blind" if root is None else f"fitted crossover budget={root:.3f}"


def _demo(args: argparse.Namespace) -> int:
    cfg, n_boot, n_trials = _runtime_config(args)
    rng = np.random.default_rng(args.seed)
    df = SCENARIOS[args.scenario](rng, cfg)
    _assert_fit_form_available(df, args.fit_form)
    projected = project_ranking(df, cfg.budgets.fit, cfg.budgets.target, fit_form=args.fit_form)
    measured_truth = truth_ranking(df, cfg.budgets.target)
    truth = synthetic_true_ranking(df, cfg.budgets.target)
    metrics = decision_report_against_truth(projected, measured_truth, k=2)
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
    rankers = _default_rankers(args.fit_form)
    ci_results = audit_with_ci(df, cfg.budgets.fit, cfg.budgets.target, rankers, n_boot, np.random.default_rng(args.seed + 10))
    ci_results["fit_form"] = args.fit_form
    _print_interval_report(ci_results)
    projected_winner = str(projected.index[0])
    true_winner = str(truth.index[0])
    if projected_winner != true_winner:
        regret = float(truth.loc[projected_winner] - truth.loc[true_winner])
        print(f"Wrong winner: {projected_winner}; true winner: {true_winner}; regret={regret:.6f}")
    else:
        print(f"Winner is correct: {projected_winner}; regret=0.000000")

    crossovers = detect_crossovers(df, cfg.budgets.fit, cfg.budgets.target, fit_form=args.fit_form)
    print("Detected crossovers:")
    if crossovers:
        for a, b, gap in crossovers:
            root_msg = _crossover_root_message(df, a, b, cfg.budgets.fit, args.fit_form)
            print(f"- {a} vs {b}: true_gap={gap:.6f}; {root_msg}")
    else:
        print("- none")

    neg = negative_controls_only(np.random.default_rng(args.seed + 1), cfg)
    neg_cross = detect_crossovers(neg, cfg.budgets.fit, cfg.budgets.target, fit_form=args.fit_form)
    print(f"Negative-control crossovers detected: {len(neg_cross)}")

    mc = monte_carlo(SCENARIOS[args.scenario], cfg, n_trials=n_trials, rng=np.random.default_rng(args.seed + 2))
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
    cfg, n_boot, _ = _runtime_config(args)
    df = load_runs(args.runs)
    _assert_fit_form_available(df, args.fit_form)
    computes = sorted(df["compute"].astype(float).unique())
    budgets = normalize_budgets(tuple(c for c in computes if c < args.target), target=args.target)
    if len(budgets) < 3:
        raise SystemExit(f"need at least 3 distinct pre-target fitting budgets; found {len(budgets)}")
    rankers = _default_rankers(args.fit_form)
    audit = audit_with_ci(df, budgets, args.target, rankers, n_boot, np.random.default_rng(args.seed))
    audit["fit_form"] = args.fit_form
    crossovers = detect_crossovers(df, budgets, args.target, fit_form=args.fit_form)
    result = {
        "audit_metadata": {
            "budgets": [float(budget) for budget in budgets],
            "fast": bool(args.fast),
            "fit_form": args.fit_form,
            "n_boot": int(n_boot),
            "rankers": sorted(rankers.keys()),
            "rng_seed": int(args.seed),
            "target": float(args.target),
        },
        "fit_form": args.fit_form,
        "rankers": audit["rankers"],
        "under_seeded_cells": audit["under_seeded_cells"],
        "noise": audit["noise"],
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
    make_figures(df, args.out, target=args.target)
    print(f"wrote figures to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="asla")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--scenario", choices=["clean_crossover", "saturation_crossover", "noise_close_call"], default="clean_crossover")
    demo.add_argument("--seed", type=int, default=1729)
    demo.add_argument("--fit-form", choices=["compute_power_law", "chinchilla"], default="compute_power_law")
    demo.add_argument("--n-boot", type=_positive_int)
    demo.add_argument("--trials", type=_positive_int)
    demo.add_argument("--fast", action="store_true")
    demo.set_defaults(func=_demo)

    val = sub.add_parser("validate")
    val.add_argument("--runs", required=True)
    val.set_defaults(func=_validate)

    audit = sub.add_parser("audit")
    audit.add_argument("--runs", required=True)
    audit.add_argument("--target", type=float, required=True)
    audit.add_argument("--out")
    audit.add_argument("--seed", type=int, default=1729)
    audit.add_argument("--fit-form", choices=["compute_power_law", "chinchilla"], default="compute_power_law")
    audit.add_argument("--n-boot", type=_positive_int)
    audit.add_argument("--fast", action="store_true")
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
    figs.add_argument("--target", type=float, help="Target budget; defaults to the largest compute in the table.")
    figs.set_defaults(func=_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
