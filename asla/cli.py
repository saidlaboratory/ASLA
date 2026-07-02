"""Command-line interface for the algorithm-selection audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from asla.analysis.audit import audit_with_ci
from asla.analysis.crossover import detect_crossovers, detect_crossovers_fdr, fitted_crossover_for_pair
from asla.analysis.ensemble import ensemble_report
from asla.analysis.fits import normalize_budgets, project_ranking, truth_ranking
from asla.analysis.metrics import decision_metrics
from asla.analysis.racing import monte_carlo_selection
from asla.analysis.rankers import Ranker, ensemble_ranker, make_projection_ranker, single_scale_ranker
from asla.config import AuditConfig
from asla.data.benchmark import FAMILIES as BENCHMARK_FAMILIES
from asla.data.benchmark import benchmark_grid, evaluate_configs
from asla.data.harvest import discover, harvest
from asla.data.io import load_runs
from asla.data.schema import SchemaError, validate
from asla.data.synthetic import SCENARIOS, negative_controls_only
from asla.data.synthetic import true_ranking as synthetic_true_ranking
from asla.figures import make_figures
from asla.report import write_report


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


def _default_rankers(fit_form: str, weighted: bool = False, n_fit_budgets: int | None = None) -> dict[str, Ranker]:
    rankers: dict[str, Ranker] = {
        "projection_ranker": make_projection_ranker(fit_form, weighted=weighted),  # type: ignore[arg-type]
        "single_scale_ranker": single_scale_ranker,
    }
    # The ensemble needs leave-largest-budget-out refits, so at least four
    # distinct fitting budgets; it is also compute-axis only.
    if fit_form == "compute_power_law" and (n_fit_budgets is None or n_fit_budgets >= 4):
        rankers["ensemble_ranker"] = ensemble_ranker
    return rankers


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
    ci_rng = np.random.default_rng(args.seed + 10)
    ci_results = audit_with_ci(df, cfg.budgets.fit, cfg.budgets.target, rankers, n_boot, ci_rng)
    ci_results["fit_form"] = args.fit_form
    _print_interval_report(ci_results)
    projected_winner = str(projected.index[0])
    true_winner = str(truth.index[0])
    if projected_winner != true_winner:
        regret = float(truth.loc[projected_winner] - truth.loc[true_winner])
        print(f"Wrong winner: {projected_winner}; true winner: {true_winner}; regret={regret:.6f}")
    else:
        print(f"Winner is correct: {projected_winner}; regret=0.000000")

    if args.fit_form == "compute_power_law" and len(cfg.budgets.fit) >= 4:
        noise_report = ci_results["noise"]
        assert isinstance(noise_report, dict)
        ens = ensemble_report(df, cfg.budgets.fit, cfg.budgets.target, noise_band=noise_report["noise_band"])
        print("Ensemble extrapolation reliability (family disagreement / target noise band):")
        for name, entry in ens.items():
            rho = entry["reliability"]
            rho_str = "unestimated" if rho is None else f"{float(rho):.2f}"
            families = entry["families"]
            assert isinstance(families, dict)
            weights = ", ".join(f"{fam}={info['weight']:.2f}" for fam, info in families.items())
            print(f"- {name}: rho={rho_str}; weights: {weights}")

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

    mc = monte_carlo_selection(SCENARIOS[args.scenario], cfg, n_trials=n_trials, rng=np.random.default_rng(args.seed + 2))
    print("Monte Carlo selection-rule comparison (plain / largest / gate / race):")
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
    rankers = _default_rankers(args.fit_form, weighted=bool(args.weighted), n_fit_budgets=len(budgets))
    audit = audit_with_ci(df, budgets, args.target, rankers, n_boot, np.random.default_rng(args.seed))
    audit["fit_form"] = args.fit_form
    crossovers = detect_crossovers(df, budgets, args.target, fit_form=args.fit_form)
    crossovers_fdr = detect_crossovers_fdr(df, budgets, args.target, q=args.crossover_q, fit_form=args.fit_form)
    ensemble: dict[str, object] | None = None
    if args.fit_form == "compute_power_law" and len(budgets) >= 4:
        ensemble = ensemble_report(df, budgets, args.target, noise_band=audit["noise"]["noise_band"])
    result = {
        "audit_metadata": {
            "budgets": [float(budget) for budget in budgets],
            "crossover_q": float(args.crossover_q),
            "fast": bool(args.fast),
            "fit_form": args.fit_form,
            "n_boot": int(n_boot),
            "rankers": sorted(rankers.keys()),
            "rng_seed": int(args.seed),
            "target": float(args.target),
            "weighted": bool(args.weighted),
        },
        "fit_form": args.fit_form,
        "rankers": audit["rankers"],
        "under_seeded_cells": audit["under_seeded_cells"],
        "noise": audit["noise"],
        "truth": audit["truth"],
        "truth_ties": audit["truth_ties"],
        "fit_diagnostics": audit["fit_diagnostics"],
        "crossovers": crossovers,
        "crossovers_fdr": crossovers_fdr,
        "ensemble": ensemble,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        if args.report:
            report_path = Path(args.out).with_suffix(".md")
            write_report(args.out, report_path)
            print(f"wrote report to {report_path}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _report(args: argparse.Namespace) -> int:
    out = args.out or str(Path(args.audit).with_suffix(".md"))
    path = write_report(args.audit, out)
    print(f"wrote report to {path}")
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


def _benchmark_heatmaps(table: pd.DataFrame, out_dir: Path) -> None:
    """Write wrong-pick-rate heat maps per rule for crossover families."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping benchmark heat maps")
        return
    for family in sorted(set(table["family"]) & {"late_crossover", "saturating"}):
        subset = table[table["family"] == family]
        for rule in sorted(subset["rule"].unique()):
            pivot = (
                subset[subset["rule"] == rule]
                .pivot_table(index="gap_to_noise", columns="crossover_position", values="wrong_pick_rate", aggfunc="mean")
                .sort_index()
            )
            if pivot.empty:
                continue
            fig, ax = plt.subplots()
            im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), [f"{c:+.2f}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [f"{g:g}" for g in pivot.index])
            ax.set_xlabel("crossover position log10(Cx / Cmax_fit)")
            ax.set_ylabel("target gap / seed noise")
            ax.set_title(f"wrong-pick rate: {rule} on {family}")
            fig.colorbar(im, ax=ax)
            name = f"benchmark_heatmap_{family}_{rule}"
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=160)
            plt.close(fig)
    print(f"wrote benchmark heat maps to {out_dir}")


def _benchmark(args: argparse.Namespace) -> int:
    n_trials = 10 if args.fast else args.trials
    n_boot = 40 if args.fast else args.n_boot
    configs = benchmark_grid(families=tuple(args.families), seeds=tuple(range(args.problem_seeds)))
    if not configs:
        raise SystemExit("benchmark grid is empty; check --families")
    print(f"evaluating {len(configs)} benchmark problems x 4 rules ({n_trials} trials each)")
    table = evaluate_configs(configs, n_trials=n_trials, n_boot=n_boot, seed=args.seed, beta=args.beta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "benchmark_results.parquet", index=False)
    table.to_csv(out / "benchmark_results.csv", index=False)
    summary = (
        table.groupby(["family", "rule"])[["mean_regret", "wrong_pick_rate", "mean_compute"]]
        .mean()
        .round(6)
        .reset_index()
        .to_dict(orient="records")
    )
    (out / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "n_problems": len(configs),
                "n_trials": int(n_trials),
                "n_boot": int(n_boot),
                "rng_seed": int(args.seed),
                "beta": float(args.beta),
                "per_family_rule_means": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _benchmark_heatmaps(table, out)
    print(json.dumps(summary, indent=2))
    print(f"wrote benchmark results to {out}")
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
    demo.add_argument(
        "--scenario",
        choices=["clean_crossover", "saturation_crossover", "noise_close_call"],
        default="clean_crossover",
    )
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
    audit.add_argument("--weighted", action="store_true", help="Weight fits by per-cell seed standard errors.")
    audit.add_argument("--crossover-q", type=float, default=0.05, help="Benjamini-Hochberg FDR level for crossover tests.")
    audit.add_argument("--report", action="store_true", help="Also write a markdown report next to --out.")
    audit.set_defaults(func=_audit)

    rep = sub.add_parser("report")
    rep.add_argument("--audit", required=True, help="Path to an audit JSON produced by 'asla audit --out'.")
    rep.add_argument("--out", help="Markdown output path; defaults to the audit path with .md suffix.")
    rep.set_defaults(func=_report)

    harv = sub.add_parser("harvest")
    harv.add_argument("--discover", action="store_true")
    harv.add_argument("--entity-project", required=True)
    harv.add_argument("--field-map")
    harv.add_argument("--out", default="runs.parquet")
    harv.set_defaults(func=_harvest)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--out", default="results/benchmark")
    bench.add_argument("--families", nargs="+", choices=list(BENCHMARK_FAMILIES), default=list(BENCHMARK_FAMILIES))
    bench.add_argument("--trials", type=_positive_int, default=100)
    bench.add_argument("--n-boot", type=_positive_int, default=300)
    bench.add_argument(
        "--problem-seeds", type=_positive_int, default=1, help="Curve-sampling seeds per knob setting."
    )
    bench.add_argument("--seed", type=int, default=1729)
    bench.add_argument("--beta", type=float, default=1.0, help="Race interval-width multiplier.")
    bench.add_argument("--fast", action="store_true")
    bench.set_defaults(func=_benchmark)

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
