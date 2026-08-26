"""Figures for the two promoted A2 findings.

Figure 1 (dose-response with control arm): excess projection flips against the
**lever arm** L = C_target / C_max_fitted, with single-scale mis-selection as
the control. If projection's excess is extrapolation variance, it must shrink
as the lever arm shortens, while the control - a rule that fits nothing and
simply reads the largest fit budget - must stay flat.

The dose variable is the lever arm, not the raw number of fit budgets. Those
two are confounded in the sweep (adding budgets usually also changes which
scales are present), and the raw budget count is only weakly related to the
excess (Spearman -0.14). Holding the ladder's start fixed and extending its
top, the relationship is clean and monotone in every case.

Figure 2 (how projection wins when it wins): for every specification where
projection ties or beats single-scale ranking, the excess flip count. If
projection ever won by *correcting a crossover*, some design would show a
positive excess alongside a lower mis-selection rate. None does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "adversarial" / "a2_specification_curve.csv"
OUT = REPO / "results" / "adversarial"


def lever_arm_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Excess flips and both mis-selection rates per lever arm (target / max fit scale).

    ``fit_scales`` is "min-max"; the max scale together with the target fixes
    the lever arm. Scales are converted to their DataDecide compute values so
    the arm is a real ratio rather than a label.
    """

    scale_compute = {
        "4M": 8.4302e15, "6M": 2.1701e16, "8M": 4.3777e16, "10M": 5.8851e16, "14M": 1.2413e17,
        "16M": 1.5376e17, "20M": 2.1909e17, "60M": 1.9555e18, "90M": 5.7581e18, "150M": 1.3439e19,
        "300M": 5.6620e19, "530M": 1.4955e20, "750M": 1.2658e20, "1B": 7.0621e20,
    }
    out = frame.copy()
    out["max_fit_scale"] = out["fit_scales"].str.split("-").str[1]
    out["min_fit_scale"] = out["fit_scales"].str.split("-").str[0]
    out["lever_arm"] = [
        scale_compute[t] / scale_compute[m] for t, m in zip(out["target"], out["max_fit_scale"])
    ]
    return out


def dose_response_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Excess flips and both mis-selection rates per number of fit budgets."""

    grouped = frame.groupby("n_fit_budgets")
    table = grouped.agg(
        n_specs=("metric", "size"),
        mean_excess_flips=("excess_projection_flips", "mean"),
        sd_excess_flips=("excess_projection_flips", "std"),
        mean_projection=("projection_mis_selection", "mean"),
        mean_single_scale=("single_scale_mis_selection", "mean"),
        mean_projection_flips=("projection_flips", "mean"),
        mean_single_flips=("single_scale_flips", "mean"),
    ).reset_index()
    table["excess_over_single"] = table["mean_projection"] - table["mean_single_scale"]
    return table


def rank_correlation(frame: pd.DataFrame) -> dict[str, float]:
    """Spearman of excess against the lever arm and against the raw budget count."""

    from scipy.stats import spearmanr

    k = frame["n_fit_budgets"].to_numpy(dtype=float)
    levered = lever_arm_table(frame)
    arm = np.log10(levered["lever_arm"].to_numpy(dtype=float))
    return {
        "spearman_excess_vs_log_lever_arm": float(spearmanr(arm, levered["excess_projection_flips"]).statistic),
        "spearman_excess_vs_log_lever_arm_p": float(spearmanr(arm, levered["excess_projection_flips"]).pvalue),
        "spearman_single_scale_vs_log_lever_arm": float(spearmanr(arm, levered["single_scale_mis_selection"]).statistic),
        "spearman_single_scale_vs_log_lever_arm_p": float(spearmanr(arm, levered["single_scale_mis_selection"]).pvalue),
        "spearman_excess_vs_budgets": float(spearmanr(k, frame["excess_projection_flips"]).statistic),
        "spearman_excess_p": float(spearmanr(k, frame["excess_projection_flips"]).pvalue),
        "spearman_projection_vs_budgets": float(spearmanr(k, frame["projection_mis_selection"]).statistic),
        "spearman_single_scale_vs_budgets": float(spearmanr(k, frame["single_scale_mis_selection"]).statistic),
        "spearman_single_scale_p": float(spearmanr(k, frame["single_scale_mis_selection"]).pvalue),
    }


def winning_designs(frame: pd.DataFrame) -> pd.DataFrame:
    ties = frame[~frame["projection_worse"].astype(bool)]
    return ties[
        ["metric", "target", "fit_scales", "fit_bounds", "n_fit_budgets", "projection_mis_selection",
         "single_scale_mis_selection", "excess_projection_flips", "n_fit_error", "n_inherited"]
    ].drop_duplicates(["metric", "target", "fit_scales", "fit_bounds"])


def lever_arm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    levered = lever_arm_table(frame)
    return (
        levered.groupby(["target", "max_fit_scale"])
        .agg(
            lever_arm=("lever_arm", "first"),
            n_specs=("metric", "size"),
            mean_excess_flips=("excess_projection_flips", "mean"),
            mean_projection=("projection_mis_selection", "mean"),
            mean_single_scale=("single_scale_mis_selection", "mean"),
        )
        .reset_index()
        .sort_values("lever_arm")
    )


def matched_ladders(frame: pd.DataFrame) -> pd.DataFrame:
    """Within one metric/target/min-scale/bounds cell, vary only the top of the ladder."""

    levered = lever_arm_table(frame)
    keys = ["metric", "target", "min_fit_scale", "fit_bounds", "ranking", "fdr"]
    rows = []
    for key, group in levered.groupby(keys):
        group = group.drop_duplicates("max_fit_scale").sort_values("lever_arm")
        if len(group) < 2:
            continue
        for _, row in group.iterrows():
            rows.append(
                {
                    "design": " | ".join(str(k) for k in key[:4]),
                    "max_fit_scale": row["max_fit_scale"],
                    "lever_arm": row["lever_arm"],
                    "n_fit_budgets": row["n_fit_budgets"],
                    "excess_projection_flips": row["excess_projection_flips"],
                    "projection_mis_selection": row["projection_mis_selection"],
                    "single_scale_mis_selection": row["single_scale_mis_selection"],
                }
            )
    return pd.DataFrame(rows)


def make_figures(frame: pd.DataFrame, table: pd.DataFrame) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []

    matched = matched_ladders(frame)
    summary = lever_arm_summary(frame)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for design, group in matched.groupby("design"):
        group = group.sort_values("lever_arm")
        ax1.plot(group["lever_arm"], group["excess_projection_flips"], "o-", alpha=0.45, lw=1.2, color="#2b6cb0")
    ax1.set_xscale("log")
    ax1.axhline(0.0, color="#718096", lw=1, ls=":")
    ax1.set_xlabel("lever arm  L = C_target / C_max fitted  (log scale)")
    ax1.set_ylabel("excess flips (projection - single-scale)")
    ax1.set_title("Dose-response: each line is one ladder\nextended upward, holding its start fixed")

    ax2.plot(summary["lever_arm"], 100 * summary["mean_projection"], "o-", color="#c53030", lw=2,
             label="projection (fits 3 parameters)")
    ax2.plot(summary["lever_arm"], 100 * summary["mean_single_scale"], "s--", color="#2f855a", lw=2,
             label="single-scale control (fits nothing)")
    slope_p = np.polyfit(np.log10(summary["lever_arm"]), 100 * summary["mean_projection"], 1)[0]
    slope_s = np.polyfit(np.log10(summary["lever_arm"]), 100 * summary["mean_single_scale"], 1)[0]
    ax2.annotate(
        f"slope per decade of L:\nprojection {slope_p:+.1f} pts\ncontrol {slope_s:+.1f} pts",
        xy=(0.05, 0.72), xycoords="axes fraction", fontsize=8,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("lever arm  L  (log scale)")
    ax2.set_ylabel("mean pairwise mis-selection (%)")
    ax2.set_title("Both rise, but the fitted rule rises ~3x faster")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT / f"fig_dose_response.{ext}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)

    wins = winning_designs(frame)
    fig2, ax = plt.subplots(figsize=(6.5, 4.0))
    margin = 100 * (wins["single_scale_mis_selection"] - wins["projection_mis_selection"])
    ax.scatter(wins["excess_projection_flips"], margin, s=45, color="#2b6cb0")
    ax.axvline(0.0, color="#c53030", lw=1.5, ls="--")
    ax.set_xlabel("excess flips (projection - single-scale)")
    ax.set_ylabel("projection's margin over single-scale (points)")
    ax.set_title("When projection wins, it wins by making fewer fit errors\n(never by correcting a crossover)")
    ax.annotate("no design lies right of 0:\nprojection never wins\nwith a positive excess",
                xy=(0.0, float(margin.max())), xytext=(-2.6, float(margin.max()) * 0.72), fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#c53030"})
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT / f"fig_projection_wins.{ext}"
        fig2.savefig(path, dpi=160, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig2)
    return written


def main() -> int:
    frame = pd.read_csv(CSV)
    table = dose_response_table(frame)
    corr = rank_correlation(frame)
    wins = winning_designs(frame)
    matched = matched_ladders(frame)
    monotone = []
    for design, group in matched.groupby("design"):
        group = group.sort_values("lever_arm")
        values = group["excess_projection_flips"].to_numpy(dtype=float)
        monotone.append(bool(np.all(np.diff(values) >= 0)))
    payload = {
        "dose_response_by_budget_count": table.to_dict("records"),
        "lever_arm_summary": lever_arm_summary(frame).to_dict("records"),
        "matched_ladders": matched.to_dict("records"),
        "n_matched_ladders": len(monotone),
        "n_monotone_in_lever_arm": int(sum(monotone)),
        "rank_correlations": corr,
        "winning_designs": wins.to_dict("records"),
        "n_winning_designs": int(len(wins)),
        "max_excess_among_winning": float(wins["excess_projection_flips"].max()),
        "all_winning_have_nonpositive_excess": bool((wins["excess_projection_flips"] <= 0).all()),
        "figures": make_figures(frame, table),
    }
    (OUT / "a2_dose_response.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(lever_arm_summary(frame).round(4).to_string(index=False))
    print(f"\nmatched ladders monotone in the lever arm: {payload['n_monotone_in_lever_arm']}/{payload['n_matched_ladders']}")
    print(json.dumps(corr, indent=2))
    print(f"winning designs: {payload['n_winning_designs']}, max excess = {payload['max_excess_among_winning']}, "
          f"all non-positive = {payload['all_winning_have_nonpositive_excess']}")
    print("figures:", payload["figures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
