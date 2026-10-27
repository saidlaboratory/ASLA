"""Render the paper's figures from committed results.

- fig_spec_curve: the specification curve (results/adversarial/a2_specification_curve.csv, summarised in
  a2_specification_summary.json): projection's excess mis-selection over single-scale for every specification.
- fig_flexibility: mis-selection against estimator flexibility (results/task_b/task_b.json, primary design).
- fig_marin_rank_intervals: each official speedrun submission's rank at 1e22 under noise at the bound and under
  the two converged forms (results/external/marin_rank_intervals.json).

Writes ``paper/figures/*.pdf``. Colours: the validated categorical slots, paired with marker shape for print.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validated (dataviz validator, light surface)
MARKERS = ("o", "s", "^", "D")
INK, MUTED, GRID = "#1b1b1a", "#6b6a63", "#e4e3dd"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def _grid(ax: plt.Axes, axis: Literal["x", "y"] = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def spec_curve() -> None:
    summary = json.loads((RESULTS / "adversarial" / "a2_specification_summary.json").read_text())
    df = pd.read_csv(RESULTS / "adversarial" / "a2_specification_curve.csv")
    assert len(df) == summary["n_specifications"]
    df["excess_pp"] = 100 * (df["projection_mis_selection"] - df["single_scale_mis_selection"])
    df = df.sort_values("excess_pp").reset_index(drop=True)
    labels = {
        "c4_en_bits_per_token": "C4 bits per token",
        "sn_c4_bits_per_byte": "C4 bits per byte (S&N)",
        "olmes_macro_error": "OLMES error",
        "olmes_macro_correct_prob_per_char_deficit": "OLMES correct-prob",
    }
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for i, (metric, label) in enumerate(labels.items()):
        sub = df[df["metric"] == metric]
        ax.scatter(sub.index, sub["excess_pp"], s=9, color=SERIES[i], marker=MARKERS[i], linewidths=0, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("specification, ordered by excess")
    ax.set_ylabel("projection minus single-scale\nmis-selection (points)")
    ax.set_xlim(-3, len(df) + 2)
    _grid(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spec_curve.pdf")
    plt.close(fig)


def flexibility() -> None:
    task = json.loads((RESULTS / "task_b" / "task_b.json").read_text())
    rankers = task["designs"]["primary_4M-300M_target1B"]["rankers"]
    order = [
        ("shared_exponent", "shared\nexponent"),
        ("shrinkage_1", "shrink\n1.0"),
        ("shrinkage_0.8", "0.8"),
        ("shrinkage_0.6", "0.6"),
        ("shrinkage_0.4", "0.4"),
        ("shrinkage_0.2", "0.2"),
        ("plain_projection", "plain"),
        ("ensemble", "ensemble"),
    ]
    assert rankers["plain_projection"]["mis_selection"]["point"] == rankers["shrinkage_0"]["mis_selection"]["point"]
    pts = [100 * rankers[k]["mis_selection"]["point"] for k, _ in order]
    lo = [100 * (rankers[k]["mis_selection"]["point"] - rankers[k]["mis_selection"]["lo"]) for k, _ in order]
    hi = [100 * (rankers[k]["mis_selection"]["hi"] - rankers[k]["mis_selection"]["point"]) for k, _ in order]
    single = rankers["single_scale"]["mis_selection"]
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.axhspan(100 * single["lo"], 100 * single["hi"], color=GRID, linewidth=0, label="single-scale (95% interval)")
    ax.axhline(100 * single["point"], color=MUTED, linewidth=0.8)
    ax.errorbar(
        range(len(order)),
        pts,
        yerr=[lo, hi],
        fmt="o",
        color=SERIES[0],
        markersize=4,
        elinewidth=1.2,
        capsize=0,
        label="estimator (95% interval)",
    )
    ax.set_xticks(range(len(order)), [lab for _, lab in order], fontsize=6.5, rotation=30, ha="right")
    ax.set_xlabel("less flexible  →  more flexible")
    ax.set_ylabel("mis-selected pairs (%)")
    _grid(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_flexibility.pdf")
    plt.close(fig)


def _short(name: str) -> str:
    return (
        name.replace("hackable_transformer_attn_sink/attnsink/", "attn_sink/")
        .replace("hackable_transformer_", "")
        .replace("_scaling", "")
        .replace("/splash/", "/")
    )


def marin_rank_intervals() -> None:
    data = json.loads((RESULTS / "external" / "marin_rank_intervals.json").read_text())
    rows = data["rows"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, r in enumerate(rows):
        ax.plot(
            [r["noise_rank_2_5"], r["noise_rank_97_5"]],
            [i, i],
            color=SERIES[0],
            linewidth=2,
            solid_capstyle="round",
            label="noise at the bound (95% of draws)" if i == 0 else None,
        )
    ax.scatter(
        [r["rank_three_parameter_optimum"] for r in rows],
        range(n),
        s=22,
        color=INK,
        marker="o",
        zorder=3,
        label="three-parameter optimum (fixed fit)",
    )
    ax.scatter(
        [r["rank_loglog"] for r in rows],
        range(n),
        s=22,
        facecolor="none",
        edgecolor=SERIES[1],
        marker="D",
        linewidths=1.2,
        zorder=3,
        label="two-parameter log-log",
    )
    ax.set_yticks(range(n), [_short(r["submission"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xticks(range(1, n + 1, 3))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("rank at the 1e22 forecast (1 = lowest BPB)")
    _grid(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_marin_rank_intervals.pdf")
    plt.close(fig)


def lever_arm() -> None:
    import numpy as np

    designs = json.loads((RESULTS / "target_scoring" / "expected_error.json").read_text())["dose_response"]["designs"]
    x = np.log10([d["lever_arm"] for d in designs])
    y = [d["expected_excess_pp"] for d in designs]
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.scatter(x, y, s=14, color=SERIES[0], linewidths=0)
    ax.set_xlabel("lever arm, log10(target / largest fitted compute)")
    ax.set_ylabel("projection minus single-scale\nmis-selection (points)")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lever_arm.pdf")
    plt.close(fig)


def certify_lever() -> None:
    import numpy as np

    data = json.loads((RESULTS / "certify" / "certify.json").read_text())
    designs = data["datadecide"]["designs"]
    alpha = 100 * data["alpha"]
    methods = (
        ("selector", "lever-arm misfit term", "o"),
        ("seed_se", "seed error only", "s"),
        ("bootstrap", "seed bootstrap", "^"),
    )
    levers = sorted({round(d["lever_arm"], 2) for d in designs})
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    ax.axhline(alpha, color=MUTED, linewidth=0.8, linestyle="--", label="nominal")
    for i, (key, label, marker) in enumerate(methods):
        pts = []
        for lv in levers:
            ds = [d for d in designs if round(d["lever_arm"], 2) == lv]
            cert = sum(d["methods"][key]["certified"] for d in ds)
            charges = sum(d["methods"][key]["charges"] for d in ds)
            pts.append(100 * charges / cert if cert else np.nan)
        ax.plot(levers, pts, color=SERIES[i], marker=marker, markersize=4, linewidth=1.5, label=label)
    ax.set_xlabel("lever arm (decades)")
    ax.set_ylabel("false certification (%)")
    _grid(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_certify_lever.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec_curve()
    flexibility()
    marin_rank_intervals()
    lever_arm()
    certify_lever()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
