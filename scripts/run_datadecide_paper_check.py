"""Task 5 of PREDICTIONS_TASK_PATTERN.md: DataDecide's released outputs and printed paper against their code.

Inputs, fetched once into the git-ignored ``data/raw`` (checksums recorded):

* ``data/raw/datadecide_single_scale/``: the public Drive folder linked from
  ``allenai/DataDecide`` ``results/readme.md`` (listing pages, and from
  ``outputs2/`` the single-scale output ``2_prediction_model_scale.csv`` and
  target table ``1_primary_transformed.csv``);
* ``data/raw/datadecide_paper/``: arXiv:2504.11393v2 (HTML and PDF).

Checks: (1) their released single-scale accuracies against our run of their
code under the three-seed and default-seed targets; (2) every macro-average
decision accuracy the paper prints, and where it shows the comparison,
against (a) their code's mixed-target output, (b) the consistent three-seed
output, (c) the consistent default-seed output; (3) the S&N agreement figure
from the committed ``results/signal_and_noise/agreement.json``.

Writes ``results/external/datadecide_paper_check.json``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_SS = REPO / "data" / "raw" / "datadecide_single_scale"
RAW_PAPER = REPO / "data" / "raw" / "datadecide_paper"
OUT = REPO / "results" / "external" / "datadecide_paper_check.json"
MACRO = "olmes_10_macro_avg"
QUOTES = {
    "printed_single_scale": "80 % of comparisons correct",
    "single_scale_target": "mean performance over 3 seed runs",
    "section_2_3_target": "mean downstream performance over 3 random seeds",
    "scaling_law_attempts": "only one prediction attempt with the default fully trained random seed",
    "figure_3": "Figure 3: Decision accuracy over 8 baseline scaling law variants",
    "table_4": "Table 4: Average prediction error",
    "size_subsets": "to explore the improvements of progressively adding larger model sizes",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_text() -> str:
    raw = (RAW_PAPER / "2504.11393v2.html").read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


def quote(text: str, phrase: str, width: int = 160) -> str | None:
    i = text.find(phrase)
    return None if i < 0 else text[max(0, i - width) : i + len(phrase) + width].strip()


def drive_listing(name: str) -> dict[str, str]:
    page = (RAW_SS / name).read_text(encoding="utf-8")
    titles = [t for t in re.findall(r'flip-entry-title">([^<]*)', page)]
    ids = re.findall(r'id="entry-([^"]+)"', page)
    return dict(zip(titles, ids))


def released_single_scale() -> dict[str, float]:
    cols = ["binary_accuracy", "metric", "model", "seed", "compute_latest"]
    parts = [
        c[c["metric"] == "primary_metric"]
        for c in pd.read_csv(RAW_SS / "2_prediction_model_scale.csv", usecols=cols, chunksize=200_000)
    ]
    frame = pd.concat(parts)
    last = frame.loc[frame.groupby(["model", "seed"])["compute_latest"].idxmax()]
    return (100 * last.groupby("model")["binary_accuracy"].mean()).to_dict()


def main() -> None:
    ours = json.loads((REPO / "results" / "external" / "datadecide_single_scale.json").read_text())
    sens = json.loads((REPO / "results" / "external" / "datadecide_sensitivity.json").read_text())
    released = released_single_scale()
    ours_three = {s: v["three_seed_target"]["mean"] for s, v in ours["their_single_scale_macro"].items()}
    ours_default = {s: v["default_seed_target"]["mean"] for s, v in ours["their_single_scale_macro"].items()}
    common = sorted(set(released) & set(ours_three))
    text = paper_text()
    quotes = {k: quote(text, v) for k, v in QUOTES.items()}
    macro_rows = [r for r in sens["task_4b"]["tested_rows"]["rows"] if r["task"] == MACRO]
    agreement = json.loads((REPO / "results" / "signal_and_noise" / "agreement.json").read_text())["metric_agreement"]
    pairwise = [r["pairwise_agreement"] for r in agreement]
    out: dict[str, Any] = {
        "inputs": {
            "drive_folder": "1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D (public; linked from allenai/DataDecide results/readme.md)",
            "drive_top_level": drive_listing("drive_listing.html"),
            "drive_outputs2": drive_listing("drive_outputs2.html"),
            "sha256": {p.name: sha256(p) for p in sorted(RAW_SS.glob("*.csv")) + sorted(RAW_PAPER.glob("2504.11393v2.*"))},
        },
        "released_single_scale_vs_our_run_of_their_code": {
            "scales": common,
            "released": {s: released[s] for s in common},
            "ours_three_seed_target": {s: ours_three[s] for s in common},
            "ours_default_seed_target": {s: ours_default[s] for s in common},
            "max_abs_diff_three_seed": max(abs(released[s] - ours_three[s]) for s in common),
            "max_abs_diff_default_seed": max(abs(released[s] - ours_default[s]) for s in common),
        },
        "paper": {
            "quotes": quotes,
            "printed_macro_decision_accuracies": {
                "single_scale_150M_approx_80": {
                    "printed": "~80%",
                    "where": "abstract and Figure 1 (right) callout",
                    "three_seed_target": ours_three["150M"],
                    "default_seed_target": ours_default["150M"],
                }
            },
            "scaling_law_macro_values": (
                "not printed: shown only as points in Figure 1 (right, 'Multi-Scale Fit' stars) and Figure 3 (8 variants); "
                "Table 4 prints prediction error, not decision accuracy. The figures include size-subset configurations "
                "({s1..sk}, {sk..s14}) that are not in the released scaling_law_fit table, so plotted points cannot be "
                "matched to released rows at the precision that separates (a) from (b)"
            ),
        },
        "code_outputs_macro_750M": {
            "single_scale_three_seed": ours_three["750M"],
            "single_scale_default_seed": ours_default["750M"],
            "variants": {
                r["setup"]: {"default_seed_target": r["implemented"], "three_seed_target": r["three_seed_mean"]}
                for r in macro_rows
            },
        },
        "sn_agreement_verified": {
            "source": "results/signal_and_noise/agreement.json metric_agreement",
            "min_pairwise_agreement": min(pairwise),
            "max_pairwise_agreement": max(pairwise),
            "by_scale": {r["scale_label"]: r["pairwise_agreement"] for r in agreement},
        },
    }
    three, default = ours_three["150M"], ours_default["150M"]
    out["verdict"] = {
        "released_single_scale_is_three_seed_target": out["released_single_scale_vs_our_run_of_their_code"][
            "max_abs_diff_three_seed"
        ]
        < 1e-6,
        "printed_80pct_consistent_with": [
            name for name, v in (("three_seed_target", three), ("default_seed_target", default)) if abs(v - 80) <= 1.5
        ],
        "printed_80pct_discriminates_targets": False,
        "mixed_comparison_where": (
            "Figure 1 (right) and Figure 3 plot single-scale points (three-seed target, per the text and the released "
            "outputs) against scaling-law points; the released scaling-law code scores against the default seed. "
            "Whether the plotted scaling-law points were scored that way cannot be established from the paper"
        ),
        "claim_kept_to_released_code": True,
    }
    out["predictions"] = {"P5_printed_numbers_match_mixed": None}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["verdict"], indent=1))
    print(
        json.dumps(
            out["released_single_scale_vs_our_run_of_their_code"]
            | {"released": None, "ours_three_seed_target": None, "ours_default_seed_target": None},
            indent=1,
        )
    )
    print(
        {k: (v is not None) for k, v in quotes.items()},
        out["sn_agreement_verified"]["min_pairwise_agreement"],
        out["sn_agreement_verified"]["max_pairwise_agreement"],
    )


if __name__ == "__main__":
    sys.exit(main())
