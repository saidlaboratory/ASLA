"""Task 3a: can the ColPret / Hitchhiker's 485-model dataset support our audit?

Answers a feasibility question with measurements, before any harvester is
written. Our decomposition needs, at minimum:

* >= 3 interventions that are genuinely being *compared* (same evaluation, same
  compute axis),
* >= 3 shared fitting budgets below a target, plus the target itself,
* >= 2 seeds per (intervention, budget) cell for the noise band and the
  crossover significance tests.

Source: https://github.com/IBM/ColPret (`test_cache/data.csv.zst`, 64 MB), the
repository the paper's footnote points to. The URL is not in the PDF text; it
was recovered from the PDF's link annotations.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asla.cli import _write_json_atomically  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "raw" / "colpret" / "data.csv.zst"
COLUMNS = ["model_name", "model_type", "scaled_set", "num_params", "tokens_seen", "flops", "seed", "loss", "domain"]
MIN_INTERVENTIONS = 3
MIN_SHARED_BUDGETS = 4  # >=3 for the fit plus 1 target
MIN_SEEDS = 2


def load() -> pd.DataFrame:
    raw = subprocess.run(["zstd", "-dc", str(ARCHIVE)], capture_output=True, check=True)
    df = pd.read_csv(io.BytesIO(raw.stdout), usecols=COLUMNS, low_memory=False)
    return df.dropna(subset=["loss", "num_params", "tokens_seen"])


def assess(df: pd.DataFrame) -> dict[str, Any]:
    language = df[df["domain"] == "LM"].copy()
    language["compute"] = 6.0 * language["num_params"] * language["tokens_seen"]
    final = language.sort_values("tokens_seen").groupby(["scaled_set", "num_params"]).tail(1)
    grid = final.pivot_table(index="scaled_set", columns="num_params", values="loss")
    sizes_per_family = grid.notna().sum(axis=1)
    multi = sizes_per_family[sizes_per_family >= 4].index.tolist()
    shared_all = list(grid.loc[multi].dropna(axis=1).columns) if multi else []

    gpt2 = final[final["scaled_set"].str.startswith("GPT2")]
    gpt2_grid = gpt2.pivot_table(index="scaled_set", columns="num_params", values="loss")
    gpt2_shared = list(gpt2_grid.dropna(axis=1).columns)
    cell_counts = final.groupby(["scaled_set", "num_params"]).size()

    checks = {
        "interventions_available": {
            "value": int(gpt2_grid.shape[0]),
            "required": MIN_INTERVENTIONS,
            "passes": bool(gpt2_grid.shape[0] >= MIN_INTERVENTIONS),
            "note": "the GPT2 subset varies pretraining data (c4 vs oscar, several mixes), a genuine intervention axis",
        },
        "shared_budgets": {
            "value": len(gpt2_shared),
            "required": MIN_SHARED_BUDGETS,
            "passes": bool(len(gpt2_shared) >= MIN_SHARED_BUDGETS),
            "note": "3 shared model sizes; a projection needs >=3 fit budgets AND a held-out target",
        },
        "seeds_per_cell": {
            "value": int(cell_counts.max()),
            "required": MIN_SEEDS,
            "passes": bool(cell_counts.max() >= MIN_SEEDS),
            "note": "one model per (family, size) cell: no replicate structure",
        },
        "cross_family_shared_ladder": {
            "value": len(shared_all),
            "required": MIN_SHARED_BUDGETS,
            "passes": bool(len(shared_all) >= MIN_SHARED_BUDGETS),
            "note": "families with >=4 sizes share zero common sizes, so they cannot be compared on one ladder",
        },
    }
    return {
        "source": "https://github.com/IBM/ColPret test_cache/data.csv.zst",
        "url_provenance": "recovered from the PDF link annotations; it does not appear in the extracted PDF text",
        "n_rows_total": int(len(df)),
        "n_rows_with_loss_params_tokens": int(len(df.dropna(subset=["loss", "num_params", "tokens_seen"]))),
        "n_models": int(df["model_name"].nunique()),
        "n_scaled_sets": int(df["scaled_set"].nunique()),
        "families_with_4plus_sizes": multi,
        "sizes_shared_by_those_families": [float(s) for s in shared_all],
        "gpt2_subset": {
            "n_families": int(gpt2_grid.shape[0]),
            "families": sorted(gpt2_grid.index.tolist()),
            "shared_sizes": [float(s) for s in gpt2_shared],
            "grid_completeness": f"{int(gpt2_grid.notna().sum().sum())}/{int(gpt2_grid.size)}",
        },
        "checks": checks,
        "verdict": "USABLE" if all(c["passes"] for c in checks.values()) else "NOT USABLE for the decomposition",
        "blocking_reasons": [name for name, c in checks.items() if not c["passes"]],
        "what_it_could_support": (
            "ColPret is built for fit-accuracy meta-analysis across heterogeneous families, which is what its "
            "paper does. It carries one model per (family, size) cell and no shared ladder across the families "
            "large enough to fit, so it cannot support a decision-level audit that needs replicate noise and a "
            "common compute axis. The 16-family GPT2 subset is the closest fit: a complete 16 x 3 grid varying "
            "pretraining data, but 3 budgets cannot support a projection with a held-out target, and single "
            "runs give no seed-noise band or crossover significance test."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/external")
    args = parser.parse_args(argv)
    result = assess(load())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(result, out / "colpret_assessment.json")
    shown = {k: v for k, v in result.items() if k not in ("families_with_4plus_sizes", "gpt2_subset")}
    print(json.dumps(shown, indent=2, default=str)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
