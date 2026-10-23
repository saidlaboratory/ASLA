"""The muonremez "-b200-" reruns and a scan for other board values taken from a different run.

Uses only metadata reachable without a W&B key: runs.json (and its history), the PR #4932 results files, and the
W&B links. Writes ``results/external/muonremez_reruns.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import marin_official  # noqa: E402
from scripts import run_break_diagnosis as bd  # noqa: E402
from scripts import run_marin_rigour as rig  # noqa: E402

OUT = REPO / "results" / "external" / "muonremez_reruns.json"
HISTORY = REPO / "data" / "raw" / "marin_speedrun" / "runs_history"
BOARD_FIELDS = (
    "model_size",
    "model_flops",
    "training_hardware_flops",
    "training_time",
    "description",
    "author",
    "run_completion_timestamp",
    "eval_paloma_c4_en_bpb",
)


def run_id(link: str | None) -> str | None:
    return link.rstrip("/").split("/runs/")[-1] if link else None


def project(link: str | None) -> str | None:
    return "/".join(link.split("wandb.ai/")[1].split("/")[:2]) if link and "wandb.ai/" in link else None


def main() -> None:
    runs = json.loads((rig.RAW / "runs.json").read_text())
    by_name = {r["run_name"]: r for r in runs}
    official = set(marin_official.official_groups(runs))
    reruns = {}
    for rung in ("520m", "1_2b"):
        board = by_name[f"muonremez_qwen3_scaling/{rung}"]
        info = json.loads(
            (bd.SRC / "experiments" / "speedrun" / "muonremez_qwen3_scaling" / rung / "speedrun_results.json").read_text()
        )["runs"][0]["run_info"]
        pr = {
            "model_size": info["model_size"],
            "model_flops": info["model_flops"],
            "training_hardware_flops": info["training_hardware_flops"],
            "training_time": info["training_time"],
            "description": info["description"],
            "author": info["author"],
            "run_completion_timestamp": info["run_completion_timestamp"],
            "eval_paloma_c4_en_bpb": info["eval/paloma/c4_en/bpb"],
        }
        differing = {k: [board.get(k), pr.get(k)] for k in BOARD_FIELDS if board.get(k) != pr.get(k)}
        reruns[rung] = {
            "board_run": run_id(board["wandb_link"]),
            "board_project": project(board["wandb_link"]),
            "pr_selected_run": run_id(info["wandb_run_link"]),
            "pr_project": project(info["wandb_run_link"]),
            "same_config_name": str(run_id(board["wandb_link"])).startswith(str(run_id(info["wandb_run_link"]))),
            "differing_board_fields": differing,
            "identical_board_fields": [k for k in BOARD_FIELDS if k not in differing],
            "throughput_flops_per_s": [
                board["training_hardware_flops"] / board["training_time"],
                info["training_hardware_flops"] / info["training_time"],
            ],
            "pr_hardware": {
                "device": (info.get("resources") or {}).get("device"),
                "num_devices": info.get("num_devices"),
                "device_flops": info.get("device_flops"),
            },
            "relative_difference": board["eval_paloma_c4_en_bpb"] / info["eval/paloma/c4_en/bpb"] - 1,
        }
    # other board values from a different run than the submission's own results file
    other = []
    for r in runs:
        g = marin_official.group_of(r["run_name"])
        if g not in official:
            continue
        f = bd.SRC / r["results_filepath"] / "speedrun_results.json"
        if not f.exists():
            other.append({"run": r["run_name"], "status": "no results file in marin@cdd4e25a8d or the PRs"})
            continue
        info = json.loads(f.read_text())["runs"][0]["run_info"]
        if (
            run_id(r["wandb_link"]) != run_id(info.get("wandb_run_link"))
            or r["eval_paloma_c4_en_bpb"] != info["eval/paloma/c4_en/bpb"]
        ):
            other.append(
                {
                    "run": r["run_name"],
                    "board_run": run_id(r["wandb_link"]),
                    "own_run": run_id(info.get("wandb_run_link")),
                    "board_bpb": r["eval_paloma_c4_en_bpb"],
                    "own_bpb": info["eval/paloma/c4_en/bpb"],
                }
            )
    rows = sorted((line.split("\t") for line in (HISTORY / "commits.tsv").read_text().splitlines()), key=lambda x: x[1])
    seen: dict[str, list[Any]] = defaultdict(list)
    for sha, *_ in rows:
        for r in json.loads((HISTORY / f"{sha[:10]}.json").read_text()):
            v = (r["eval_paloma_c4_en_bpb"], run_id(r.get("wandb_link")))
            if not seen[r["run_name"]] or seen[r["run_name"]][-1][1] != list(v):
                seen[r["run_name"]].append([sha[:10], list(v)])
    history_changes = {k: v for k, v in seen.items() if len(v) > 1 and marin_official.group_of(k) in official}
    out = {
        "reruns": reruns,
        "b200": "inferred from the board descriptions: the reruns are 'centered on Muon-style lr' 1.60e-2 (520m) and "
        "8.00e-3 (1.2b) against 1.20e-2 and 6.00e-3 for the PR's selected runs. The PR sweeps covered 0.5x-1.5x a "
        "base LR of 8e-3 (520m) and 4e-3 (1.2b); the reruns are at exactly 2.00x base, so 'b200' reads as 200% of the "
        "base LR: a sweep extension past the PR's range, named after the run it was forked from. (In Marin's code "
        "B200 is also an NVIDIA GPU, but the board records carry the PR runs' TPU throughput.)",
        "differ_only_in_seed_or_hardware": False,
        "differing_setting": "learning rate (Muon-style LR x4/3 at 520m and 1.2b; Adam LR unchanged), per the "
        "descriptions; the full configs need the W&B key (anonymous GraphQL returns no project)",
        "implied_relative_sd": None,
        "implied_relative_sd_reason": "not replicates: they differ in learning rate, so no noise value is implied and "
        "the both-rung count and 1e22 resolution are not rerun",
        "consequence": "the board's muonremez 520m and 1.2b points come from a sixth learning rate outside the submitted "
        "five-run sweep, which scored below all five; the 130m and 300m points are the best of the five",
        "other_board_values_from_a_different_run": other,
        "official_runs_whose_value_or_link_changed_in_history": history_changes,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
