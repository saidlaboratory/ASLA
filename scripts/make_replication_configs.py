"""Write one training config per replication run, and the budget (PREDICTIONS_REPLICATION_PROTOCOL.md).

Reads replication_protocol/ladder.json (DataDecide's setup at the eight sizes) and recipes.json (the ten
selected corpora), and writes replication_protocol/configs/<recipe>__<size>__seed<seed>.json plus
replication_protocol/budget.json (runs, total tokens, total FLOPs = sum of 6 N D, GPU-hours under the same
utilisation assumptions as results/replication/untuned_spec.json). Writes files only; trains nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "replication_protocol"
HARDWARE = {"H100 SXM": {"peak_flops": 989.4e12, "mfu": 0.40}, "RTX 3090": {"peak_flops": 71.0e12, "mfu": 0.30}}


def configs() -> list[dict]:
    ladder = json.loads((PROTOCOL / "ladder.json").read_text(encoding="utf-8"))
    recipes = json.loads((PROTOCOL / "recipes.json").read_text(encoding="utf-8"))["selected"]
    sh = ladder["shared"]
    out = []
    for r in recipes:
        for s in ladder["sizes"]:
            n_rule = s.get("nominal_n_for_rules", s["params_n"])
            for seed in ladder["seeds"]:
                out.append(
                    {
                        "run_id": f"{r['id']}__{s['size']}__seed{seed}",
                        "recipe": r["id"],
                        "corpus": {"hf_id": r["hf_id"], "revision_sha": r["revision_sha"], "processing": r["processing"]},
                        "size": s["size"],
                        "seed": seed,
                        "seed_label": ladder["seed_labels"][str(seed)],
                        "model": {
                            "d_model": s["d_model"],
                            "n_layers": s["n_layers"],
                            "n_heads": s["n_heads"],
                            "architecture": sh["architecture"],
                            "tokenizer": sh["tokenizer"],
                            "sequence_length": sh["sequence_length"],
                        },
                        "params_n": s["params_n"],
                        "tokens_d": s["tokens_d"],
                        "flops": 6.0 * s["params_n"] * s["tokens_d"],
                        "train": {
                            "batch_seqs": s["batch_seqs"],
                            "steps": s["steps"],
                            "peak_lr": s["peak_lr"],
                            "warmup_steps": round(n_rule / (s["batch_seqs"] * sh["sequence_length"])),
                            "optimizer": sh["optimizer"],
                            "schedule": sh["schedule"],
                            "grad_clip": sh["grad_clip"],
                            "precision": sh["precision"],
                            "framework": sh["framework"],
                        },
                        "eval": {"metric": "c4_en_bits_per_token", **ladder["evaluation"]},
                    }
                )
    return out


def budget(runs: list[dict]) -> dict:
    flops = sum(r["flops"] for r in runs)
    return {
        "runs": len(runs),
        "recipes": len({r["recipe"] for r in runs}),
        "sizes": sorted({r["size"] for r in runs}, key=lambda s: next(x["params_n"] for x in runs if x["size"] == s)),
        "seeds": sorted({r["seed"] for r in runs}),
        "total_tokens": sum(r["tokens_d"] for r in runs),
        "total_flops": flops,
        "gpu_hours": {k: flops / (h["peak_flops"] * h["mfu"]) / 3600 for k, h in HARDWARE.items()},
        "hardware_assumptions": HARDWARE,
        "tokens_per_corpus_largest_run": max(r["tokens_d"] for r in runs),
    }


def main() -> dict:
    runs = configs()
    d = PROTOCOL / "configs"
    d.mkdir(exist_ok=True)
    for old in d.glob("*.json"):
        old.unlink()
    for r in runs:
        (d / f"{r['run_id']}.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    b = budget(runs)
    (PROTOCOL / "budget.json").write_text(json.dumps(b, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: b[k] for k in ("runs", "total_tokens", "total_flops", "gpu_hours")}, indent=2))
    return b


if __name__ == "__main__":
    main()
