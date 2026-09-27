"""Task 2 of PREDICTIONS_TASK_CRN.md: the data-order share of seed variance from PolyPythias (an upper bound for CRN).

``pythia-160m-data-seed{1,2,3}`` vary only the data order (initialisation fixed);
``pythia-160m-weight-seed{1,2,3}`` vary only the initialisation (data order
fixed). Each final checkpoint is downloaded alone (375 MB), evaluated on a fixed
post-Pile held-out sample, and deleted before the next. The metric is mean token
cross-entropy (nats) over the first 20 chunks of 1,024 tokens of
``data/raw/crn/heldout_text.txt`` (DataDecide arXiv:2504.11393v2, then
arXiv:2609.10702v1). rho_D = s2_D / (s2_D + s2_W) with an F(2, 2) interval.

Run with a Python that has ``transformers`` (the scratch venv); the estimation
functions need only numpy and scipy.

    <venv>/bin/python scripts/run_crn_polypythias.py --resume --scratch <dir>

Writes ``results/crn/polypythias_rho_d.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TEXT = REPO / "data" / "raw" / "crn" / "heldout_text.txt"
CACHE = REPO / "data" / "cache" / "crn_polypythias.jsonl"
OUT = REPO / "results" / "crn" / "polypythias_rho_d.json"
ARMS = {
    "data": [f"EleutherAI/pythia-160m-data-seed{i}" for i in (1, 2, 3)],
    "weight": [f"EleutherAI/pythia-160m-weight-seed{i}" for i in (1, 2, 3)],
}
TOKENIZER_REPO = "EleutherAI/pythia-160m"
CHUNK, N_CHUNKS = 1024, 20


def rho_d(data_losses: list[float], weight_losses: list[float], level: float = 0.95) -> dict[str, Any]:
    """Data-order share of seed variance, s2_D / (s2_D + s2_W), with an F-distribution interval."""

    s2_d, s2_w = float(np.var(data_losses, ddof=1)), float(np.var(weight_losses, ddof=1))
    df_d, df_w = len(data_losses) - 1, len(weight_losses) - 1
    ratio = s2_d / s2_w if s2_w > 0 else float("inf")
    alpha = 1 - level
    lo_ratio = ratio / stats.f.ppf(1 - alpha / 2, df_d, df_w)
    hi_ratio = ratio / stats.f.ppf(alpha / 2, df_d, df_w)
    return {
        "s2_data": s2_d,
        "s2_weight": s2_w,
        "rho_d": ratio / (1 + ratio),
        "rho_d_ci": [lo_ratio / (1 + lo_ratio), hi_ratio / (1 + hi_ratio)],
        "variance_ratio": ratio,
        "variance_ratio_ci": [lo_ratio, hi_ratio],
        "df": [df_d, df_w],
    }


def token_chunks(scratch: Path) -> list[list[int]]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(TOKENIZER_REPO, allow_patterns=["tokenizer*", "special_tokens_map.json"], cache_dir=scratch)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer(TEXT.read_text(encoding="utf-8"))["input_ids"]
    if len(ids) < CHUNK * N_CHUNKS:
        raise ValueError(f"held-out text has {len(ids)} tokens, fewer than {CHUNK * N_CHUNKS}")
    return [ids[i * CHUNK : (i + 1) * CHUNK] for i in range(N_CHUNKS)]


def evaluate(repo: str, chunks: list[list[int]], scratch: Path) -> dict[str, Any]:
    """Download one model, score every chunk, delete the download."""

    import torch
    from huggingface_hub import snapshot_download
    from transformers import GPTNeoXForCausalLM

    target = scratch / "models"
    path = Path(snapshot_download(repo, allow_patterns=["config.json", "pytorch_model.bin"], cache_dir=target))
    try:
        model = GPTNeoXForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        model.eval()
        losses = []
        with torch.no_grad():
            for chunk in chunks:
                ids = torch.tensor([chunk])
                losses.append(float(model(ids, labels=ids).loss))
        weights_sha = hashlib.sha256((path / "pytorch_model.bin").read_bytes()).hexdigest()
    finally:
        shutil.rmtree(target, ignore_errors=True)
    return {"repo": repo, "chunk_losses": losses, "mean_loss": float(np.mean(losses)), "weights_sha256": weights_sha}


def _load() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {r["repo"]: r for r in (json.loads(line) for line in CACHE.read_text().splitlines() if line.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and CACHE.exists():
        CACHE.unlink()
    chunks = token_chunks(args.scratch)
    done = _load()
    for repo in [r for arm in ARMS.values() for r in arm]:
        if repo in done:
            continue
        row = evaluate(repo, chunks, args.scratch)
        with CACHE.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row) + "\n")
        print(f"{repo}: {row['mean_loss']:.5f}", flush=True)
    done = _load()
    data = [done[r]["mean_loss"] for r in ARMS["data"]]
    weight = [done[r]["mean_loss"] for r in ARMS["weight"]]
    per_chunk = [
        rho_d([done[r]["chunk_losses"][k] for r in ARMS["data"]], [done[r]["chunk_losses"][k] for r in ARMS["weight"]])[
            "rho_d"
        ]
        for k in range(N_CHUNKS)
    ]
    out = {
        "sample": {
            "file": "data/raw/crn/heldout_text.txt",
            "sha256": hashlib.sha256(TEXT.read_bytes()).hexdigest(),
            "tokens": CHUNK * N_CHUNKS,
            "chunks": N_CHUNKS,
        },
        "runs": {r: {k: v for k, v in done[r].items() if k != "chunk_losses"} for r in done},
        "arms": {"data_seed_losses": data, "weight_seed_losses": weight},
        "primary": rho_d(data, weight),
        "secondary_per_chunk_rho_d": {"median": float(np.median(per_chunk)), "values": per_chunk},
        "interpretation": (
            "upper bound on CRN's gap-variance reduction when only data order is shared and its effect is perfectly "
            "correlated across recipes; if initialisation is also shared (same architecture, parameters copied "
            "explicitly), the bound is the whole seed variance"
        ),
    }
    primary = out["primary"]
    out["predictions"] = {
        "P2_rho_d_in_0_3_to_0_7": bool(0.3 <= primary["rho_d"] <= 0.7),
        "P2_interval_covers_most_of_0_1": bool((primary["rho_d_ci"][1] - primary["rho_d_ci"][0]) >= 0.5),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(primary, indent=1), out["predictions"])


if __name__ == "__main__":
    main()
