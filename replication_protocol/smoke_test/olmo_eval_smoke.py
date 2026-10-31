"""Smoke test of the ASLA evaluator patch against OLMo itself (replication protocol v2.5). Trains nothing.

Run inside an environment with allenai/OLMo@7094aab installed and replication_protocol/olmo_patch/
olmo_per_document.patch applied:

    python olmo_eval_smoke.py run --eval-file c4_en_val.npy --metadata c4_en_val.csv.gz --out DIR \\
        --world-size W --chunk 2048 --batch 1 --weights DIR/weights.pt
    python olmo_eval_smoke.py reference --eval-file ... --metadata ... --out DIR --chunk 2048 --weights ...

``run`` initializes torch.distributed with the gloo backend on CPU at world size W, builds the patched
evaluator through olmo.eval.build_evaluators, wraps a tiny randomly initialized OLMo model (2 layers, d_model
64) in DDP, and calls the patched Trainer.eval(), which writes per-document logs for every rank. The weights
are created once from a fixed seed and loaded by every run, so all world sizes evaluate the same model.

``reference`` scores the same file independently of OLMo's evaluation code: it walks the virtual stream
[EOS] + file in windows of ``chunk`` tokens with stride ``chunk - 1``, one window at a time, computes the
per-token cross-entropy of the model's logits with torch directly, and writes the total loss, the token count,
the bits per token and per-document totals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

EOS, PAD, VOCAB, EMB = 50279, 1, 50280, 50304
LABEL = "c4_en-validation"


def model_config(chunk: int):
    from olmo.config import ModelConfig

    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=8,
        max_sequence_length=chunk,
        vocab_size=VOCAB,
        embedding_size=EMB,
        eos_token_id=EOS,
        pad_token_id=PAD,
        rope=True,
        weight_tying=False,
        include_bias=False,
        init_device="cpu",
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )


def make_model(chunk: int, weights: Path):
    from olmo.model import OLMo

    model = OLMo(model_config(chunk))
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        torch.manual_seed(20261031)
        model.reset_parameters()
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights)
    model.eval()
    return model


def _worker(rank: int, world: int, a: dict, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(max(1, 4 // world))
    from torch.nn.parallel import DistributedDataParallel as DDP

    from olmo.config import DataConfig, EvaluatorConfig, TokenizerConfig, TrainConfig
    from olmo.eval import build_evaluators
    from olmo.train import Trainer

    chunk = a["chunk"]
    model = make_model(chunk, Path(a["weights"]))
    log_dir = Path(a["out"]) / f"ws{world}"
    cfg = TrainConfig(
        model=model_config(chunk),
        tokenizer=TokenizerConfig(identifier="tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"),  # the ladder's
        device_eval_batch_size=a["batch"],
        precision="fp32",
        fused_loss=False,
        eval_subset_num_batches=-1,
        seed=6198,
        evaluators=[
            EvaluatorConfig(
                label="asla-c4_en-complete",
                complete_eval=True,
                subset_num_batches=-1,
                per_document_log_dir=str(log_dir),
                per_document_metadata={LABEL: a["metadata"]},
                data=DataConfig(drop_last=False, datasets={LABEL: [a["eval_file"]]}, num_workers=0),
            )
        ],
    )
    evaluators = build_evaluators(cfg, torch.device("cpu"))
    trainer = Trainer(
        cfg=cfg,
        model=model,
        dist_model=DDP(model),
        optim=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        train_loader=None,
        device=torch.device("cpu"),
        evaluators=evaluators,
        global_step=a["step"],
    )
    loader = evaluators[0].eval_loader
    info = {"rank": rank, "batches": len(loader), "instances_on_rank": len(loader.sampler), "padded_total": loader.sampler.total, "real_instances": len(loader.dataset.base)}
    metrics = trainer.eval()
    if rank == 0:
        info["olmo_reported_metrics"] = metrics
    Path(log_dir / f"rank{rank}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def run(a: argparse.Namespace) -> None:
    make_model(a.chunk, Path(a.weights))  # create the shared weights before spawning
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(a.world_size, vars(a), port), nprocs=a.world_size, join=True)


def reference(a: argparse.Namespace) -> None:
    import csv
    import gzip
    import io

    torch.set_num_threads(4)
    model = make_model(a.chunk, Path(a.weights))
    toks = np.fromfile(a.eval_file, dtype=np.uint16).astype(np.int64)
    v = np.concatenate([[EOS], toks])
    n, stride = toks.size, a.chunk - 1
    rows = list(csv.reader(io.TextIOWrapper(gzip.open(a.metadata), encoding="utf-8")))
    starts = np.array([int(r[0]) for r in rows])
    ids = [r[2] for r in rows]
    per_tok = np.zeros(n)
    seen = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for k in range(-(-n // stride)):
            off = k * stride
            x = torch.tensor(v[off : off + a.chunk])[None, :]
            logits = model(input_ids=x).logits[0, :-1].float()
            m = min(stride, n - off)
            ce = F.cross_entropy(logits[:m], x[0, 1 : 1 + m], reduction="none").double().numpy()
            per_tok[off : off + m] = ce
            seen[off : off + m] += 1
    doc = np.searchsorted(starts, np.arange(n), side="right") - 1
    totals = {ids[d]: [float(per_tok[doc == d].sum()), int((doc == d).sum())] for d in range(len(ids))}
    out = {
        "tokens": int(n),
        "every_token_predicted_once": bool((seen == 1).all()),
        "loss_sum": float(per_tok.sum()),
        "bits_per_token": float(per_tok.sum() / n / math.log(2)),
        "per_document": totals,
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "reference.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_document"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reference"])
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--step", type=int, default=0)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else reference(a)


if __name__ == "__main__":
    main()
