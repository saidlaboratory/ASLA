"""Minimal deterministic GPT trainer for the copied-initialisation CRN pilot.

Common random numbers are made explicit rather than inferred from seeds:

* ``make_init`` builds the model once for a seed index and saves its exact state;
  every learning-rate run loads that file (``load_state_dict(strict=True)``) and
  records the SHA-256 of its parameters before the first step;
* ``make_order`` draws the permutation of training chunks once for a seed index
  and saves it; every run reads batches from that file in order.

Determinism: ``torch.use_deterministic_algorithms(True)``,
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (set before torch is imported),
deterministic cuDNN without benchmarking, no dropout, no data-loader workers,
explicit (non-fused) attention. Each run records the SHA-256 of the batch
sequence it consumed.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

BASE_LR = 1e-3
LR_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5)
SEED_INDICES = (0, 1, 2, 3)


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    context: int = 256
    vocab: int = 256


@dataclass(frozen=True)
class RunConfig:
    lr_multiplier: float
    seed_index: int
    steps: int
    batch_size: int = 64
    base_lr: float = BASE_LR
    warmup_fraction: float = 0.01
    min_lr_fraction: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_checkpoints: int = 10
    eval_batch_size: int = 64
    model: ModelConfig = field(default_factory=ModelConfig)

    def name(self) -> str:
        return f"lr{self.lr_multiplier:g}_seed{self.seed_index}"


def set_determinism(device: str) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda"):
        # TF32 is deterministic but changes numerics between drivers; keep full fp32 matmuls outside autocast.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.out = nn.Linear(4 * cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:t, :t], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss


def _init_weights(model: GPT) -> None:
    """GPT-2 style: matrices N(0, 0.02), residual projections scaled by 1/sqrt(2L); biases 0; LayerNorm weights 1."""

    for name, p in model.named_parameters():
        if p.dim() >= 2:
            std = 0.02 / math.sqrt(2 * model.cfg.n_layer) if name.endswith(("proj.weight", "out.weight")) else 0.02
            nn.init.normal_(p, mean=0.0, std=std)
        elif name.endswith("bias"):
            nn.init.zeros_(p)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_init(seed_index: int, cfg: ModelConfig, path: Path) -> str:
    """Build the model for one seed index and save its exact initial state; returns the parameter hash."""

    torch.manual_seed(1_000_003 * (seed_index + 1))
    model = GPT(cfg)
    _init_weights(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, path)
    return state_hash(model)


def n_chunks(train_bin: Path, context: int) -> int:
    return os.path.getsize(train_bin) // (context + 1)


def make_order(seed_index: int, train_bin: Path, context: int, needed: int, path: Path) -> str:
    """Draw the chunk permutation for one seed index once and save it; returns its hash."""

    total = n_chunks(train_bin, context)
    if needed > total:
        raise ValueError(f"need {needed} training chunks but the corpus has {total}")
    rng = np.random.default_rng([20260927, seed_index])
    order = rng.permutation(total)[:needed].astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, order)
    return file_hash(path)


def _batches(train: np.ndarray, order: np.ndarray, context: int, batch: int, step: int) -> np.ndarray:
    idx = order[step * batch : (step + 1) * batch]
    rows = [train[i * (context + 1) : (i + 1) * (context + 1)] for i in idx]
    return np.stack(rows).astype(np.int64)


def _eval_loss(model: GPT, evaluation: np.ndarray, context: int, batch: int, device: str) -> float:
    model.eval()
    n = len(evaluation) // (context + 1)
    rows = evaluation[: n * (context + 1)].reshape(n, context + 1).astype(np.int64)
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, n, batch):
            chunk = torch.from_numpy(rows[start : start + batch]).to(device)
            with _autocast(device):
                _, loss = model(chunk[:, :-1], chunk[:, 1:])
            assert loss is not None
            total += float(loss) * chunk.size(0)
            count += chunk.size(0)
    model.train()
    return total / count


def _autocast(device: str) -> Any:
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def lr_at(step: int, cfg: RunConfig) -> float:
    peak = cfg.base_lr * cfg.lr_multiplier
    warmup = max(1, int(cfg.warmup_fraction * cfg.steps))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    return peak * (cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


def checkpoints(steps: int, n: int) -> list[int]:
    return sorted({0, *(round(steps * (k + 1) / n) for k in range(n))})


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return "apple-mps" if device.startswith("mps") else "cpu"


def train_run(
    cfg: RunConfig,
    work: Path,
    device: str = "cpu",
    state_dir: Path | None = None,
    stop_after_checkpoint: int | None = None,
) -> dict[str, Any]:
    """One learning-rate run from the saved initial state and data order for its seed index.

    ``work`` holds train.bin and eval.bin; ``state_dir`` (default ``work``) holds the per-seed-index
    initial states and orders. ``stop_after_checkpoint`` ends the run, with the full schedule, after
    that many checkpoints (used by the determinism gate).
    """

    set_determinism(device)
    state_dir = state_dir or work
    context = cfg.model.context
    train = np.memmap(work / "train.bin", dtype=np.uint8, mode="r")
    evaluation = np.fromfile(work / "eval.bin", dtype=np.uint8)
    order = np.load(state_dir / f"order_seed{cfg.seed_index}.npy")
    if len(order) < cfg.steps * cfg.batch_size:
        raise ValueError("saved order is shorter than the run needs")
    torch.manual_seed(0)  # any construction seed: the state is overwritten by the saved one
    model = GPT(cfg.model)
    state = torch.load(state_dir / f"init_seed{cfg.seed_index}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    init_hash = state_hash(model)
    model.to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.base_lr * cfg.lr_multiplier,
        betas=(0.9, 0.95),
        foreach=False,
        fused=False,
    )
    marks = checkpoints(cfg.steps, cfg.n_checkpoints)
    curve = [
        {"step": 0, "eval_loss": _eval_loss(model, evaluation, context, cfg.eval_batch_size, device), "train_loss": None}
    ]
    batch_digest = hashlib.sha256()
    running, running_n = 0.0, 0
    start = time.time()
    for step in range(cfg.steps):
        batch = _batches(train, order, context, cfg.batch_size, step)
        batch_digest.update(batch.tobytes())
        x = torch.from_numpy(batch).to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        with _autocast(device):
            _, loss = model(x[:, :-1], x[:, 1:])
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running += float(loss.detach())
        running_n += 1
        if step + 1 in marks:
            curve.append(
                {
                    "step": step + 1,
                    "eval_loss": _eval_loss(model, evaluation, context, cfg.eval_batch_size, device),
                    "train_loss": running / running_n,
                }
            )
            running, running_n = 0.0, 0
            if stop_after_checkpoint is not None and len(curve) - 1 >= stop_after_checkpoint:
                break
    return {
        "name": cfg.name(),
        "config": asdict(cfg),
        "device": device,
        "device_name": _device_name(device),
        "torch_threads": torch.get_num_threads(),
        "stopped_after_checkpoint": stop_after_checkpoint,
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "init_param_sha256": init_hash,
        "init_file_sha256": file_hash(state_dir / f"init_seed{cfg.seed_index}.pt"),
        "order_file_sha256": file_hash(state_dir / f"order_seed{cfg.seed_index}.npy"),
        "batch_sequence_sha256": batch_digest.hexdigest(),
        "curve": curve,
        "final_eval_loss": curve[-1]["eval_loss"],
        "wall_seconds": time.time() - start,
    }


def probe_step_seconds(
    cfg: RunConfig, work: Path, device: str, n: int = 60, skip: int = 10, state_dir: Path | None = None
) -> float:
    """Mean wall seconds per training step (no evaluation) over ``n - skip`` steps, after ``skip`` warm-up steps."""

    set_determinism(device)
    state_dir = state_dir or work
    train = np.memmap(work / "train.bin", dtype=np.uint8, mode="r")
    order = np.load(state_dir / f"order_seed{cfg.seed_index}.npy")
    model = GPT(cfg.model)
    model.load_state_dict(torch.load(state_dir / f"init_seed{cfg.seed_index}.pt", map_location="cpu", weights_only=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.95), foreach=False, fused=False)
    start = 0.0
    for step in range(n):
        if step == skip:
            _sync(device)
            start = time.time()
        x = torch.from_numpy(_batches(train, order, cfg.model.context, cfg.batch_size, step)).to(device)
        with _autocast(device):
            _, loss = model(x[:, :-1], x[:, 1:])
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _sync(device)
    return (time.time() - start) / (n - skip)


def write_run(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="One pilot run (the launcher calls this in a fresh process).")
    parser.add_argument("--lr-multiplier", type=float, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--n-layer", type=int, default=ModelConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=ModelConfig.n_head)
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--context", type=int, default=ModelConfig.context)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--stop-after-checkpoint", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model = ModelConfig(n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model, context=args.context)
    cfg = RunConfig(
        lr_multiplier=args.lr_multiplier,
        seed_index=args.seed_index,
        steps=args.steps,
        batch_size=args.batch_size,
        model=model,
    )
    result = train_run(cfg, args.work, args.device, args.state_dir, args.stop_after_checkpoint)
    if args.tag:
        result["name"] = f"{result['name']}_{args.tag}"
    print(
        json.dumps(
            {"name": result["name"], "final_eval_loss": result["final_eval_loss"], "wall_seconds": result["wall_seconds"]}
        )
    )
    write_run(result, args.out)


if __name__ == "__main__":
    main()
