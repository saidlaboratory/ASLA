"""Deterministic, complete perplexity evaluation for OLMo (ASLA replication protocol v2.5).

Shipped as olmo/eval/complete_eval.py by olmo_per_document.patch. Each evaluation file is read in fixed order
as overlapping windows (olmo/eval/per_document.py, complete_layout): no shuffling, no dropped chunks, and the
trailing partial window is padded, with the padding masked out of the labels. The sampler gives rank r the
instances r, r + W, r + 2W, ... of a list padded with fully masked dummy instances to a multiple of
W x batch size, so every rank runs the same number of batches and every evaluation token is scored exactly
once whatever the world size W.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..config import DataConfig, TrainConfig
from ..data.collator import DataCollator
from ..torch_util import get_global_rank, get_world_size
from ..util import file_size, get_bytes_range
from .per_document import complete_layout


class CompleteEvalDataset(Dataset):
    def __init__(self, label_paths: List[Tuple[str, str]], chunk_size: int, pad_token_id: int, eos_token_id: int, dtype=np.uint16):
        self.chunk_size, self.pad, self.eos, self.dtype = int(chunk_size), int(pad_token_id), int(eos_token_id), dtype
        self.item_size = np.dtype(dtype).itemsize
        self.items: List[Tuple[str, str, int, int, int, int]] = []
        for label, path in label_paths:
            n = file_size(path) // self.item_size
            for k, (offset, n_pred) in enumerate(complete_layout(n, self.chunk_size)):
                self.items.append((label, str(path), n, k, offset, n_pred))
        self.first_label = label_paths[0][0]

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, path: str, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.zeros(0, dtype=self.dtype)
        buf = get_bytes_range(path, start * self.item_size, (stop - start) * self.item_size)
        return np.frombuffer(buf, dtype=self.dtype)

    def instance(self, i: int) -> Dict[str, Any]:
        label, path, n, k, offset, n_pred = self.items[i]
        # the window is virtual positions [offset, offset + chunk) of [EOS] + file
        if offset == 0:
            toks = np.concatenate([[self.eos], self._read(path, 0, min(n, self.chunk_size - 1))])
        else:
            toks = self._read(path, offset - 1, min(n, offset - 1 + self.chunk_size))
        ids = np.full(self.chunk_size, self.pad, dtype=np.int64)
        ids[: toks.size] = toks
        label_mask = np.zeros(self.chunk_size, dtype=bool)
        label_mask[1 : 1 + n_pred] = True  # label position t is predicted by loss index t - 1 = file offset + t - 1
        return {
            "input_ids": torch.tensor(ids),
            "label_mask": torch.tensor(label_mask),
            "metadata": {"label": label, "instance": k, "predict_offset": offset, "dummy": False},
        }

    def dummy(self) -> Dict[str, Any]:
        return {
            "input_ids": torch.full((self.chunk_size,), self.pad, dtype=torch.long),
            "label_mask": torch.zeros(self.chunk_size, dtype=torch.bool),
            "metadata": {"label": self.first_label, "instance": -1, "predict_offset": -1, "dummy": True},
        }


class _Padded(Dataset):
    def __init__(self, base: CompleteEvalDataset, total: int):
        self.base, self.total = base, total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.base.instance(i) if i < len(self.base) else self.base.dummy()


class PaddedRankSampler(Sampler):
    def __init__(self, n: int, world_size: int, rank: int, batch_size: int):
        per_round = world_size * batch_size
        self.total = -(-n // per_round) * per_round
        self.world_size, self.rank = world_size, rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.total, self.world_size))

    def __len__(self) -> int:
        return self.total // self.world_size


def build_complete_eval_dataloader(train_config: TrainConfig, data_config: DataConfig, batch_size: int) -> DataLoader:
    if data_config.datasets:
        label_paths = [(label, str(p)) for label in sorted(data_config.datasets) for p in data_config.datasets[label]]
    else:
        label_paths = [(str(p), str(p)) for p in (data_config.paths or [])]
    base = CompleteEvalDataset(
        label_paths,
        train_config.model.max_sequence_length,
        train_config.model.pad_token_id,
        train_config.model.eos_token_id,
        dtype=data_config.effective_memmap_dtype,
    )
    sampler = PaddedRankSampler(len(base), get_world_size(), get_global_rank(), batch_size)
    return DataLoader(
        _Padded(base, sampler.total),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id),
        num_workers=data_config.num_workers,
        drop_last=False,
    )


__all__ = ["CompleteEvalDataset", "PaddedRankSampler", "build_complete_eval_dataloader"]
