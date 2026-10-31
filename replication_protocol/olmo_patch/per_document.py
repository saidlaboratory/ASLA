"""Per-document loss logging for a complete, deterministic evaluation (ASLA replication protocol v2.5).

Shipped as olmo/eval/per_document.py by olmo_per_document.patch. The evaluation file (N tokens) is read as the
virtual stream [EOS] + file, in windows of ``chunk_size`` tokens with stride ``chunk_size - 1``; window k starts
at virtual position k * stride, and its loss at position j predicts file token ``offset + j`` with
``offset = k * stride``, for j < min(stride, N - offset). So every file token is predicted exactly once, each
with up to chunk_size - 1 tokens of preceding context. Each predicted token is attributed to the document whose
[start, end) span (from the file's part-0-00000.csv.gz metadata) contains it; rows are (instance, document,
summed natural-log loss, token count). Depends on numpy only.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

SCHEMA = "asla-per-document-v2"


def complete_layout(n_tokens: int, chunk_size: int) -> List[Tuple[int, int]]:
    """(offset, number of predicted tokens) for every window covering an n_tokens file."""

    stride = chunk_size - 1
    return [(k * stride, min(stride, n_tokens - k * stride)) for k in range(-(-n_tokens // stride))]


def load_doc_spans(metadata_csv_gz: Union[str, Path, bytes]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(starts, ends, ids) from an OLMo memmap metadata file (start,end,id,source,...)."""

    raw = metadata_csv_gz if isinstance(metadata_csv_gz, bytes) else Path(metadata_csv_gz).read_bytes()
    starts, ends, ids = [], [], []
    for row in csv.reader(io.TextIOWrapper(io.BytesIO(gzip.decompress(raw)), encoding="utf-8")):
        starts.append(int(row[0]))
        ends.append(int(row[1]))
        ids.append(row[2])
    order = np.argsort(starts, kind="stable")
    return np.asarray(starts)[order], np.asarray(ends)[order], [ids[i] for i in order]


class PerDocumentLog:
    def __init__(self, starts: np.ndarray, ends: np.ndarray, ids: Sequence[str], chunk_size: int):
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.ids = list(ids)
        self.chunk_size = int(chunk_size)
        self.rows: Dict[Tuple[int, int], List[float]] = {}

    def reset(self) -> None:
        self.rows = {}

    def add(self, instance: int, offset: int, token_losses: np.ndarray, valid: np.ndarray) -> None:
        """token_losses[j] is the loss predicting file position offset + j; only valid positions are kept."""

        loss = np.asarray(token_losses, dtype=np.float64).reshape(-1)
        ok = np.asarray(valid, dtype=bool).reshape(-1)
        pos = int(offset) + np.arange(loss.size, dtype=np.int64)
        doc = np.searchsorted(self.starts, pos, side="right") - 1
        inside = ok & (doc >= 0) & (pos < self.ends[np.clip(doc, 0, None)])
        for d in np.unique(doc[inside]):
            m = inside & (doc == d)
            key = (int(instance), int(d))
            if key in self.rows:
                raise ValueError(f"instance {instance} logged twice")
            self.rows[key] = [float(loss[m].sum()), int(m.sum())]

    def records(self) -> List[dict]:
        return [
            {"instance": c, "doc": self.ids[d], "loss_sum": v[0], "tokens": v[1]}
            for (c, d), v in sorted(self.rows.items())
        ]

    def write(self, path: Union[str, Path], label: str, step: int, rank: int) -> None:
        header = {"schema": SCHEMA, "label": label, "step": int(step), "rank": int(rank), "chunk_size": self.chunk_size}
        lines = [json.dumps(header)] + [json.dumps(r) for r in self.records()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


__all__ = ["SCHEMA", "PerDocumentLog", "complete_layout", "load_doc_spans"]
