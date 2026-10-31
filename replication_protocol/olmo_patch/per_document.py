"""Per-document loss logging for OLMo's in-loop perplexity evaluator (ASLA replication protocol v2.4).

Shipped as olmo/eval/per_document.py by olmo_per_document.patch. Evaluation instances are fixed chunks of
``chunk_size`` tokens of the evaluation memmap; the loss at position j of a chunk (j = 0 .. chunk_size - 2)
predicts the token at file position chunk_index * chunk_size + j + 1. Each predicted token is attributed to
the document whose [start, end) token span (from the evaluation file's part-0-00000.csv.gz metadata) contains
that position. Rows are (chunk, document): the chunk index, the document id, the summed natural-log loss and
the token count, so a rescore can drop chunks scored twice (across ranks) and any excluded documents.
Depends on numpy only.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

SCHEMA = "asla-per-document-v1"


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

    def add(self, chunk_index: int, token_losses: np.ndarray, valid: np.ndarray) -> None:
        """token_losses[j] is the loss predicting file position chunk_index * chunk_size + j + 1."""

        loss = np.asarray(token_losses, dtype=np.float64).reshape(-1)
        ok = np.asarray(valid, dtype=bool).reshape(-1)
        pos = int(chunk_index) * self.chunk_size + 1 + np.arange(loss.size, dtype=np.int64)
        doc = np.searchsorted(self.starts, pos, side="right") - 1
        inside = ok & (doc >= 0) & (pos < self.ends[np.clip(doc, 0, None)])
        for d in np.unique(doc[inside]):
            m = inside & (doc == d)
            key = (int(chunk_index), int(d))
            cur = self.rows.setdefault(key, [0.0, 0])
            cur[0] += float(loss[m].sum())
            cur[1] += int(m.sum())

    def records(self) -> List[dict]:
        return [
            {"chunk": c, "doc": self.ids[d], "loss_sum": v[0], "tokens": v[1]}
            for (c, d), v in sorted(self.rows.items())
        ]

    def write(self, path: Union[str, Path], label: str, step: int, rank: int) -> None:
        header = {"schema": SCHEMA, "label": label, "step": int(step), "rank": int(rank), "chunk_size": self.chunk_size}
        lines = [json.dumps(header)] + [json.dumps(r) for r in self.records()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


__all__ = ["SCHEMA", "PerDocumentLog", "load_doc_spans"]
