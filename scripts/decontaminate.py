"""Remove evaluation-set text from a training corpus (PREDICTIONS_REPLICATION_PROTOCOL.md, protocol v2.2).

The evaluation set is DataDecide's in-loop c4_en-validation file, listed in replication_protocol/eval_set.jsonl
and indexed by 13-gram in replication_protocol/eval_13grams.npz. Two passes, in order:

1. exact: a training document whose SHA-256 of NFC-normalized text equals an evaluation document's is removed;
2. 13-gram: a remaining training document that shares at least one 13-gram with any evaluation document is
   removed. Text is normalized before tokenizing (NFC, casefold, every character that is not a letter or digit
   replaced by a space) and tokenized on whitespace; a 13-gram is 13 consecutive tokens joined by single
   spaces, hashed to 64 bits (blake2b, digest_size 8, little-endian). Evaluation documents shorter than 13
   tokens have no 13-grams and are covered by the exact pass only.

The --report file gives, for the corpus, how many documents each pass removed and which evaluation ids each
pass hit (with counts).

Run: .venv/bin/python scripts/decontaminate.py --corpus NAME --report removed.json < corpus.jsonl > clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVAL_SET = REPO / "replication_protocol" / "eval_set.jsonl"
EVAL_INDEX = REPO / "replication_protocol" / "eval_13grams.npz"
N = 13


def normalize_tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch if ch.isalnum() else " " for ch in t).split()


def ngram_hashes(text: str, n: int = N) -> np.ndarray:
    tok = normalize_tokens(text)
    out = np.empty(max(0, len(tok) - n + 1), dtype=np.uint64)
    for i in range(len(out)):
        d = hashlib.blake2b(" ".join(tok[i : i + n]).encode("utf-8"), digest_size=8).digest()
        out[i] = int.from_bytes(d, "little")
    return out


def nfc_hash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def build_index(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique 13-gram hashes and, for each, the index of the first evaluation document containing it."""

    hs, docs = [], []
    for j, t in enumerate(texts):
        h = np.unique(ngram_hashes(t))
        hs.append(h)
        docs.append(np.full(h.size, j, dtype=np.int32))
    h = np.concatenate(hs) if hs else np.empty(0, dtype=np.uint64)
    d = np.concatenate(docs) if docs else np.empty(0, dtype=np.int32)
    order = np.lexsort((d, h))
    h, d = h[order], d[order]
    first = np.ones(h.size, dtype=bool)
    first[1:] = h[1:] != h[:-1]
    return h[first], d[first]


class Decontaminator:
    def __init__(self, ids: Sequence[str], exact: dict[str, int], grams: np.ndarray, gram_doc: np.ndarray):
        self.ids, self.exact, self.grams, self.gram_doc = list(ids), exact, grams, gram_doc
        self.stats = {"documents_in": 0, "kept": 0, "removed_exact": 0, "removed_13gram": 0}
        self.hits_exact: dict[str, int] = {}
        self.hits_13gram: dict[str, int] = {}

    @classmethod
    def from_protocol(cls) -> "Decontaminator":
        rows = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()]
        idx = np.load(EVAL_INDEX, allow_pickle=False)
        return cls([r["id"] for r in rows], {r["sha256_nfc"]: j for j, r in enumerate(rows)}, idx["hashes"], idx["doc"])

    @classmethod
    def from_texts(cls, ids: Sequence[str], texts: Sequence[str]) -> "Decontaminator":
        g, d = build_index(texts)
        return cls(ids, {nfc_hash(t): j for j, t in enumerate(texts)}, g, d)

    def check(self, text: str) -> tuple[str, str] | None:
        """(pass, evaluation id) that removes this document, or None to keep it."""

        j = self.exact.get(nfc_hash(text))
        if j is not None:
            return "exact", self.ids[j]
        h = ngram_hashes(text)
        if h.size and self.grams.size:
            pos = np.searchsorted(self.grams, h)
            pos = np.minimum(pos, self.grams.size - 1)
            match = self.grams[pos] == h
            if match.any():
                return "13gram", self.ids[int(self.gram_doc[pos[np.argmax(match)]])]
        return None

    def filter(self, lines: Iterable[str]) -> Iterator[str]:
        for line in lines:
            if not line.strip():
                continue
            self.stats["documents_in"] += 1
            hit = self.check(json.loads(line)["text"])
            if hit is None:
                self.stats["kept"] += 1
                yield line if line.endswith("\n") else line + "\n"
                continue
            rule, eid = hit
            self.stats[f"removed_{rule}"] += 1
            book = self.hits_exact if rule == "exact" else self.hits_13gram
            book[eid] = book.get(eid, 0) + 1

    def report(self, corpus: str) -> dict:
        return {
            "corpus": corpus,
            "eval_set": "replication_protocol/eval_set.jsonl",
            "eval_index": "replication_protocol/eval_13grams.npz",
            "rule": "exact NFC-text SHA-256 match, then any shared 13-gram over normalized whitespace tokens",
            **self.stats,
            "eval_ids_hit_exact": dict(sorted(self.hits_exact.items())),
            "eval_ids_hit_13gram": dict(sorted(self.hits_13gram.items())),
        }


def main(argv: list[str] | None = None, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    dc = Decontaminator.from_protocol()
    for line in dc.filter(stdin):
        stdout.write(line)
    rep = dc.report(a.corpus)
    a.report.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
