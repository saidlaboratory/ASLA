"""Remove evaluation-set documents from a training corpus (PREDICTIONS_REPLICATION_PROTOCOL.md, protocol v2.1).

Reads JSON lines with a "text" field on stdin and writes the lines whose NFC-normalized text does not hash
(SHA-256) to a document of replication_protocol/eval_set.jsonl, DataDecide's in-loop c4_en-validation file.
The ids of the evaluation documents that were matched are written to the --report file, so each corpus's
removals can be audited against the evaluation set.

Run: .venv/bin/python scripts/decontaminate.py --report removed.json < corpus.jsonl > clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, TextIO

REPO = Path(__file__).resolve().parents[1]
EVAL_SET = REPO / "replication_protocol" / "eval_set.jsonl"


def eval_hashes(path: Path = EVAL_SET) -> dict[str, str]:
    """NFC text hash -> evaluation document id."""

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["sha256_nfc"]] = r["id"]
    return out


def nfc_hash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def filter_lines(lines: Iterable[str], hashes: dict[str, str], removed: dict[str, int]) -> Iterator[str]:
    for line in lines:
        if not line.strip():
            continue
        h = nfc_hash(json.loads(line)["text"])
        if h in hashes:
            removed[hashes[h]] = removed.get(hashes[h], 0) + 1
            continue
        yield line if line.endswith("\n") else line + "\n"


def main(argv: list[str] | None = None, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> dict[str, int]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    removed: dict[str, int] = {}
    for line in filter_lines(stdin, eval_hashes(), removed):
        stdout.write(line)
    a.report.write_text(json.dumps({"eval_set": "replication_protocol/eval_set.jsonl", "removed_eval_ids": removed}, indent=2) + "\n", encoding="utf-8")
    return removed


if __name__ == "__main__":
    main()
