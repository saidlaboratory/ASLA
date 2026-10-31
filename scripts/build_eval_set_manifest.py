"""Write replication_protocol/eval_set.jsonl: the replication's evaluation set, one line per document.

The evaluation set is DataDecide's in-loop c4_en-validation file (OLMo eval-data/perplexity/
v3_small_gptneox20b/c4_en/val/part-0-00000.npy with its part-0-00000.csv.gz metadata). Each line gives the
document's id (from the metadata), its token span, and the SHA-256 of its decoded text, exact and after
Unicode NFC normalization; decontamination (scripts/decontaminate.py) matches training documents on the NFC
hash, then on any shared 13-gram via replication_protocol/eval_13grams.npz (sorted 64-bit 13-gram hashes and the
index of the first evaluation document holding each; protocol v2.2). Decoding is scripts/check_c4_eval_file.py's pure-Python GPT-NeoX-20B decoder. Downloaded files are read
as data only; paths come in as arguments.

Run: .venv/bin/python -I scripts/build_eval_set_manifest.py TOKENS.npy META.csv.gz TOKENIZER.json
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_c4_eval_file import decoder  # noqa: E402
from decontaminate import build_index, normalize_tokens  # noqa: E402

FILE_SHA256 = "c37d6b115de9120bfaefe9ca6a20a310eb6738c20bc26abf469dcada66100ab4"
OUT = REPO / "replication_protocol" / "eval_set.jsonl"
INDEX = REPO / "replication_protocol" / "eval_13grams.npz"


def main(tokens: str, meta: str, tok: str) -> None:
    raw = Path(tokens).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != FILE_SHA256:
        raise SystemExit(f"in-loop file hash {got} is not the pinned {FILE_SHA256}")
    ids = np.frombuffer(raw, dtype=np.uint16)
    decode = decoder(Path(tok))
    lines, texts = [], []
    for start, end, doc_id, *_ in csv.reader(io.TextIOWrapper(gzip.open(meta), encoding="utf-8")):
        text, _ = decode(ids[int(start) : int(end)])
        texts.append(text)
        lines.append(
            json.dumps(
                {
                    "id": doc_id,
                    "token_start": int(start),
                    "token_end": int(end),
                    "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "sha256_nfc": hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest(),
                    "normalized_tokens": len(normalize_tokens(text)),
                }
            )
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    grams, doc = build_index(texts)
    np.savez(INDEX, hashes=grams, doc=doc)
    short = sum(len(normalize_tokens(t)) < 13 for t in texts)
    print(f"{len(lines)} documents -> {OUT}; {grams.size} distinct 13-grams -> {INDEX}; {short} documents under 13 tokens")


if __name__ == "__main__":
    main(*sys.argv[1:4])
