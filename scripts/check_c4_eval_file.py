"""Content check of DataDecide's in-loop C4 evaluation file against public C4-en validation.

The OLMo DataDecide ladder evaluates c4_en-validation on eval-data/perplexity/v3_small_gptneox20b/c4_en/val/
part-0-00000.npy (raw uint16 GPT-NeoX-20B token ids, one EOS-terminated document after another; the
accompanying part-0-00000.csv.gz gives each document's token span, id and source file). This script decodes
every document with the public GPT-NeoX-20B byte-level vocabulary (pure Python, no tokenizer library) and looks
each one up, by SHA-256 of its exact text, in allenai/c4 en validation (all eight shards). Downloaded files are
read as data only; paths come in as arguments.

Run: .venv/bin/python -I scripts/check_c4_eval_file.py TOKENS.npy META.csv.gz TOKENIZER.json C4VAL_DIR OUT.json [PALOMA_C4_VAL_DIR]

With the optional PALOMA_C4_VAL_DIR (the val-*.jsonl.gz files of allenai/paloma c4_en, a gated dataset), the
in-loop documents are also compared with Paloma's by document id and by text hash.
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


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def decoder(tok_path: Path):
    tj = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab = {i: t for t, i in tj["model"]["vocab"].items()}
    added = {a["id"]: a for a in tj.get("added_tokens", [])}
    special = {i for i, a in added.items() if a.get("special")}
    plain = {i: a["content"] for i, a in added.items() if not a.get("special")}  # e.g. whitespace runs
    inv = {c: b for b, c in bytes_to_unicode().items()}

    def decode(ids: np.ndarray) -> tuple[str, int]:
        out, unknown = bytearray(), 0
        for i in ids.tolist():
            if i in special:
                continue  # EOS and other specials are not document text
            if i in plain:
                out.extend(plain[i].encode("utf-8"))
                continue
            t = vocab.get(i)
            if t is None:
                unknown += 1
                continue
            out.extend(inv[c] for c in t)
        return out.decode("utf-8", errors="replace"), unknown

    return decode


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paloma_compare(docs: list[dict], paloma_dir: str) -> dict:
    by_id, texts, n = {}, set(), 0
    for f in sorted(Path(paloma_dir).glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                by_id[str(r.get("id"))] = r["text"]
                texts.add(sha(unicodedata.normalize("NFC", r["text"])))
    ids_match = sum(d["id"] in by_id for d in docs)
    text_match = sum(sha(unicodedata.normalize("NFC", d["text"])) in texts for d in docs)
    return {"paloma_documents": n, "inloop_ids_found": ids_match, "inloop_texts_found_nfc": text_match, "identical_document_set": bool(n == len(docs) == ids_match == text_match)}


def main(tokens: str, meta: str, tok: str, c4dir: str, out: str, paloma_dir: str | None = None) -> dict:
    raw = Path(tokens).read_bytes()
    ids = np.frombuffer(raw, dtype=np.uint16)
    rows = list(csv.reader(io.TextIOWrapper(gzip.open(meta), encoding="utf-8")))
    decode = decoder(Path(tok))
    docs = []
    for start, end, doc_id, src, *_ in rows:
        text, unk = decode(ids[int(start) : int(end)])
        docs.append({"id": doc_id, "source": src, "tokens": int(end) - int(start), "text": text, "unknown_ids": unk})
    exact, stripped, prefix, nfc = {}, {}, {}, {}
    files = sorted(Path(c4dir).glob("c4-validation.*.json.gz"))
    n_c4 = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)["text"]
                n_c4 += 1
                exact[sha(t)] = 1
                stripped[sha(t.strip())] = 1
                prefix[sha(t.strip()[:200])] = 1
                nfc[sha(unicodedata.normalize("NFC", t))] = 1
    hit_exact = sum(sha(d["text"]) in exact for d in docs)
    hit_strip = sum(sha(d["text"].strip()) in stripped for d in docs)
    hit_nfc = sum(sha(unicodedata.normalize("NFC", d["text"])) in nfc for d in docs)
    hit_prefix = sum(sha(d["text"].strip()[:200]) in prefix for d in docs)
    res = {
        "inloop_file": "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy",
        "inloop_sha256": hashlib.sha256(raw).hexdigest(),
        "inloop_tokens": int(ids.size),
        "inloop_documents": len(docs),
        "inloop_metadata_sources": sorted({d["source"] for d in docs}),
        "tokens_in_spans": int(sum(d["tokens"] for d in docs)),
        "ids_outside_neox_vocab": int(sum(d["unknown_ids"] for d in docs)),
        "c4_validation_shards": [f.name for f in files],
        "c4_validation_documents": n_c4,
        "matched_exact": hit_exact,
        "matched_after_strip": hit_strip,
        "matched_after_nfc_both_sides": hit_nfc,
        "matched_first_200_chars": hit_prefix,
        "eos_note": "the one id per document outside the GPT-NeoX-20B vocabulary is 50279, the OLMo tokenizer's EOS appended to each document",
        "unmatched_examples": [{"id": d["id"], "tokens": d["tokens"], "head": d["text"][:120]} for d in docs if sha(d["text"].strip()[:200]) not in prefix][:5],
    }
    res["paloma"] = paloma_compare(docs, paloma_dir) if paloma_dir else {
        "status": "not checked: allenai/paloma is gated (auto) and no Hugging Face credential is available on this machine",
        "to_complete": "accept Paloma's terms, download c4_en/val/val-*.jsonl.gz, rerun with PALOMA_C4_VAL_DIR",
    }
    Path(out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("unmatched_examples", "c4_validation_shards")}, indent=2))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:7])
