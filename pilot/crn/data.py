"""Fixed corpus sample for the CRN pilot: TinyStories as bytes.

Training text: the first training shard of ``roneneldan/TinyStories`` at a pinned
revision (249 MB parquet). Evaluation text: the first ``EVAL_SEQUENCES``
sequences of the validation split (10 MB parquet). Stories are UTF-8 encoded
and joined with a blank line. Both downloads are checked against pinned
SHA-256 digests; the byte files' digests go into the manifest.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILES = {
    "train": (
        "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b",
    ),
    "validation": (
        "data/validation-00000-of-00001-869c898b519ad725.parquet",
        "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab",
    ),
}
SEPARATOR = b"\n\n"
EVAL_SEQUENCES = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, raw_dir: Path) -> Path:
    remote, expected = FILES[name]
    target = raw_dir / Path(remote).name
    if not target.exists() or sha256(target) != expected:
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}"
        urllib.request.urlretrieve(url, target)
    if sha256(target) != expected:
        raise RuntimeError(f"{target} does not match its pinned SHA-256")
    return target


def stories_to_bytes(stories: Iterable[str]) -> bytes:
    return SEPARATOR.join(s.encode("utf-8") for s in stories)


def write_bytes(data: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.frombuffer(data, dtype=np.uint8).tofile(path)
    return sha256(path)


def prepare(work: Path, context: int) -> dict[str, str]:
    """Download (if needed), convert to byte files, and return their digests."""

    import pyarrow.parquet as pq

    raw = work / "raw"
    train_stories = pq.read_table(download("train", raw), columns=["text"]).column("text").to_pylist()
    valid_stories = pq.read_table(download("validation", raw), columns=["text"]).column("text").to_pylist()
    eval_bytes = stories_to_bytes(valid_stories)[: EVAL_SEQUENCES * (context + 1)]
    if len(eval_bytes) < EVAL_SEQUENCES * (context + 1):
        raise RuntimeError("validation split is shorter than the evaluation sample")
    return {
        "train_bin_sha256": write_bytes(stories_to_bytes(train_stories), work / "train.bin"),
        "eval_bin_sha256": write_bytes(eval_bytes, work / "eval.bin"),
        "source": f"{REPO}@{REVISION}",
    }
