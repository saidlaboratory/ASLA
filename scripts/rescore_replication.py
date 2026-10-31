"""Rescore the replication's runs from per-document evaluation logs (PREDICTIONS_REPLICATION_PROTOCOL.md, v2.5).

Each run writes, through replication_protocol/olmo_patch/olmo_per_document.patch (complete, deterministic
evaluation), one gzipped JSON-lines file per rank and evaluation:
<logs>/<run_id>/c4_en-validation_step<step>_rank<rank>.jsonl.gz, a header line then (instance, doc, loss_sum,
tokens) rows. Rescoring uses the final checkpoint (the config's train.steps) and is strict:

* rows from all ranks are pooled; every window of the complete layout must be present exactly once, and every
  evaluation document of replication_protocol/eval_set.jsonl must have exactly its full token count logged,
  or the rescore refuses (every token of every evaluation document is scored exactly once);
* bits per token = sum of loss over sum of tokens over the documents not excluded, taken in sorted
  document-id order, divided by ln 2.

With an empty exclusion list this is the primary c4_en_bits_per_token: ``build`` writes the runs table the entry
point reads. With the v2.3 exclusion list it is the sensitivity rescore. ``exclusions`` derives that list from
scripts/decontaminate.py reports under the v2.3 rule; ``score`` scores one log directory.

Run:
  .venv/bin/python scripts/rescore_replication.py build --logs DIR --out runs.csv [--exclude EXCL.json]
  .venv/bin/python scripts/rescore_replication.py score --dir RUN_LOG_DIR --step STEP [--exclude EXCL.json]
  .venv/bin/python scripts/rescore_replication.py exclusions --reports DIR --out EXCL.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "replication_protocol"
LABEL = "c4_en-validation"
SCHEMA = "asla-per-document-v2"
EVAL_SET = PROTOCOL / "eval_set.jsonl"
TRIGGER_SHARE = 0.03
TRIGGER_MEDIAN_MULTIPLE = 2.0


def protocol_runs() -> dict[str, dict]:
    out = {}
    for f in sorted((PROTOCOL / "configs").glob("*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        out[c["run_id"]] = c
    return out


def log_files_in(run_dir: Path, step: int) -> list[Path]:
    return sorted(Path(run_dir).glob(f"{LABEL}_step{step}_rank*.jsonl.gz"), key=lambda p: int(p.name.split("_rank")[1].split(".")[0]))


def log_files(logs: Path, run_id: str, step: int) -> list[Path]:
    return log_files_in(logs / run_id, step)


def eval_spans(path: Path = EVAL_SET) -> dict[str, int]:
    """Evaluation document id -> its token count."""

    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["id"]] = int(r["token_end"]) - int(r["token_start"])
    return out


def read_rows(files: Iterable[Path], step: int) -> tuple[dict[int, list[tuple[str, float, int]]], int]:
    """instance -> its (doc, loss_sum, tokens) rows, pooled over ranks; refuses an instance logged twice."""

    chunks: dict[int, list[tuple[str, float, int]]] = {}
    chunk_size = None
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("schema") != SCHEMA or header.get("label") != LABEL or int(header.get("step")) != step:
                raise ValueError(f"{f}: unexpected header {header}")
            chunk_size = header["chunk_size"] if chunk_size is None else chunk_size
            if header["chunk_size"] != chunk_size:
                raise ValueError(f"{f}: chunk size {header['chunk_size']} differs from {chunk_size}")
            mine: dict[int, list[tuple[str, float, int]]] = {}
            for line in fh:
                r = json.loads(line)
                mine.setdefault(int(r["instance"]), []).append((r["doc"], float(r["loss_sum"]), int(r["tokens"])))
        for c, rows in mine.items():
            if c in chunks:
                raise ValueError(f"instance {c} logged on more than one rank")
            chunks[c] = rows
    if chunk_size is None:
        raise ValueError("no per-document log files")
    return chunks, int(chunk_size)


def check_complete(chunks: dict[int, list[tuple[str, float, int]]], chunk_size: int, spans: dict[str, int]) -> None:
    n_tokens = sum(spans.values())
    want = -(-n_tokens // (chunk_size - 1))
    if sorted(chunks) != list(range(want)):
        missing = sorted(set(range(want)) - set(chunks))
        raise ValueError(f"incomplete evaluation: {len(missing)} of {want} windows missing, e.g. {missing[:3]}")
    got: dict[str, int] = {}
    for rows in chunks.values():
        for doc, _, n in rows:
            got[doc] = got.get(doc, 0) + n
    bad = [d for d, n in spans.items() if got.get(d, 0) != n]
    extra = sorted(set(got) - set(spans))
    if bad or extra:
        raise ValueError(f"incomplete evaluation: {len(bad)} documents not scored exactly once, {len(extra)} unknown ids")


def doc_totals(chunks: dict[int, list[tuple[str, float, int]]]) -> dict[str, tuple[float, int]]:
    tot: dict[str, list] = {}
    for c in sorted(chunks):
        for doc, loss, n in chunks[c]:
            cur = tot.setdefault(doc, [0.0, 0])
            cur[0] += loss
            cur[1] += n
    return {d: (v[0], v[1]) for d, v in tot.items()}


def bits_per_token(totals: dict[str, tuple[float, int]], exclude: set[str] = frozenset()) -> float:
    loss, n = 0.0, 0
    for d in sorted(totals):
        if d in exclude:
            continue
        loss += totals[d][0]
        n += totals[d][1]
    if n == 0:
        raise ValueError("no tokens left after exclusion")
    return loss / n / math.log(2)


def missing_logs(logs: Path, run_ids: Iterable[str], runs: dict[str, dict] | None = None) -> list[str]:
    runs = runs or protocol_runs()
    return [r for r in run_ids if not log_files(logs, r, runs[r]["train"]["steps"])]


def score_dir(run_dir: Path, step: int, exclude: set[str] = frozenset(), spans: dict[str, int] | None = None) -> float:
    chunks, chunk_size = read_rows(log_files_in(run_dir, step), step)
    check_complete(chunks, chunk_size, spans if spans is not None else eval_spans())
    return bits_per_token(doc_totals(chunks), exclude)


def run_bits(logs: Path, run: dict, exclude: set[str] = frozenset(), spans: dict[str, int] | None = None) -> float:
    return score_dir(logs / run["run_id"], run["train"]["steps"], exclude, spans)


def build_table(logs: Path, run_ids: Iterable[str] | None = None, exclude: set[str] = frozenset()) -> pd.DataFrame:
    runs = protocol_runs()
    ids = list(run_ids) if run_ids is not None else list(runs)
    gone = missing_logs(logs, ids, runs)
    if gone:
        raise SystemExit(f"{len(gone)} runs lack per-document logs, e.g. {gone[:3]}")
    rows = []
    spans = eval_spans()
    for rid in ids:
        r = runs[rid]
        rows.append(
            {
                "recipe": r["recipe"],
                "size": r["size"],
                "seed": r["seed"],
                "params_n": r["params_n"],
                "tokens_d": r["tokens_d"],
                "c4_en_bits_per_token": run_bits(logs, r, exclude, spans),
                "run_id": rid,
            }
        )
    return pd.DataFrame(rows)


def load_exclusion(path: Path | None) -> set[str]:
    if path is None:
        return set()
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(d["exclude_eval_ids"] if isinstance(d, dict) else d)


def exclusions(reports: list[dict]) -> dict:
    """The v2.3 rule: triggering corpora and, per corpus, the eval ids behind over half its 13-gram removals."""

    share = {r["corpus"]: r["removed_13gram"] / r["documents_in"] for r in reports if r["documents_in"]}
    med = statistics.median(share.values()) if share else 0.0
    out = {"median_13gram_share": med, "corpora": {}}
    for r in reports:
        c = r["corpus"]
        s = share.get(c, 0.0)
        if not (s > TRIGGER_SHARE or s > TRIGGER_MEDIAN_MULTIPLE * med):
            continue
        hits = sorted(r["eval_ids_hit_13gram"].items(), key=lambda kv: (-kv[1], kv[0]))
        need, acc, ids = r["removed_13gram"] / 2.0, 0, []
        for eid, n in hits:
            ids.append(eid)
            acc += n
            if acc > need:
                break
        out["corpora"][c] = {"share_13gram": s, "exclude_eval_ids": ids}
    union = sorted({e for v in out["corpora"].values() for e in v["exclude_eval_ids"]})
    out["union"] = union if len(out["corpora"]) > 1 else None
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--logs", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--exclude", type=Path)
    sc = sub.add_parser("score")
    sc.add_argument("--dir", type=Path, required=True)
    sc.add_argument("--step", type=int, required=True)
    sc.add_argument("--exclude", type=Path)
    e = sub.add_parser("exclusions")
    e.add_argument("--reports", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_table(a.logs, exclude=load_exclusion(a.exclude)).to_csv(a.out, index=False, float_format="%.17g")
    elif a.cmd == "score":
        print(json.dumps({"dir": str(a.dir), "step": a.step, "c4_en_bits_per_token": score_dir(a.dir, a.step, load_exclusion(a.exclude))}))
    else:
        reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(a.reports.glob("*.json"))]
        a.out.write_text(json.dumps(exclusions(reps), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
