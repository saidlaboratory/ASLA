"""Dry run of per-document logging and the decontamination-sensitivity rescore (protocol v2.4). Trains nothing.

Synthetic per-document logs for all 240 protocol runs are written through the evaluator patch's own
accumulator (replication_protocol/olmo_patch/per_document.py), on the real evaluation file's layout: the 2,099
document spans of replication_protocol/eval_set.jsonl, 2048-token chunks, 498 scored chunks. Run-level losses
follow one simulated world of the frozen design (scripts/dry_run_replication.py, world 0). Each run is scored on
a simulated world size, drops a few chunks (DistributedSampler drop_last) and scores one chunk on two ranks.

Checks:
1. rescoring with an empty exclusion list reproduces the primary bits per token exactly (bitwise, also after the
   CSV round trip), and agrees with the evaluator's mean of per-chunk means to rounding;
2. a planted exclusion (documents wholly inside one chunk, which one recipe's runs never score, and which one
   contaminated recipe's runs find easy) changes exactly the runs that scored those documents, by the amount an
   independent recomputation from the raw token losses gives, and leaves the others bitwise unchanged;
3. the v2.3 exclusion rule, applied to synthetic decontamination reports, recovers the planted documents;
4. the entry point's sensitivity mode runs, and refuses when one run's logs are missing.

Run: nice -n 19 .venv/bin/python scripts/dry_run_rescore.py LOG_DIR
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import rescore_replication as rs  # noqa: E402

OUT = REPO / "results" / "replication" / "rescore_dry_run.json"
CHUNK = 2048
SEED = 20261031
CONTAMINATED = "cc100_en"  # its runs find the planted documents easy
NEVER_SCORES = "txt360"  # its runs drop the planted chunk


def per_document_module():
    spec = importlib.util.spec_from_file_location("per_document", REPO / "replication_protocol" / "olmo_patch" / "per_document.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_layout():
    rows = [json.loads(l) for l in (REPO / "replication_protocol" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = np.array([r["token_start"] for r in rows])
    ends = np.array([r["token_end"] for r in rows])
    ids = [r["id"] for r in rows]
    n_chunks = int(ends.max()) // CHUNK
    return starts, ends, ids, n_chunks


def planted_chunk(starts, ends, n_chunks) -> tuple[int, list[int]]:
    """The first chunk holding at least two documents wholly inside its predicted positions."""

    for c in range(n_chunks):
        lo, hi = c * CHUNK + 1, (c + 1) * CHUNK
        inside = [i for i in range(len(starts)) if starts[i] >= lo and ends[i] <= hi]
        if len(inside) >= 2:
            return c, inside
    raise RuntimeError("no chunk with two whole documents")


def run_bpb_targets() -> dict[str, float]:
    d = importlib.import_module("scripts.dry_run_replication")
    cells = d.draw(d.spec(), np.random.default_rng([d.SEED, 0]))
    table = d.runs_table(cells)
    return {f"{r}__{s}__seed{k}": float(b) for r, s, k, b in zip(table["recipe"], table["size"], table["seed"], table["c4_en_bits_per_token"])}


def synth_logs(logs: Path, pd_mod, starts, ends, ids, n_chunks, planted_c, planted_docs) -> dict:
    runs = rs.protocol_runs()
    target = run_bpb_targets()
    rng = np.random.default_rng(SEED)
    doc_effect = rng.normal(0.0, 0.15, len(ids))
    pos_all = np.arange(1, n_chunks * CHUNK, dtype=np.int64)
    doc_of = np.searchsorted(starts, pos_all, side="right") - 1
    raw = {}
    for rid, cfg in runs.items():
        b = target[rid] * math.log(2)
        tok = b * (1.0 + doc_effect[doc_of]) * np.exp(0.1 * rng.standard_normal(doc_of.size))
        if cfg["recipe"] == CONTAMINATED:
            tok[np.isin(doc_of, planted_docs)] *= 0.3  # memorized evaluation text
        losses = np.zeros((n_chunks, CHUNK - 1))
        for c in range(n_chunks):
            losses[c] = tok[c * CHUNK : (c + 1) * CHUNK - 1]  # position c*CHUNK+1+j -> index c*CHUNK+j
        world = int(rng.choice([1, 2, 4, 8, 16]))
        dropped = set(rng.choice(n_chunks, size=int(rng.integers(0, 8)), replace=False).tolist())
        if cfg["recipe"] == NEVER_SCORES:
            dropped.add(planted_c)
        else:
            dropped.discard(planted_c)
        scored = [c for c in range(n_chunks) if c not in dropped]
        dup = int(rng.choice(scored))
        per_rank = {k: [] for k in range(world + 1)}
        for i, c in enumerate(scored):
            per_rank[i % world].append(c)
        per_rank[world].append(dup)  # the same chunk scored again on an extra rank
        step = cfg["train"]["steps"]
        for rank, chunks in per_rank.items():
            log = pd_mod.PerDocumentLog(starts, ends, ids, CHUNK)
            for c in chunks:
                log.add(c, losses[c], np.ones(CHUNK - 1, dtype=bool))
            log.write(logs / rid / f"{rs.LABEL}_step{step}_rank{rank}.jsonl.gz", rs.LABEL, step, rank)
        raw[rid] = {"losses": losses, "scored": scored}
    return raw


def main(log_dir: str) -> dict:
    logs = Path(log_dir)
    if logs.exists():
        shutil.rmtree(logs)
    pd_mod = per_document_module()
    starts, ends, ids, n_chunks = eval_layout()
    pc, pdocs = planted_chunk(starts, ends, n_chunks)
    planted_ids = [ids[i] for i in pdocs]
    raw = synth_logs(logs, pd_mod, starts, ends, ids, n_chunks, pc, pdocs)
    runs = rs.protocol_runs()
    run_ids = list(runs)

    # 1. empty exclusion = primary, exactly
    primary = rs.build_table(logs, run_ids)
    again = rs.build_table(logs, run_ids, set())
    csv_path = logs / "runs.csv"
    primary.to_csv(csv_path, index=False, float_format="%.17g")
    reread = pd.read_csv(csv_path, float_precision="round_trip")
    evaluator_style = np.array(
        [np.mean([raw[r]["losses"][c].mean() for c in raw[r]["scored"]]) / math.log(2) for r in run_ids]
    )
    rel = np.abs(primary["c4_en_bits_per_token"].to_numpy() / evaluator_style - 1)

    # 2. planted exclusion
    excl = set(planted_ids)
    sens = rs.build_table(logs, run_ids, excl)
    changed = primary["c4_en_bits_per_token"].to_numpy() != sens["c4_en_bits_per_token"].to_numpy()
    scored_planted = np.array([pc in raw[r]["scored"] for r in run_ids])
    pos_all = np.arange(1, n_chunks * CHUNK, dtype=np.int64)
    doc_of = np.searchsorted(starts, pos_all, side="right") - 1
    keep_tok = ~np.isin(doc_of, pdocs)
    indep = []
    for r in run_ids:
        tot = n = 0.0
        for c in raw[r]["scored"]:
            m = keep_tok[c * CHUNK : (c + 1) * CHUNK - 1]
            tot += raw[r]["losses"][c][m].sum()
            n += m.sum()
        indep.append(tot / n / math.log(2))
    indep_rel = np.abs(sens["c4_en_bits_per_token"].to_numpy() / np.array(indep) - 1)
    contaminated = np.array([runs[r]["recipe"] == CONTAMINATED for r in run_ids])
    shift = sens["c4_en_bits_per_token"].to_numpy() - primary["c4_en_bits_per_token"].to_numpy()

    # 3. the v2.3 rule on synthetic decontamination reports
    reps = []
    for j, rec in enumerate(sorted({c["recipe"] for c in runs.values()})):
        if rec == CONTAMINATED:
            hits = {planted_ids[0]: 40, planted_ids[1]: 30, ids[0]: 20, ids[1]: 10}
            n13 = 100
        else:
            hits = {ids[2 + j]: 5}
            n13 = 5
        reps.append({"corpus": rec, "documents_in": 2000 if rec == CONTAMINATED else 1000, "removed_13gram": n13, "eval_ids_hit_13gram": hits})
    rule = rs.exclusions(reps)
    excl_path = logs / "exclusions.json"
    excl_path.write_text(json.dumps(rule, indent=2), encoding="utf-8")

    # 4. entry point: sensitivity mode, and refusal on missing logs
    entry = importlib.import_module("scripts.analyze_replication")
    with contextlib.redirect_stdout(io.StringIO()):
        res = entry.main([str(csv_path), "--no-stability", "--per-document-logs", str(logs), "--sensitivity-exclude", str(excl_path)])
    broken = logs.parent / (logs.name + "_missing_one")
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(logs, broken)
    shutil.rmtree(broken / run_ids[0])
    refused = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            entry.main([str(csv_path), "--no-stability", "--per-document-logs", str(broken), "--sensitivity-exclude", str(excl_path)])
    except SystemExit as e:
        refused = str(e)
    shutil.rmtree(broken)

    sens_set = res["decontamination_sensitivity"]["sets"]
    out = {
        "note": "synthetic per-document logs through the evaluator patch's accumulator; nothing trained",
        "runs": len(run_ids),
        "scored_chunks_per_run": [int(min(len(raw[r]["scored"]) for r in run_ids)), int(max(len(raw[r]["scored"]) for r in run_ids))],
        "empty_exclusion_reproduces_primary_bitwise": bool((primary["c4_en_bits_per_token"].to_numpy() == again["c4_en_bits_per_token"].to_numpy()).all()),
        "csv_round_trip_bitwise": bool((reread["c4_en_bits_per_token"].to_numpy() == primary["c4_en_bits_per_token"].to_numpy()).all()),
        "max_rel_diff_vs_evaluator_mean_of_chunk_means": float(rel.max()),
        "planted": {
            "chunk": pc,
            "eval_ids": planted_ids,
            "runs_that_scored_them": int(scored_planted.sum()),
            "runs_changed": int(changed.sum()),
            "changed_exactly_the_runs_that_scored_them": bool((changed == scored_planted).all()),
            "unchanged_runs_bitwise_identical": bool((~changed[~scored_planted]).all()),
            "max_rel_diff_vs_independent_recomputation": float(indep_rel.max()),
            "mean_shift_contaminated_recipe": float(shift[contaminated & scored_planted].mean()),
            "mean_shift_other_scoring_runs": float(shift[~contaminated & scored_planted].mean()),
        },
        "v2_3_rule": {
            "triggered": sorted(rule["corpora"]),
            "median_13gram_share": rule["median_13gram_share"],
            "recovered_planted_ids": rule["corpora"].get(CONTAMINATED, {}).get("exclude_eval_ids") == planted_ids,
        },
        "entry_point": {
            "primary_verdict": res["verdict"],
            "primary_estimate": res["estimate"],
            "primary_runs_match_logs_exactly": res["decontamination_sensitivity"]["primary_runs_match_logs_exactly"],
            "sensitivity": {k: {"estimate": v["estimate"], "verdict": v["verdict"]} for k, v in sens_set.items()},
            "refused_when_one_run_lacks_logs": refused is not None and refused.startswith("refusing the sensitivity analysis"),
            "refusal_message": refused,
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
