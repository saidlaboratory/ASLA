"""The frozen replication protocol is complete and its analysis entry point is pinned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "replication_protocol"


def test_selection_is_ten_eligible_corpora_none_from_datadecide():
    rec = json.loads((PROTOCOL / "recipes.json").read_text(encoding="utf-8"))
    assert len(rec["selected"]) == 10
    eligible = {c["hf_id"] for c in rec["candidates"] if c["eligible"]}
    assert {s["hf_id"] for s in rec["selected"]} <= eligible
    banned = {"allenai/c4", "mlfoundations/dclm-baseline-1.0", "allenai/dolma", "tiiuae/falcon-refinedweb", "HuggingFaceFW/fineweb-edu"}
    assert not banned & {s["hf_id"] for s in rec["selected"]}
    assert all(s["revision_sha"] for s in rec["selected"])
    # v2: one RedPajama family member, at least three ordered reserves, all eligible and not selected
    assert sum("RedPajama" in s["hf_id"] for s in rec["selected"]) <= 1
    assert len(rec["reserves"]) >= 3
    assert {r["hf_id"] for r in rec["reserves"]} <= eligible
    assert not {r["hf_id"] for r in rec["reserves"]} & {s["hf_id"] for s in rec["selected"]}


def test_configs_cover_the_design_and_match_datadecide_compute():
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.make_replication_configs import budget, configs

    runs = configs()
    assert len(runs) == 240 and len({r["run_id"] for r in runs}) == 240
    for r in runs:
        assert r["tokens_d"] == r["train"]["steps"] * r["train"]["batch_seqs"] * 2048
    b = budget(runs)
    assert b["total_flops"] == pytest.approx(2.344e21, rel=1e-3)


def test_entry_point_pins_match_and_refuses_incomplete(tmp_path):
    import sys

    sys.path.insert(0, str(REPO))
    from scripts import analyze_replication as entry

    entry.check_pins()
    (tmp_path / "runs.csv").write_text(",".join(entry.RUN_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        entry.load_runs(tmp_path / "runs.csv")


def test_evaluation_and_decontamination_reference_the_in_loop_file():
    import hashlib

    ladder = json.loads((PROTOCOL / "ladder.json").read_text(encoding="utf-8"))
    es = ladder["evaluation"]["eval_set"]
    assert hashlib.sha256((REPO / es["file"]).read_bytes()).hexdigest() == es["sha256"]
    rows = [json.loads(l) for l in (REPO / es["file"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == es["documents"] == 2099 and len({r["sha256_nfc"] for r in rows}) == 2099
    assert "eval_set.jsonl" in ladder["evaluation"]["decontamination"]
    rec = json.loads((PROTOCOL / "recipes.json").read_text(encoding="utf-8"))
    assert "eval_set.jsonl" in rec["rule"]["per_corpus_processing"]
    for f in (PROTOCOL / "configs").glob("*.json"):
        text = f.read_text(encoding="utf-8")
        cfg = json.loads(text)
        assert "paloma" not in text.lower()
        assert cfg["eval"]["eval_set_sha256"] == es["sha256"] == cfg["decontamination"]["eval_set_sha256"]


def _planted():
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.decontaminate import Decontaminator

    eval_a = (
        "The committee met on Tuesday to review the proposed budget for the regional library system, "
        "and members agreed that the reading room renovation should be completed before the winter term begins."
    )
    eval_b = "Short note."  # under 13 tokens: only the exact pass can catch it
    dc = Decontaminator.from_texts(["eval-a", "eval-b"], [eval_a, eval_b])
    return dc, eval_a, eval_b


def test_planted_contamination_is_removed():
    dc, eval_a, eval_b = _planted()
    passage = " ".join(eval_a.split()[3:22])  # a 19-word passage from inside the evaluation document
    embedded = (
        "Local news roundup. Several events took place across the county this week, including a bake sale. "
        + passage
        + " In other news, the football team won its third straight match on Friday night."
    )
    near_dup = (
        "the  committee met on tuesday -- to review the proposed budget, for the regional\nlibrary system; "
        "and members agreed that the reading-room renovation should be completed before the winter term begins!!"
    )
    clean = "An unrelated document about gardening: water tomatoes early in the morning and mulch the beds well."
    lines = [json.dumps({"text": t}) for t in (eval_a, embedded, near_dup, clean, eval_b)]
    kept = list(dc.filter(lines))
    assert [json.loads(l)["text"] for l in kept] == [clean]
    rep = dc.report("planted")
    assert rep["removed_exact"] == 2 and rep["removed_13gram"] == 2 and rep["kept"] == 1
    assert rep["eval_ids_hit_exact"] == {"eval-a": 1, "eval-b": 1}
    assert rep["eval_ids_hit_13gram"] == {"eval-a": 2}


def test_twelve_shared_words_are_not_enough():
    dc, eval_a, _ = _planted()
    twelve = " ".join(eval_a.split()[3:15])
    assert dc.check("Prefix words here. " + twelve + " but then something else entirely.") is None


def test_protocol_index_matches_eval_set():
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.decontaminate import Decontaminator

    dc = Decontaminator.from_protocol()
    assert len(dc.ids) == 2099 and len(dc.exact) == 2099
    assert dc.grams.size == dc.gram_doc.size > 0 and np.all(np.diff(dc.grams.astype(np.float64)) >= 0)
    assert dc.gram_doc.min() >= 0 and dc.gram_doc.max() < 2099


def test_stability_diagnostic_is_output_only():
    import sys

    sys.path.insert(0, str(REPO))
    from scripts import analyze_replication as entry

    src = (REPO / "scripts" / "analyze_replication.py").read_text(encoding="utf-8")
    body = src.split("def main(")[1]
    # the verdict is set by analyse() before stability() runs, and stability's result is only stored
    assert body.index("res = analyse(frame)") < body.index("stability(frame, res)")
    assert 'res["numerical_stability"] = stability(frame, res)' in body
    assert entry.STABILITY_DRAWS == 20 and entry.STABILITY_REL == 1e-12


def _per_document_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("per_document", PROTOCOL / "olmo_patch" / "per_document.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_document_log_attributes_predicted_tokens_to_documents(tmp_path):
    mod = _per_document_module()
    # three documents over two 8-token chunks; position 0 is never predicted
    log = mod.PerDocumentLog(np.array([0, 5, 11]), np.array([5, 11, 16]), ["a", "b", "c"], 8)
    log.add(0, np.arange(1.0, 8.0), np.ones(7, bool))  # predicts positions 1..7
    log.add(1, np.full(7, 10.0), np.array([1, 1, 1, 0, 1, 1, 1], bool))  # positions 9..15, one masked
    rec = {(r["chunk"], r["doc"]): (r["loss_sum"], r["tokens"]) for r in log.records()}
    assert rec == {(0, "a"): (1 + 2 + 3 + 4, 4), (0, "b"): (5 + 6 + 7, 3), (1, "b"): (20.0, 2), (1, "c"): (40.0, 4)}
    log.write(tmp_path / "x.jsonl.gz", "c4_en-validation", 5, 0)


def test_rescore_dedupes_ranks_and_excludes(tmp_path):
    import sys

    sys.path.insert(0, str(REPO))
    from scripts import rescore_replication as rs

    mod = _per_document_module()
    for rank, chunks in ((0, [0]), (1, [1, 0])):  # chunk 0 scored twice
        log = mod.PerDocumentLog(np.array([0, 5, 11]), np.array([5, 11, 16]), ["a", "b", "c"], 8)
        for c in chunks:
            log.add(c, np.full(7, 2.0 + c), np.ones(7, bool))
        log.write(tmp_path / "run" / f"{rs.LABEL}_step9_rank{rank}.jsonl.gz", rs.LABEL, 9, rank)
    files = rs.log_files(tmp_path, "run", 9)
    tot = rs.doc_totals(rs.read_rows(files, 9))
    # chunk 0 (loss 2.0) predicts positions 1..7: a gets 1..4, b gets 5..7; chunk 1 (loss 3.0) predicts 9..15:
    # b gets 9..10, c gets 11..15; the second copy of chunk 0 (rank 1) is ignored
    assert tot == {"a": (8.0, 4), "b": (6.0 + 6.0, 5), "c": (15.0, 5)}
    import math

    assert rs.bits_per_token(tot) == (8.0 + 12.0 + 15.0) / 14 / math.log(2)
    assert rs.bits_per_token(tot, {"b"}) == (8.0 + 15.0) / 9 / math.log(2)


def test_configs_require_the_pinned_evaluator_patch():
    import hashlib

    ladder = json.loads((PROTOCOL / "ladder.json").read_text(encoding="utf-8"))
    pdl = ladder["shared"]["per_document_logging"]
    assert hashlib.sha256((REPO / pdl["patch"]).read_bytes()).hexdigest() == pdl["patch_sha256"]
    patch = (REPO / pdl["patch"]).read_text(encoding="utf-8")
    for f in ("olmo/eval/per_document.py", "olmo/eval/evaluator.py", "olmo/train.py", "olmo/data/memmap_dataset.py", "olmo/config.py", "olmo/eval/__init__.py", "scripts/ladder.py"):
        assert f"+++ b/{f}" in patch
    for f in (PROTOCOL / "configs").glob("*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        p = c["per_document_logging"]
        assert p["required"] and p["patch_sha256"] == pdl["patch_sha256"]
        assert p["env"]["ASLA_PER_DOCUMENT_LOG_DIR"].endswith(c["run_id"])
        assert f"step{c['train']['steps']}_" in p["final_log"]


def test_sensitivity_refuses_without_logs(tmp_path):
    import sys

    sys.path.insert(0, str(REPO))
    from scripts import analyze_replication as entry

    runs = tmp_path / "runs.csv"
    runs.write_text("recipe,size,seed,params_n,tokens_d,c4_en_bits_per_token\ncc100_en,4M,6198,3744832,375193600,1.5\n", encoding="utf-8")
    (tmp_path / "ex.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing the sensitivity analysis"):
        entry.sensitivity(runs, tmp_path / "no_logs", tmp_path / "ex.json")
