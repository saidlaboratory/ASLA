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
