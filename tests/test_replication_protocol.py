"""The frozen replication protocol is complete and its analysis entry point is pinned."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_decontaminate_removes_eval_documents_by_nfc_hash(tmp_path):
    import io
    import sys
    import unicodedata

    sys.path.insert(0, str(REPO))
    from scripts import decontaminate

    hashes = decontaminate.eval_hashes()
    target_hash, target_id = next(iter(hashes.items()))
    # a synthetic document whose NFC hash is forced to match, plus an ordinary one
    fake = {target_hash: target_id, decontaminate.nfc_hash(unicodedata.normalize("NFD", "café")): "x"}
    removed: dict = {}
    lines = [json.dumps({"text": "café"}), json.dumps({"text": "kept"})]
    out = list(decontaminate.filter_lines(lines, fake, removed))
    assert removed == {"x": 1} and len(out) == 1 and "kept" in out[0]


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
