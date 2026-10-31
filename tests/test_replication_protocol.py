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
