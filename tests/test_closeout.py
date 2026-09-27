"""Tests for the close-out records: the correction-list locator and its entries."""

from __future__ import annotations

import importlib
from pathlib import Path

corrections = importlib.import_module("scripts.build_prose_corrections")


def test_locate_matches_across_line_breaks(tmp_path: Path) -> None:
    doc = tmp_path / "doc.tex"
    doc.write_text("first line\nThere is an open\n  discrepancy here.\nlast\n", encoding="utf-8")
    assert corrections.locate(doc, "There is an open discrepancy here.") == 2
    assert corrections.locate(doc, "not present") is None


def test_every_correction_has_a_replacement_or_is_a_deletion() -> None:
    for entry in corrections.CORRECTIONS:
        assert entry["action"] in ("replace", "delete")
        assert (entry["action"] == "delete") == (entry["replacement"] == "")
        assert entry["category"]


def test_paper_replacements_use_only_macros_for_numbers() -> None:
    import re

    for entry in corrections.CORRECTIONS:
        if entry["file"] == "paper/main.tex":
            text = entry["replacement"].replace("\\texttt{3\\_param-1\\_step}", "")  # a setup name
            text = re.sub(r"\\[A-Za-z]+", "", text)  # macros
            text = re.sub(r"\b\d+[MB]\b", "", text)  # scale labels are structural
            assert not re.search(r"\d", text), entry["id"]
