"""The correction list for the prose pass: every sentence that frames the DataDecide non-reproduction as open.

Found by searching every tracked ``.md`` and ``.tex`` file, and the scripts that
generate result documents, for statements that assert a code/paper discrepancy,
call the non-reproduction open, or say the released fits contradict the
paper's description. Each entry gives the existing text (checked verbatim,
whitespace-normalised, in its file), its location, and an accurate
replacement. Paper replacements use generated macros only. Nothing is
rewritten here.

Writes ``results/external/prose_corrections.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "external" / "prose_corrections.json"
SEARCHED = (
    "every git-tracked .md and .tex file (PREDICTIONS_* excluded: pre-registrations are not revised), and scripts/ "
    "and asla/ for generated-document text, with the patterns: discrepan, investigating with the authors, "
    "under investigation, no convention, makes no claim about the paper, reproduc, contradict"
)
CORRECTIONS: list[dict[str, Any]] = [
    {
        "id": "paper-published-1",
        "file": "paper/main.tex",
        "category": ["asserts_discrepancy", "calls_non_reproduction_open"],
        "existing": "There is an open discrepancy, which we are investigating with the authors.",
        "action": "replace",
        "replacement": (
            "DataDecide's own decision-accuracy function, run unmodified on the released predictions, reproduces "
            "\\codeCheckTested\\ of \\codeCheckTestedN\\ task--variant rows and \\codeCheckAll\\ of \\codeCheckAllN\\ "
            "rows overall; each of the other \\ddRemainingRows\\ matches under an ordering of tied predictions, which "
            "their sort leaves arbitrary."
        ),
        "macros": ["codeCheckTested", "codeCheckTestedN", "codeCheckAll", "codeCheckAllN", "ddRemainingRows"],
    },
    {
        "id": "paper-published-2",
        "file": "paper/main.tex",
        "category": ["asserts_discrepancy"],
        "existing": (
            "Using the released target column, \\gateExact\\ of \\gateRows\\ task--variant rows match our "
            "recomputation, and the largest difference is \\gateMaxError."
        ),
        "action": "replace",
        "replacement": (
            "Our earlier recomputation, which scored tied pairs as one half, matched \\gateExact\\ of \\gateRows; "
            "their function orders tied values by its sort and scores every pair as agreeing or not, and differs "
            "from ours only there."
        ),
        "macros": ["gateExact", "gateRows"],
    },
    {
        "id": "paper-published-3",
        "file": "paper/main.tex",
        "category": ["calls_non_reproduction_open", "says_released_fits_contradict_paper"],
        "existing": ("No convention we tried (prediction column, target column or seed mean, tie rule) matches them all."),
        "action": "delete",
        "replacement": "",
        "macros": [],
        "note": "superseded by paper-published-1 and paper-published-2",
    },
    {
        "id": "paper-published-4",
        "file": "paper/main.tex",
        "category": ["calls_non_reproduction_open"],
        "existing": "Until it is resolved, the re-test below is exploratory and makes no claim about the paper's numbers.",
        "action": "replace",
        "replacement": (
            "Their protocol, as implemented in the released code and consistent with the paper's description, scores "
            "single-scale ranking against the three-seed mean at 1B and each scaling-law variant against the default "
            "seed. The re-test below scores both against the three-seed mean. On that consistent target, "
            "\\texttt{3\\_param-1\\_step} reaches \\ddFindingVariant\\ against \\ddFindingSingle\\ for single-scale "
            "ranking at 750M, a difference of \\ddFindingDiff\\ points inside its candidate-level interval "
            "([\\ddFindingLow, \\ddFindingHigh])."
        ),
        "macros": ["ddFindingVariant", "ddFindingSingle", "ddFindingDiff", "ddFindingLow", "ddFindingHigh"],
        "note": (
            "the paper states the single-scale target (section 2.3, Figure 1) and that each scaling-law variant is one "
            "attempt with the default seed (section 3.2); it does not state the scaling-law target separately, hence "
            "'consistent with the paper's description' rather than 'stated in the paper'"
        ),
    },
    {
        "id": "paper-limitations-1",
        "file": "paper/main.tex",
        "category": ["calls_non_reproduction_open", "asserts_discrepancy"],
        "existing": (
            "The scaling-law re-test is exploratory while the discrepancy between the released decision accuracies "
            "and our recomputation from the released predictions is under investigation."
        ),
        "action": "replace",
        "replacement": (
            "The released decision accuracies are reproduced by DataDecide's own code "
            "(\\S\\ref{sec:published-comparisons}); the scaling-law re-test scores both methods against one target, "
            "the three-seed mean, where their protocol uses two."
        ),
        "macros": [],
    },
    {
        "id": "generated-restated-claims-label",
        "file": "scripts/run_restated_claims.py",
        "category": ["calls_non_reproduction_open"],
        "existing": 'f"DataDecide {setup} vs single-scale at 750M (exploratory: open discrepancy)"',
        "action": "replace",
        "replacement": 'f"DataDecide {setup} vs single-scale at 750M (consistent three-seed target)"',
        "macros": [],
        "generated_outputs": ["results/target_scoring/restated_claims.json", "paper/macros.tex (restated claims rows)"],
        "occurrences_in_outputs": {
            "results/target_scoring/restated_claims.json": "(exploratory: open discrepancy)",
            "paper/macros.tex": "(exploratory: open discrepancy)",
        },
        "note": "regenerate restated_claims.json, then paper/macros.tex, in the prose pass",
    },
    {
        "id": "generated-published-comparisons-gate-note",
        "file": "scripts/run_published_comparisons.py",
        "category": ["asserts_discrepancy", "calls_non_reproduction_open"],
        "existing": (
            '"Open discrepancy under investigation with the authors: no convention tried (prediction column, target "'
        ),
        "action": "replace",
        "replacement": (
            "Resolved: DataDecide's compute_decision_accuracy, run unmodified on the released predictions, "
            "reproduces 88/88 tested rows and 5,271/5,280 overall; each of the other 9 matches under an ordering of "
            "tied predictions. Their protocol scores single-scale against the three-seed 1B mean and scaling-law "
            "variants against the default seed; D1 uses the three-seed target for both. See "
            "results/external/datadecide_code_check.json, datadecide_sensitivity.json, datadecide_single_scale.json "
            "and datadecide_paper_check.json."
        ),
        "macros": [],
        "generated_outputs": ["results/external/published_comparisons.json (d1_gate_note)"],
        "occurrences_in_outputs": {"results/external/published_comparisons.json": "Open discrepancy under investigation"},
        "note": "the existing note spans three string literals (lines of d1_gate_note); the whole note is replaced",
    },
]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate(path: Path, existing: str) -> int | None:
    """1-based line on which ``existing`` starts, matching across line breaks; None if absent."""

    raw = path.read_text(encoding="utf-8")
    flat: list[str] = []
    starts: list[int] = []  # offset in ``raw`` of each character of the whitespace-collapsed text
    for i, ch in enumerate(raw):
        if ch.isspace():
            if flat and flat[-1] == " ":
                continue
            ch = " "
        flat.append(ch)
        starts.append(i)
    flat_text = "".join(flat)
    pos = flat_text.find(_normalise(existing))
    if pos < 0:
        return None
    return raw.count("\n", 0, starts[pos]) + 1


def main() -> None:
    entries = []
    for entry in CORRECTIONS:
        line = locate(REPO / entry["file"], entry["existing"])
        if line is None:
            raise SystemExit(f"{entry['id']}: existing text not found in {entry['file']}")
        checked = dict(entry, line=line, verified_present=True)
        for output, snippet in entry.get("occurrences_in_outputs", {}).items():
            path = REPO / output
            checked.setdefault("verified_in_outputs", {})[output] = snippet in path.read_text(encoding="utf-8")
        entries.append(checked)
    macros = (REPO / "paper" / "macros.tex").read_text(encoding="utf-8")
    missing = sorted({m for e in entries for m in e["macros"] if f"\\newcommand{{\\{m}}}" not in macros})
    if missing:
        raise SystemExit(f"replacement macros not generated: {missing}")
    out = {
        "purpose": "correction list for the prose pass; nothing here rewrites prose",
        "searched": SEARCHED,
        "not_in_scope": (
            "KNOWN_ANSWER.md tabulates the released decision_acc values without asserting a discrepancy; "
            "RELATED_WORK.md describes the paper's three-seed target, which is accurate for single-scale ranking"
        ),
        "entries": entries,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for e in entries:
        print(f"{e['id']:44s} {e['file']}:{e['line']} {e['action']} {e.get('verified_in_outputs', '')}")


if __name__ == "__main__":
    main()
