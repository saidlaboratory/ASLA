"""Verify that every generated document reproduces from committed JSON.

Regenerates each rendered document into a temporary directory and compares it
byte-for-byte with the committed copy. Nothing is written to the repository, so
this is safe to run on a clean checkout as a reviewer check:

    python scripts/verify_reproducibility.py

Exits non-zero if any document has drifted from the results it claims to be
generated from.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (renderer, arguments, {committed path: regenerated path}). The mapping is
# explicit because two committed documents are both named RESULTS.md, so any
# name-based matching silently compares one against the other.
RENDERERS: list[tuple[str, list[str], dict[str, str]]] = [
    (
        "render_consolidated_findings.py",
        [],
        {
            "results/consolidated/METHODS_PRACTICE.md": "",
            "results/consolidated/MODEL_ERROR_STRUCTURE.md": "",
        },
    ),
    (
        "render_confidence_failures.py",
        ["--out-dir", "{tmp}/confidence"],
        {
            "results/confidence/STUDENT_T_DEFECT.md": "{tmp}/confidence/STUDENT_T_DEFECT.md",
            "results/confidence/CONFIDENCE_FAILURES.md": "{tmp}/confidence/CONFIDENCE_FAILURES.md",
            "results/confidence/CERTIFICATION_COST.md": "{tmp}/confidence/CERTIFICATION_COST.md",
        },
    ),
    (
        "render_allocation_results.py",
        [
            "--results",
            "results/allocation/allocation_1B.json",
            "results/allocation/allocation_60M.json",
            "--out",
            "{tmp}/allocation_RESULTS.md",
        ],
        {"results/allocation/RESULTS.md": "{tmp}/allocation_RESULTS.md"},
    ),
    (
        "render_abstention_results.py",
        ["--results", "results/abstention/abstention_1B.json", "--out", "{tmp}/abstention_RESULTS.md"],
        {"results/abstention/RESULTS.md": "{tmp}/abstention_RESULTS.md"},
    ),
    (
        "render_curvature_results.py",
        ["--results", "results/curvature/curvature_gate.json", "--out", "{tmp}/curvature_RESULTS.md"],
        {"results/curvature/RESULTS.md": "{tmp}/curvature_RESULTS.md"},
    ),
    (
        "render_macros.py",
        ["--out", "{tmp}/macros.tex"],
        {"paper/macros.tex": "{tmp}/macros.tex"},
    ),
]

# Lines that legitimately differ between two renders of the same content.
# macros.tex records when and from what commit it was generated; those lines are
# the point of the header, not drift, so they are excluded from the comparison
# while every \newcommand line is compared exactly.
VOLATILE_PREFIXES = ("% generated:", "% source commit:")


def _comparable(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(VOLATILE_PREFIXES))


def _run(script: str, arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *arguments],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"{script} failed:\n{result.stdout}\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    checked = 0

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        # Snapshot the committed documents, then restore them afterwards, so a
        # renderer that writes in place cannot mask drift by overwriting it.
        snapshot: dict[Path, str] = {}
        for _, _, outputs in RENDERERS:
            for relative in outputs:
                path = REPO / relative
                if path.exists():
                    snapshot[path] = path.read_text(encoding="utf-8")

        try:
            for script, arguments, outputs in RENDERERS:
                _run(script, [a.format(tmp=tmp) for a in arguments])
                for relative, target in outputs.items():
                    committed = REPO / relative
                    if committed not in snapshot:
                        failures.append(f"{relative}: committed copy missing")
                        continue
                    checked += 1
                    if target:
                        regenerated = Path(target.format(tmp=tmp))
                        if not regenerated.exists():
                            failures.append(f"{relative}: renderer produced no output")
                            continue
                        produced = regenerated.read_text(encoding="utf-8")
                    else:
                        # Renderer writes in place; the snapshot is the reference.
                        produced = committed.read_text(encoding="utf-8")
                    if _comparable(produced) != _comparable(snapshot[committed]):
                        failures.append(f"{relative}: differs from committed copy")
                    elif args.verbose:
                        print(f"OK  {relative}")
        finally:
            for path, text in snapshot.items():
                path.write_text(text, encoding="utf-8")

    if failures:
        print(f"{len(failures)} document(s) did not reproduce:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"all {checked} generated documents reproduce from committed JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
