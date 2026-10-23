"""Apply every entry of ``results/external/prose_corrections.json`` and record what was done.

- paper/main.tex entries: the existing text is located across line breaks (whitespace-normalised) and replaced or
  deleted.
- script entries: the script text is checked to carry the replacement (the script edits are part of the same
  commit); their generated outputs are patched in place on exactly the corrected field, after checking that the
  file re-serialises byte-identically, so no other value is regenerated.

Writes ``results/external/prose_corrections_applied.json``. Run ``scripts/render_macros.py`` afterwards.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CORRECTIONS = REPO / "results" / "external" / "prose_corrections.json"
OUT = REPO / "results" / "external" / "prose_corrections_applied.json"


def _span(raw: str, existing: str) -> tuple[int, int] | None:
    """Raw-text span of ``existing``, matching runs of whitespace to any whitespace."""

    words = existing.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, raw)
    return (m.start(), m.end()) if m else None


def apply_text(raw: str, entry: dict[str, Any]) -> tuple[str, int]:
    span = _span(raw, entry["existing"])
    if span is None:
        raise SystemExit(f"{entry['id']}: existing text not found")
    start, end = span
    line = raw.count("\n", 0, start) + 1
    if entry["action"] == "delete":
        while end < len(raw) and raw[end] == " ":
            end += 1
        return raw[:start] + raw[end:], line
    return raw[:start] + entry["replacement"] + raw[end:], line


def patch_json(path: Path, mutate: Any, **dump: Any) -> bool:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if json.dumps(data, **dump) + "\n" != raw:
        raise SystemExit(f"{path} does not re-serialise byte-identically; regenerate it instead")
    changed = mutate(data)
    path.write_text(json.dumps(data, **dump) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    from scripts.run_published_comparisons import gate_note

    entries = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["entries"]
    paper = REPO / "paper" / "main.tex"
    raw = paper.read_text(encoding="utf-8")
    applied = []
    for entry in entries:
        if entry["file"] == "paper/main.tex":
            raw, line = apply_text(raw, entry)
            applied.append({"id": entry["id"], "file": entry["file"], "action": entry["action"], "line_before": line})
            continue
        script = (REPO / entry["file"]).read_text(encoding="utf-8")
        if entry["id"] == "generated-restated-claims-label":
            assert entry["replacement"] in script and entry["existing"] not in script, entry["id"]
            old, new = "(exploratory: open discrepancy)", "(consistent three-seed target)"

            def relabel(node: Any) -> int:
                n = 0
                items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
                for k, v in list(items):
                    if isinstance(v, str) and old in v:
                        node[k] = v.replace(old, new)
                        n += 1
                    elif isinstance(v, (dict, list)):
                        n += relabel(v)
                return n

            n = patch_json(REPO / "results" / "target_scoring" / "restated_claims.json", relabel, indent=2)
            applied.append(
                {
                    "id": entry["id"],
                    "file": entry["file"],
                    "action": "replace",
                    "outputs_patched": {"results/target_scoring/restated_claims.json": f"{n} claim labels"},
                }
            )
        elif entry["id"] == "generated-published-comparisons-gate-note":
            assert '"d1_gate_note": gate_note(),' in script and entry["existing"] not in script, entry["id"]
            note = gate_note()
            assert note == entry["replacement"], "gate_note() differs from the correction list"

            def renote(d: dict[str, Any]) -> bool:
                d["d1_gate_note"] = note
                return True

            patch_json(REPO / "results" / "external" / "published_comparisons.json", renote, indent=2, sort_keys=True)
            applied.append(
                {
                    "id": entry["id"],
                    "file": entry["file"],
                    "action": "replace",
                    "outputs_patched": {"results/external/published_comparisons.json": "d1_gate_note"},
                }
            )
        else:
            raise SystemExit(f"no handler for {entry['id']}")
    paper.write_text(raw, encoding="utf-8")
    OUT.write_text(
        json.dumps(
            {
                "source": "results/external/prose_corrections.json",
                "applied": applied,
                "count": len(applied),
                "entries": len(entries),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for a in applied:
        print(a)


if __name__ == "__main__":
    main()
