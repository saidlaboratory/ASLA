"""Numeric claims in paper/main.tex whose source JSON changed after prose_corrections.json was written.

For every macro the paper uses, the source named in its generated comment is resolved to a results file; if that
file has commits after the commit that wrote prose_corrections.json, the macro's value then (paper/macros.tex at
that commit) and now (the current render) are reported.

Writes ``results/external/paper_macro_drift.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "external" / "paper_macro_drift.json"
MACRO = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\}  % (.+)$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def macros(text: str) -> dict[str, tuple[str, str]]:
    out = {}
    for line in text.splitlines():
        m = MACRO.match(line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def resolve(origin: str) -> list[str]:
    names = re.findall(r"[\w.-]+\.json", origin)
    found = []
    for name in names:
        found += [str(p.relative_to(REPO)) for p in (REPO / "results").rglob(name)]
    return sorted(set(found))


def main() -> None:
    written = git("log", "-1", "--format=%H", "--", "results/external/prose_corrections.json").strip()
    # the commit that first wrote the list (later commits only revise its replacement text)
    first = git("log", "--format=%H", "--diff-filter=A", "--", "results/external/prose_corrections.json").split()[-1]
    then = macros(git("show", f"{first}:paper/macros.tex"))
    now = macros((REPO / "paper" / "macros.tex").read_text(encoding="utf-8"))
    used = set(re.findall(r"\\([A-Za-z]+)", (REPO / "paper" / "main.tex").read_text(encoding="utf-8"))) & set(now)
    rows: list[dict[str, Any]] = []
    for name in sorted(used):
        value, origin = now[name]
        sources = resolve(origin)
        # changed since the reference: committed after it, or modified in the working tree
        changed = {}
        for src in sources:
            if git("diff", "--name-only", first, "--", src).strip():
                last = git("log", "-1", "--format=%h %cs", f"{first}..HEAD", "--", src).strip()
                changed[src] = last or "working tree"
        if changed:
            old = then.get(name, (None, None))[0]
            rows.append(
                {
                    "macro": name,
                    "sources_changed": changed,
                    "old": old,
                    "new": value,
                    "new_claim": old is None,
                    "value_changed": old is not None and old != value,
                }
            )
    out = {
        "reference_commit": first[:10],
        "reference": "the commit that first wrote results/external/prose_corrections.json",
        "latest_commit_touching_list": written[:10],
        "macros_used_in_main_tex": len(used),
        "claims_with_changed_source": len(rows),
        "claims_whose_value_changed": sum(r["value_changed"] for r in rows),
        "new_claims_since_reference": sum(r["new_claim"] for r in rows),
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})
    for r in rows:
        print(f"  {r['macro']:28s} {r['old']!s:>12.40} -> {r['new']!s:.40}  {list(r['sources_changed'])}")


if __name__ == "__main__":
    main()
